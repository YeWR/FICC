from tools import consist_loss_func
import torch.nn.functional as F
from model import ResidualBlock, DecoderMultiScope

import torch
from torch import nn
from torch import optim
from config import config
from model import RepresentationNetwork
import matplotlib.pyplot as plt
from matplotlib.pyplot import close
from model import Decoder, LatentActionGen, Dynamic
from transform import Transforms
import os


class FICC(nn.Module):
	def __init__(self, name='naive', num_channels=None, num_blocks=None, transform=None):
		if num_channels is None:
			num_channels = config.channel
		if num_blocks is None:
			num_blocks = config.num_blocks
		super(FICC, self).__init__()
		self.name = name
		self.encoder = RepresentationNetwork(config.observation_shape,
		                                     num_blocks=num_blocks,
		                                     num_channels=num_channels,
		                                     downsample=True,
		                                     momentum=config.bn_momentum)
		self.decoder = Decoder()
		self.dynamic = Dynamic(num_channels, config.latent_action_dim, num_blocks=num_blocks)
		self.delta_dynamic = Dynamic(num_channels, config.latent_action_dim, num_blocks=num_blocks)
		self.recon_dynamic = ResidualBlock(num_channels, num_channels)
		self.decoder_delta = DecoderMultiScope()
		self.lag = LatentActionGen(num_embeddings=config.num_embeddings,
		                           in_channel=num_channels,
		                           vq_in_channel=5,
		                           embedding_channel=config.latent_action_dim,
		                           num_blocks=num_blocks)

		self.num_channels = num_channels
		self.transform = transform
		self.resize = Transforms(['resize'])

		self.optim = optim.Optimizer(self.parameters(), {})
		self.loss = nn.CosineSimilarity()

		# Atari
		self.proj_hid = 512
		self.proj_out = 512
		self.pred_hid = 256
		self.pred_out = 512

		# # default
		# self.proj_hid = 256
		# self.proj_out = 256
		# self.pred_hid = 64
		# self.pred_out = 256

		self.projection_in_dim = num_channels * config.state_size
		official = False
		self.projection = nn.Sequential(
			nn.Linear(self.projection_in_dim, self.proj_hid, bias=not official),
			nn.BatchNorm1d(self.proj_hid),
			nn.ReLU(),
			nn.Linear(self.proj_hid, self.proj_hid, bias=not official),
			nn.BatchNorm1d(self.proj_hid),
			nn.ReLU(),
			nn.Linear(self.proj_hid, self.proj_out),
			nn.BatchNorm1d(self.proj_out, affine=not official)
		)
		self.projection_head = nn.Sequential(
			nn.Linear(self.proj_out, self.pred_hid, bias=not official),
			nn.BatchNorm1d(self.pred_hid),
			nn.ReLU(),
			nn.Linear(self.pred_hid, self.pred_out),
		)

		self.img_cnt = 0

	def set_optimizer(self, lr=None, momentum=None, weight_decay=None):
		if lr is None:
			lr = config.lr
		if momentum is None:
			momentum = config.momentum
		if weight_decay is None:
			weight_decay = config.weight_decay
		# log(self.name + ' setting optimizer ', config.optim, lr, momentum, weight_decay))
		if config.optim is optim.SGD:
			self.optim = optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
		elif config.optim is optim.Adam:
			self.optim = optim.Adam(self.parameters(), lr=lr)
		elif config.optim is optim.AdamW:
			self.optim = optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
		else:
			raise NotImplementedError(str(config.optim))

	def project(self, hidden_state, with_grad=True):
		# only the branch of proj + pred can share the gradients
		hidden_state = hidden_state.view(-1, self.projection_in_dim)
		proj = self.projection(hidden_state)

		# with grad, use proj_head
		if with_grad:
			proj = self.projection_head(proj)
			return proj
		else:
			# TODO: use eval mode?
			return proj.detach()

	def save(self, file_name=''):
		if not file_name:
			file_name = self.name
		if not os.path.exists('save/%s' % file_name):
			os.makedirs('save/%s' % file_name)

		torch.save(self.encoder.state_dict(), 'save/%s/representation.pkl' % file_name)
		torch.save(self.decoder.state_dict(), 'save/%s/decoder.pkl' % file_name)
		torch.save(self.dynamic.state_dict(), 'save/%s/dynamics.pkl' % file_name)
		torch.save(self.lag.state_dict(), 'save/%s/lag.pkl' % file_name)
		torch.save(self.projection.state_dict(), 'save/%s/proj.pkl' % file_name)
		torch.save(self.projection_head.state_dict(), 'save/%s/proj_h.pkl' % file_name)
		torch.save(self.state_dict(), 'save/%s/model.pkl' % file_name)

	def restore(self, file_name='', strict=True):
		if not file_name:
			file_name = self.name
		if not os.path.exists('save/%s' % file_name):
			raise FileNotFoundError('restore(): can not find file [%s].' % 'save/%s' % file_name)
		self.load_state_dict(torch.load('save/%s/model.pkl' % file_name), strict=strict)

	def loss(self, obs, action, mask, visual):
		T = config.max_dynamic_timestep
		batch_size = obs.shape[0]

		obs_T = self.resize(obs.transpose(0, 1))  # reshape to [T, B, ...]
		if config.observation_shape[1:] == (84, 84):
			obs_pad_T = F.pad(obs_T[:, :, -1:], (6, 6, 6, 6))
		else:
			obs_pad_T = obs_T[:, :, -1:]
		if self.transform is not None:
			obs_0 = self.transform(obs_T)[0]
			obs_T = self.transform(obs_T)
		else:
			raise NotImplementedError()

		clip_unroll_repr_grad = False

		if clip_unroll_repr_grad:
			with torch.no_grad():
				mode = self.encoder.training
				self.encoder.eval()
				s_T = [self.encoder(obs_T[t]) for t in range(T)]
				self.encoder.train(mode=mode)
		else:
			s_T = [self.encoder(obs_T[t]) for t in range(T)]

		loss_func = nn.BCELoss(reduction='none')
		loss_repr, loss_delta_repr, loss_dyna, loss_lag, loss_adapter = (torch.zeros(1, device=obs.device) for _ in
		                                                                 range(5))

		_s = self.encoder(obs_0)
		_s_T = [_s.clone()]
		_d_T = []  # delta_dynamics
		_r_T = []  # recon_dynamics
		_encoding_index_T = []

		for t in range(1, T):
			s_p = s_T[t - 1]
			s_t = s_T[t]
			z, _loss_lag, perp, encoding_index = self.lag(s_p, s_t)
			_loss_lag = (_loss_lag * mask[:, t]).mean()
			loss_lag += _loss_lag

			_d = self.delta_dynamic(_s, z)
			_r = self.recon_dynamic(_s)
			if config.use_action:
				action_one_hot = torch.ones((batch_size, 1, *_s.shape[-2:]), dtype=torch.float, device=_s.device)
				action_one_hot = action[:, t - 1: t, None, None] * action_one_hot / config.action_space_size
				_s = self.dynamic(_s, action_one_hot)
			else:
				_s = self.dynamic(_s, z)

			_s_T.append(_s.clone())
			_d_T.append(_d.clone())
			_r_T.append(_r.clone())
			_encoding_index_T.append(encoding_index)
		_r_T.append(self.recon_dynamic(_s))

		_obs_T = self.decoder(torch.cat(_r_T, dim=0)).chunk(T)
		_obs_delta_T = self.decoder_delta(torch.cat(_d_T, dim=0))  # .chunk(T)
		for i in range(len(_obs_delta_T)):
			_obs_delta_T[i] = _obs_delta_T[i].chunk(T - 1)

		# Penalty for hat s
		# TODO: forgot the mask?
		l1_penalty = (_s_T[0].abs().mean() + torch.cat(_s_T[1:], dim=0).abs().mean()) / 2
		l2_penalty = ((_s_T[0] ** 2).mean() + (torch.cat(_s_T[1:], dim=0) ** 2).mean()) / 2

		tps = []
		dts = []
		for t in range(T):
			_obs = _obs_T[t]
			_obs_pad = obs_pad_T[t]
			_loss_repr = ((loss_func(_obs, _obs_pad) - loss_func(_obs_pad, _obs_pad)).sum(dim=(1, 2, 3)) * mask[:,
			                                                                                               t]).mean()  # TODO MASK
			loss_repr += _loss_repr
			if t != 0:
				s_t = s_T[t]
				_s_t = _s_T[t]
				if config.consistency == 'contrastive':
					proj0 = self.project(s_t, with_grad=False)
					proj1 = self.project(_s_t, with_grad=True)
					_loss_dyna = (consist_loss_func(proj0, proj1) * mask[:, t]).mean()  # TODO MASK
				elif config.consistency == 'mse':
					_loss_dyna = ((s_t - _s_t) ** 2).mean()
				else:
					raise NotImplementedError(
						'consistency loss [%s] has not been implemented yet.' % config.consistency)
				loss_dyna += _loss_dyna

				_obs_pad_pre = obs_pad_T[t - 1]
				pool = nn.MaxPool2d(kernel_size=2)
				eps = 0.
				scopes_0 = (_obs_pad - _obs_pad_pre).clip(min=eps, max=1. - eps)
				scopes_1 = (_obs_pad_pre - _obs_pad).clip(min=eps, max=1. - eps)
				bce = nn.BCELoss(reduction='none')
				_loss_delta = torch.zeros(batch_size, device=obs.device)

				tps.append([])
				dts.append([])
				for i in range(len(_obs_delta_T) - 1, -1, -1):
					_obs_delta = _obs_delta_T[i][t - 1]
					tps[-1].append(scopes_0)
					dts[-1].append(_obs_delta)
					o0 = _obs_delta[:, 0]
					o1 = _obs_delta[:, 1]
					g0 = scopes_0[:, 0]
					g1 = scopes_1[:, 0]
					loss_0 = (bce(o0, g0) - bce(g0, g0)).sum(dim=(1, 2))
					loss_1 = (bce(o1, g1) - bce(g1, g1)).sum(dim=(1, 2))
					scopes_0 = pool(scopes_0)
					scopes_1 = pool(scopes_1)
					_loss_delta += loss_0 + loss_1
					if config.single_scope:
						# use single delta scope rather than multi scope
						break

				_loss_delta = (_loss_delta * mask[:, t]).mean()  # TODO MASK
				loss_delta_repr += _loss_delta
				# print('%d: %.5f %.5f %.5f' % (t, _loss_repr, _loss_delta, _loss_dyna), end='; ')
				pass
			else:
				# print('%d: %.5f' % (t, _loss_repr), end='; ')
				pass

		print()
		print('%.5f %.5f %.5f %.5f %.5f %.5f' % (
			loss_repr / T, loss_delta_repr / T, loss_dyna / T, loss_lag, loss_adapter, l1_penalty))

		c_loss_repr = 1.
		c_loss_dyna = 1.
		c_loss_delta = 1.
		c_loss_lag = 1.

		if config.no_delta:
			c_loss_delta = 0.
		if config.no_repr:
			c_loss_repr = 0.
		loss = (c_loss_repr * loss_repr + c_loss_delta * loss_delta_repr + c_loss_dyna * loss_dyna) / T

		loss += c_loss_lag * loss_lag
		loss += config.l1_penalty_coeff * l1_penalty + config.l2_penalty_coeff * l2_penalty

		if visual:
			from matplotlib.colors import NoNorm
			no_norm = NoNorm()
			fig, axs = plt.subplots(T, 10, figsize=(10 * 5, T * 5))
			for _t in range(T):
				axs[_t, 0].imshow(obs_pad_T[_t][0, -1].detach().cpu().numpy(), cmap='gray')
				axs[_t, 1].imshow(_obs_T[_t][0, -1].detach().cpu().numpy(), cmap='gray')
				if _t != 0:
					for i in range(len(_obs_delta_T)):
						axs[_t, 2 + i * 2].imshow(tps[_t - 1][i][0, 0].detach().cpu().numpy(), cmap='gray',
						                          norm=no_norm)
						axs[_t, 2 + i * 2 + 1].imshow(dts[_t - 1][i][0, 0].detach().cpu().numpy(), cmap='gray',
						                              norm=no_norm)
			plt.show()
			path = 'results/%s/' % self.name
			if not os.path.exists(path):
				os.makedirs(path)
			plt.savefig(path + '%d.png' % self.img_cnt)
			self.img_cnt += 1
			close(fig)
		return loss, loss_repr, loss_delta_repr, loss_dyna, loss_lag, loss_adapter, l1_penalty, l2_penalty

	def learn(self, obs, action, mask, visual=False):
		self.optim.zero_grad()
		self.train()

		loss, loss_repr, loss_delta_repr, loss_dyna, loss_lag, loss_adapter, l1_penalty, l2_penalty = self.loss(obs,
		                                                                                                        action,
		                                                                                                        mask,
		                                                                                                        visual)
		loss.backward()

		for p in [self.encoder.parameters(),
		          self.decoder.parameters(),
		          self.decoder_delta.parameters(),
		          self.lag.parameters(),
		          self.dynamic.parameters(),
		          self.delta_dynamic.parameters(),
		          self.recon_dynamic.parameters(),
		          self.projection.parameters(),
		          self.projection_head.parameters()]:
			total_norm = nn.utils.clip_grad_norm_(p, max_norm=config.clip_max)
			# print('grad_norm:', total_norm)
			pass
		self.optim.step()

		return loss.item(), \
		       loss_repr.item(), \
		       loss_delta_repr.item(), \
		       loss_dyna.item(), \
		       loss_lag.item(), \
		       loss_adapter.item(), \
		       l1_penalty.item(), \
		       l2_penalty.item()

	def test(self, obs, action, mask, visual=False):
		self.eval()
		with torch.no_grad():
			loss, loss_repr, loss_delta_repr, loss_dyna, loss_lag, loss_adapter, l1_penalty, l2_penalty = self.loss(obs,
			                                                                                                        action,
			                                                                                                        mask,
			                                                                                                        visual)
		return loss.item(), \
		       loss_repr.item(), \
		       loss_delta_repr.item(), \
		       loss_dyna.item(), \
		       loss_lag.item(), \
		       loss_adapter.item(), \
		       l1_penalty.item(), \
		       l2_penalty.item()

	def forward(self):
		pass

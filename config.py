import numpy as np
import torch
from torch import optim
import argparse
import gym

torch.set_printoptions(linewidth=10000000, precision=3, threshold=100000000, sci_mode=False)
np.set_printoptions(linewidth=10000, precision=3)

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='breakout')
parser.add_argument('--dataset_path', type=str, default='./atari_replay_dataset/')
parser.add_argument('--model_name', type=str, default='NONE')
parser.add_argument('--device', type=int, default=1)
parser.add_argument('--restore', action='store_true', default=False)
parser.add_argument('--lr', type=float, default=0.02)
parser.add_argument('--lr_min', type=float, default=0.0002)
parser.add_argument('--channel', type=int, default=64)
parser.add_argument('--num_blocks', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--optimizer', type=str, default='SGD')
parser.add_argument('--clip_max', type=float, default=10.0)
parser.add_argument('--state_norm', action='store_true', default=False)
parser.add_argument('--max_dynamic_timestep', type=int, default=6)
parser.add_argument('--resolution', type=tuple, default=(84, 84))
parser.add_argument('--latent_action_dim', type=int, default=5)
parser.add_argument('--num_embeddings', type=int, default=20)
parser.add_argument('--consistency', type=str, default='contrastive', choices=['mse', 'contrastive', 'none'])
parser.add_argument('--discount', type=float, default=0.997)
parser.add_argument('--extra_info', type=str, default='')
parser.add_argument('--use_action', action='store_true', default=False)  # TODO: use ground truth action

parser.add_argument('--l1_penalty_coeff', type=float, default=0.0)
parser.add_argument('--l2_penalty_coeff', type=float, default=0.0)
parser.add_argument('--stage_interval', type=int, default=1000)
parser.add_argument('--single_scope', action='store_true', default=False)
parser.add_argument('--no_delta', action='store_true', default=False)
parser.add_argument('--no_repr', action='store_true', default=False)


args = parser.parse_args()


def to_camel(snake):
	components = snake.split('_')
	return ''.join(x.title() for x in components)


def get_action_space_size(snake_env_name):
	camel_env_name = to_camel(snake_env_name)
	game = gym.make(camel_env_name + 'NoFrameskip-v4')
	return game.action_space.n


class DiscreteSupport(object):
	def __init__(self, min: int, max: int, delta=1.):
		assert min < max
		self.min = min
		self.max = max
		self.range = np.arange(min, max + 1, delta)
		self.size = len(self.range)
		self.delta = delta


class Config:
	def __init__(self):
		self.extra_info = args.extra_info
		self.batch_size = args.batch_size
		self.channel = args.channel
		self.num_blocks = args.num_blocks
		self.lr = args.lr
		self.lr_min = args.lr_min
		self.model_name = args.model_name

		self.momentum = args.momentum
		self.weight_decay = args.weight_decay
		self.clip_max = args.clip_max
		self.device = 'cuda:%d' % args.device
		self.restore = args.restore

		self.dataset = args.dataset
		self.dataset_path = args.dataset_path
		self.observation_shape = (4,) + args.resolution
		self.state_shape = (64, 6, 6)
		self.state_size = self.state_shape[1] * self.state_shape[2]

		self.latent_action_dim = args.latent_action_dim
		self.num_embeddings = args.num_embeddings
		self.state_norm = args.state_norm

		self.max_dynamic_timestep = args.max_dynamic_timestep
		self.bn_momentum = 0.01
		self.dataset = args.dataset

		self.frame_stack = 4
		self.frame_skip = 4
		self.discount = args.discount ** self.frame_skip

		self.action_space_size = get_action_space_size(self.dataset)
		self.consistency = args.consistency

		self.value_support: DiscreteSupport = DiscreteSupport(-300, 300, delta=1)
		self.reward_support: DiscreteSupport = DiscreteSupport(-300, 300, delta=1)
		self.__reward = torch.from_numpy(np.array([x for x in self.reward_support.range])).to(self.device)
		self.__value = torch.from_numpy(np.array([x for x in self.value_support.range])).to(self.device)

		self.l1_penalty_coeff = args.l1_penalty_coeff
		self.l2_penalty_coeff = args.l2_penalty_coeff
		self.stage_interval = args.stage_interval

		self.use_action = args.use_action

		assert args.optimizer in ['SGD', 'Adam', 'AdamW']
		if args.optimizer == 'SGD':
			self.optim = optim.SGD
		if args.optimizer == 'Adam':
			self.optim = optim.Adam
		if args.optimizer == 'AdamW':
			self.optim = optim.AdamW

		self.single_scope = args.single_scope
		self.no_delta = args.no_delta
		self.no_repr = args.no_repr

	def scalar_transform(self, x):
		""" Reference from MuZerp: Appendix F => Network Architecture
		& Appendix A : Proposition A.2 in https://arxiv.org/pdf/1805.11593.pdf (Page-11)
		"""
		delta = self.value_support.delta
		assert delta == 1
		epsilon = 0.001
		sign = torch.ones(x.shape, device=x.device).float()
		sign[x < 0] = -1.0
		output = sign * (torch.sqrt(torch.abs(x / delta) + 1) - 1) + epsilon * x / delta
		return output

	def inverse_reward_transform(self, reward_logits):
		return self.inverse_scalar_transform(reward_logits, self.reward_support, self.__reward)

	def inverse_value_transform(self, value_logits):
		return self.inverse_scalar_transform(value_logits, self.value_support, self.__value)

	def inverse_scalar_transform(self, logits, scalar_support, __):
		""" Reference from MuZerp: Appendix F => Network Architecture
		& Appendix A : Proposition A.2 in https://arxiv.org/pdf/1805.11593.pdf (Page-11)
		"""
		delta = self.value_support.delta
		value_probs = torch.softmax(logits, dim=1)
		value_support = torch.ones(value_probs.shape, device=value_probs.device)
		value_support[:, :] = __.clone()
		value = (value_support * value_probs).sum(1, keepdim=True) / delta

		epsilon = 0.001
		sign = torch.ones(value.shape, device=value.device).float()
		sign[value < 0] = -1.0
		output = (((torch.sqrt(1 + 4 * epsilon * (torch.abs(value) + 1 + epsilon)) - 1) / (2 * epsilon)) ** 2 - 1)
		output = sign * output * delta

		nan_part = torch.isnan(output)
		output[nan_part] = 0.
		output[torch.abs(output) < epsilon] = 0.
		return output

	def value_phi(self, x):
		return self._phi(x, self.value_support.min, self.value_support.max, self.value_support.size)

	def reward_phi(self, x):
		return self._phi(x, self.reward_support.min, self.reward_support.max, self.reward_support.size)

	def _phi(self, x, min, max, set_size: int):
		delta = self.value_support.delta

		x.clamp_(min, max)
		x_low = x.floor()
		x_high = x.ceil()
		p_high = x - x_low
		p_low = 1 - p_high

		target = torch.zeros(x.shape[0], set_size, device=x.device)
		x_high_idx, x_low_idx = x_high - min / delta, x_low - min / delta
		target.scatter_(1, x_high_idx.long().unsqueeze(-1), p_high.unsqueeze(-1))
		target.scatter_(1, x_low_idx.long().unsqueeze(-1), p_low.unsqueeze(-1))
		return target


config = Config()

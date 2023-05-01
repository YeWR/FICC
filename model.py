import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from config import config
from tools import renormalize


def mlp(input_size,
        layer_sizes,
        output_size,
        output_activation=nn.Identity,
        activation=nn.ReLU,
        momentum=0.1,
        init_zero=False):
	sizes = [input_size] + layer_sizes + [output_size]
	layers = []
	for i in range(len(sizes) - 1):
		if i < len(sizes) - 2:
			act = activation
			layers += [nn.Linear(sizes[i], sizes[i + 1]),
			           nn.BatchNorm1d(sizes[i + 1], momentum=momentum),
			           act()]
		else:
			act = output_activation
			layers += [nn.Linear(sizes[i], sizes[i + 1]),
			           act()]

	if init_zero:
		layers[-2].weight.data.fill_(0)
		layers[-2].bias.data.fill_(0)

	return nn.Sequential(*layers)


def conv3x3(in_channels, out_channels, stride=1):
	return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels, downsample=None, stride=1, momentum=0.1):
		super(ResidualBlock, self).__init__()
		self.conv1 = conv3x3(in_channels, out_channels, stride)
		self.bn1 = nn.BatchNorm2d(out_channels, momentum=momentum)
		self.conv2 = conv3x3(out_channels, out_channels)
		self.bn2 = nn.BatchNorm2d(out_channels, momentum=momentum)
		self.downsample = downsample
		self.act = nn.ReLU()

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.act(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.act(out)
		return out


class DownSample(nn.Module):
	def __init__(self, in_channels, out_channels, momentum=0.1):
		super(DownSample, self).__init__()
		self.conv1 = conv3x3(in_channels, out_channels // 2, stride=2)
		self.bn1 = nn.BatchNorm2d(out_channels // 2, momentum=momentum)
		self.resblocks1 = nn.ModuleList(
			[ResidualBlock(out_channels // 2, out_channels // 2, momentum=momentum) for _ in range(1)]
		)
		self.conv2 = conv3x3(out_channels // 2, out_channels, stride=2)
		self.downsample_block = ResidualBlock(out_channels // 2, out_channels, momentum=momentum, stride=2,
		                                      downsample=self.conv2)
		self.resblocks2 = nn.ModuleList(
			[ResidualBlock(out_channels, out_channels, momentum=momentum) for _ in range(1)]
		)
		self.pooling1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
		self.resblocks3 = nn.ModuleList(
			[ResidualBlock(out_channels, out_channels, momentum=momentum) for _ in range(1)]
		)
		self.pooling2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = nn.functional.relu(x)
		for block in self.resblocks1:
			x = block(x)
		x = self.downsample_block(x)
		for block in self.resblocks2:
			x = block(x)
		x = self.pooling1(x)
		for block in self.resblocks3:
			x = block(x)
		x = self.pooling2(x)
		return x


class RepresentationNetwork(nn.Module):
	def __init__(
			self,
			observation_shape,
			num_blocks,
			num_channels,
			downsample,
			momentum=0.1):
		super(RepresentationNetwork, self).__init__()
		self.downsample = downsample
		if self.downsample:
			self.downsample_net = DownSample(
				observation_shape[0],
				num_channels,
				momentum=momentum
			)
		self.conv = conv3x3(
			observation_shape[0],
			num_channels,
		)
		self.bn = nn.BatchNorm2d(num_channels, momentum=momentum)
		self.resblocks = nn.ModuleList(
			[ResidualBlock(num_channels, num_channels, momentum=momentum) for _ in range(num_blocks)]
		)

	def forward(self, x):
		if self.downsample:
			x = self.downsample_net(x)
		else:
			x = self.conv(x)
			x = self.bn(x)
			x = nn.functional.relu(x)

		for block in self.resblocks:
			x = block(x)
		if config.state_norm:
			x = renormalize(x)
		return x

	def get_param_mean(self):
		mean = []
		for name, param in self.named_parameters():
			mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
		mean = sum(mean) / len(mean)
		return mean


class Decoder(nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()
		modules = []

		# hidden_dims = [64, 64, 64, 64]
		hidden_dims = [config.channel] * 4
		hidden_dims.reverse()

		for i in range(len(hidden_dims) - 1):
			modules.append(
				nn.Sequential(
					nn.ConvTranspose2d(hidden_dims[i],
					                   hidden_dims[i + 1],
					                   kernel_size=3,
					                   stride=2,
					                   padding=1,
					                   output_padding=1),
					nn.BatchNorm2d(hidden_dims[i + 1], momentum=config.bn_momentum),
					nn.ReLU())
			)

		self.dropout = nn.Dropout2d(p=0.5)
		self.decoder = nn.Sequential(*modules)

		self.final_layer = nn.Sequential(
			nn.ConvTranspose2d(hidden_dims[-1],
			                   hidden_dims[-1],
			                   kernel_size=3,
			                   stride=2,
			                   padding=1,
			                   output_padding=1),
			nn.BatchNorm2d(hidden_dims[-1], momentum=config.bn_momentum),
			nn.ReLU(),
			nn.Conv2d(hidden_dims[-1], out_channels=1,
			          kernel_size=3, padding=1),
			nn.Sigmoid())

	def forward(self, z):
		result = self.decoder(z)
		result = self.final_layer(result)
		return result


class DecoderMultiScope(nn.Module):
	def __init__(self):
		super(DecoderMultiScope, self).__init__()
		self.decoder = nn.ModuleList()
		self.output_layer = nn.ModuleList()

		# hidden_dims = [64, 64, 64, 64]
		self.hidden_dims = [config.channel] * 4
		# hidden_dims.reverse()

		for i in range(len(self.hidden_dims) - 1):
			self.decoder.append(
				nn.Sequential(
					nn.ConvTranspose2d(self.hidden_dims[i],
					                   self.hidden_dims[i + 1],
					                   kernel_size=3,
					                   stride=2,
					                   padding=1,
					                   output_padding=1),
					nn.BatchNorm2d(self.hidden_dims[i + 1], momentum=config.bn_momentum),
					nn.ReLU())
			)

		for i in range(len(self.hidden_dims)):
			self.output_layer.append(
				nn.Sequential(
					nn.ConvTranspose2d(self.hidden_dims[i],
					                   self.hidden_dims[i],
					                   kernel_size=3,
					                   stride=2,
					                   padding=1,
					                   output_padding=1),
					nn.BatchNorm2d(self.hidden_dims[i], momentum=config.bn_momentum),
					nn.ReLU(),
					nn.Conv2d(self.hidden_dims[i], out_channels=2,
					          kernel_size=3, padding=1),
					nn.Sigmoid())
			)

	def forward(self, z):
		# z = self.dropout(z)
		outputs = [self.output_layer[0](z)]
		for i in range(len(self.hidden_dims) - 1):
			z = self.decoder[i](z)
			outputs.append(self.output_layer[i + 1](z))
		return outputs


class VectorQuantizer1D(nn.Module):
	def __init__(self, num_embeddings, input_sizes, embedding_dim, commitment_cost):
		super(VectorQuantizer1D, self).__init__()

		self._input_sizes = input_sizes
		self._embedding_dim = embedding_dim
		self._num_embeddings = num_embeddings

		self.embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
		self.embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
		self._commitment_cost = commitment_cost

	def get_embedding(self, index):
		return self.embedding.weight[index]

	def get_embeddings(self):
		return self.embedding.weight

	def forward(self, flat_input):
		device = flat_input.device

		distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
		             + torch.sum(self.embedding.weight ** 2, dim=1)
		             - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

		encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
		encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings).to(device)
		encodings.scatter_(1, encoding_indices, 1)

		quantized = torch.matmul(encodings, self.embedding.weight)

		e_latent_loss = ((quantized.detach() - flat_input) ** 2).mean(dim=1)
		q_latent_loss = ((quantized - flat_input.detach()) ** 2).mean(dim=1)
		loss = q_latent_loss + self._commitment_cost * e_latent_loss

		quantized = flat_input + (quantized - flat_input).detach()

		avg_probs = torch.mean(encodings, dim=0)
		perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
		return quantized, loss, perplexity, encoding_indices[:, 0]


class LatentActionGen(nn.Module):
	def __init__(self, num_embeddings, in_channel, vq_in_channel, embedding_channel, num_blocks):
		super(LatentActionGen, self).__init__()
		self._input_size = vq_in_channel * 36
		self.quantizer = VectorQuantizer1D(num_embeddings, vq_in_channel * 36, embedding_channel, 1.0)

		self.conv = conv3x3(in_channel, in_channel)
		self.conv_s = conv3x3(in_channel, in_channel)
		self.conv_a = conv3x3(1, in_channel)
		# self.conv = conv3x3(in_channel * 2, embedding_channel) # sample
		self.bn = nn.BatchNorm2d(in_channel)
		self.act = nn.ReLU()
		self.resblocks = nn.ModuleList(
			[ResidualBlock(in_channel, in_channel) for _ in range(num_blocks)]
		)
		self.conv_out = conv3x3(in_channel, vq_in_channel)
		self.fc = nn.Linear(self._input_size, embedding_channel)
		self.bn_o = nn.BatchNorm1d(embedding_channel, affine=False)

	def forward(self, s0, s1):
		x = self.conv(s0) + self.conv_s(s1)
		# x = self.bn(x) TODO: delete this
		# x += s0
		x = self.act(x)
		for block in self.resblocks:
			x = block(x)
		x = self.conv_out(x)

		flat_x = x.view(-1, self._input_size)
		flat_x = self.fc(flat_x)
		flat_x = self.bn_o(flat_x)
		z, loss, perplexity, encoding_indices = self.quantizer(flat_x)
		z = z.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, *s0.shape[-2:])
		# print('perp: %.4f; LAG loss: %.4f;' % (perplexity, loss))
		return z, loss, perplexity, encoding_indices


class Dynamic(nn.Module):
	def __init__(self, s_channel, z_channel, num_blocks):
		super(Dynamic, self).__init__()
		self.conv = conv3x3(s_channel + z_channel, s_channel)
		self.bn = nn.BatchNorm2d(s_channel, momentum=config.bn_momentum)
		self.act = nn.ReLU()
		self.resblocks = nn.ModuleList(
			[ResidualBlock(s_channel, s_channel) for _ in range(num_blocks)]
		)

	def forward(self, s, z):
		sz = torch.cat([s, z], dim=1)
		x = self.conv(sz)
		x = self.bn(x)
		x += s
		x = self.act(x)
		for block in self.resblocks:
			x = block(x)
		if config.state_norm:
			x = renormalize(x)
		return x

	def get_dynamic_mean(self):
		dynamic_mean = np.abs(self.conv.weight.detach().cpu().numpy().reshape(-1)).tolist()

		for block in self.resblocks:
			for name, param in block.named_parameters():
				dynamic_mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
		dynamic_mean = sum(dynamic_mean) / len(dynamic_mean)
		return dynamic_mean

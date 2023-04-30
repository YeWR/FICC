import os
import re
import time
import torch
import numpy as np
import torch.nn.functional as F
from config import config
from torch.utils.data import DataLoader
import cv2


def check(files, new_file):
	ptn = re.compile('^' + new_file)
	for file in files:
		if ptn.search(file) is not None:
			return False
	return True


class Logger:
	def __init__(self):
		self.model = None
		self.file = None
		self.path = None
		self.cnt = None

	def set_model(self, model):
		self.model = model
		self.path = 'log/%s/' % self.model
		if not os.path.exists(self.path):
			os.makedirs(self.path)
		files = os.listdir(self.path)
		cnt = 0
		while True:
			cnt += 1
			file = 'log_%d' % cnt
			if check(files, file):
				self.file = 'log/%s/%s.txt' % (self.model, file)
				self.cnt = cnt
				break
		self.write_line('Model: %s' % self.model)
		self.write_line('Time: %s' % time.strftime('%Y.%m.%d-%H:%M:%S', time.localtime()))
	
	def write(self, text):
		if self.file is None:
			raise RuntimeError('Logger: have not set model name')
		print('[Log] %s: %s' % (self.file, text))
		logfile = open(self.file, 'a')
		logfile.write(text)
		logfile.close()
	
	def write_line(self, text):
		self.write(text + '\n')

	def __call__(self, text):
		self.write_line(text)


def momentum_update(model0, model1, tau=0.95):
	with torch.no_grad():
		dict0 = model0.state_dict()
		dict1 = model1.state_dict()
		
		for name in dict0:
			# print(name)
			dict0[name] = dict0[name] * tau + dict1[name] * (1. - tau)
		model0.load_state_dict(dict0)


def bisect(a, x):
	lo, hi = -1, len(a) - 1
	while lo < hi:
		mid = -(-(lo + hi) // 2)
		if a[mid] > x:
			hi = mid - 1
		else:
			lo = mid
	return lo


class AddGaussianNoise:
	def __init__(self, mean=0., std=1.):
		self.std = std
		self.mean = mean
	
	def __call__(self, tensor: torch.Tensor):
		tensor = tensor + torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean
		return tensor.clip(0., 1.)
	
	def __repr__(self):
		return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def get_data_loader(dataset):
	return DataLoader(dataset,
	                  batch_size=config.batch_size,
	                  drop_last=True,
	                  shuffle=True,
	                  num_workers=1)


def renormalize(tensor, first_dim=1):
	# normalize the tensor (states)
	if first_dim < 0:
		first_dim = len(tensor.shape) + first_dim
	flat_tensor = tensor.view(*tensor.shape[:first_dim], -1)
	max = torch.max(flat_tensor, first_dim, keepdim=True).values
	min = torch.min(flat_tensor, first_dim, keepdim=True).values
	flat_tensor = (flat_tensor - min) / (max - min)
	# print(max.mean(), min.mean(), max.shape)
	return flat_tensor.view(*tensor.shape)


def consist_loss_func(f1, f2):
	f1 = F.normalize(f1, p=2., dim=-1, eps=1e-5)
	f2 = F.normalize(f2, p=2., dim=-1, eps=1e-5)
	return 1. - (f1 * f2).sum(dim=1)


def str_to_arr(s, gray_scale=False):
	"""To reduce memory usage, we choose to store the jpeg strings of image instead of the numpy array in the buffer.
	This function decodes the observation numpy arr from the jpeg strings
	Parameters
	----------
	s: string
		the inputs
	gray_scale: bool
		True -> the inputs observation is gray not RGB.
	"""
	nparr = np.frombuffer(s, np.uint8)
	if gray_scale:
		arr = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
		arr = np.expand_dims(arr, -1)
	else:
		arr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
	
	return arr


log = Logger()


def set_learning_rate(lr):
	config.lr = lr
	log('set learning rate: ' + str(lr))


def log_setting():
	log('--------------')
	for n in config.__dict__:
		log('==Config== %s: %s' % (str(n), str(config.__dict__[n])))
	log('--------------')


if __name__ == '__main__':
	pass


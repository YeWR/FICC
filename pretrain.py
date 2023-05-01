import torch
from config import config
import numpy as np
from tools import log, log_setting, get_data_loader
from atari import AtariDatasetMultistep
import tools
from transform import Transforms
from ficc import FICC


class TrainInfo:
	def __init__(self):
		self.step = 0
		self.stage = 0
		self.previous_stage = -1
		self.max_stage = 50

		self.lr_init = config.lr
		self.lr_min = config.lr_min
		self.warm_up_stage = 4
		self.max_stage = 50
		self.lr_decay = (self.lr_min / self.lr_init) ** (1 / (self.max_stage - self.warm_up_stage))
		self.lr_warmup_start = 0.00001
		self.lr_increase = (self.lr_init / self.lr_warmup_start) ** (1 / self.warm_up_stage)
		self.stage_interval = config.stage_interval

		print('TrainInfo:', self.lr_decay, self.lr_increase)


def train_epoch(model, dataset, test_dataset, train_info: TrainInfo):
	data_loader = get_data_loader(dataset)
	cnt = 0
	loss_list = [[] for _ in range(8)]
	for data in data_loader:
		if train_info.previous_stage != train_info.stage:
			log('EPOCH: %d' % train_info.stage)
			if train_info.stage < train_info.warm_up_stage:
				lr = train_info.lr_warmup_start * train_info.lr_increase ** train_info.stage
			else:
				lr = train_info.lr_init * train_info.lr_decay ** (train_info.stage - train_info.warm_up_stage)
			tools.set_learning_rate(lr)
			model.set_optimizer()
			train_info.previous_stage = train_info.stage

		# print(data.shape)
		obs, action, reward, value, mask = data
		obs = obs.type(torch.float32).to(config.device) / 255
		action = action.to(config.device)
		mask = mask.to(config.device)

		cnt += 1
		# visual = cnt % 200 == 0
		visual = False
		loss = model.learn(obs, action, mask, visual=visual)
		print('#', loss[0])

		for tp in range(8):
			loss_list[tp].append(loss[tp])

		train_info.step += 1
		if train_info.step % train_info.stage_interval == 0:
			log(('train:' + ' %.5f' * 8) % tuple(np.mean(loss_list[tp]) for tp in range(8)))
			loss_list = [[] for _ in range(8)]

			test(model, test_dataset)
			model.save()

			train_info.stage += 1
			if train_info.stage > train_info.max_stage:
				return


def test(model, dataset):
	log('Repr Mean: ' + str(model.encoder.get_param_mean()))
	log('Dynamic Mean: ' + str(model.dynamic.get_dynamic_mean()))

	data_loader = get_data_loader(dataset)
	cnt = 0
	loss_list = [[] for _ in range(8)]
	for data in data_loader:
		obs, action, reward, value, mask = data
		obs = obs.type(torch.float32).to(config.device) / 255
		action = action.to(config.device)
		mask = mask.to(config.device)
		visual = False
		loss = model.test(obs, action, mask, visual=visual)
		print('# test:', loss[0])

		for tp in range(8):
			loss_list[tp].append(loss[tp])

		cnt += 1
		if cnt >= 200:
			break
	log(('test:' + ' %.5f' * 8) % tuple(np.mean(loss_list[tp]) for tp in range(8)))


def get_dataset(subdir, block_id):
	dataset = AtariDatasetMultistep(subdir, block_id)
	return dataset


def pretrain():
	transform = Transforms()
	model = FICC(config.model_name, transform=transform).to(config.device)
	log.set_model(model.name)
	log_setting()

	if config.restore:
		model.restore()

	# train_dataset = get_dataset([1, 2, 3], [1, 25, 49])
	# test_dataset = get_dataset(4, [1, 25, 49])
	train_dataset = get_dataset(1, [25])  # for debugging
	test_dataset = get_dataset(2, [25])   # for debugging
	test(model, test_dataset)

	train_info = TrainInfo()

	while train_info.stage < train_info.max_stage:
		train_epoch(model, train_dataset, test_dataset, train_info)


if __name__ == '__main__':
	pretrain()

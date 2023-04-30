import numpy as np
import gzip
from typing import List
from config import config
from torch.utils.data import Dataset
from tools import log


class AtariDataset(Dataset):
	def __init__(self, subdir=1, block_id=1, cut=None, limit=3000*1000):
		self.path = config.dataset_path + '%s/' % config.dataset.lower()
		if type(subdir) is int:
			self.subdir = [subdir]
		else:
			self.subdir = subdir
		if type(block_id) is int:
			self.block_id = [block_id]
		else:
			self.block_id = block_id

		self.observation_buffer: np.ndarray
		self.action_buffer: np.ndarray
		self.reward_buffer: np.ndarray
		self.terminal_buffer: np.ndarray
		self.terminal_idx: List

		self.cut = cut
		self.limit = limit
		self._load_buffer()

	def _load_buffer(self):
		observation_buffer_list = []
		action_buffer_list = []
		reward_buffer_list = []
		terminal_buffer_list = []

		trajectory_info_list = []
		for subdir in self.subdir:
			for block_id in self.block_id:
				path = self.path + '%d/replay_logs/' % subdir
				with gzip.open(path + '$store$_terminal_ckpt.%d.gz' % block_id, 'rb') as f:
					terminal_buffer = np.load(f)
					if self.cut is not None:
						terminal_buffer = terminal_buffer[: self.cut]
					terminal_buffer[-1] = 1
					terminal_idx = np.where(terminal_buffer == 1)[0] + 1
					pre_idx = terminal_idx[0]
					for idx in terminal_idx[1:-1]:
						length = idx - pre_idx
						trajectory_info = (subdir, block_id, idx, pre_idx, length)
						trajectory_info_list.append(trajectory_info)
						pre_idx = idx
		np.random.shuffle(trajectory_info_list)

		total_length = 0
		approved_info_set = set()
		for trajectory_info in trajectory_info_list:
			length = trajectory_info[4]
			total_length += length
			approved_info_set.add(trajectory_info)
			if total_length >= self.limit:
				break
		print('Limit:', self.limit, total_length)

		for subdir in self.subdir:
			for block_id in self.block_id:
				print('GameLoader:_load_buffer (%d, %d)' % (subdir, block_id))
				path = self.path + '%d/replay_logs/' % subdir

				with gzip.open(path + '$store$_observation_ckpt.%d.gz' % block_id, 'rb') as f:
					observation_buffer = np.load(f)
					observation_buffer = observation_buffer.reshape((-1, *config.observation_shape[1:]))
				with gzip.open(path + '$store$_action_ckpt.%d.gz' % block_id, 'rb') as f:
					action_buffer = np.load(f)
				with gzip.open(path + '$store$_reward_ckpt.%d.gz' % block_id, 'rb') as f:
					reward_buffer = np.load(f)
				with gzip.open(path + '$store$_terminal_ckpt.%d.gz' % block_id, 'rb') as f:
					terminal_buffer = np.load(f)

				if self.cut is not None:
					observation_buffer = observation_buffer[: self.cut]
					action_buffer = action_buffer[: self.cut]
					reward_buffer = reward_buffer[: self.cut]
					terminal_buffer = terminal_buffer[: self.cut]

				terminal_buffer[-1] = 1
				terminal_idx = np.where(terminal_buffer == 1)[0] + 1
				pre_idx = terminal_idx[0]
				for idx in terminal_idx[1:-1]:
					length = idx - pre_idx
					trajectory_info = (subdir, block_id, idx, pre_idx, length)
					if trajectory_info in approved_info_set:
						observation_buffer_list.append(observation_buffer[pre_idx: idx])
						action_buffer_list.append(action_buffer[pre_idx: idx])
						reward_buffer_list.append(reward_buffer[pre_idx: idx])
						terminal_buffer_list.append(terminal_buffer[pre_idx: idx])
					pre_idx = idx
				print('GameLoader:_load_buffer (%d, %d) done.' % (subdir, block_id))

		self.observation_buffer = np.concatenate(observation_buffer_list, axis=0)
		self.action_buffer = np.concatenate(action_buffer_list, axis=0)
		self.reward_buffer = np.concatenate(reward_buffer_list, axis=0).astype(np.float32)
		self.terminal_buffer = np.concatenate(terminal_buffer_list, axis=0)
		self.terminal_idx = [-1] + list(np.where(self.terminal_buffer == 1)[0])
		self.value_buffer = np.zeros_like(self.reward_buffer)

		self.buffer_length = self.observation_buffer.shape[0]

		value = 0.
		for idx in range(self.buffer_length - 1, -1, -1):
			if self.terminal_buffer[idx]:
				value = 0.
			value = self.reward_buffer[idx] + config.discount * value
			self.value_buffer[idx] = value

	def __len__(self):
		return self.buffer_length

	def get_stack_num(self, i, frame_stack):
		for t in range(1, frame_stack):
			# if self.terminal_buffer[i - t]:
			if i - t < 0 or self.terminal_buffer[i - t]:
				return t
		return frame_stack

	def __getitem__(self, i_cd):
		raise NotImplementedError('')


class AtariDatasetMultistep(AtariDataset):
	def __init__(self, subdir=1, block_id=1, cut=None, limit=3000*1000):
		super(AtariDatasetMultistep, self).__init__(subdir, block_id, cut, limit)

		log('AtariDatasetMultistep: (%s, %s) load done. total_length = %d.' % (str(self.subdir), str(self.block_id), self.buffer_length))

		self.__stack_num = np.array([self.get_stack_num(i, config.frame_stack) for i in range(self.buffer_length)])

	def __len__(self):
		return self.buffer_length - config.max_dynamic_timestep

	def check_valid(self, i):
		T = config.max_dynamic_timestep
		return not self.terminal_buffer[i: i + T - 1].any()

	def __getitem__(self, i):
		T = config.max_dynamic_timestep

		obs = np.zeros((T, config.frame_stack, *config.observation_shape[1:]), dtype=np.uint8)

		terminal = -1
		for t in range(T):
			stack_num = self.__stack_num[i + t]
			obs[t, -stack_num:] = self.observation_buffer[i + t - stack_num + 1: i + t + 1]
			if self.terminal_buffer[i + t]:
				terminal = t + 1
				break

		action = self.action_buffer[i: i + T].copy()
		reward = self.reward_buffer[i: i + T].copy()
		value = self.value_buffer[i: i + T].copy()
		mask = np.ones(T, dtype=np.uint8)

		if terminal != -1:
			action[terminal:] = 0
			reward[terminal:] = 0
			value[terminal:] = 0
			mask[terminal:] = 0
		return obs, action, reward, value, mask

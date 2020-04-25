import json
import numpy as np
import random
import pickle


class DataStore:
	def __init__(self):
		self.prev_states = [];
		self.actions = [];
		self.current_states = [];

	def sample(self, sample_size):
		bin_size = len(self.prev_states)
		if sample_size <= bin_size:
			rand_indices = random.sample(range(0, bin_size-1), sample_size)
		else:
			rand_indices = random.sample(range(0, bin_size-1), bin_size-1)
		# print(rand_indices)
		sample_prev_states = self.prev_states[rand_indices, :]
		sample_current_states = self.current_states[rand_indices, :]
		sample_actions = self.actions[rand_indices, :]
		return (sample_prev_states, sample_actions, sample_current_states)

	def add(self, prev_states, actions, current_states):
		self.prev_states.append(prev_states)
		self.actions.append(actions)
		self.current_states.append(current_states)

	def build(self):
		self.prev_states = np.array(self.prev_states).astype(np.float32)
		self.actions = np.array(self.actions).astype(np.float32)
		self.current_states = np.array(self.current_states).astype(np.float32)
		

	def load_data(self, file_name="env_data.jsl", limit=None):

		with open(file_name, "r") as f:
			line_count = 0
			for line in f.readlines():
				line = json.loads(line)
				prev_state = np.fromstring(line["prev_state"][1:-1], sep=",")
				current_state = np.fromstring(line["current_state"][1:-1], sep=",")
				info = np.fromstring(line["info"][1:-1], sep=",")
				action = np.fromstring(line["action"][1:-1], sep=",")
				prev_state = np.append(prev_state, info)
				current_state = np.append(current_state, info)
				self.add(prev_state, action, current_state)
				line_count += 1
				if limit:
					if line_count >= limit:
						print(line_count)
						break


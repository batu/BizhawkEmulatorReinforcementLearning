import gym
from gym import spaces

import numpy as np
from scipy import spatial

import pyautogui

from keras.models import load_model
# from keras.applications.inception_v3 import InceptionV3
from scipy.misc import imread
from skimage.transform import resize
from glob import glob
import subprocess
import win_unicode_console
import os
import sys

pyautogui.PAUSE = 1
pyautogui.FAILSAFE = True

# Director paths
bizhawk_dirs = 'BizHawk/'
rom_dirs = 'Rom/'
rom_name = 'SuperMarioWorld.smc'
data_dirs = 'Data/'
model_dirs = 'Model/'
state_dirs = 'States/'
results_dir = "Results/"


class BizHawk(gym.Env):
	metadata = {'render.modes': ['human']}

	# state_representation SS or RAM
	def __init__(self, logging_folder_path, algorithm_name="DQN", state_representation="SS", reward_representation="DISTANCE", state_frame_count=1, no_action=False, human_warm_up_episode=0, active_debug_text=True):
		self.__version__ = "1.0.0"
		print("BizHawk - Version {}".format(self.__version__))

		state_frame_count = 1 if state_frame_count < 1 else state_frame_count
		self.state_representation = state_representation
		self.reward_representation = reward_representation

		self.target_vector = []
		self.current_vector = []
		self.last_distance = 0
		self.last_state = []

		self.paths = []
		self.data_paths = glob(data_dirs + '*')
		self.no_action = no_action
		self.active_debug_text = active_debug_text

		self.logging_folder_path = logging_folder_path
		self.run_name = ""

		print("Initilized variables.")
		print("Started loading models.")

		self.original_embedding = np.load(model_dirs + 'embedding.npy')
		# self.input_model = load_model(model_dirs + 'input_model.h5')
		self.embedded_model = load_model(model_dirs + 'embedded_model.h5')
		# self.inception = InceptionV3(weights='imagenet')
		self.max_embedding = np.amax(self.original_embedding, axis=0)
		self.min_embedding = np.amin(self.original_embedding, axis=0)
		print("Done loading models.")

		# BREADCRUMBS_START
		self.EPISODE_LENGTH = 512
		self.ACTION_LENGTH = 12
		# BREADCRUMBS_END

		self.curr_action_step = 0

		self.last_cos_similarity = 0
		self.cumulative_reward = 0
		self.max_cumulative_reward = 0

		# This will probably be Discrete(33) for all decided actionsself.
		# Currently:
		# 0 : Noop
		# 1 : Jump
		# 2 : Left
		# 3 : Right

		# BREADCRUMBS_START
		# This is the action space.
		self.action_dict = {
			0: "A",
			1: "Right",
			2: "Left",
			3: "B",
			# 3: "Down",
			# 4: "B"
		}
		# BREADCRUMBS_END
		self.action_space = spaces.Discrete(len(self.action_dict))

		self.memory_vectors = []
		self.memory_count = state_frame_count

		self.current_RAM_state = []
		if self.state_representation == "RAM":
			unit_state_size = 4096
			self.current_RAM_state = np.zeros(4096)

		elif self.state_representation == "SS":
			unit_state_size = 256
			for _ in range(20):
				print("REMEMBER TO TURN ON UPDATE RESET IF YOU HAVE A TARGET VECTOR.")

		for _ in range(self.memory_count):
			self.memory_vectors.append(np.zeros(unit_state_size))

		high = np.ones(unit_state_size * self.memory_count)
		low = np.zeros(unit_state_size * self.memory_count)
		self.observation_space = spaces.Box(low, high)

		# Store what the agent tried
		self.curr_episode = 0
		self.curr_step = 0
		self.action_episode_memory = []

		print("Starting BizHawk.")
		self.proc = self.start_bizhawk_process()

		print("Load the game state.")
		self.start_bizhawk_game()
		# print("Load succesful!")
		# self.start_recording_bizhawk()

	def step(self, action):
		"""
		ob, reward, episode_over, info : tuple
			ob (object) :
			reward (float) :
			episode_over (bool) :
			debug_info (dict) :
		"""
		if self.no_action:
			self._take_action(4)
		else:
			self._take_action(action)
		self.curr_step += 1
		reward = self._get_reward()
		self.cumulative_reward += reward
		ob = self._get_state()
		episode_over = self.curr_step >= self.EPISODE_LENGTH
		action_code = self.action_dict[action]
		if self.active_debug_text:
			sys.stdout.write(f"Reward: {reward:4.2f}   Action Taken: {action_code}           \r")
			sys.stdout.flush()
		return ob, reward, episode_over, {}

	def reset(self):
		print(f"For episode {self.curr_episode} the cumulative_reward was {self.cumulative_reward:4.2f} and the max reward was {self.max_cumulative_reward:4.2f}.")
		self.curr_episode += 1
		self.curr_step = 0
		self.write_graphs()
		self.cumulative_reward = 0
		self.max_cumulative_reward = 0
		# self.update_target_vector()

		self.proc.stdin.write(b'client.speedmode(400) ')
		self.proc.stdin.write(b'savestate.loadslot(1) ')
		self.proc.stdin.write(b'emu.frameadvance() ')
		self.proc.stdin.flush()
		self.last_distance = self.get_distance()
		return self._get_state()

	def render(self, mode='human', close=False):
		return

	def _get_state(self):
		def only_current_vector_from_ss():
			self.update_current_vector_bizhawk_screenshot()
			return self.current_vector

		def current_vector_with_memory_from_ss():
			self.update_current_vector_bizhawk_screenshot()
			np_vectors = np.array(self.memory_vectors)
			flat = np_vectors.reshape((256 * self.memory_count))
			return flat

		def current_vector_plus_target_from_ss():
			if self.memory_count > 1:
				combined_state = np.append(self.memory_vectors, self.target_vector)
			else:
				combined_state = np.append(self.current_vector, self.target_vector)
			return combined_state

		def RAM_4096_state():
			ram_state = self.get_ram_state()
			if ram_state:
				self.current_RAM_state = ram_state
			return self.current_RAM_state

		def pixel_state():
			# Possibly do some more data munging.
			return self.get_pixel_data()

		# BREADCRUMBS_START
		# This is the state
		return current_vector_with_memory_from_ss()
		# BREADCRUMBS_END

	def get_ram_state(self, normalize=True):
		self.send_byte_read_command()
		byte_values = self.receive_bytes_from_lua()[1:-1]
		RAM_state = []
		# For some reason I cant get 4096 numbers, so reading 97 and slicing one off
		for byte_value in byte_values[:-1]:
			RAM_state.append(int(byte_value[:-1]))
#            RAM_state.append(int.from_bytes(byte_value, byteorder="little", signed=True))
		if normalize:
			max_val = max(RAM_state)
			min_val = min(RAM_state)
			RAM_state_normalized = [val - min_val / max_val - min_val for val in RAM_state]
			return RAM_state_normalized
		else:
			return RAM_state

	def get_pixel_data(self):
		self.proc.stdin.flush()
		self.proc.stdin.write(b'client.screenshot("temp_screenshot.png") ')
		self.proc.stdin.write(b'io.stdout:write("continue\\n") ')
		self.proc.stdin.flush()

		new_line = self.proc.stdout.readline()
		while new_line != b'continue\n':
			new_line = self.proc.stdout.readline()
		temp_img = np.expand_dims(resize(imread('temp_screenshot.png'),
												(224, 256),
												mode='reflect'),
												axis=0)
		# (1, 224, 256, 3)
		return temp_img

	def _take_action(self, action):
		# This is for  algorithms that epress action probablitiy distributionself.
		# Current it is greedy
		try:
			selected_action = self.action_dict[action]
		except TypeError:
			key = action.tolist().index(max(action))
			selected_action = self.action_dict[key]

		for i in range(self.ACTION_LENGTH):
			action_code = b''
			action_code += b'buttons = {} '

			# Create the action code that is selected.
			action_code += b'buttons["' + str.encode(selected_action) + b'"] = 1 '
			action_code += b'joypad.set(buttons, 1) '
			action_code += b'emu.frameadvance() '

			# Actually send the action
			self.proc.stdin.write(action_code)
			self.proc.stdin.flush()

	def get_distance(self):
		self.proc.stdin.write(b'io.stdout:write("Start\\n") ')
		code = b'io.stdout:write(serialize(mainmemory.readbyterange(147, 3)) )'
		self.proc.stdin.write(code)
		self.proc.stdin.flush()

		new_line = self.proc.stdout.readline()
		while new_line != b'Start\n':
			new_line = self.proc.stdout.readline()

		new_line = self.proc.stdout.readline().split()[1:-1]
		dist_1 = new_line[0][:-1]
		dist_2 = new_line[1]
		distance = (int(dist_1) + int(dist_2) * 255) / 10

		self.proc.stdin.write(b'io.stdout:write("DoneReward\\n") ')
		self.proc.stdin.flush()

		new_line = self.proc.stdout.readline()
		while new_line not in b'DoneReward\n':
			new_line = self.proc.stdout.readline()

		return distance

	def _get_reward(self):
		def plus_one_for_positivedelta():
			cosine_similarity = 1 - spatial.distance.cosine(self.current_vector, self.target_vector)
			if self.last_cos_similarity < cosine_similarity:
				self.last_cos_similarity = cosine_similarity
				return 1
			else:
				self.last_cos_similarity = cosine_similarity
				return -1

		def delta_in_similarity():
			cosine_similarity = 1 - spatial.distance.cosine(self.current_vector, self.target_vector)
			delta = cosine_similarity - self.last_cos_similarity
			self.last_cos_similarity = cosine_similarity
			return delta

		def one_for_any_increase_in_distance():
			distance = self.get_distance()
			delta = distance - self.last_distance

			reward = 0
			# The reward amounts
			if delta > 0:
				reward = 1
			elif delta < 0:
				reward = -1

			self.last_distance = distance
			return reward

		# BREADCRUMBS_START
		def plusone_increase_minus_half_decrease_or_stationary():
			distance = self.get_distance()
			delta = distance - self.last_distance

			reward = -.5
			# The reward amounts
			if delta > 0:
				reward = 1
			elif delta < 0:
				reward = -.5

			self.last_distance = distance
			return reward
		# BREADCRUMBS_END

		def distance_traveled_between_frames():
			distance = self.get_distance()
			delta = distance - self.last_distance
			if delta > 10000:
				delta = 0
				exit()
			self.last_distance = distance
			return delta

		# BREADCRUMBS_START
		# The reward is:
		reward = distance_traveled_between_frames()
		# BREADCRUMBS_END
		self.cumulative_reward += reward
		if self.cumulative_reward > self.max_cumulative_reward:
			self.max_cumulative_reward = self.cumulative_reward
		return reward

	def update_current_vector_bizhawk_screenshot(self):
		self.proc.stdin.flush()
		self.proc.stdin.write(b'client.screenshot("temp_screenshot.png") ')
		self.proc.stdin.write(b'io.stdout:write("continue\\n") ')
		self.proc.stdin.flush()

		new_line = self.proc.stdout.readline()
		while new_line != b'continue\n':
			new_line = self.proc.stdout.readline()
		temp_img = np.expand_dims(resize(imread('temp_screenshot.png'),
												(224, 256),
												mode='reflect'),
												axis=0)
		self.current_vector = self.embedded_model.predict(temp_img)[0]
		if self.memory_vectors:
			self.memory_vectors.pop()
			self.memory_vectors.append(self.current_vector)

	def update_target_vector(self):
		def random_screenshot_embedding_vector(MovingTarget=False):
			if not MovingTarget:
				np.random.seed(41)
			goal_img = imread(self.data_paths[np.random.randint(len(self.data_paths))])
			self.target_vector = self.embedded_model.predict(np.expand_dims(goal_img, axis=0))[0]

		def normalized_random_embedding_vector(MovingTarget=False):
			goal = np.zeros(self.max_embedding.shape)
			if not MovingTarget:
				np.random.seed(41)
			for i in range(len(self.max_embedding)):
				goal[i] = np.random.uniform(self.min_embedding[i], self.max_embedding[i])
			self.target_vector = goal

		def random_embedding_vector(MovingTarget=False):
			if not MovingTarget:
				np.random.seed(41)
			goal = np.zeros(self.max_embedding.shape)
			for i in range(len(self.max_embedding)):
				goal[i] = np.random.uniform(0, 1)
			self.target_vector = goal

		random_screenshot_embedding_vector()

	def read_byte_lua(self, num):
		code = b''
		code += b'io.stdout:write(mainmemory.readbyte(' + str.encode(str(num)) + b')) '
		code += b'io.stdout:write(" ") '
		return code

	def send_byte_read_command(self):
		self.proc.stdin.write(b'io.stdout:write("pass\\n") ')
		code = b''
		code += b'io.stdout:write(serialize(mainmemory.readbyterange(0, 4098)) )'
		code += b'io.stdout:write(" ") '
		code += b'io.stdout:write("\\n") io.stdout:write("continue\\n") '
		self.proc.stdin.write(code)
		self.proc.stdin.flush()

	def receive_bytes_from_lua(self):
		new_line = b''

		while new_line != b'pass\n':
			new_line = self.proc.stdout.readline()

		new_line = self.proc.stdout.readline()
		nums = new_line[:-1].split()
		state_num = nums

		new_line = self.proc.stdout.readline()
		while new_line != b'continue\n':
			new_line = self.proc.stdout.readline()

		self.proc.stdin.write(b'io.stdout:write("DoneReward\\n") ')
		self.proc.stdin.flush()

		new_line = self.proc.stdout.readline()
		while new_line not in b'DoneReward\n':
			new_line = self.proc.stdout.readline()
		return state_num

	def start_bizhawk_process(self):
		win_unicode_console.enable()
		if not os.path.exists(bizhawk_dirs + state_dirs):
			os.mkdir(bizhawk_dirs + state_dirs)

		proc = subprocess.Popen([bizhawk_dirs + 'EmuHawk.exe',
								rom_dirs + rom_name,
								'--lua=../rrt.lua'],
								stdout=subprocess.PIPE,
								stdin=subprocess.PIPE)
		return proc

	def start_bizhawk_game(self):
		started = False
		while True:
			out_line = self.proc.stdout.readline()

			# get rom name
			if out_line[:5] == b'start':
				started = True

			# started
			if started:
				self.proc.stdin.write(b'client.speedmode(400) ')
				self.proc.stdin.write(b'savestate.loadslot(1) ')
				self.proc.stdin.write(b'emu.frameadvance() ')

				self.proc.stdin.flush()
				self.update_current_vector_bizhawk_screenshot()
				break
			else:
				pass
				# print(out_line)

	def start_recording_bizhawk(self):

		for _ in range(30):
			self.proc.stdin.write(b'emu.frameadvance() ')
			self.proc.stdin.flush()

		pyautogui.moveTo(229, 180, duration=0.25)
		pyautogui.click()
		pyautogui.moveTo(229, 190, duration=0.25)
		pyautogui.click()
		pyautogui.moveTo(317, 396, duration=0.25)
		pyautogui.click()
		pyautogui.moveTo(504, 450, duration=0.25)
		pyautogui.click()
		pyautogui.moveTo(504, 350, duration=0.25)
		pyautogui.click()
		pyautogui.click()

	def save_recording_bizhawk(self, dest: str):
		self.proc.stdin.write(b'movie.save()')
		self.proc.stdin.flush()

	def write_graphs(self):
		target = self.logging_folder_path + self.run_name
		with open(f"{target}/max_reward.txt", "a+") as file:
			file.write(f"{self.curr_episode},{self.max_cumulative_reward:4.4f}\n")

		with open(f"{target}/cumulative_reward.txt", "a+") as file:
			file.write(f"{self.curr_episode},{self.cumulative_reward:4.4f}\n")

	def shut_down_bizhawk_game(self):
		print("Exiting bizhawk.")
		try:
			for _ in range(1000):
				self.proc.stdin.write(b'emu.frameadvance() ')
				self.proc.stdin.flush()
				self.proc.stdin.write(b'client.exit() ')
				self.proc.stdin.flush()
		except OSError:
			return

import gym
from gym import spaces

import numpy as np
from scipy import spatial

from keras.models import load_model
from keras.applications.inception_v3 import InceptionV3
from scipy.misc import imread
from skimage.transform import resize
from glob import glob
import subprocess
import win_unicode_console
import os

# Director paths
bizhawk_dirs = 'BizHawk/'
rom_dirs = 'Rom/'
rom_name = 'SuperMarioWorld.smc'
data_dirs = 'Data/'
model_dirs = 'Model/'
state_dirs = 'States/'


class BizHawk(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.__version__ = "0.1.0"
        print("BizHawk - Version {}".format(self.__version__))

        self.target_vector = []
        self.current_vector = []

        self.paths = []
        self.data_paths = glob(data_dirs + '*')

        print("Initilized variables.")
        print("Started loading models.")

        self.original_embedding = np.load(model_dirs + 'embedding.npy')
        # self.input_model = load_model(model_dirs + 'input_model.h5')
        self.embedded_model = load_model(model_dirs + 'embedded_model.h5')
        self.inception = InceptionV3(weights='imagenet')
        self.max_embedding = np.amax(self.original_embedding, axis=0)
        self.min_embedding = np.amin(self.original_embedding, axis=0)

        print("Done loading models.")

        self.EPISODE_LENGTH = 512
        self.ACTION_LENGTH = 8

        self.last_cos_similarity = np.inf
        self.cumulative_reward = 0

        # This will probably be Discrete(33) for all decided actionsself.
        # Currently:
        # 0 : Noop
        # 1 : Jump
        # 2 : Left
        # 3 : Right

        self.action_dict = {
            0: "",
            1: "A",
            2: "Left",
            3: "Right",
            4: "B"
        }

        self.action_space = spaces.Discrete(len(self.action_dict))
        high = np.ones(512)
        low = np.zeros(512)
        self.observation_space = spaces.Box(low, high)
        # Store what the agent tried
        self.curr_episode = -1
        self.action_episode_memory = []

        print("Starting BizHawk.")
        self.proc = self.start_bizhawk_process()

        print("Load the game state.")
        self.start_bizhawk_game()
        print("Load succesful!")

    def step(self, action):
        """
        ob, reward, episode_over, info : tuple
            ob (object) :
            reward (float) :
            episode_over (bool) :
            debug_info (dict) :
        """

        self.curr_episode += 1
        self._take_action(action)
        reward = self._get_reward()
        self.cumulative_reward += reward
        ob = self._get_state()
        episode_over = self.curr_episode >= self.EPISODE_LENGTH
        return ob, reward, episode_over, {}

    def reset(self):
        print("For episode {} the cumulative_reward was {}.".format(self.curr_episode, self.cumulative_reward))
        self.curr_episode = -1
        self.cumulative_reward = 0
        self.update_target_vector_normalized_random()

        self.proc.stdin.write(b'client.speedmode(400) ')
        self.proc.stdin.write(b'savestate.loadslot(1) ')
        self.proc.stdin.flush()
        return self._get_state()

    def render(self, mode='human', close=False):
        return

    def _get_state(self):
        self.update_current_vector_bizhawk_screenshot()
        # self.update_target_vector_normalized_random()
        combined_state = np.append(self.current_vector, self.target_vector)
        # print(self.cumulative_reward)
        return combined_state

    def _take_action(self, action):
        selected_action = self.action_dict[action]

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

    def _get_reward(self):
        cosine_similarity = 1 - spatial.distance.cosine(self.current_vector, self.target_vector)
        if self.last_cos_similarity < cosine_similarity:
            self.last_cos_similarity = cosine_similarity
            return 1
        else:
            self.last_cos_similarity = cosine_similarity
            return -1
        # return cosine_similarity

    def update_current_vector_bizhawk_screenshot(self):
        self.proc.stdin.write(b'client.screenshot("temp_screenshot.png") ')
        self.proc.stdin.write(b'io.stdout:write("continue\\n") ')
        self.proc.stdin.flush()
        new_line = b''
        new_line = self.proc.stdout.readline()
        while new_line != b'continue\n':
            new_line = self.proc.stdout.readline()
        temp_img = np.expand_dims(resize(imread('temp_screenshot.png'),
                                                (224, 256),
                                                mode='reflect'),
                                                axis=0)
        self.current_vector = self.embedded_model.predict(temp_img)[0]

    def update_target_vector_random_screenshot(self):
        goal_img = imread(self.data_paths[np.random.randint(len(self.data_paths))])
        self.target_vector = self.embedded_model.predict(np.expand_dims(goal_img, axis=0))[0]

    def update_target_vector_normalized_random(self):
        goal = np.zeros(self.max_embedding.shape)
        np.random.seed(41)
        for i in range(len(self.max_embedding)):
            goal[i] = np.random.uniform(self.min_embedding[i], self.max_embedding[i])
        self.target_vector = goal

    def update_target_vector_random(self):
        goal = np.zeros(self.max_embedding.shape)
        for i in range(len(self.max_embedding)):
            goal[i] = np.random.uniform(0, 1)
        self.target_vector = goal

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
                self.proc.stdin.flush()
                self.update_current_vector_bizhawk_screenshot()
                break
            else:
                pass
                # print(out_line)

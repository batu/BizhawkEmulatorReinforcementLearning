import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np


class BizHawk(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.__version__ = "0.1.0"
        print("BizHawk - Version {}".format(self.__version__))

        self.target_embdeding = np.zeroes(256)
        self.current_embedding = np.zeroes(256)

        self.EPISODE_LENGTH = 100

        # This will probably be Discrete(33) for all decided actionsself.
        # Currently:
        # 0 : Noop
        # 1 : Jump
        # 2 : Left
        # 3 : Down
        # 4 : Right

        self.action_space = spaces.Discrete(5)

        # Store what the agent tried
        self.curr_episode = -1
        self.action_episode_memory = []
        return

    def step(self, action):
        """
        The agent takes a step in the environment.
        Parameters
        ----------
        action : int
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """

        self.curr_step += 1
        self._take_action(action)
        reward = self._get_reward()
        ob = self._get_state()

        return ob, reward, episode_over, diognostic_dict

    def reset(self):
        return

    def render(self, mode='human', close=False):
        return

    def _get_state():
        combined_state = self.current_embedding + self.target_embdeding;
        return combined_state

    def _get_reward():
        return

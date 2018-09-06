# Copyright 2017 reinforce.io. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import time

import numpy as np

from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

from support_utils import save_hyperparameters, send_email, visualize_cumulative_reward, visualize_max_reward

import gym_bizhawk

REPLAY = False
run_number = 0
experiment_name = "V4_PPONetwork"

TB_path = f"Results/TensorBoard/{experiment_name}/"

try:
    os.mkdir(TB_path[:-1])
except:
    pass

try:
    os.mkdir(f"{TB_path}README")
except:
    pass

models_path = "Results/Models/"
ENV_NAME = 'BizHawk-v0'

changes = """ Changed to the new and improved RL library. Started using PPO"""
reasoning = """PPO is the go to algorithm."""
hypothesis = """Better results. Not intially but after tweaking. """


if not REPLAY:
    if len(hypothesis) + len(changes) + len(reasoning) < 10:
        print("NOT ENOUGH LOGGING INFO")
        print("Please write more about the changes and reasoning.")
        exit()

    with open(f"{TB_path}/README/README.txt", "w") as readme:
        start_time_ascii = time.asctime(time.localtime(time.time()))
        algorithm = os.path.basename(__file__)[:-2]
        print(f"Experiment start time: {start_time_ascii}", file=readme)
        print(f"\nAlgorithm:\n{algorithm}", file=readme)
        print(f"\nThe Changes:\n{changes}", file=readme)
        print(f"\nReasoning:\n{reasoning}", file=readme)
        print(f"\nHypothesis:\n{hypothesis}", file=readme)
        print(f"\nResults:\n", file=readme)


# Create an OpenAIgym environment.
environment = OpenAIGym('BizHawk-v0', visualize=False)
environment.gym.logging_folder_path = TB_path

# Network as list of layers
# - Embedding layer:
#   - For Gym environments utilizing a discrete observation space, an
#     "embedding" layer should be inserted at the head of the network spec.
#     Such environments are usually identified by either:
#     - class ...Env(discrete.DiscreteEnv):
#     - self.observation_space = spaces.Discrete(...)

# Note that depending on the following layers used, the embedding layer *may* need a
# flattening layer

# BREADCRUMBS_START
network_spec = [
    # dict(type='embedding', indices=100, size=32),
    # dict(type'flatten'),
    dict(type='dense', size=14),
    dict(type='dense', size=14),
    dict(type='dense', size=8)
]

agent = PPOAgent(
    states=environment.states,
    actions=environment.actions,
    network=network_spec,
    # Agent
    states_preprocessing=None,
    actions_exploration=None,
    reward_preprocessing=None,
    # MemoryModel
    update_mode=dict(
        unit='timesteps',
        # 10 episodes per update
        batch_size=128,
        # Every 10 episodes
        frequency=1
    ),
    memory=dict(
        type='latest',
        include_next_states=False,
        capacity=5000
    ),
    # DistributionModel
    distributions=None,
    entropy_regularization=0.01,
    # PGModel
    baseline_mode='states',
    baseline=dict(
        type='mlp',
        sizes=[32, 32]
    ),
    baseline_optimizer=dict(
        type='multi_step',
        optimizer=dict(
            type='adam',
            learning_rate=1e-4
        ),
        num_steps=5
    ),
    gae_lambda=0.97,
    # PGLRModel
    likelihood_ratio_clipping=0.2,
    # PPOAgent
    step_optimizer=dict(
        type='adam',
        learning_rate=1e-3
    ),
    subsampling_fraction=0.2,
    optimization_steps=25,
    execution=dict(
        type='single',
        session_config=None,
        distributed_spec=None
    )
)
# BREADCRUMBS_END

folder_count = len([f for f in os.listdir(TB_path) if not os.path.isfile(os.path.join(models_path, f))])
run_name = f"run{folder_count}"
run_path = f'{TB_path}{run_name}'
environment.gym.run_name = run_name

os.mkdir(run_path)

# This function saves all the important hypterparameters to the run summary file.
save_hyperparameters(["ppo.py", "gym_bizhawk\\envs\\bizhawk_env.py"], f"{run_path}/run_summary.txt")

start_time_ascii = time.asctime(time.localtime(time.time()))
start_time = time.time()
print("Training has started!")


# Create the runner
runner = Runner(agent=agent, environment=environment)


# Callback function printing episode statistics
def episode_finished(r):
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                 reward=r.episode_rewards[-1]))
    return True


# Start learning
runner.run(episodes=32, max_episode_timesteps=768, episode_finished=episode_finished)

environment.gym.shut_down_bizhawk_game()
runner.close()

# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-100:]))
)


visualize_cumulative_reward(input_file=f"{run_path}/cumulative_reward.txt",
                            ouput_destionation=f"{run_path}/cumulative_reward_{experiment_name}_R{folder_count}_plot.png",
                            readme_dest=f"{TB_path}/README/cumulative_reward_{experiment_name}_R{folder_count}_plot.png",
                            experiment_name=experiment_name, run_count=folder_count)

visualize_max_reward(input_file=f"{run_path}/max_reward.txt",
                            ouput_destionation=f"{run_path}/max_reward_{experiment_name}_R{folder_count}_plot.png",
                            readme_dest=f"{TB_path}/README/max_reward_{experiment_name}_R{folder_count}_plot.png",
                            experiment_name=experiment_name, run_count=folder_count)

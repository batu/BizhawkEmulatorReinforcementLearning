import gym_bizhawk
import os
import time
import json
from support_utils import save_hyperparameters, send_email, visualize_cumulative_reward, visualize_max_reward

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Reshape
from keras.optimizers import Adam
from keras import callbacks
from keras.models import model_from_json

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, BoltzmannQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory

REPLAY = True
run_number = 2
experiment_name = "V3_Level"

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
ENV_NAME = 'BizHawk-v1'

changes = """Changed polcy to EpsGrededy to reach 0."""
reasoning = """It was never reaching 0. So it still had the problem we inistiallt had. Fixed it."""
hypothesis = """I expect the floor of max distance to rise even further."""

if not REPLAY:
    if len(hypothesis) + len(changes) + len(reasoning) < 140:
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

# Get the environment and extract the number of actions.
# env = gym.make(ENV_NAME)
# BREADCRUMBS_START
window_length = 1
# BREADCRUMBS_END
env = gym_bizhawk.BizHawk(logging_folder_path=TB_path, replay=REPLAY)
nb_actions = env.action_space.n

# Sequential Model
print(env.observation_space.shape)

# BREADCRUMBS_START
model = Sequential()
model.add(Flatten(input_shape=(window_length,) + env.observation_space.shape))
model.add(Dense(nb_actions, activation='linear'))
# BREADCRUMBS_END
model.summary()

# CONV Model
# print(env.observation_space.shape)
# model = Sequential()
# model.add(Reshape((224, 256, 3), input_shape=(1, 1, 224, 256, 3)))
# print(model.summary())
# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!

# enable the dueling network
# you can specify the dueling_type to one of {'avg','max','naive'}

# BREADCRUMBS_START
memory = SequentialMemory(limit=2048, window_length=window_length)
# policy = BoltzmannQPolicy()
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=0, value_test=0,
                          nb_steps=1024 * 32)

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
       nb_steps_warmup=1024, gamma=.98, target_model_update=1e-3,
       train_interval=1, delta_clip=1.)

dqn.compile(Adam(lr=1e-3), metrics=['mae'])
# BREADCRUMBS_END

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
folder_count = len([f for f in os.listdir(TB_path) if not os.path.isfile(os.path.join(models_path, f))])
run_name = f"run{folder_count}"
run_path = f'{TB_path}{run_name}'
env.run_name = run_name

if REPLAY:
    run_name = f"run{run_number}"
    run_path = f'{TB_path}{run_name}'
    with open(f"{run_path}/config.json", "r") as config:
        json_string = json.load(config)
        model = model_from_json(json_string)
        dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               nb_steps_warmup=1024, gamma=.99, target_model_update=1,
               train_interval=1, delta_clip=1.)
        dqn.compile(Adam(lr=1e-3), metrics=['mae'])
        dqn.load_weights('{}\{}_run{}_weights.h5f'.format(run_path, ENV_NAME, run_number))
        print("Testing has started!")
        dqn.test(env, nb_episodes=1, visualize=False)
        print("Testing has started!")
        env.shut_down_bizhawk_game()
        exit()

os.mkdir(run_path)

with open(f"{run_path}/run_summary.txt", "w") as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

with open(f"{run_path}/config.json", "w") as outfile:
    json_string = model.to_json()
    json.dump(json_string, outfile)

# This function saves all the important hypterparameters to the run summary file.
save_hyperparameters(["DQN.py", "gym_bizhawk.py"], f"{run_path}/run_summary.txt")

start_time_ascii = time.asctime(time.localtime(time.time()))
start_time = time.time()
print("Training has started!")

# BREADCRUMBS_START
dqn.fit(env, nb_steps=1024 * 32, visualize=True, verbose=0, callbacks=[callbacks.TensorBoard(log_dir=run_path, write_graph=False)])
# BREADCRUMBS_END

# After training is done, we save the final weights.
dqn.save_weights('{}\{}_run{}_weights.h5f'.format(run_path, ENV_NAME, folder_count), overwrite=True)

total_run_time = round(time.time() - start_time, 2)
print("Training is done.")

visualize_cumulative_reward(input_file=f"{run_path}/cumulative_reward.txt",
                            ouput_destionation=f"{run_path}/cumulative_reward_{experiment_name}_R{folder_count}_plot.png",
                            readme_dest=f"{TB_path}/README/cumulative_reward_{experiment_name}_R{folder_count}_plot.png",
                            experiment_name=experiment_name, run_count=folder_count)

visualize_max_reward(input_file=f"{run_path}/max_reward.txt",
                            ouput_destionation=f"{run_path}/max_reward_{experiment_name}_R{folder_count}_plot.png",
                            readme_dest=f"{TB_path}/README/max_reward_{experiment_name}_R{folder_count}_plot.png",
                            experiment_name=experiment_name, run_count=folder_count)

send_email(f"The training of {run_name} finalized!\nIt started at {start_time_ascii} and took {total_run_time/60} minutes .",
            run_path=run_path, experiment_name=experiment_name, run_number=folder_count)

env.shut_down_bizhawk_game()
# Finally, evaluate our algorithm for 5 episodes.
# movie.save("C:/Users/user/Desktop/VideoGame Ret/RL Retrieval/movie")
# dqn.test(env, nb_episodes=1, visualize=False)

time.sleep(5)

import gym_bizhawk
import os
import time
from support_utils import save_hyperparameters, send_email

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Reshape
from keras.optimizers import Adam
from keras import callbacks

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, BoltzmannQPolicy
from rl.memory import SequentialMemory

TB_path = "Results/TensorBoard/V1/"
models_path = "Results/Models/"
ENV_NAME = 'BizHawk-v1'

# Get the environment and extract the number of actions.
# env = gym.make(ENV_NAME)
env = gym_bizhawk.BizHawk()
nb_actions = env.action_space.n
# BREADCRUMBS_START
window_length = 1
# BREADCRUMBS_END

# Sequential Model
print(env.observation_space.shape)

model = Sequential()
model.add(Flatten(input_shape=(window_length,) + env.observation_space.shape))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(nb_actions, activation='linear'))
print(model.summary())


# CONV Model
# print(env.observation_space.shape)
# model = Sequential()
# model.add(Reshape((224, 256, 3), input_shape=(1, 1, 224, 256, 3)))
# model.add(Conv2D(16, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Flatten())
# model.add(Dense(32, activation='relu'))
# model.add(Dense(nb_actions, activation='linear'))
# print(model.summary())
# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!

# enable the dueling network
# you can specify the dueling_type to one of {'avg','max','naive'}

# BREADCRUMBS_START
memory = SequentialMemory(limit=50000, window_length=window_length)
policy = BoltzmannQPolicy()

dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100, enable_dueling_network=True, dueling_type='avg', target_model_update=1e-3, policy=policy)

dqn.compile(Adam(lr=1e-3), metrics=['mae'])
# BREADCRUMBS_END

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
folder_count = len([f for f in os.listdir(TB_path)])
run_name = f"DQN_{ENV_NAME}_run{folder_count + 1}"
tb_folder_path = f'{TB_path}{run_name}'

os.mkdir(tb_folder_path)

with open(f"{tb_folder_path}/run_summary.txt", "w") as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

# This function saves all the important hypterparameters to the run summary file.
save_hyperparameters(["duelingDQN.py", "gym_bizhawk.py"], f"{tb_folder_path}/run_summary.txt")

start_time_ascii = time.asctime(time.localtime(time.time()))
start_time = time.time()
# BREADCRUMBS_START
dqn.fit(env, nb_steps=1025, visualize=True, verbose=0, callbacks=[callbacks.TensorBoard(log_dir=tb_folder_path, write_graph=False)])
# BREADCRUMBS_END

file_count = len([f for f in os.listdir(models_path) if os.path.isfile(os.path.join(models_path, f))])


# After training is done, we save the final weights.
dqn.save_weights('{}\DQN_{}_run{}_weights.h5f'.format(tb_folder_path, ENV_NAME, file_count + 1), overwrite=True)

total_run_time = round(time.time() - start_time, 2)
print("Training is done.")
send_email(f"The training of {run_name} finalized!\nIt started at {start_time_ascii} and took {total_run_time/60} minutes .")
# Finally, evaluate our algorithm for 5 episodes.
# dqn.test(env, nb_episodes=0, visualize=True)

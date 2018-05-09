import gym_bizhawk
import os

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
window_length = 1

# Sequential Model
# print(env.observation_space.shape)
# model = Sequential()
# model.add(Flatten(input_shape=(window_length,) + env.observation_space.shape))
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dense(256))
# model.add(Activation('relu'))
# model.add(Dense(256))
# model.add(Activation('relu'))
# model.add(Dense(nb_actions, activation='linear'))
# print(model.summary())


# CONV Model
print(env.observation_space.shape)
model = Sequential()
model.add(Reshape((224, 256, 3), input_shape=(1, 1, 224, 256, 3)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))
print(model.summary())
# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=window_length)
policy = BoltzmannQPolicy()
# enable the dueling network
# you can specify the dueling_type to one of {'avg','max','naive'}
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100, enable_dueling_network=True, dueling_type='avg', target_model_update=1e-3, policy=policy)

dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
folder_count = len([f for f in os.listdir(TB_path)])
tb_folder_path = f'{TB_path}DQN_{ENV_NAME}_run{folder_count + 1}'
os.mkdir(tb_folder_path)
dqn.fit(env, nb_steps=100000, visualize=True, verbose=0, callbacks=[callbacks.TensorBoard(log_dir=tb_folder_path, write_graph=False)])


file_count = len([f for f in os.listdir(models_path) if os.path.isfile(os.path.join(models_path, f))])


# After training is done, we save the final weights.
dqn.save_weights('{}\DQN_{}_run{}_weights.h5f'.format(models_path, ENV_NAME, file_count + 1), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True)

import gym_bizhawk
import os
import gym
import time
import json

import cv2
import matplotlib as plt
import numpy as np
from shutil import copyfile
from support_utils import save_hyperparameters, send_email, visualize_cumulative_reward, visualize_max_reward

import tensorflow as tf_out
import keras
from keras import backend as K

from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Reshape, Input, GlobalAveragePooling2D, Lambda
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet import MobileNet
from keras.utils.generic_utils import CustomObjectScope


from keras.optimizers import Adam
from keras import callbacks
from keras.models import model_from_json

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, BoltzmannQPolicy, LinearAnnealedPolicy, BoltzmannGumbelQPolicy
from rl.memory import SequentialMemory

REPLAY = False
run_number = 0
experiment_name = "Qbert_V2_mobilenetworking"

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
ENV_NAME = 'Qbert-v0'

changes = """ Switched to the mobilenet to increase the speed. """
reasoning = """ The 24fps is reasonably close to the human play experience."""
hypothesis = """ The mobilenet with 3m params should make it at least function."""


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

def lambda_resize(input_tensor):
    import tensorflow as tf_out
    return tf_out.image.resize_images(input_tensor, (128, 128))

def relu6(x):
  return K.relu(x, max_value=6)

with CustomObjectScope({'relu6': relu6}):
        window_length = 5
        memory_size = 2048
        batch_size = 1
        env = gym.make(ENV_NAME)

        # BREADCRUMBS_END
        nb_actions = env.action_space.n

        # Sequential Model
        print(env.observation_space.shape)

        # BREADCRUMBS_START
        input_tensor = Input((5, 210, 160, 3,))
        input_tensor = Lambda(lambda frame: frame[0,0:1], output_shape=(210, 160, 3))(input_tensor)
        input_tensor = Lambda(lambda_resize)(input_tensor)

        mobilenet_model = MobileNet(weights='imagenet', include_top=False, input_shape=(128, 128, 3), input_tensor=input_tensor )

        for layer in mobilenet_model.layers:
            layer.trainable = False

        # add a global spatial average pooling layer
        x = mobilenet_model.output
        x = Flatten()(x)
        x = Dense(32, activation='relu',  trainable=False)(x)
        x = Dense(16, activation='relu',  trainable=True)(x)
        x = Dense(8, activation='relu',  trainable=True)(x)

        predictions = Dense(nb_actions, activation='linear',  trainable=True)(x)

        # this is the model we will train
        model = Model(inputs=mobilenet_model.input, outputs=predictions)

        # BREADCRUMBS_END
        #
        # model = Sequential()
        # model.add(Flatten(input_shape=(5, 210, 160, 3)))
        # model.add(Dense(nb_actions, activation='linear', trainable=False))

        model.summary()


        # BREADCRUMBS_START
        episode_count = 8
        step_count = 21600 # 24 frames per second time 15 minutes
        memory = SequentialMemory(limit=memory_size, window_length=window_length)
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=.75, value_min=0.075, value_test=0,
                                     nb_steps=episode_count * 512 / 2)

        dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               nb_steps_warmup=256, gamma=.925, target_model_update=1e-2,
               train_interval=4, batch_size=batch_size, delta_clip=1., enable_dueling_network=True, dueling_type="avg")

        dqn.compile(Adam(lr=1e-3), metrics=['mae'])
        # BREADCRUMBS_END

        # Okay, now it's time to learn something! We visualize the training here for show, but this
        # slows down training quite a lot. You can always safely abort the training prematurely using
        # Ctrl + C.
        folder_count = len([f for f in os.listdir(TB_path) if not os.path.isfile(os.path.join(models_path, f))])
        run_name = f"run{folder_count}"
        run_path = f'{TB_path}{run_name}'
        env.run_name = run_name

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
        dqn.fit(env, nb_steps=step_count, visualize=True, verbose=2, callbacks=[callbacks.TensorBoard(log_dir=run_path, write_graph=True, write_images=True)])

        # BREADCRUMBS_END

        # After training is done, we save the final weights.
        dqn.save_weights('{}\{}_run{}_weights.h5f'.format(run_path, ENV_NAME, folder_count), overwrite=True)

        total_run_time = round(time.time() - start_time, 2)
        print("Training is done.")

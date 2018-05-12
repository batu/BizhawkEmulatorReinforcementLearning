import gym_bizhawk
import os
import time
import json
from support_utils import save_hyperparameters, send_email

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Reshape
from keras.optimizers import Adam
from keras import callbacks
from keras.models import model_from_json

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, BoltzmannQPolicy
from rl.memory import SequentialMemory

REPLAY = False
run_number = 21
TB_path = "Results/TensorBoard/DDQN-SmallNetworks/"
models_path = "Results/Models/"
ENV_NAME = 'BizHawk-v1'

# Get the environment and extract the number of actions.
# env = gym.make(ENV_NAME)
for k in [1, 2, 4, 8, 16, 32, 64, 128]:
    env = gym_bizhawk.BizHawk()
    nb_actions = env.action_space.n
    # BREADCRUMBS_START
    window_length = 1
    # BREADCRUMBS_END

    # Sequential Model
    print(env.observation_space.shape)

# BREADCRUMBS_START
    model = Sequential()
    model.add(Flatten(input_shape=(window_length,) + env.observation_space.shape))
    model.add(Dense(k, activation='relu'))
    model.add(Dense(nb_actions, activation='softmax'))
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
    memory = SequentialMemory(limit=50000, window_length=window_length)
    policy = BoltzmannQPolicy()

    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100, enable_dueling_network=True, dueling_type='avg', target_model_update=1e-3, policy=policy)

    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    # BREADCRUMBS_END

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    folder_count = len([f for f in os.listdir(TB_path) if not os.path.isfile(os.path.join(models_path, f))])
    run_name = f"DQN_{ENV_NAME}_run{folder_count}"
    tb_folder_path = f'{TB_path}{run_name}'

    if REPLAY:
        run_name = f"DQN_{ENV_NAME}_run{run_number}"
        tb_folder_path = f'{TB_path}{run_name}'
        with open(f"{tb_folder_path}/config.json", "r") as config:
            json_string = json.load(config)
        model = model_from_json(json_string)
        dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100, enable_dueling_network=True, dueling_type='avg', target_model_update=1e-3, policy=policy)
        dqn.compile(Adam(lr=1e-3), metrics=['mae'])
        dqn.load_weights('{}\DQN_{}_run{}_weights.h5f'.format(tb_folder_path, ENV_NAME, run_number))
        print("Testing has started!")
        dqn.test(env, nb_episodes=1, visualize=False)
        print("Testing has started!")
        env.shut_down_bizhawk_game()
        exit()

    os.mkdir(tb_folder_path)

    with open(f"{tb_folder_path}/run_summary.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    with open(f"{tb_folder_path}/config.json", "w") as outfile:
        json_string = model.to_json()
        json.dump(json_string, outfile)

    # This function saves all the important hypterparameters to the run summary file.
    save_hyperparameters(["duelingDQN.py", "gym_bizhawk.py"], f"{tb_folder_path}/run_summary.txt")

    start_time_ascii = time.asctime(time.localtime(time.time()))
    start_time = time.time()
    print("Training has started!")
    # BREADCRUMBS_START
    # try:
    dqn.fit(env, nb_steps=512 * 16, visualize=True, verbose=0, callbacks=[callbacks.TensorBoard(log_dir=tb_folder_path, write_graph=False)])
    # except OSError:
    #     print("OS ERROR OCCURED.")
    #     print("If this is not a emulator switch.")
    # BREADCRUMBS_END

    # After training is done, we save the final weights.
    dqn.save_weights('{}\DDQN_{}_run{}_weights.h5f'.format(tb_folder_path, ENV_NAME, folder_count), overwrite=True)

    total_run_time = round(time.time() - start_time, 2)
    print("Training is done.")
    send_email(f"The training of {run_name} finalized!\nIt started at {start_time_ascii} and took {total_run_time/60} minutes .")

    env.shut_down_bizhawk_game()
    # Finally, evaluate our algorithm for 5 episodes.
    # movie.save("C:/Users/user/Desktop/VideoGame Ret/RL Retrieval/movie")
    # dqn.test(env, nb_episodes=1, visualize=False)

    time.sleep(5)

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
from rl.policy import EpsGreedyQPolicy, BoltzmannQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory

REPLAY = False
run_number = 21
TB_path = "WorkhorseResults/TensorBoard/W1/"
models_path = "Results/Models/"
ENV_NAME = 'BizHawk-v1'

changes = """Made the model 1 layer deeper."""
reasoning = """The single layer, even after the model, seems too shallow. I want to fine tune a bit better, and specifically, I am curious if there are any differences between 1 layer and 2 layers."""
hypothesis = """There will not be any significant changes between a single layer and double layer set up."""

if len(hypothesis) + len(changes) + len(reasoning) < 140:
    print("NOT ENOUGH LOGGING INFO")
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
for k in range(10):
    env = gym_bizhawk.BizHawk(logging_folder_path=TB_path)
    nb_actions = env.action_space.n
    # BREADCRUMBS_START
    window_length = 1
    # BREADCRUMBS_END

    # Sequential Model
    print(env.observation_space.shape)

    # BREADCRUMBS_START
    model = Sequential()
    model.add(Flatten(input_shape=(window_length,) + env.observation_space.shape))
    model.add(Dense(2**k + 1, activation='relu'))
    model.add(Dense(2**k, activation='relu'))
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
    memory = SequentialMemory(limit=50000, window_length=window_length)
    policy = BoltzmannQPolicy()

    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
           nb_steps_warmup=1024, gamma=.9, target_model_update=1,
           train_interval=1, delta_clip=1.)

    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    # BREADCRUMBS_END

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    folder_count = len([f for f in os.listdir(TB_path) if not os.path.isfile(os.path.join(models_path, f))])
    run_name = f"run{folder_count}"
    tb_folder_path = f'{TB_path}{run_name}'
    env.run_name = run_name

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
    save_hyperparameters(["DQN.py", "gym_bizhawk.py"], f"{tb_folder_path}/run_summary.txt")

    start_time_ascii = time.asctime(time.localtime(time.time()))
    start_time = time.time()
    print("Training has started!")
    # BREADCRUMBS_START
    # try:
    dqn.fit(env, nb_steps=512 * 32, visualize=True, verbose=0, callbacks=[callbacks.TensorBoard(log_dir=tb_folder_path, write_graph=False)])
    # except OSError:
    #     print("OS ERROR OCCURED.")
    #     print("If this is not a emulator switch.")
    # BREADCRUMBS_END

    # After training is done, we save the final weights.
    dqn.save_weights('{}\DQN_{}_run{}_weights.h5f'.format(tb_folder_path, ENV_NAME, folder_count), overwrite=True)

    total_run_time = round(time.time() - start_time, 2)
    print("Training is done.")
    send_email(f"The training of {run_name} finalized!\nIt started at {start_time_ascii} and took {total_run_time/60} minutes .")

    env.shut_down_bizhawk_game()
    # Finally, evaluate our algorithm for 5 episodes.
    # movie.save("C:/Users/user/Desktop/VideoGame Ret/RL Retrieval/movie")
    # dqn.test(env, nb_episodes=1, visualize=False)

    time.sleep(5)

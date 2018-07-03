import gym_bizhawk
import os
import time
import json
from support_utils import save_hyperparameters, send_email

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam
from keras import callbacks
from keras.models import model_from_json

from rl.agents import DDPGAgent
from rl.policy import EpsGreedyQPolicy, BoltzmannQPolicy
from rl.random import OrnsteinUhlenbeckProcess
from rl.memory import SequentialMemory

REPLAY = False
run_number = 21
TB_path = "Results/TensorBoard/DDPG-minus/"
models_path = "Results/Models/"
ENV_NAME = 'BizHawk-v1'

changes = """Reduced the state space reprsenetation from 1024, 4 concatination to 1."""
reasoning = """The timeline information between frames might not be preserved between frames.
This might be detrimental to the training as it might just add noise."""
hypothesis = """The training will be more stable."""

if len(hypothesis) + len(changes) + len(reasoning) < 140:
    print("NOT ENOUGH LOGGING INFO")
    exit()

with open(f"{TB_path}/README.txt", "w") as readme:
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
for k in range(20):
    env = gym_bizhawk.BizHawk()
    nb_actions = env.action_space.n
    # BREADCRUMBS_START
    window_length = 1
    # BREADCRUMBS_END

    # Sequential Model
    print(env.observation_space.shape)

    # BREADCRUMBS_START
    actor = Sequential()
    actor.add(Flatten(input_shape=(window_length,) + env.observation_space.shape))
    actor.add(Dense(2**(k + 1)))
    actor.add(Activation('relu'))
    actor.add(Dense(2**(k)))
    actor.add(Activation('relu'))
    actor.add(Dense(nb_actions))
    actor.add(Activation('tanh'))
    # BREADCRUMBS_END
    actor.summary()
    # BREADCRUMBS_END
    print(actor.summary())

    actor.summary()

    # BREADCRUMBS_START
    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    # BREADCRUMBS_END
    print(critic.summary())

    # BREADCRUMBS_START
    memory = SequentialMemory(limit=100000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=10, nb_steps_warmup_actor=10,
                      random_process=random_process, gamma=.99, target_model_update=1e-3)
    agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])
    # BREADCRUMBS_END

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    folder_count = len([f for f in os.listdir(TB_path) if not os.path.isfile(os.path.join(models_path, f))])
    run_name = f"run{folder_count}"
    tb_folder_path = f'{TB_path}{run_name}'

    if REPLAY:
        run_name = f"DDPG_{ENV_NAME}_run{run_number}"
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
        actor.summary(print_fn=lambda x: f.write(x + '\n'))

    with open(f"{tb_folder_path}/run_summary.txt", "a") as f:
        critic.summary(print_fn=lambda x: f.write(x + '\n'))

    with open(f"{tb_folder_path}/actor_config.json", "w") as outfile:
        json_string = actor.to_json()
        json.dump(json_string, outfile)

    with open(f"{tb_folder_path}/critic_config.json", "w") as outfile:
        json_string = critic.to_json()
        json.dump(json_string, outfile)

    # This function saves all the important hypterparameters to the run summary file.
    save_hyperparameters(["DDPG.py", "gym_bizhawk.py"], f"{tb_folder_path}/run_summary.txt")

    start_time_ascii = time.asctime(time.localtime(time.time()))
    start_time = time.time()
    print("Training has started!")
    # BREADCRUMBS_START
    callback = [callbacks.TensorBoard(log_dir=tb_folder_path, write_graph=False)]
    agent.fit(env, nb_steps=8192 * 2, visualize=True, verbose=1, nb_max_episode_steps=512, callbacks=callback)
    # BREADCRUMBS_END

    # After training is done, we save the final weights.
    agent.save_weights('{}\{}_run{}_weights.h5f'.format(tb_folder_path, ENV_NAME, folder_count), overwrite=True)

    total_run_time = round(time.time() - start_time, 2)
    print("Training is done.")
    send_email(f"The training of {run_name} finalized!\nIt started at {start_time_ascii} and took {total_run_time/60} minutes .")

    env.shut_down_bizhawk_game()
    # Finally, evaluate our algorithm for 5 episodes.
    # movie.save("C:/Users/user/Desktop/VideoGame Ret/RL Retrieval/movie")
    # dqn.test(env, nb_episodes=1, visualize=False)

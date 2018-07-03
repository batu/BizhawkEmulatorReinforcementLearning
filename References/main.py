import numpy as np
from keras.models import load_model
from keras.backend import clear_session
from keras.applications.inception_v3 import InceptionV3
import matplotlib.pyplot as plt
from scipy.misc import imread
from skimage.transform import resize
from glob import glob
from rrt import explore_with_rrt
import subprocess
import win_unicode_console
import os


def img_lua():
    proc.stdin.write(b'client.screenshot("temp_screenshot.png") ')
    proc.stdin.write(b'io.stdout:write("continue\\n") ')
    proc.stdin.flush()
    new_line = b''
    new_line = proc.stdout.readline()
    while new_line != b'continue\n':
        new_line = proc.stdout.readline()
    temp_img = np.expand_dims(resize(imread('temp_screenshot.png'),
                                            (224, 256),
                                            mode='reflect'),
                                            axis=0)
    embeddings.append(embedded_model.predict(temp_img)[0])
    scores.append(inception.predict(temp_img)[0])


def state_projection(state):
    return embeddings[state]


def get_random_successor(state, goal, available_action):
    action_name = ["Up", "Down", "Left", "Right", "Select",
                   "Start", "B", "A", "X", "Y", "L", "R"]
    temp_action = 4096
    temp_inputs = input_model.predict([embeddings[state].reshape((1, 256)),
                                      goal.reshape((1, 256))])[0]
    action_dict_copy = action_dict.copy()
    while temp_action not in available_action:
        if not np.any(temp_inputs):
            if all(value == 0 for value in action_dict_copy.values()):
                temp_action = np.random.choice(4096)
            else:
                temp_action = max(action_dict_copy, key=action_dict_copy.get)
                action_dict_copy[temp_action] = 0
        else:
            temp_action = np.argmax(temp_inputs)
            temp_inputs[temp_action] = 0
    action = "{0:b}".format(temp_action).rjust(12, '0')[::-1]

    next_state = len(embeddings)
    interval = 30

    # do action
    for i in range(interval + 1):
        action_code = b''
        action_code += b'buttons = {} '

        for j, name in enumerate(action_name):
            if action[j] == '1':
                action_code += b'buttons["' + str.encode(name) + b'"] = 1 '
                print(b'buttons["' + str.encode(name) + b'"] = 1 ')
        # send the action to BizHawk
        action_code += b'joypad.set(buttons, 1) '
        action_code += b'emu.frameadvance() '

        if i == 0:
            paths.append(
                bizhawk_dirs + state_dirs + str(len(embeddings)) + '.State')
            proc.stdin.write(
                b'savestate.save("' + str.encode(paths[-1]) + b'") ')
            proc.stdin.write(
                b'savestate.load("' + str.encode(paths[state]) + b'") ')
            proc.stdin.flush()

        elif i == 1:
            proc.stdin.write(action_code)
            proc.stdin.flush()

        elif i == interval:
            img_lua()

        else:
            proc.stdin.write(b'joypad.set(buttons, 1) ')
            proc.stdin.write(b'emu.frameadvance() ')
            proc.stdin.flush()

    return (temp_action, next_state)


def random_goal():
    goal_img = imread(data_paths[np.random.randint(len(data_paths))])
    goal = embedded_model.predict(np.expand_dims(goal_img, axis=0))[0]
    '''
    goal = np.zeros(max_embedding.shape)
    for i in range(len(max_embedding)):
        goal[i] = np.random.uniform(min_embedding[i], max_embedding[i])
    '''
    return goal


def render_video(edge_list):
    return


def make_chart(scores):
    num = len(scores)
    ratio = np.zeros((num,))
    scores = np.array(scores)
    for i in range(1, num):
        temp = scores[:i]
        ratio[i] = sum(np.amax(temp, axis=0))
    '''
    for i in range(1, num):
        temp = scores[:i]
        diff = np.amax(temp, axis=0) - np.amin(temp, axis=0)
        ratio[i-1] = sum(np.ma.log(diff).filled(0))
    '''
    '''
    embedding_num = len(embeddings)
    embedding_size = len(embeddings[0])
    diff = max_embedding - min_embedding
    prev = 0
    for i in range(embedding_num):
        if i != 0:
            total = 0
            for j in range(embedding_size):
                total += abs(embeddings[i][j] - embeddings[0][j]) / diff[j]
            ratio[i] = max(total / embedding_size, prev)
            prev = ratio[i]
    '''

    plt.plot(ratio)
    plt.ylabel('Progress Ratio')
    plt.xlabel('Time')
    plt.show()


bizhawk_dirs = 'BizHawk/'
rom_dirs = 'Rom/'
rom_name = 'SuperMarioWorld.smc'
data_dirs = 'Data/'
model_dirs = 'Model/'
state_dirs = 'States/'

preparation = False
paths = []
embeddings = []
scores = []
outliers = []
action_dict = {}
data_paths = glob(data_dirs + '*')

original_embedding = np.load(model_dirs + 'embedding.npy')
input_model = load_model(model_dirs + 'input_model.h5')
embedded_model = load_model(model_dirs + 'embedded_model.h5')
inception = InceptionV3(weights='imagenet')
max_embedding = np.amax(original_embedding, axis=0)
min_embedding = np.amin(original_embedding, axis=0)

if __name__ == '__main__':
    actions = ['U', 'D', 'L', 'R', 's', 'S', 'Y', 'B', 'X', 'A', 'l', 'r']
    with open('Input Log.txt', 'r') as f:
        for i, line in enumerate(f):
            temp_action = 0
            for i, action in enumerate(actions):
                if line.find(action) != -1:
                    temp_action += 2 ** i
            if temp_action == 0 or i < 3:
                continue
            if temp_action in action_dict:
                action_dict[temp_action] += 1
            else:
                action_dict[temp_action] = 1

    win_unicode_console.enable()
    if not os.path.exists(bizhawk_dirs + state_dirs):
        os.mkdir(bizhawk_dirs + state_dirs)

    proc = subprocess.Popen([bizhawk_dirs + 'EmuHawk.exe',
                            rom_dirs + rom_name,
                            '--lua=../rrt.lua'],
                            stdout=subprocess.PIPE,
                            stdin=subprocess.PIPE)

    while True:
        out_line = proc.stdout.readline()

        # get rom name
        if out_line[:5] == b'start':
            print('Got the rom name')
            rom_name = out_line[6:-1]
            preparation = True

        # started
        if preparation:

            proc.stdin.write(b'client.speedmode(400) ')
            proc.stdin.write(b'savestate.loadslot(1) ')
            proc.stdin.flush()
            print('line 203')
            img_lua()

            thelist = explore_with_rrt(0, get_random_successor,
                                       random_goal, state_projection,
                                       render_video, max_samples=1000)

            break

        else:
            print(out_line)

        # to terminate program after closing EmuHawk
        if out_line == b'':
            break

    proc.terminate()
    clear_session()
    # os.remove('temp_screenshot.png')
    # np.save('../thelist.npy', thelist)

    make_chart(scores)

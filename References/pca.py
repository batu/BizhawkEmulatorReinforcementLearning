import subprocess
import numpy as np
import sklearn.decomposition
from os import remove
from scipy.misc import imsave, imread
from moviepy.editor import *
from skimage.draw import line, set_color, ellipse, circle_perimeter
from cv2 import line, LINE_AA
import rrt

def get_byte():
	new_line = b''
	while new_line != b'pass\n':
		new_line = proc.stdout.readline()

	new_line = proc.stdout.readline()
	nums = new_line[:-1].split()
	state_num.append(nums)
	new_line = proc.stdout.readline()
	while new_line != b'continue\n':
		new_line = proc.stdout.readline()

def read_byte_lua(num):
	code = b''
	code += b'io.stdout:write(mainmemory.readbyte(' + str.encode(str(num)) + b')) '
	code += b'io.stdout:write(" ") '
	return code

def pass_byte_lua():
	proc.stdin.write(b'io.stdout:write("pass\\n") ')
	code = b''
	for i in range(ram_size):
		code += read_byte_lua(i)
	code += b'io.stdout:write("\\n") io.stdout:write("continue\\n") '
	proc.stdin.write(code)

def state_projection(state):
	p = np.zeros(ram_size, dtype=int)
	for i in range(ram_size):
		p[i] = int(str(state_num[state][i])[2:-1])
	temp_pca = pca.transform(p.reshape(1, -1))
	return temp_pca[0]

def get_random_successor(state, goal):
	real_action = np.random.randint(4096) #4095
	action = "{0:b}".format(real_action).rjust(12, '0')
	next_state = len(state_num)
	interval = 60

	#do action
	for i in range(interval + 1):
		action_code = b''
		action_code += b'buttons = {} '
		if action[4] == '1':
			action_code += b'buttons["Select"] = 1 '
		if action[5] == '1':
			action_code += b'buttons["Start"] = 1 '
		if action[6] == '1':
			action_code += b'buttons["B"] = 1 '
		if action[7] == '1':
			action_code += b'buttons["A"] = 1 '
		if action[8] == '1':
			action_code += b'buttons["X"] = 1 '
		if action[9] == '1':
			action_code += b'buttons["Y"] = 1 '
		if action[10] == '1':
			action_code += b'buttons["L"] = 1 '
		if action[11] == '1':
			action_code += b'buttons["R"] = 1 '

		if action[0] == '1' and action[1] != '1':
			action_code += b'buttons["Up"] = 1 '
		if action[1] == '1' and action[0] != '1':
			action_code += b'buttons["Down"] = 1 '
		if action[2] == '1' and action[3] != '1':
			action_code += b'buttons["Left"] = 1 '
		if action[3] == '1' and action[2] != '1':
			action_code += b'buttons["Right"] = 1 '
		if action[0] == '1' and action[1] == '1':
			if np.random.randint(2) == 0:
				action_code += b'buttons["Up"] = 1 '
			else:
				action_code += b'buttons["Down"] = 1 '
		if action[2] == '1' and action[3] == '1':
			if np.random.randint(2) == 0:
				action_code += b'buttons["Left"] = 1 '
			else:
				action_code += b'buttons["Right"] = 1 '

		action_code += b'joypad.set(buttons, 1) '
		action_code += b'emu.frameadvance() '

		if i == 0:
			paths.append('../States/node_' + str(len(state_num)) + '.State')
			proc.stdin.write(b'savestate.save("' + str.encode(paths[-1]) + b'") ')
			proc.stdin.write(b'savestate.load("' + str.encode(paths[state]) + b'") ')
			proc.stdin.flush()

		elif i == 1:
			proc.stdin.write(action_code)
			proc.stdin.flush()

		elif i == interval:
			pass_byte_lua()
			proc.stdin.flush()
			get_byte()

		else:
			proc.stdin.write(b'joypad.set(buttons, 1) ')
			proc.stdin.write(b'emu.frameadvance() ')
			proc.stdin.flush()

	return (real_action, next_state)

def render_video_(edge_list):
	return

def render_video(edge_list):
	print('Video Rendering')
	length = len(edge_list)
	projection_num = edge_list[0][4].shape[0]
	starts = np.zeros(length, dtype=int)
	ends = np.zeros(length, dtype=int)
	for i, edge in enumerate(edge_list):
	    _, starts[i], _, ends[i], _, _ = edge

	clip_time = 0.1
	map_img = imread('../Files/Super Metroid (Japan, USA) (En,Ja)/SuperMetroid-SpaceColony.png')
	maps = np.zeros((length, map_img.shape[0], map_img.shape[1], map_img.shape[2]))
	clip_list = []
	temp_path = 'temp.png'

	#original image
	imsave(temp_path, map_img)
	temp_clip = ImageClip(temp_path).set_duration(clip_time).resize(0.5)
	clip_list.append(temp_clip)
	remove(temp_path)

	#draw
	for i in range(length):
		print('\r{}'.format(i), end='')

		last = starts[i]
		start = [int(str(state_num[last][2806])[2:-1]) + int(str(state_num[last][2807])[2:-1]) * 256,
				 int(str(state_num[last][2810])[2:-1]) + int(str(state_num[last][2811])[2:-1]) * 256]
		end = [int(str(state_num[i+1][2806])[2:-1]) + int(str(state_num[i+1][2807])[2:-1]) * 256,
			   int(str(state_num[i+1][2810])[2:-1]) + int(str(state_num[i+1][2811])[2:-1]) * 256]

		if i != 0:
			#set_color(map_img, (previous_rr, previous_cc), [1, 1, 1])
			map_img[previous_rr_node, previous_cc_node] = previous_node_img
			map_img = line(map_img,
						   (int(previous_start[0]), int(previous_start[1])),
						   (int(previous_end[0]), int(previous_end[1])),
						   (255, 255, 255), 10, LINE_AA)

		#show edges
		map_img = line(map_img, (int(start[0]), int(start[1])),
					   (int(end[0]), int(end[1])), (255, 0, 0), 10, LINE_AA)
		previous_start = start
		previous_end = end

		#show node
		rr_node, cc_node = circle_perimeter(int(end[1]), int(end[0]), 20)
		previous_node_img = map_img[rr_node, cc_node]
		map_img[rr_node, cc_node] = [255, 1, 255]
		previous_rr_node, previous_cc_node = rr_node, cc_node
		#map_img = circle(map_img, (end[0], end[1]), 10, (32, 128, 0), -1,
		#				 LINE_AA)

		maps[i] = map_img

	#combine text and image
	for i, one_map in enumerate(maps):
		print('\r{}'.format(i), end='')

		imsave(temp_path, one_map)
		temp_clip = ImageClip(temp_path).set_duration(clip_time).resize(0.5)
		step_clip = TextClip('Step: ' + str(i+1), font='Amiri-Bold',
							 fontsize=30, color='white')
		step_clip = step_clip.set_duration(clip_time).set_pos(('left', 'bottom'))
		temp_clip = CompositeVideoClip([temp_clip, step_clip])
		clip_list.append(temp_clip)

	#save as mp4
	remove(temp_path)
	print("Concatenating Videoclips")
	video = concatenate_videoclips(clip_list)
	video.write_videofile('../a.mp4', fps=25)

preparation = 0
paths = []
state_num = []

ram_size = 4096
PCA_Num = 256
npy = np.load('../Files/Super Metroid (Japan, USA) (En,Ja)/npy/backbone_states.npy')[:, :ram_size]
pca = sklearn.decomposition.PCA(n_components=PCA_Num)
npy_pca = pca.fit_transform(npy)
max_pca = np.amax(npy_pca, axis=0)
min_pca = np.amin(npy_pca, axis=0)

if __name__ == '__main__':
	proc = subprocess.Popen(['EmuHawk.exe', 'SNES/mario.smc',
							'--lua=' + 'Lua/test_rrt.lua'],
							stdout=subprocess.PIPE,
							stdin=subprocess.PIPE)

	while True:
		out_line = proc.stdout.readline()

		#get rom name
		if out_line[:5] == b'start':
			rom_name = out_line[6:-1]

		#preparation
		if out_line[:22] == b'Selecting display size':
			preparation += 1

		#started
		if preparation == 6:
			proc.stdin.write(b'savestate.loadslot(2) ')
			proc.stdin.flush()
			proc.stdin.write(b'client.speedmode(400) ')
			pass_byte_lua()
			proc.stdin.flush()
			get_byte()

			thelist = rrt.explore_with_rrt(0, get_random_successor,
									   	   (min_pca,max_pca),
										   state_projection,
										   render_video_, max_samples=100)
			break

		else:
			print(out_line)

		#to terminate program after closing EmuHawk
		if out_line == b'':
			break

	proc.terminate()
	np.save('../thelist.npy', thelist)

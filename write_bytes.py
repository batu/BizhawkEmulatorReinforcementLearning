# $7E:000
import subprocess
import win_unicode_console
import os

bizhawk_dirs = 'BizHawk/'
rom_dirs = 'Rom/'
rom_name = 'SuperMarioWorld.smc'
data_dirs = 'Data/'
model_dirs = 'Model/'
state_dirs = 'States/'

ram_size = 4096


def start_bizhawk_process():
	win_unicode_console.enable()
	if not os.path.exists(bizhawk_dirs + state_dirs):
		os.mkdir(bizhawk_dirs + state_dirs)

	proc = subprocess.Popen([bizhawk_dirs + 'EmuHawk.exe',
							rom_dirs + rom_name,
							'--lua=../rrt.lua'],
							stdout=subprocess.PIPE,
							stdin=subprocess.PIPE)
	print("Start completed.")
	return proc


def start_bizhawk_game(proc):
	started = False
	while True:
		out_line = proc.stdout.readline()

		# get rom name
		if out_line[:5] == b'start':
			print("Started!")
			started = True

		# started
		if started:
			proc.stdin.write(b'client.speedmode(400) ')
			proc.stdin.write(b'savestate.loadslot(1) ')
			proc.stdin.flush()
			break
		else:
			pass


def read_byte_lua(num):
	code = b''
	code += b'io.stdout:write(mainmemory.readbyte(' + str.encode(str(num)) + b')) '
	code += b'io.stdout:write(" ") '
	return code


# 0F34 for score
# 0094(148) 0095(149)
def get_x_location():
	proc.stdin.write(b'io.stdout:write("Start\\n") ')
	code = b'io.stdout:write(serialize(mainmemory.readbyterange(147, 3)) )'
	proc.stdin.write(code)
	proc.stdin.flush()

	new_line = proc.stdout.readline()
	while new_line != b'Start\n':
		new_line = proc.stdout.readline()

	new_line = proc.stdout.readline().split()[1:-1]
	dist_1 = new_line[0][:-1]
	dist_2 = new_line[1]
	print((int(dist_1) + int(dist_2) * 255) / 10)


def send_byte_read_command():
	proc.stdin.write(b'io.stdout:write("pass\\n") ')
	code = b''
	for i in range(ram_size):
		code += b'io.stdout:write(mainmemory.readbyte(' + str.encode(str(i)) + b')) '
		code += b'io.stdout:write(" ") '
	code += b'io.stdout:write("\\n") io.stdout:write("continue\\n") '
	proc.stdin.write(code)
	proc.stdin.flush()


def send_byte_bulk_read_command():
	proc.stdin.write(b'io.stdout:write("pass\\n") ')
	code = b''
	code += b'io.stdout:write(serialize(mainmemory.readbyterange(0, 4098)) )'
	code += b'io.stdout:write(" ") '
	code += b'io.stdout:write("\\n") io.stdout:write("continue\\n") '
	proc.stdin.write(code)
	proc.stdin.flush()


def receive_bytes_from_lua():
	global state_num
	new_line = b''

	while new_line != b'pass\n':
		new_line = proc.stdout.readline()

	new_line = proc.stdout.readline()
	nums = new_line[:-1].split()
	state_num = nums

	new_line = proc.stdout.readline()
	while new_line != b'continue\n':
		new_line = proc.stdout.readline()
	return state_num

	# print(int.from_bytes(read_bytes[1], byteorder="little"))


def get_ram_state():
	send_byte_bulk_read_command()
	byte_values = receive_bytes_from_lua()[1:-1]
	results = []
	for byte_value in byte_values:
		results.append(int(byte_value[:-1]))
	print(byte_values)
	print(results)
	# byte_values = receive_bytes_from_lua()
	# RAM_state = []
	# for byte_value in byte_values:
	#	RAM_state.append(int.from_bytes(byte_value, byteorder="little", signed=False))


proc = start_bizhawk_process()
start_bizhawk_game(proc)

# 0F34 for score
# 0094 0095

state_num = []
while True:
	proc.stdin.write(b'emu.frameadvance() ')
	# send_byte_read_command()
	# receive_bytes_from_lua()
	get_x_location()
	for _ in range(12):
		proc.stdin.write(b'emu.frameadvance() ')
		proc.stdin.flush()

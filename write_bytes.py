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
			proc.stdin.write(b'client.speedmode(100) ')
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


def send_byte_read_command():
	proc.stdin.write(b'io.stdout:write("pass\\n") ')
	code = b''
	for i in range(ram_size):
		code += b'io.stdout:write(mainmemory.readbyte(' + str.encode(str(i)) + b')) '
		code += b'io.stdout:write(" ") '
	code += b'io.stdout:write("\\n") io.stdout:write("continue\\n") '
	proc.stdin.write(code)
	proc.stdin.flush()


def receive_bytes_from_lua():
	global state_num
	new_line = b''

	while new_line != b'pass\n':
		new_line = proc.stdout.readline()

	print("Got the pass.")
	new_line = proc.stdout.readline()
	nums = new_line[:-1].split()
	print(nums)
	state_num = nums

	new_line = proc.stdout.readline()
	while new_line != b'continue\n':
		new_line = proc.stdout.readline()
	print("Got the cont.")


# 0F34 for score
# 0094(148) 0095(149)
def get_x_location():
	code = b'io.stdout:write(mainmemory.readbyte(' + str.encode(str(149)) + b')) '
	code += b'io.stdout:write(" ") '
	proc.stdin.write(code)
	proc.stdin.flush()
	new_line = proc.stdout.readline()

	try:
		read_byte = new_line[:-1].split()[0]
		print(int.from_bytes(read_byte, byteorder="little", signed=False))
	except IndexError:
		print("Errored!")
	# print(int.from_bytes(read_bytes[1], byteorder="little"))


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
	for _ in range(16):
		proc.stdin.write(b'emu.frameadvance() ')
		proc.stdin.flush()

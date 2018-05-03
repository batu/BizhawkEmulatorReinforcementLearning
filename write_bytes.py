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
            print(out_line)
            pass

proc = start_bizhawk_process()
start_bizhawk_game(proc)
proc.stdin.write(b'emu.frameadvance() ')
proc.stdin.write(b'mainmemory.readbyterange(10, 10)')

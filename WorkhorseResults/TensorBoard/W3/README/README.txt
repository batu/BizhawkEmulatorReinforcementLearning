Experiment start time: Thu May 17 20:24:13 2018

Algorithm:
DQN.

The Changes:
Reduce the gamma from .99 to .9

Reasoning:
I want the system to make shorter term plans and be more reactive.

Hypothesis:
.9 might be a bit too much. Especially because the -reward for death doesnt kick in until the screen fades to black (not at collision) But I expect less running away from Bill.

Results:
Seem to have more peaks but the floor has not risen. I will increase it a little but not to .99

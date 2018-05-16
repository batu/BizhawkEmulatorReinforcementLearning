Experiment start time: Wed May 16 15:15:33 2018

Algorithm:
DQN.

The Changes:
Reduced the state space reprsenetation from 1024, 4 concatination to 1.

Reasoning:
The timeline information between frames might not be preserved between frames.
This might be detrimental to the training as it might just add noise.

Hypothesis:
The training will be more stable.

Results:


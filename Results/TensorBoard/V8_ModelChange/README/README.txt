Experiment start time: Wed May 23 16:39:58 2018

Algorithm:
DQN.

The Changes:
Changed Model to have different structure that condenses information from each frame. Chaned epmax to .5

Reasoning:
Adam suggested it, and the fact that we have seperated the temporal aspect of the frames from the learning frames themselves makes a lot of sense. This also controls the param size b removing the 1024 direct step

Hypothesis:
No specific expecations apart from increased performance.

Results:


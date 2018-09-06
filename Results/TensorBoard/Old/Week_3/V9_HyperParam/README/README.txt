Experiment start time: Thu May 24 11:22:58 2018

Algorithm:
DQN.

The Changes:
Keeping the model constant but changing several parameters. batch_size is 32 -> 256. model update faster. 1024 memory and epsilon reducing faster

Reasoning:
Tried to preserve the DQN paper hyperparameters  ratios with the memory and epsilon reducing. with model update and batch size I want to increase time spent training in TF compared to bizhawk running.

Hypothesis:
No specific expecations apart from increased performance.

Results:


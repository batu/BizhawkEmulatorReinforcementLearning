Experiment start time: Thu May 24 15:55:01 2018

Algorithm:
DQN.

The Changes:
 Chaned the policy to BoltzmannGumbelQPolicy

Reasoning:
So we want the agent to act confidently where it is confident (initial parts) but then explore when it gets to areas it doesnt know well. That seems to be captured by
boltzman. The Gumbel is a "improed versin" apparently. Other hyperparameters are set as defaults 

Hypothesis:
 Mario shall not stop.

Results:


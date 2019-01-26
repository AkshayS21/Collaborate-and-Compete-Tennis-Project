# Collaborate-and-Compete-Tennis-Project
Implementation of the MultiAgent Reinforcement Learning Algorithm 


This is the repository for the Tennis Project. It is an implementation of the Multi-Agent DDPG paper by OpenAI [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf) to solve the Tennis environment which is similar to, but not identical to the Tennis environment on the Unity ML-Agents GitHub page.


## Environment

![](gifs/1.218.gif)

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

For example:
```
Number of agents: 2
Size of each action: 2
There are 2 agents. Each observes a state with length: 24
The state for the first agent looks like: [ 0.          0.          0.          0.          0.          0.         0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.         -6.65278625 -1.5        -0.          0.
  6.83172083  6.         -0.          0.        ]
rewards shape: (2,)

```

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). 

After each episode, the rewards that each agent received for each step are added (without discounting), to get a score for each agent. This yields two scores. The maximum of these two scores is then the episode score.
The environment is considered solved, when the average of 100 episode scores is at least +0.5.

## Instructions

The environment can be downloaded from the following links depending on the OS:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

(_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.

After downloading, please place the contents of the zip file in your workspace directory.

## The Algorithm
- Please read the report.md file for complete description of the algorithm including the networks, the multi-agent module (maddpg), the   individual agent module (ddpg) and the training code (marl).
- Please use the Tennis.ipynb file to explore the environment and training steps of the algorithm. 
- Please see Tennis_Evaluate.ipynb file to see how the different levels of solved environment weights perform.

Run ``` python train.py ``` to train new weights and run ``` python evaluate.py``` to evaluate your trained weights.

I have made this project as much modular as possible with the following modules:

- model.py - contains the network for both Actor and Critic models.
- ddpg.py - contains the code for an individual agent to execute its fucntions.
- maddpg.py - code for creating a meta agent (I called it "Parent") and executing its functions.
- buffer.py - code for adding and sampling experiences for the agents.
- OUNoise.py - code for creating and samping noise for the agent actions.
- workspace_utils.py - includes code for keeping the session active.
- train.py - includes code to train your own weights.
- evaluate.py - includes code to evaluate trained weights.

Please see the report.md file for a complete description of the algorithm, the network architecture, the steps taken to reach the final results, the interpretation of the results and how I decided the most stable weights. The report.md file concludes with ideas for future work on this environment.


The plot of the final results showing #Episodes vs Average score over last hundred episodes is as follows:

![](gifs/results.JPG)



### Note:
- If the jupyter notebook does not display for some reason, please copy the link to the notebook and use this website -   https://nbviewer.jupyter.org/







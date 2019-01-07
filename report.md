# Collaborate and Compete – Tennis Project

## Implementation of the Multi-Agent Reinforcement Learning Algorithm


### Networks

Actor model:

![](images/actor_model.PNG)

Critic Model:

![](images/critic_model.PNG)



### Hyperparameters

- Learning_rate = 0.001
- Tau = 0.001
- noise_start = 2, noise_decay = 0.999
- batch_size = 128, buffer_size = 1e6
- gamma = 1
- OU Noise parameters
  - mu=0, theta=0.15, sigma=0.2 


### Algorithm

I have named my Multi-Agent clas as Parent. When an instance of Parent is created, it creates a 2-Agent DDPG agent as follows:
```
self.madagents = [ddpg.ddpg(24, 2, 256, 128 , 64 ), ddpg.ddpg(24, 2, 256, 128, 64)]

```
Each DDPG agent has two actor models(local and target) and two critic models(local and target). During an episode, the environment generates two states(one for each agent). These states are fed to the parent agent to generate respective actions of each DDPG agent.
If there are enough experiences in the buffer memory(more than the batch size) then the parent agent samples batch size memories for each agent, so that it can learn and update its actor and critic local networks.

Updating the networks:


Each DDPG agent updates its local actor and critic networks during every update step of the parent agent. The update algorithm for a DDPG agent, say agent_0, is as folows:
- Computing the Advantage function:
  - The sampled next_states of both agents are fed to their current version of actor_target models to generate next_actions after
    next_states.
  - The next_actions are concatenated with next_states and fed to the critic_target model of agent_0 to generate action_value, Q_next       given next_states and next_actions.
  - The Advantage function is computed as rewards(for agent_0 for taking actions after observing states) + gamma * Q_next.
  
    Hence Q_next acts like the cumulative reward for next_states.
   
- Training the critic_local model:
  - states and actions sampled are concatenated and fed to the current version of critic_local model of agent_0 to predict Q_expected       for states.
  
  - The critic_local model of agent_0 is then trained such that Q_expected matches the Advantage function. The difference in their           values is computed as the mean squared error loss and the critic_local network is optimized using Adam optimizer.
    
- Training the actor_local model:
  - The sampled states for both agents are fed to their current version of the actor_local model to predict actions,say actions_pred,       for both agents.
  
  - These actions_pred are concatenated with states and fed to the current version of critic_local model of agent_0 to compute               Q_current.
  
  - The actor_local model of agent_0 is then optimized by minimizing the value -1* Q_current, in other words by maximizing Q_current.       Again I used Adam optimizer.
    
- Soft Updating the Target Networks:

  The actor_target and critic_target networks of agent_0 are soft_updated to gradually match their repective local networks.
  
The same steps are followed for the second agent as well.

### Randomness

The OU Noise that is usually added to actions to generate more random states wasn't exploratory enough for this environment. Hence I made the actions 100% random by modifying a uniform distribution (0,1] to (-1 ,1] and randomly sampling actions from this distribution. This helped generate random consecutive states and made the algorithm much more exploratory.


### Attempts

1. In the first attempt, I set the randomness to 100% for first 1000 episodes and 50% until 2000 episodes, updated the networks every 2 time steps and didn't set any limit on the number of time steps per episode. The result was as follows,

![](images/first.JPG)

The environment was solved but was unstable. The maximum average score per 100 episodes was +0.543.

2. In the scond attempt, I kept all parameters the same but updated the networks every time step. 

![](images/second.JPG)

As expected, the environment was solved with much higher score, but was also more unstable.The maximum average score per 100 episodes was +0.890.

3. In the third attempt, in order to make the algorithm more stable, I decided to update the networks less often. Hence I updated the network every 2 time steps and limited the number of time steps per episode to 1000. To compensate for less often training, I increased the exploratory characteristics of the agents by selecting actions with 100% randomness for first 1500 epsiodes and 50% thereafter until 2500 episodes. The results were as follows,

![](images/third.JPG)

The maximum average score per 100 episodes is +1.969.

### Evaluation

In order to select the most stable weights, we need to look for the score at which the algorithm was stable and stayed for a long time.
On a side note, the average score per 100 episodes for solving this environemnt is +0.5 but the Benchmark mean score per episode is +2.5. 

In the plot above, we see that the maximum average score per 100 episodes was +1.969. But the algorithm crashes right after achieving this score. When I assigned the weights corresponding to this score to the agents and ran 5 episodes, I got a very fluctuating mean score as seen below. The benchmark score of +2.5 was never reached.

![](images/1.969.JPG)

We can see in the plot that the algorithm fluctuates between approximately 1 and 1.5 from episodes 3800 to 5000. The algorithm never stayes stable at 1 or 1.5 but fluctuates in between these scores. This means that it is stable somewhere in between these scores. This can be verified from the mean scores below.

Evaluation of weights corresponding to +1.595.

![](images/1.595.JPG)

Evaluation of weights corresponding to +1.002.

![](images/1.002.JPG)

As we can see, the weight above exceed the benchmark mean scores per episode of +2.5 but they are not consistent.
Hence in order to find the stable weights, I looked for weights corresponding to a score that is around the mean of 1 and 1.5. The closest weights were those corresponding to score of +1.218. Following was their evaluation result.

![](images/1.218.JPG)

The mean episode score consistently exceeded +2.5. So we can conclude that the corresponding weights are stable and can be used for the environment. Following is their performance,

Stable +1.218

![](gifs/1.218.gif)

Less Stable +1.595

![](gifs/1.595.gif)


Less Stable +1.002

![](gifs/1.002.gif)


Unstable +1.969

![](gifs/1.969.gif)

The evaluation codes can be found in the file Tennis_Evaluate.ipynb.


### Ideas for Future





























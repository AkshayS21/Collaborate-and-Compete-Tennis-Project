# Collaborate and Compete – Tennis Project

## Implementation of the Multi-Agent Reinforcement Learning Algorithm


### 1.Networks

Actor model:

![](images/actor_model.PNG)

Critic Model:

![](images/critic_model.PNG)


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
 - next_states of both agents are fed to their current version of actor_target models to generate next_actions after next_states.
 - The next_actions are concatenated with next_states and fed to the critic_target model of agent_0 to action_value Q_next given   
   next_states and next_actions.
 - The Advantage function is computed as rewards(for agent_0 for taking actions after observing states) + gamma * Q_next
   Hence Q_next acts like the cumulative reward for next_states.
   
- Training the critic_local model
 - states and actions sampled are concatenated and fed to the current version of critic_local model of agent_0 to predict Q_expected for    states.
 - The critic_local model of agent_0 is then trained such that Q_expected matches the Advantage function. The difference in their values    is computed as the mean squared error loss and the critic_local network is optimized using Adam optimizer.











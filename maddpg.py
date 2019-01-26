
# coding: utf-8

# In[1]:


import ddpg
import importlib
importlib.reload(ddpg)
import numpy as np
import buffer
importlib.reload(buffer)

import torch
import torch.nn as nn
import torch.nn.functional as f

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


buffer_size = int(1e6)
batch_size = 128


class Parent():
    def __init__(self, action_size = 2, buffer_size = buffer_size , n_agents = 2 ,\
                 batch_size = batch_size , seed = 2, update_every = 1 , gamma = 1):
        
        self.madagents = [ddpg.ddpg(24, 2, 256, 128 , 64 ), ddpg.ddpg(24, 2, 256, 128, 64)]
        
        
        self.update_every = update_every
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.memory = buffer.ReplayBuffer(action_size, buffer_size, batch_size, seed = 2)
        #self.t_step = 0
        self.n_agents = n_agents
        self.gamma = gamma
    
    
    
    def step(self, states, actions, rewards, next_states, dones, t_step):
        
        #states = states.reshape(1,-1)
        #next_states = next_states.reshape(1,-1)
        
        self.memory.add(states, actions, rewards, next_states, dones)
        
        if len(self.memory) > self.batch_size:
            if t_step % self.update_every == 0:
                
                for a_i in range(self.n_agents):
                    experience = self.memory.sample()
                    self.update(experience,agent_number = a_i )
        #self.t_step +=1
        
    def get_local_actors(self):
        actors = [self.madagents[i].actor_local for i in range(self.n_agents)]
        return actors
    
    def get_target_actors(self):
        actors = [self.madagents[i].actor_target for i in range(self.n_agents)]
        
        
    def local_actions(self, states,noise,rand):
        #print("sx",states.shape)
        #actors = self.get_local_actors()
        actions = [agent.local_act(state,noise,rand) for agent,state in zip(self.madagents,states)]
        #print("ax",len(actions))
        return actions
        
    
    
                   
    def target_actions(self, next_states):
        #print("nx:",next_states.shape)
        #next_states = next_states.reshape(self.batch_size,self.n_agents,24)
        
        batch_actions = []
        for i, agent in enumerate(self.madagents):
        
            index = torch.tensor([i]).to(device)
            next_state = next_states.reshape(-1, 2, 24).index_select(1, index).squeeze(1)     
            
            batch_actions.append(agent.actor_target(next_state))
        
        return batch_actions #torch.from_numpy(actions).float().to(device)
    
    def updated_actions(self,states):
        #index = torch.tensor([agent_number]).to(device)
        #states = states.reshape(self.batch_size,self.n_agents,24)
        
        batch_actions = []
        for i, agent in enumerate(self.madagents):
            index = torch.tensor([i]).to(device)
            state = states.reshape(-1, 2, 24).index_select(1, index).squeeze(1)
        
        #batch_actions = []
            batch_actions.append(agent.actor_local(state))
       
        return batch_actions #torch.from_numpy(actions).float().to(device)
        
        
    
    
    def update(self, experience,agent_number):
        
        states, actions, rewards, next_states, dones = experience
        
        #print("actions:",actions.shape)
        #print("actions:",states.shape)
        
        
        flat_states = states.reshape(self.batch_size,-1)  #1x48
        
        flat_next_states = next_states.reshape(self.batch_size,-1) # 128x48
        
        
        ### make q_target_next from critic target.Input size is 2x(states+actions)flattened
        
        
        
        critic_target_actions = self.target_actions(next_states)# change if reqd remove from_numpy in act
        
        critic_target_actions = torch.cat(critic_target_actions, dim = 1).to(device) # 1x4
        critic_input_target = torch.cat((flat_next_states,critic_target_actions), dim = 1).to(device) # 128x52
        
        
        agent = self.madagents[agent_number]
        
        agent.critic_optimizer.zero_grad()
        
        with torch.no_grad():
            q_target_next = agent.critic_target(critic_input_target)  ### 64x1
            
        
        
        index = torch.tensor([agent_number]).to(device)
        
        y = rewards.index_select(1, index) + (self.gamma * q_target_next * (1 - dones.index_select(1, index)))
        
        
        actions_reshaped= actions.reshape(self.batch_size,-1)
        critic_input_local = torch.cat((flat_states,actions_reshaped), dim = 1).to(device)
        
        
        q_pred = agent.critic_local(critic_input_local)
        
        #huber_loss = torch.nn.SmoothL1Loss()
        
        critic_loss = f.mse_loss( q_pred , y.detach())
        
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(agent.critic_local.parameters(), 1)
        agent.critic_optimizer.step()
        
        
        ###actor update###
        
        agent.actor_optimizer.zero_grad()
        
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        
        critic_local_actions = self.updated_actions(states)
        critic_local_actions = torch.cat(critic_local_actions, dim = 1).to(device)
        
        critic_input_l2 = torch.cat((flat_states,critic_local_actions), dim = 1).to(device)  #make to device
        
        expected_return = -agent.critic_local(critic_input_l2).mean()
        
        expected_return.backward()
        agent.actor_optimizer.step()
        
        ### soft_update###
        
        agent.soft_update(agent.actor_target,agent.actor_local)
        agent.soft_update(agent.critic_target,agent.critic_local)
        
    def save_models(self,last_max):
        
        for i,agent in enumerate(self.madagents):
            torch.save(agent.actor_local.state_dict(),f"actor_agent_{i}_{last_max:.3f}.pth")
            torch.save(agent.critic_local.state_dict(),f"critic_agent_{i}_{last_max:.3f}.pth")
            
            
    def reset_actions(self):
        
        for agent in (self.madagents):
            agent.reset_action()
            
            
        
        
        
     

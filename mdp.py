# Import libraries 
import numpy as np
import pandas as pd
import glob as glob

# PyTorch 
import torch
import torch.nn as nn
import torch.nn.functional as F

class mdp(object):
    '''
    Framing of financial portfolio management problem as a Markov Decision Process. 
    This is based on the mathematical framework outlined in:
    A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem
    https://arxiv.org/abs/1706.10059
    '''
    
    def __init__(self,
                 df,
                 look_back = 30,
                 eval_time = "Close",
                 features = 5,
                 seed = 0):
        '''
        @param look_back, (int) Number of previous time periods to include in state
        @param features, (int) Number of features per asset
        @param seed, (int) seed for random number generators
        '''
        
        # Set seed
        np.random.seed(seed)
        
        # Store data
        self.df = df
        self.data = np.array(data.drop(columns=['Date']))
        
        # Indicies in self.data with eval_time information
        self.eval_idx = [i - 1 for i, s in enumerate(list(df.columns)) 
                         if eval_time in s]
        
        # Dimension of continuous action space
        self.action_dim = int((len(df.columns) - 1) / features) + 1
        
        # Dimension of state space
        self.state_dim = (len(df.columns) - 1) * look_back
        
        # Initial portfolio value
        self.value = 1.0
        self.values = [self.value]
        
        # First row with full state information
        self.look_back = look_back
        self.start_row = look_back + 1
        
        # Last row with next state still available
        self.end_row = df.shape[0] - 2
        
        # Initialize time
        self.time = self.start_row
        
    def unnormalized_prev_state(self):
        '''
        Returns unnormalized previous state
        '''
        return self.data[self.time - self.look_back - 1: self.time - 1].flatten()
        
    def unnormalized_state(self):
        '''
        Returns unnormalized current state
        '''
        return self.data[self.time - self.look_back: self.time].flatten()
    
    def unnormalized_next_state(self):
        '''
        Returns unnormalized next state
        '''
        return self.data[self.time - self.look_back + 1: self.time + 1].flatten()
    
    def state(self):
        '''
        Returns normalized state
        '''
        return self.unnormalized_state() / self.unnormalized_prev_state()
    
    def next_state(self):
        '''
        Returns normalized next state
        '''
        return self.unnormalized_next_state() / self.unnormalized_state()
    
    def reward(self, u_s, a, u_sp):
        '''
        Returns reward measured as log percent increase of portfolio 
        
        @param u_s, unnormalized state
        @param a, action
        @param u_sp, unnormalized next state
        
        @returns r, reward
        '''
        
        # Assert action is valid
        assert(np.isclose(a.sum(), 1.0))
        
        # Extract prices at evaluation time
        old_price = u_s[self.eval_idx]
        new_price = u_sp[self.eval_idx]
        
        # Append cash, percent change in each stock
        delta = np.concatenate(([1.0], new_price / old_price), axis = None)
        
        # Log percent change in portfolio
        reward = np.log(np.dot(delta, a))
        
        return reward
    
    def sample_action(self):
        '''
        Returns a random action
        '''
        
        # Generate random numbers
        a = np.random.random(self.action_dim)
        
        # Normalize
        a = a / a.sum()
        
        return a
    
    
    def step(self, a):
        '''
        Take a step in the environment
        '''
        
        # Extract transition information
        u_s = self.unnormalized_state()
        s = self.state()
        u_sp = self.unnormalized_next_state()
        sp = self.next_state()
        
        # Collect reward
        r = self.reward(u_s, a, u_sp)
        
        # Update portfolio value
        self.value *= np.exp(r)
        
        # Record portfolio value
        self.values.append(self.value)
        
        # Check if episode done
        done = (self.time == self.end_row)
        
        # If episode done, reset time
        if done:
            self.reset()
        # Otherwise, increment time
        else:
            self.time += 1
            
        return s, a, sp, r, done
    
    def reset(self):
        
        # Reset time and value
        self.t = 0
        self.value = 1.0     
        
        return self.state(), False
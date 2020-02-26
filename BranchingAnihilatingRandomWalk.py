# BranchingAnihilatingRandomWalk.py

''' The code creates the Mammary ductal glands evolution by the method of
    statistical branching-annihilating random walk.
'''

import numpy as np
import time
import matplotlib.pyplot as plt

# Class creating the 2D space and random walkers

class BARandomWalk:
    
    def __init__(self, N, sigma):
        # N - dimensions of the 2d space
        # sigma - ratio of frequency between branching and random walking
        
        self.N = N
        self.prob_branch = sigma / (1 + sigma)
        
        # 2d plane which contain information about trace
        self.board = np.zeros((N,N))
        # list of random walkers which are numpy arrays of two numbers - their
        # coordinates on the board
        self.r_walkers = []
        
        # assigning random position to the first walker
        self.r_walkers.append(np.random.randint(N, size=(2)))
        
    def one_step_evolution(self):
        
        # position of walker in r_walkers
        i_walker = 0
        for walker in self.r_walkers:
            r_branch = np.random.rand()
            # Random walking
            if r_branch > self.prob_branch:
                r_move = 4 * np.random.rand()
                move_vector = ((r_move < 1) * np.array([1,0]) 
                              +(r_move >= 1 and r_move < 2) * np.array([0,1])
                              +(r_move >= 2 and r_move < 3) * np.array([-1,0])
                              +(r_move >=3) * np.array([0,-1]))
                # leaving a trace
                self.board[walker[0], walker[1]] = 1
                walker += move_vector
                # deleting walker if it hits the trace or the ends of board
                if ((self.board[walker[0],walker[1]] == 1) or (walker[0] < 0) or 
                    (walker[0] >= self.N) or (walker[1] < 0) or 
                    (walker[1] >= self.N)): 
                    self.r_walkers.pop(i_walker)
                    i_walker -= 1
            else:
                r_spawn = 4*np.random.rand()
                spawn_vector = ((r_spawn < 1) * np.array([1,0]) 
                              +(r_spawn >= 1 and r_spawn < 2) * np.array([0,1])
                              +(r_spawn >= 2 and r_spawn < 3) * np.array([-1,0])
                              +(r_spawn >=3) * np.array([0,-1]))
                # spawning new walker
                new_walker = walker + spawn_vector
                self.r_walkers.append(new_walker)
                if ((self.board[new_walker[0],new_walker[1]] == 1) or (new_walker[0] < 0) or 
                    (new_walker[0] >= self.N) or (new_walker[1] < 0) or 
                    (new_walker[1] >= self.N)):
                    self.r_walkers.pop(-1)
            i_walker += 1
        
        
    def whole_evolution(self, no_steps):
        
        n_iter = 0
        while(len(self.r_walkers) > 0 and n_iter < no_steps):
            self.one_step_evolution()
            n_iter += 1
            
            
    def return_trace_matrix(self):
        
        return self.board
    
    
    
# main code

def main():
    N = 100
    sigma = 0.5
    no_steps = 200000
    
    randomWalkGenerator = BARandomWalk(N, sigma)
    randomWalkGenerator.whole_evolution(no_steps)
    
    trace = randomWalkGenerator.return_trace_matrix()
    
    plt.figure(1)
    plt.imshow(trace)
    plt.show()
    
main()
        
        
        
        
        
        
        
        
        
        
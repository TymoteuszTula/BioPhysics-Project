# BranchingAnihilatingRandomWalk.py

''' The code creates the Mammary ductal glands evolution by the method of
    statistical branching-annihilating random walk.
'''

import numpy as np
import time
import matplotlib.pyplot as plt

# Constants

VEC_UP = np.array([1,0])
VEC_DOWN = np.array([-1,0])
VEC_LEFT = np.array([0,-1])
VEC_RIGHT = np.array([0,1])

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
                if ((walker[0] < 0) or 
                    (walker[0] >= self.N) or (walker[1] < 0) or 
                    (walker[1] >= self.N)): 
                    self.r_walkers.pop(i_walker)
                    i_walker -= 1
                elif (self.board[walker[0],walker[1]] == 1):
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
                if ((new_walker[0] < 0) or 
                    (new_walker[0] >= self.N) or (new_walker[1] < 0) or 
                    (new_walker[1] >= self.N)):
                    self.r_walkers.pop(-1)
                elif (self.board[new_walker[0],new_walker[1]] == 1):
                    self.r_walkers.pop(-1)
            i_walker += 1
        
        
    def whole_evolution(self, no_steps):
        
        n_iter = 0
        while(len(self.r_walkers) > 0 and n_iter < no_steps):
            self.one_step_evolution()
            n_iter += 1
            
            
    def return_trace_matrix(self):
        
        return self.board
    


class BARandomWalk2:
    
    def __init__(self, N, sigma):
        
        self.N = N
        self.prob_branch = sigma / (1 + sigma)
        self.board = np.zeros((N,N))
        self.r_walkers = []
        
        self.r_walkers.append([np.random.randint(N, size=(2)), 0])
        
    def one_step_evolution(self):
        
        i_walker = 0
        for walker in self.r_walkers:
            r_branch = np.random.rand()
            if r_branch > self.prob_branch:
                move_vector, last_move = self.move_walker(walker)
                self.board[walker[0][0], walker[0][1]] = 1
                walker[0] += move_vector
                walker[1] = last_move
                
                if ((walker[0][0] < 0) or (walker[0][0] >= self.N) or (
                    walker[0][1] < 0) or (walker[0][1] >= self.N)):
                    self.r_walkers.pop(i_walker)
                    i_walker -= 1
                elif (self.board[walker[0][0], walker[0][1]] == 1):
                    self.r_walkers.pop(i_walker)
                    i_walker -= 1
                    
            else:
                r_spawn = 4*np.random.rand()
                spawn_vector = ((r_spawn < 1) * VEC_LEFT
                              +(r_spawn >= 1 and r_spawn < 2) * VEC_RIGHT
                              +(r_spawn >= 2 and r_spawn < 3) * VEC_UP
                              +(r_spawn >=3) * VEC_DOWN)
                # spawning new walker
                new_walker = [np.copy(walker[0]), 0]
                new_walker[0] += spawn_vector
                self.r_walkers.append(new_walker)
                if ((new_walker[0][0] < 0) or 
                    (new_walker[0][0] >= self.N) or (new_walker[0][1] < 0) or 
                    (new_walker[0][1] >= self.N)):
                    self.r_walkers.pop(-1)
                elif (self.board[new_walker[0][0],new_walker[0][1]] == 1):
                    self.r_walkers.pop(-1)

            i_walker += 1
        
    
    def whole_evolution(self, no_steps):
        
        n_iter = 0
        while(len(self.r_walkers) > 0 and n_iter < no_steps):
            self.one_step_evolution()
            n_iter += 1
            print("hey")
            
            
    def return_trace_matrix(self):
        
        return self.board
    
    
    
    
    def move_walker(self, walker):
        
        if walker[1] == 0:
            rand = np.random.randint(4)
            print(rand)
            return (((rand == 0) * VEC_UP + (rand == 1) * VEC_DOWN +
                    (rand == 2) * VEC_RIGHT + (rand == 3) * VEC_LEFT),
                    rand + 1)
        if walker[1] == 1:
            rand = np.random.randint(3)
            print(rand)
            return (((rand == 0) * VEC_DOWN +
                    (rand == 1) * VEC_RIGHT + (rand == 2) * VEC_LEFT),
                    rand + 2)
        if walker[1] == 2:
            rand = np.random.randint(3)
            print(rand)
            return (((rand == 0) * VEC_UP +
                    (rand == 1) * VEC_RIGHT + (rand == 2) * VEC_LEFT),
                    (rand == 0) * (rand + 1) + (rand > 0) *
                    (rand + 2))
        if walker[1] == 3:
            rand = np.random.randint(3)
            print(rand)
            return (((rand == 0) * VEC_UP + (rand == 1) * VEC_DOWN 
                    + (rand == 2) * VEC_LEFT),
                    (rand < 2) * (rand + 1) + (rand == 2) *
                    (rand + 2))
        if walker[1] == 4:
            rand = np.random.randint(3)
            print(rand)
            return (((rand == 0) * VEC_UP + (rand == 1) * VEC_DOWN +
                    (rand == 2) * VEC_RIGHT,
                    rand + 1))
        
        return (0,0)

class BARandomWalk3:

    def __init__(self, N, sigma):
        self.prob_branch = sigma / (1 + sigma)
        self.board = np.zeros((N,N))
        self.r_walkers = []
        self.r_walkers.append(np.random.randint(N, size=(2)))
        self.N = N

    def one_epoch_evolution(self):
        i_walker = 0
        new_walkers = []
        
        while i_walker < len(self.r_walkers):
            r_branch = np.random.rand()

            if r_branch > self.prob_branch:
                r_move = np.random.randint(4)
                next_move = ((r_move == 0) * VEC_UP +
                             (r_move == 1) * VEC_DOWN +
                             (r_move == 2) * VEC_LEFT +
                             (r_move == 3) * VEC_RIGHT)
                self.board[(self.r_walkers[i_walker][0], 
                            self.r_walkers[i_walker][1])] = 1
                self.r_walkers[i_walker] += next_move

                if ((self.r_walkers[i_walker][0] < 0) or
                    (self.r_walkers[i_walker][0] >= self.N) or
                    (self.r_walkers[i_walker][1] < 0) or
                    (self.r_walkers[i_walker][1] >= self.N)):
                    self.r_walkers.pop(i_walker)
                    i_walker -= 1
                elif (self.board[(self.r_walkers[i_walker][0] , 
                            self.r_walkers[i_walker][1])] == 1):
                    self.r_walkers.pop(i_walker)
                    i_walker -= 1
            else:
                r_born = np.random.randint(4)
                born_vector = ((r_born == 0) * VEC_UP +
                             (r_born == 1) * VEC_DOWN +
                             (r_born == 2) * VEC_LEFT +
                             (r_born == 3) * VEC_RIGHT)
                new_walker = np.copy(self.r_walkers[i_walker]) + born_vector

                if ((new_walker[0] < 0) or
                    (new_walker[0] >= self.N) or
                    (new_walker[1] < 0) or
                    (new_walker[1] >= self.N)):
                    pass
                elif (self.board[new_walker[0], new_walker[1]] == 1):
                    pass
                else:
                    new_walkers.append(new_walker)
            i_walker += 1
            
        self.r_walkers = self.r_walkers + new_walkers

    def whole_evolution(self):
        while self.r_walkers != []:
            self.one_epoch_evolution()

    def show_trace(self):
        return self.board

VEC_DOWN_RIGHT = np.array([-1, 1])
VEC_DOWN_LEFT = np.array([-1, -1])
VEC_UP_RIGHT = np.array([1, 1])
VEC_UP_LEFT = np.array([1, -1])


class BARandomWalk4:
    def __init__(self, N, sigma):
        self.prob_branch = sigma / (1 + sigma)
        self.board = np.zeros((N,N))
        self.r_walkers = []
        self.r_walkers.append(np.random.randint(N, size=(2)))
        self.N = N

    def one_epoch_evolution(self):
        i_walker = 0
        new_walkers = []
        
        while i_walker < len(self.r_walkers):
            r_branch = np.random.rand()

            if r_branch > self.prob_branch:
                r_move = np.random.randint(8)
                next_move = ((r_move == 0) * VEC_UP +
                             (r_move == 1) * VEC_DOWN +
                             (r_move == 2) * VEC_LEFT +
                             (r_move == 3) * VEC_RIGHT +
                             (r_move == 4) * VEC_DOWN_LEFT +
                             (r_move == 5) * VEC_DOWN_RIGHT +
                             (r_move == 6) * VEC_UP_LEFT +
                             (r_move == 7) * VEC_UP_RIGHT)
                self.board[(self.r_walkers[i_walker][0], 
                            self.r_walkers[i_walker][1])] = 1
                self.r_walkers[i_walker] += next_move

                if ((self.r_walkers[i_walker][0] < 0) or
                    (self.r_walkers[i_walker][0] >= self.N) or
                    (self.r_walkers[i_walker][1] < 0) or
                    (self.r_walkers[i_walker][1] >= self.N)):
                    self.r_walkers.pop(i_walker)
                    i_walker -= 1
                elif (self.board[(self.r_walkers[i_walker][0] , 
                            self.r_walkers[i_walker][1])] == 1):
                    self.r_walkers.pop(i_walker)
                    i_walker -= 1
            else:
                r_born = np.random.randint(8)
                born_vector = ((r_born == 0) * VEC_UP +
                             (r_born == 1) * VEC_DOWN +
                             (r_born == 2) * VEC_LEFT +
                             (r_born == 3) * VEC_RIGHT +
                             (r_born == 4) * VEC_DOWN_LEFT +
                             (r_born == 5) * VEC_DOWN_RIGHT +
                             (r_born == 6) * VEC_UP_LEFT +
                             (r_born == 7) * VEC_UP_RIGHT)
                new_walker = np.copy(self.r_walkers[i_walker]) + born_vector

                if ((new_walker[0] < 0) or
                    (new_walker[0] >= self.N) or
                    (new_walker[1] < 0) or
                    (new_walker[1] >= self.N)):
                    pass
                elif (self.board[new_walker[0], new_walker[1]] == 1):
                    pass
                else:
                    new_walkers.append(new_walker)
            i_walker += 1
            
        self.r_walkers = self.r_walkers + new_walkers

    def whole_evolution(self):
        while self.r_walkers != []:
            self.one_epoch_evolution()

    def show_trace(self):
        return self.board

                
def fnn_phase_transition(sigma_min = 0, sigma_max = 2,
                         sigma_step = 0.01, av_no = 100,
                         N = 100):
    sigma = np.arange(sigma_min, sigma_max, sigma_step)
    density = np.zeros(sigma.shape[0])
    h = 0

    for i_sigma in sigma:
        for _ in range(av_no):
            tempRandomWalk = BARandomWalk3(N, i_sigma)
            tempRandomWalk.whole_evolution()
            density[h] += np.sum(tempRandomWalk.show_trace())
        density[h] /= av_no
        h += 1
        print("Progress: ", i_sigma * 100 / sigma_max, 
              "% completed")

    return sigma, density

def snn_phase_transition(sigma_min = 0, sigma_max = 2,
                         sigma_step = 0.01, av_no = 10,
                         N = 100):
    sigma = np.arange(sigma_min, sigma_max, sigma_step)
    density = np.zeros(sigma.shape[0])
    h = 0

    for i_sigma in sigma:
        for _ in range(av_no):
            tempRandomWalk = BARandomWalk4(N, i_sigma)
            tempRandomWalk.whole_evolution()
            density[h] += np.sum(tempRandomWalk.show_trace())
        density[h] /= av_no
        h += 1
        print("Progress: ", i_sigma * 100 / sigma_max, 
              "% completed")

    return sigma, density  



# main code

def main(case):

    if case == 1:
        N = 100
        sigma = 0.94
        
        randomWalkGenerator = BARandomWalk4(N, sigma)
        randomWalkGenerator.whole_evolution()
        
        trace = randomWalkGenerator.show_trace()
        
        plt.figure(1)
        plt.imshow(trace)
        plt.show()
    elif case == 2:
        sigma1, density1 = fnn_phase_transition()
        sigma2, density2 = snn_phase_transition()

        plt.figure(1)
        plt.plot(sigma1, density1)
        plt.title("Nearest neighbour phase transition")

        plt.figure(2)
        plt.plot(sigma2, density2)
        plt.title("Second nearest neighbour phase transition")

        plt.show()
    
main(2)
        
     
        
        
        
        
        
        
        
        

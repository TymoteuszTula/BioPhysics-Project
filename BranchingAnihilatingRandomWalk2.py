import numpy as np
import time
import matplotlib.pyplot as plt

# Constants

VEC_UP = np.array([1,0])
VEC_DOWN = np.array([-1,0])
VEC_LEFT = np.array([0,-1])
VEC_RIGHT = np.array([0,1])


class BARandomWalk3:

    def __init__(self, N, sigma):
        self.prob_branch = sigma / (1 + sigma)
        self.board = np.zeros((N,N))
        self.r_walkers = []
        self.r_walkers.append(np.random.randint(N, size=(2)))
        self.N = N
        self.term_probability = []

    def one_epoch_evolution(self):
        i_walker = 0
        new_walkers = []
        init_no_of_walkers = len(self.r_walkers)
        
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
        
        final_no_of_walkers = len(self.r_walkers)
        self.term_probability.append((init_no_of_walkers - final_no_of_walkers) /
                                     init_no_of_walkers)
            
        self.r_walkers = self.r_walkers + new_walkers
        

    def whole_evolution(self):
        while self.r_walkers != []:
            self.one_epoch_evolution()

    def show_trace(self):
        return self.board
    
    def show_term_probability(self):
        return self.term_probability
    
class BARandomWalk:
    
    def __init__(self, N, sigma):
        self.prob_branch = sigma / (1 + sigma)
        self.board = np.zeros((N,N))
        self.r_walkers = []
        self.r_walkers.append([np.random.randint(N, size=(2)), 0, 0])
        self.N = N
        self.generation = []

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
                self.board[(self.r_walkers[i_walker][0][0], 
                            self.r_walkers[i_walker][0][1])] = 1
                self.r_walkers[i_walker][0] += next_move

                if ((self.r_walkers[i_walker][0][0] < 0) or
                    (self.r_walkers[i_walker][0][0] >= self.N) or
                    (self.r_walkers[i_walker][0][1] < 0) or
                    (self.r_walkers[i_walker][0][1] >= self.N)):
                    self.generation.append(np.copy(self.r_walkers[i_walker][1:]))
                    self.r_walkers.pop(i_walker)
                    i_walker -= 1
                elif (self.board[(self.r_walkers[i_walker][0][0] , 
                            self.r_walkers[i_walker][0][1])] == 1):
                    self.generation.append(np.copy(self.r_walkers[i_walker][1:]))
                    self.r_walkers.pop(i_walker)
                    i_walker -= 1
            else:
                self.r_walkers[i_walker][2] += 1
                r_born = np.random.randint(4)
                born_vector = ((r_born == 0) * VEC_UP +
                             (r_born == 1) * VEC_DOWN +
                             (r_born == 2) * VEC_LEFT +
                             (r_born == 3) * VEC_RIGHT)
                new_walker = (np.copy(self.r_walkers[i_walker]) +
                            [born_vector, 0, 0])
                new_walker[1] = new_walker[2] - 1

                if ((new_walker[0][0] < 0) or
                    (new_walker[0][0] >= self.N) or
                    (new_walker[0][1] < 0) or
                    (new_walker[0][1] >= self.N)):
                    pass
                elif (self.board[new_walker[0][0], new_walker[0][1]] == 1):
                    pass
                else:
                    new_walkers.append(np.copy(new_walker))
            i_walker += 1
            
        self.r_walkers = self.r_walkers + new_walkers
        

    def whole_evolution(self):
        while self.r_walkers != []:
            self.one_epoch_evolution()

    def show_trace(self):
        return self.board
    
    def show_generations(self):
        return self.generation
    
def main(case):

    if case == 1:
        N = 100
        sigma = 1
        
        randomWalkGenerator = BARandomWalk(N, sigma)
        randomWalkGenerator.whole_evolution()
        
        trace = randomWalkGenerator.show_trace()
        
        plt.figure(1)
        plt.imshow(trace)
        plt.show()
        
        gen_array = np.array(randomWalkGenerator.show_generations())
        print(gen_array[:100])
        term_probability = []
        
        for gen in range(1, 35):
            no_of_bi = np.sum(gen == gen_array[:,0])
            no_of_an = np.sum(gen == gen_array[:,1])
            term_probability.append(no_of_an / (no_of_bi + no_of_an))
            
        plt.figure(2)
        plt.plot(term_probability)
        plt.show()
            
    if case == 2:
        N = 100
        sigma = 1.5
        mean = 1000
        
        mean_term_probability = np.zeros(36)
        no_of_mean = np.zeros(36)
        
        for m in range(mean):
            randomWalkGenerator = BARandomWalk(N, sigma)
            randomWalkGenerator.whole_evolution()
            
            gen_array = np.array(randomWalkGenerator.show_generations())
            for gen in range(0, 35):
                no_of_bi = np.sum(gen == gen_array[:,0])
                no_of_an = np.sum(gen == gen_array[:,1])
                if (no_of_bi + no_of_an):
                    mean_term_probability[gen] += no_of_an / (no_of_bi + no_of_an)
                    no_of_mean[gen] += 1
            print("m = ", m)
                
        
        mean_term_probability /= no_of_mean
        
        print(mean_term_probability)
        
        plt.figure(1)
        plt.plot(mean_term_probability)
        plt.show()
        
            
        
        
main(2)

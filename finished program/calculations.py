# -*- coding: utf-8 -*-
 
import numpy as np

'''Calculate the energy/hamiltonian of the given configuration'''
def energy(state, Jh, Jv, B): 
    L = len(state)
    h = 0
    for i in range(L):
        for j in range(L):
            # central/chosen spin @ site
            s = state[i,j]
            
            # nearest neighbours of the chosen spin
            '''% modulo for next index to be the first resp. the last in the column/line if out of bounds
            only upper and right neighbour --> enought for complete the energy calculation'''
            
            nnb = Jv[i,j]*state[(i+1)%L, j] + Jh[i,j]*state[i, (j+1)%L]
            h -= s*(nnb + B)
    return h

'''Calculate Magnetization of the given configuration (sum of all spins)'''
def magnet(state): 
    mm = np.sum(state)
    return mm
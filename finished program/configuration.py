# -*- coding: utf-8 -*-

import numpy as np

'''Create and save a random LxL matrix with -1, 1 elements/spins:'''
def random_config(L):
     a = 2*np.random.randint(2, size=(L,L)) - 1
     return a

def randomJ(L):
     sigma = 1.
     jh = np.random.normal(0, sigma, size=(L,L))
     jv = np.random.normal(0, sigma, size=(L,L))
     return jh, jv
 
'''Metropolis Monte Carlo random importance sampling (sweeps) with pass conditions and magnetic field B'''                       
def sflip(state, beta, Jh, Jv, B): 
    L = len(state)
    dEs = 0
    for i in range(L**2):
        x = np.random.randint(0, L)
        y = np.random.randint(0, L)
        s = state[x,y]
        nnb = Jv[x,y]*state[(x+1)%L, y] + Jv[(x-1)%L,y]*state[(x-1)%L, y] + Jh[x,y]*state[x, (y+1)%L] + Jh[x,(y-1)%L]*state[x, (y-1)%L]
        
        '''Difference between energy of flipped spin site and original site in magnetic field B
        # an attempt to determining a formula for calculating the hamiltonian per spin yields:'''
        dE = 2*s*(nnb + B)
        
        if dE < 0 or np.random.rand() < np.exp(-dE*beta):
            
            #spin @state(x,y) flipped if above condition met
            s *= -1
            dEs += dE
            
        # passing either initial or flipped spin
        state[x,y] = s 
    return state, dEs
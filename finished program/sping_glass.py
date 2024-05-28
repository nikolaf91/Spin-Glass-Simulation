# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 23:26:04 2022

@author: StudentIn
"""

import numpy as np
import configuration
import calculations
import animations
import matplotlib.pyplot as plt
from tqdm import tqdm #for progessbar and timer
from multiprocessing import Pool, cpu_count


#############################################################################
#                                                                           #
# (Metropolis) Monte Carlo random sampling/spin flips with pass conditions  #
#                                                                           #
#############################################################################


def main():
    '''reciprocal temperature b = ß(J) = 1/T with Boltzamnn constant k_b set to 1'''

    '''Animation for temperature dependent evolution of systems for 2D Ising/Spin-Glas model'''
    def plot1():  
        L = 50 # size of LxL lattice
        nT = 31 # number of temperature points
        T = np.linspace(0.1, 5.1 , nT)
        equ = 100
        A1 = configuration.random_config(L)
        X = np.zeros([L,L,len(T)])
        
        # choose initial Spin-Glas or Ising system:
            
        #Jh, Jv = configuration.randomJ(L) #spin-glass
        Jh, Jv = np.ones((L,L)), np.ones((L,L)) #ising
        
        for num2, t in tqdm(enumerate(T)):
            A = A1.copy()
            beta = 1./t
            for j in range(equ):
                A, dE = configuration.sflip(A, beta, Jh, Jv, 0.)
            X[:,:,num2] = A 
        anim = animations.bw(X, T)
        #change filename and/or directory if needed
        anim.save("anim.gif")
    #plot1()

    '''average energy/magnetization per spin, heat capacity C and Edwards-Anderson parameter q'''
    def plot2():
        
        L = 30
        L2 = L**2
        nT = 31 # number of temperature points
        T = np.linspace(0.1, 5.1 , nT)
        beta = 1./T
        equ1 = 300 # equilibration steps/time for parllel temp. 
        equ2 = 500 # equilibration steps/time for C and q
        N = 100 # number of states to average per run
        
        # Magnetic field strength
        #B = np.linspace(0., 1., 5)
        B = [0.]
        
        #parallel tempering threshold temperature array
        tc = T[:10] 
        
        # choose (randomized) initial Spin-Glas or Ising system:
            
        #Jh, Jv = configuration.randomJ(L) #spin-glass
        Jh, Jv = np.ones((L,L)), np.ones((L,L)) #ising
        A1 = configuration.random_config(L) 
        
        fig, axes = plt.subplots(2, 2, figsize=(24,16))
        
        '''Parallel tempering / Replica exchange algorithm for low temperatures (due to energy barrier)'''
        for num, b in enumerate(B):
            print("Field strength:", b)
            
            def para_temp(t_start, t_end):
                print("Parallel tempering from T =", t_start,"to T =", t_end)
                Ax = np.empty([L, L, len(tc)])
                Ex, dEx = np.empty([len(tc)]), np.empty([len(tc)])
                
                #intialising len(tc) copies for parallel tempering algorithm
                for i in range(len(tc)): 
                    Ax[:,:,i] = A1 
                Ex[:] = calculations.energy(A1, Jh, Jv, b)
                
                for i in tqdm(range(equ1)):
                    for num0 in range(len(tc)):
                        Ax[:,:,num0], dEx[num0] = configuration.sflip(Ax[:,:,num0], beta[num0], Jh, Jv, b)
                    Ex[:] += dEx[:]
                    
                    for num1 in range(20):
                        u = np.random.randint(1,len(tc))
                        delta = (beta[u] - beta[u-1])*(Ex[u] - Ex[u-1])
                        #accepted when when the energy of the higher temperature replica is 
                        #lower than the one of the colder replica or random[0,1] < exp(..)
                        
                        if delta > 0 or np.random.rand() < np.exp(delta):
                            sw = Ax[:,:,u - 1] #swapping arrays
                            Ax[:,:,u - 1] = Ax[:,:,u].copy()
                            Ax[:,:,u] = sw.copy()
                            
                return Ax
            Ax = para_temp(tc[0], tc[-1])
            
            def process(t_start, t_end, A1, Ax): 
                en, en2, m = np.array([]), np.array([]), np.array([])
                c = np.array([])
                for num2, t in tqdm(enumerate(T)):
                    k = 0
                    X3d, X23d = np.empty([L,L,int(N/10)+1]), np.empty([L,L,int(N/10)+1])
                    A0 = np.zeros([L,L])
                    E2s, Es, M = 0, 0, 0
                    start_anim = False
                    
                    if t<tc[-1]:
                        A = Ax[:,:,num2]
                    if t>=tc[-1]:
                        A = A1.copy()
                        for j in range(equ2):
                            A, dE = configuration.sflip(A, beta[num2], Jh, Jv, b)
                    E = calculations.energy(A, Jh, Jv, b)
                    for i in range(N):
                        A, dE = configuration.sflip(A, beta[num2], Jh, Jv, b) 
                        M += np.abs(calculations.magnet(A))
                        E += dE
                        Es += E
                        E2s += E**2
                        At = A.copy()
                        A0 += At  
                       
                        '''Colorgrid Animation'''
                        if t==T[4] or t==T[10] or t==T[20]:
                            if i>0 and i%10==0:
                                Xx = A0.copy()/10
                                X3d[:,:,k] = Xx
                                X23d[:,:,k] = np.square(Xx)
                                A0 = np.zeros([L,L])
                                k+=1
                                start_anim = True
                    if start_anim:    
                        anim = animations.color(X3d, X23d, t)
                        anim.save("animforwT{:.5}.gif".format(t))
                
                    X = A0/N
                    X2 = np.square(X)
                    en = np.append(en, Es)
                    en2 = np.append(en2, E2s)
                    m = np.append(m, M)
                    c = np.append(c, calculations.magnet(X2))
                    
                dH2 = en2/N - np.square(en/N) #energy fluctuations/variance
                C = dH2/np.square(T)
                
                ax1, ax2, ax3, ax4 = axes[0,0], axes[0,1], axes[1,0], axes[1,1]
                colors = ["r","b", "y", "g", "m"]
                ax1.plot(T, en/(L2*N), linestyle="-" , color=colors[num], label = "B = {0}".format(b))
                ax2.plot(T, m/(L2*N), linestyle="-" ,color=colors[num], label = "B = {0}".format(b))
                ax3.plot(T, C/L2, linestyle="-" ,color=colors[num], label = "B = {0}".format(b))
                ax3.plot(T, np.gradient(en, T[1]-T[0])/(L2*N), linestyle="dotted" ,color=colors[num], label = "B = {0}".format(b))
                ax4.plot(T, c/L2, linestyle="-" ,color=colors[num], label = "B = {0}".format(b))
                return A, ax1, ax2, ax3, ax4 
            
            A, ax1, ax2, ax3, ax4 = process(T[0], T[-1], A1, Ax)
            
        ax1.set_xlabel("Temperature (T)", fontsize=20)
        ax1.set_ylabel("<$H_B$>", fontsize=20)  
        ax1.set_xticks(np.linspace(0.1, 5.1, 11))
        ax1.grid()        
        ax1.legend(prop={"size":20})
        ax2.set_xlabel("Temperature (T)", fontsize=20)
        ax2.set_ylabel("<$M_B$>", fontsize=20) 
        ax2.set_xticks(np.linspace(0.1, 5.1 , 11))
        ax2.grid()
        ax2.legend(prop={"size":20})
        ax3.set_xlabel("Temperature (T)", fontsize=20)
        ax3.set_ylabel(r"(solid: $c= \frac{<(\delta H^2)>}{k_b T^2}$)(dotted:$c= \frac{d<H>}{dT}$)", fontsize=20) 
        ax3.set_xticks(np.linspace(0.1, 5.1 , 11))
        ax3.grid()
        ax3.legend(prop={"size":20})
        ax4.set_xlabel("Temperature (T)", fontsize=20)
        ax4.set_ylabel("Edwards-And. param. q", fontsize=20) 
        ax4.set_xticks(np.linspace(0.1, 5.1 , 11))
        ax4.grid()
        ax4.legend(prop={"size":20})
        plt.show()
        
    #plot2()
        
    '''Evolution of magnetization for different temperatures'''
    def plot3():
        L = 15
        L2 = L**2
        B = 0.
        N = 1000 # number of averaged states
        T = np.linspace(0.1, 5.1 , 6)
        A1 = configuration.random_config(L)
        # choose Spin-Glas or Ising
        #Jh, Jv = configuration.randomJ(L) #spin-glass
        Jh, Jv = np.ones((L,L)), np.ones((L,L)) #ising
        plt.figure(figsize=(12, 8))
        for j in tqdm(T):
            print("Temperature:", j)
            A = A1.copy()
            m = np.array([])
            b = 1./j
            for i in range(N):
                A, dE = configuration.sflip(A, b, Jh, Jv, B)
                M = calculations.magnet(A)
                m = np.append(m, M)
            plt.plot(m/L2, label="T = {0}".format(j))
    
        plt.xlabel("Timings", fontsize=15)
        plt.ylabel("Magnetization", fontsize=15)
        plt.grid()
        plt.legend()
        plt.show()
    #plot3()
    
    '''Hysterisis (curve) for different temperatures t and system size L'''
    def plot4(): 
        L = 5
        T = [0.5, 1., 1.5, 2.5, 4., 6.]
        N = 100 # number of averaged states
        n = 1./(N*L**2)
        B = np.linspace(-1.5, 1.5,101)
        A1 = configuration.random_config(L)
        # choose Spin-Glas or Ising system:
        #Jh, Jv = configuration.randomJ(L) #spin-glass
        Jh, Jv = np.ones((L,L)), np.ones((L,L)) #ising
        colors = ["r","b", "y", "g", "m", "k"]
        plt.figure(figsize=(12, 8)) 
        for num, t in enumerate(T):
            beta = 1./t
            A = A1.copy()
            m = np.array([])
            print("forward loop for T =", t)
            for b in tqdm(B):
                M = 0
                for i in range(N):
                    A, dE = configuration.sflip(A, beta, Jh, Jv, b)
                    M += calculations.magnet(A)
                m = np.append(m, M)
            plt.plot(B, m*n, linestyle="-" , color=colors[num], label = "T = {0}".format(t))
            print("backwards loop: for T =", t)
            m = np.array([])
            for b in tqdm(B[::-1]):
                M = 0
                for l in range(N):
                    A, dE = configuration.sflip(A, beta, Jh, Jv, b)
                    M += calculations.magnet(A)
                m = np.append(m, M)
            plt.plot(B, m[::-1]*n, linestyle="-" , color=colors[num])
    
        plt.xlabel("B", fontsize=15)
        plt.ylabel("<M$_T$>", fontsize=15)
        plt.grid()
        plt.axvline(0, color="k")
        plt.legend(prop={"size":12})
        plt.show()
    #plot4()
    
    '''Magnetic susceptibility'''
    def plot5(): 
        #L = np.array([6, 9, 12, 15, 18, 21])
        L = np.array([5])
        L2 = L**2
        T = np.linspace(0.1, 5.1 , 12)
        beta = 1/T
        B = 0
        global glassJ, IsingJ #make global to avoid multiprocessing.Pool picking Error
        colors = ["r","b","g", "y","c","m"]
        
        '''Spin-Glass system'''
        def glassJ(r):
            N = 700 # number of averaged states
            equ1 = 1500
            equ2 = 500
            Jn = 10
            tc = T[:25] #temperature array for parallel tempering
            d = np.array([])
            fig, axes = plt.subplots(1, 2, figsize=(24, 7))
            ax1, ax2 = axes[0], axes[1]
            Xl = np.empty([len(T),len(L)])
            for num0, l in enumerate(L):
                Xn = np.zeros(len(T))
                for j in range(Jn):
                    A1 = configuration.random_config(l)
                    Jh, Jv = configuration.randomJ(l)
                    
                    def para_temp():
                        Ax = np.empty([l, l, len(tc)])
                        Ex, dEx = np.empty([len(tc)]), np.empty([len(tc)])
                        for i in range(len(tc)): #intialising len(tc) copies for parallel tempering algorithm
                            Ax[:,:,i] = A1.copy()
                            Ex[:] = calculations.energy(A1, Jh, Jv, B)
                        for i in range(equ1):
                            for num0 in range(len(tc)):
                                Ax[:,:,num0], dEx[num0] = configuration.sflip(Ax[:,:,num0], beta[num0], Jh, Jv, B)
                            Ex[:] += dEx[:]
                            for num1 in range(20): 
                                u = np.random.randint(1,len(tc))
                                #from higher temperatures to lower for effecitve implementation
                                delta = (beta[u] - beta[u-1])*(Ex[u] - Ex[u-1])
                                #accepted when when the energy of the higher temperature replica is 
                                #lower than the one of the colder replica or random[0,1] < exp(..)
                                if delta > 0 or np.random.rand() < np.exp(delta):
                                    sw = Ax[:,:,u - 1] #swapping arrays
                                    Ax[:,:,u - 1] = Ax[:,:,u].copy()
                                    Ax[:,:,u] = sw.copy()
                        return Ax
                            
                    Ax = para_temp()
                            
                    def process(Ax): 
                                
                        m , m2 = np.array([]), np.array([])
                        for num2, t in enumerate(T):
                            M, M2 = 0, 0
                            if t<tc[-1]:
                                A = Ax[:,:,num2]
                            if t>=tc[-1]:
                                A = A1.copy()
                                for j in range(equ2):
                                    A, dE = configuration.sflip(A, beta[num2], Jh, Jv, B)
                            for i in range(N):
                                A, dE = configuration.sflip(A, beta[num2], Jh, Jv, B) 
                                dM = calculations.magnet(A)
                                M += dM
                                M2 += dM**2
                                        
                            m = np.append(m, M)
                            m2 = np.append(m2, M2)
                                    
                        dM2 = m2/N - np.square(m/N)
                        X = dM2*beta # usually times L**2 but since one must divide by the same factor we leave it out
                        return X
                    X = process(Ax)
                    Xn += X
                
                Xl[:,num0] = Xn/(Jn*L2[num0])
                
                ax1.plot(T, Xl[:,num0], linestyle="-" , color=colors[num0], label="L = {0}".format(l))  
                
                ax1.vlines(T[np.argmax(Xl[:,num0])], 0., np.amax(Xl[:,num0]), color=colors[num0])
                d = np.append(d, T[np.argmax(Xl[:,num0])])
                    
                ax2.scatter(1./l, d[num0], color=colors[num0], label="L = {0}".format(l))

            ax1.set_xlabel("Temperature (T)", fontsize=15)
            ax1.set_ylabel(r"$\chi$", fontsize=15)
            ax1.set_xticks(np.linspace(0.1, 5.1 , 11))
            ax1.legend(prop={"size":12})
            ax1.grid()
            
            k, b = np.polyfit(1./L, d, 1)        
            ax2.plot(1./L, k/L + b, color = "k")
            ax2.set_xlabel("1/L", fontsize=15)
            ax2.set_ylabel("Temperature (T)", fontsize=15)
            ax2.legend(prop={"size":12})
            ax2.grid()
            plt.show()
            return Xl
 
        '''Ising system'''
        def IsingJ(r):
            T = np.linspace(2., 3. , 51)
            beta = 1/T
            N = 700 # number of averaged states
            equ = 1500 #equilibration time
            J = 5
            plt.subplots(figsize=(12, 7)) 
            fig, axes = plt.subplots(1, 2, figsize=(24, 7))
            ax1, ax2 = axes[0], axes[1]
            Xl = np.empty([len(T),len(L)])
            d = np.array([])
            for num0, l in enumerate(L):
                Jh, Jv = np.ones([l,l]), np.ones([l,l])
                Xn = np.zeros(len(T))
                for j in range(J):
                    A1 = configuration.random_config(l)
                    m , m2 = np.array([]), np.array([])
                    for num, t in enumerate(T):
                        M, M2 = 0, 0
                        A = A1.copy()
                        for j in range(equ):
                            A, dE = configuration.sflip(A, beta[num], Jh, Jv, B)
                        for i in range(N):
                            A, dE = configuration.sflip(A, beta[num], Jh, Jv, B) 
                            dM = calculations.magnet(A)
                            M2 += dM**2
                            M += dM
                        m = np.append(m, M)  
                        m2 = np.append(m2, M2)
                    dM2 = m2/N - np.square(m/N)
                    X = dM2*beta # usually times L**2 but since one must divide by the same factor we leave it out
                    
                    Xn += X
                    
                Xl[:,num0] = Xn/(J*L2[num0])
                    
                ax1.plot(T, Xl[:,num0], linestyle="-" , color=colors[num0], label="L = {0}".format(l))
                ax1.vlines(T[np.argmax(Xl[:,num0])], 0., np.amax(Xl[:,num0]), color=colors[num0])
                d = np.append(d, T[np.argmax(Xl[:,num0])])
                
                ax2.scatter(1./l, d[num0], color=colors[num0], label="L = {0}".format(l))
                    
            ax1.set_xlabel("Temperature (T)", fontsize=15)
            ax1.set_ylabel(r"$\chi$", fontsize=15)
            ax1.set_xticks(np.linspace(0.1, 5.1 , 11))
            ax1.legend(prop={"size":12})
            ax1.grid()
            
            k, b = np.polyfit(1./L, d, 1)        
            ax2.plot(1./L, k/L + b, color = "k")
            ax2.set_xlabel("1/L", fontsize=15)
            ax2.set_ylabel("Temperature (T)", fontsize=15)
            ax2.legend(prop={"size":12})
            ax2.grid()
            plt.show()
            return Xl
                
        def multi(check):
            threads = cpu_count()
            with Pool(threads) as p:
                if check == 1:
                    pG = p.map(glassJ, range(threads))
                    results = np.array(pG)
                   
                elif check == 2:
                    pI = p.map(IsingJ, range(threads))
                    results = np.array(pI)
                    
            p.close()
            p.join()
            Xn = np.average(results, axis=0)
            d = np.array([])
            fig, axes = plt.subplots(1, 2, figsize=(24, 7))
            ax1, ax2 = axes[0], axes[1]
            for num0, l in enumerate(L):
                ax1.plot(T, Xn[:, num0], linestyle="-", color=colors[num0], label="L = {0}".format(l))
                ax1.vlines(T[np.argmax(Xn[:, num0])], 0., np.amax(Xn[:, num0]), color=colors[num0])
                d = np.append(d, T[np.argmax(Xn[:, num0])])
                ax2.scatter(1. / l, d[num0], color=colors[num0], label="L = {0}".format(l))
            
            ax1.set_xlabel("Temperature (T)", fontsize=15)
            ax1.set_ylabel(r"$\chi$", fontsize=15)
            ax1.set_xticks(np.linspace(0.1, 5.1, 11))
            ax1.legend(prop={"size": 12})
            ax1.grid()
            
            k, b = np.polyfit(1. / L, d, 1)
            ax2.plot(1. / L, k / L + b, color="k")
            ax2.set_xlabel("1/L", fontsize=15)
            ax2.set_ylabel("Temperature (T)", fontsize=15)
            ax2.legend(prop={"size": 12})
            ax2.grid()
            plt.show()
            
        
        '''averaging over parallel processes/results'''
        if __name__ == '__main__':
            multi(1)  # run for spin glass susceptibility
            #multi(2) # run for standard Ising susceptibility

    #plot5()
    
    '''calculating Edwards-Anderson parameter q for different coupling configs J and averaged over these'''
    def plot6(): 
        L = 15
        L2 = L**2
        J = 6
        nT = 31 # number of temperature points
        T = np.linspace(0.1, 5.1 , nT)
        beta = 1./T
        equ = 80
        N = 10
        A1 = configuration.random_config(L)   
        c0 = np.zeros([len(T)])
        colors = ["r", "g", "y", "b", "c", "m"]
        plt.figure(figsize=(12, 8)) 
        for j in tqdm(range(J)):
            Jh, Jv = configuration.randomJ(L)
            print("J config{0}".format(j))
            c = np.array([])
            for num2, t in enumerate(T):
                A0 = np.zeros([L,L])
                A = A1.copy()
                for k in range(equ):
                    A, dE = configuration.sflip(A, beta[num2], Jh, Jv, 0)
                for i in range(N):
                    A, dE = configuration.sflip(A, beta[num2], Jh, Jv, 0) 
                    At = A.copy()
                    A0 += At  
                
                X2 = np.square(A0/N)
                c = np.append(c, calculations.magnet(X2))
                
            plt.plot(T, c/L2, linestyle="-" , linewidth=1, color=colors[j], label="q: J{0}".format(j))
            ct = c.copy()
            c0 += ct
        
        plt.plot(T, c0/(J*L2), marker="o", linestyle="-" ,color="k", label="$<q>_J$")
        plt.xlabel("Temperature (T)", fontsize=20)
        plt.ylabel("$q$", fontsize=20) 
        plt.xticks(np.linspace(0.1, 5.1 , 11))
        plt.legend()
        plt.grid()
        plt.show()
    #plot6()
        
    '''calculating q for different number of Monte Carlo sweeps and each averaged over 
    different coupling configs J Edwards Anderson Model'''
    def plot7():
        L = 8
        L2 = L**2
        J = 10
        nT = 26 # number of temperature points
        T = np.linspace(0.1, 3.6 , nT)
        beta = 1./T
        equ = 500
        N = np.array([10, 100, 1000, 10000])
        A1 = configuration.random_config(L)
        colors = ["r", "g", "y", "b", "c", "m"]
        plt.figure(figsize=(12, 8)) 
        for num, n in tqdm(enumerate(N)):
            print("N = {0}".format(n))
            c0 = np.zeros([len(T)])
            for k in range(J):
                Jh, Jv = configuration.randomJ(L) 
                c = np.array([])
                for num2, t in enumerate(T):
                    A0 = np.zeros([L,L])
                    A = A1.copy()
                    for k in range(equ):
                        A, dE = configuration.sflip(A, beta[num2], Jh, Jv, 0)
                    for i in range(n):
                        A, dE = configuration.sflip(A, beta[num2], Jh, Jv, 0) 
                        At = A.copy()
                        A0 += At  
                    
                    X2 = np.square(A0/n)
                    
                    c = np.append(c, calculations.magnet(X2))
                ct = c.copy()
                c0 += ct
                
            plt.plot(T, c0/(J*L2), linestyle="-" , linewidth=1, color=colors[num], label="{0} sweeps".format(n))

         
        plt.xlabel("Temperature (T)", fontsize=20)
        plt.ylabel("$q$", fontsize=20) 
        plt.xticks(np.linspace(0.1, 3.6 , 11))
        plt.legend()
        plt.grid()
        plt.show()
    #plot7()
    
    '''magnetic susceptibility by means of the method 2 formula'''
    def plot8(): 
        L = 10
        L2 = L**2
        nT = 51 # number of temperature points
        T = np.linspace(0.1, 5.1 , nT)
        beta = 1./T
        equ1 = 3500
        equ2 = 3000
        N = 3000 # number of averaged states
        tc = T[:13] #parallel tempering threshold temperature array
        Jh, Jv = configuration.randomJ(L)
        #Jh, Jv = np.ones((L,L)), np.ones((L,L))
        A1 = configuration.random_config(L) 
        fig, axes = plt.subplots(1, 2, figsize=(24,8))
        # parallel tempering/replica exchange for low temperatures (due to energy barrier)
        def para_temp(t_start, t_end):
            print("Parallel tempering from T =", t_start,"to T =", t_end)
            Ax = np.empty([L, L, len(tc)])
            Ex, dEx = np.empty([len(tc)]), np.empty([len(tc)])
            for i in range(len(tc)): #intialising len(tc) copies for parallel tempering algorithm
                Ax[:,:,i] = A1 
            Ex[:] = calculations.energy(A1, Jh, Jv, 0)
            for i in tqdm(range(equ1)):
                for num0 in range(len(tc)):
                    Ax[:,:,num0], dEx[num0] = configuration.sflip(Ax[:,:,num0], beta[num0], Jh, Jv, 0)
                Ex[:] += dEx[:]
                for num1 in range(20):
                    u = np.random.randint(1,len(tc))
                    delta = (beta[u] - beta[u-1])*(Ex[u] - Ex[u-1])
                    #accepted when when the energy of the higher temperature replica is 
                    #lower than the one of the colder replica or random[0,1] < exp(..)
                    if delta > 0 or np.random.rand() < np.exp(delta):
                        sw = Ax[:,:,u - 1] #swapping arrays
                        Ax[:,:,u - 1] = Ax[:,:,u].copy()
                        Ax[:,:,u] = sw.copy()
            return Ax
        Ax = para_temp(tc[0], tc[-1])
            
        def process(t_start, t_end, A1, Ax): 
            m = np.array([])
            c = np.array([])
            for num2, t in tqdm(enumerate(T)):
                A0 = np.zeros([L,L])
                M = 0
                if t<tc[-1]:
                    A = Ax[:,:,num2]
                if t>=tc[-1]:
                    A = A1.copy()
                    for j in range(equ2):
                        A, dE = configuration.sflip(A, beta[num2], Jh, Jv, 0)
                for i in range(N):
                    A, dE = configuration.sflip(A, beta[num2], Jh, Jv, 0) 
                    M += np.abs(calculations.magnet(A))
                    At = A.copy()
                    A0 += At  
                X = A0/N
                X2 = np.square(X)
                m = np.append(m, M)
                c = np.append(c, calculations.magnet(X2))
            
            q = c/L2
            S = beta*(1-q)
            
            ax1, ax2 = axes[0], axes[1]
            ax1.plot(T, S, linestyle="-" , color="r", label = "L = {0}".format(L))
            ax2.plot(T, q, linestyle="-" ,color="r", label = "L = {0}".format(L))
            return A, ax1, ax2
            
        A, ax1, ax2 = process(T[0], T[-1], A1, Ax)
            
        ax1.set_xlabel("Temperature (T)", fontsize=20)
        ax1.set_ylabel(r"$\chi = \frac{1}{k_B T}(1 - q(T))$", fontsize=20) 
        ax1.set_xticks(np.linspace(0.1, 5.1 , 11))
        ax1.grid()
        ax1.legend(prop={"size":20})
        ax2.set_xlabel("Temperature (T)", fontsize=20)
        ax2.set_ylabel("Edwards-And. param. q", fontsize=20) 
        ax2.set_xticks(np.linspace(0.1, 5.1 , 11))
        ax2.grid()
        ax2.legend(prop={"size":20})
        plt.tight_layout()
        plt.show()
        
    #plot8()
main()
            
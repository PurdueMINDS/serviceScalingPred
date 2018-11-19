'''
Created on Sep 3, 2018

@author: mohame11
'''
import math
import numpy as np
from scipy.optimize import *


def M_M_1_K_log_solve(x, K, PK):
    print x
    logp = math.log(1-x) + K*math.log(x) - math.log(1-(x**(K+1)))
    return logp - math.log(PK)

def M_M_1_K_solve(x, K, PK):
    p = (1.0-x) * (x**K) / (1.0 - (x**(K+1)))
    return p - PK


def M_M_1_K(x, K):
    p = (1.0-x) * (x**K) / (1.0 - (x**(K+1)))
    return p


def M_M_m_K_log(x, m, K):
    #print x
    c1 = K * math.log(x)
    c2 = math.log(math.factorial(m))
    c3 = (K-m) * math.log(m)
    logC_K = c1 - c2 - c3
    
    part2 = 0.0
    for n in range(1,m):
        part2 += float(x**n) / float(math.factorial(n))
        
    part3 = 0.0
    for n in range(m,K+1):
        part3 += float(x**n) / float(m**(n-m))
        
    logP0 = -1 * math.log(1.0 + part2 + part3)
    
    logP_K = logC_K + logP0
    
    return logP_K


def M_M_m_K_log_solve(x, m, K, PK):
    #print x
    c1 = K * math.log(x)
    c2 = math.log(math.factorial(m))
    c3 = (K-m) * math.log(m)
    logC_K = c1 - c2 - c3
    
    part2 = 0.0
    for n in range(1,m):
        part2 += float(x**n) / float(math.factorial(n))
        
    part3 = 0.0
    for n in range(m,K+1):
        part3 += float(x**n) / float(m**(n-m))
        
    logP0 = -1 * math.log(1.0 + part2 + part3)
    
    logP_K = logC_K + logP0
    
    return logP_K - math.log(PK)


def f(x, m, K):
    #print x
    part1 = (x**K) / (math.factorial(m) * m**(K-m))
    
    #part1 = K * math.log(x) - math.log((math.factorial(m) * m**(K-m)))
    #part1 = math.exp(part1) 
    
    part2 = 0.0
    for n in range(1,m):
        part2 += float(x**n) / float(math.factorial(n))
        
    part3 = 0.0
    for n in range(m,K+1):
        part3 += float(x**n) / float(m**(n-m))
        
    part3 *= 1.0/math.factorial(m)
      
    tot = part1 * (1.0/(1.0+part2+part3))
    #tot = part1 + math.log(1.0/(1.0+part2+part3))
    #tot = math.exp(tot) 
    
    return tot
    #return tot - PK


def M_M_m_K(x, m, K):
    #print x
    part1 = (x**K) / (math.factorial(m) * m**(K-m))
    
    #part1 = K * math.log(x) - math.log((math.factorial(m) * m**(K-m)))
    #part1 = math.exp(part1) 
    
    part2 = 0.0
    for n in range(1,m):
        part2 += float(x**n) / float(math.factorial(n))
        
    part3 = 0.0
    for n in range(m,K+1):
        part3 += float(x**n) / float(m**(n-m))
        
    part3 *= 1.0/math.factorial(m)
      
    tot = part1 * (1.0/(1.0+part2+part3))
    #tot = part1 + math.log(1.0/(1.0+part2+part3))
    #tot = math.exp(tot) 
    
    #return tot
    return tot

def M_M_m_K_solve(x, m, K, PK):
    #print x
    part1 = (x**K) / (math.factorial(m) * m**(K-m))
    
    #part1 = K * math.log(x) - math.log((math.factorial(m) * m**(K-m)))
    #part1 = math.exp(part1) 
    
    part2 = 0.0
    for n in range(1,m):
        part2 += float(x**n) / float(math.factorial(n))
        
    part3 = 0.0
    for n in range(m,K+1):
        part3 += float(x**n) / float(m**(n-m))
        
    part3 *= 1.0/math.factorial(m)
      
    tot = part1 * (1.0/(1.0+part2+part3))
    #tot = part1 + math.log(1.0/(1.0+part2+part3))
    #tot = math.exp(tot) 
    
    #return tot
    return tot - PK




def main():
    m = 1 # #workers
    K = 972 # queue size in terms of #request
    my_lambda = 8400 #arrival per unit time
    PK = 1775.0/124259 #prob of failure
    #PK = 0.001
    rho_0 = 0.1 #lambda/mu (initial point to start)
    #mu Amit's intuition is 8400 packet/sec
    
    print 'PK=', PK
    
    rho = fsolve(M_M_m_K_log_solve, rho_0, (m, K, PK)) #solve for rho
    
    diff = M_M_m_K_log_solve(rho, m, K, PK)
    pkk = f(rho, m, K, PK)
    mu = my_lambda / rho
    print 'using M/M/m/K \nrho_0=%.5f, rho_final=%.5f, PK\'=%.5f, PK\'-PK=%.5f, mu=%.5f, lambda=%.5f' % (rho_0, rho, pkk, diff, mu, my_lambda)
    
    '''
    rho = fsolve(M_M_1_K_log_solve, rho_0, (K, PK)) #solve for rho
    diff = M_M_1_K_log_solve(rho, K, PK)
    pkk = M_M_1_K(rho, K)
    mu = my_lambda / rho
    print 'using M/M/1/K\nrho_0=%.5f, rho_final=%.5f, PK\'=%.5f, PK\'-PK=%.5f, mu=%.5f' % (rho_0, rho, pkk, diff, mu)
    '''
    
     
    #sol = newton(f, x, (m, K, PK))
    #diff = f(sol, m, K, PK)
    #print 'using newton\nrho_0=%.5f, rho_final=%.5f, PK\'-PK=%.5f' % (x, sol, diff)
    

    



if __name__ == "__main__":
    main()
    print('DONE!')
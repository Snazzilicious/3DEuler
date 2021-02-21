#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:48:44 2019

@author: ian
"""

import numpy as np

gamma = 1.4


def pressure(rho,en):
    return (gamma-1)*rho*en


def gradP(rho,en):
    numNodes = len(rho)
    
    dP = np.column_stack([(gamma-1)*en, np.zeros([numNodes,3]), (gamma-1)*rho])
    
    return dP


def df123dU(rho,v1,v2,v3,en):
    numNodes = len(rho)
    numVars = 5
    from scipy.sparse import csr_matrix
    
    df1rho,df1v1,df1v2,df1v3,df1E,\
        df2rho,df2v1,df2v2,df2v3,df2E,\
            df3rho,df3v1,df3v2,df3v3,df3E = dfAll(rho,v1,v2,v3,en)
    

    #number of entries in Jacobian = num vars (rho - e) squared 
    #times numNodes = 5*5*numNodes
    numEntries = numVars*numVars*numNodes
    
    rows = np.arange(0,numEntries+1,numVars)
    
    cols = np.tile( np.arange(numVars), numVars*numNodes)
    cols = cols.reshape([-1,numVars])
    cols += np.tile( np.arange(numNodes), numVars).reshape([-1,1])
    cols = cols.reshape([-1])


    vals1 = np.column_stack([df1rho,df1v1,df1v2,df1v3,df1E]).reshape([-1])
    vals2 = np.column_stack([df2rho,df2v1,df2v2,df2v3,df2E]).reshape([-1])
    vals3 = np.column_stack([df3rho,df3v1,df3v2,df3v3,df3E]).reshape([-1])
    
    
    df1dU = csr_matrix( (vals1,cols,rows), shape=(numVars*numNodes, numVars*numNodes) )
    df2dU = csr_matrix( (vals2,cols,rows), shape=(numVars*numNodes, numVars*numNodes) )
    df3dU = csr_matrix( (vals3,cols,rows), shape=(numVars*numNodes, numVars*numNodes) )
    
    return df1dU, df2dU, df3dU



#This might be done now
def dfAll(rho,v1,v2,v3,en):
    numNodes = len(rho)

    P = pressure(rho,en)
    dP = gradP(rho,en)
    
    allZero = np.zeros(numNodes)
    
    E = en + 0.5*( v1*v1 + v2*v2, v3*v3 )
    dE = np.column_stack([allZero, v1, v2, v3, np.ones(numNodes)])
    
    
    df1_rho = np.column_stack([v1, rho, allZero, allZero, allZero])
    df1_v1 = np.column_stack([v1*v1, 2*rho*v1, allZero, allZero, allZero ]) + dP
    df1_v2 = np.column_stack([v2*v1, rho*v2, rho*v1, allZero, allZero])
    df1_v3 = np.column_stack([v3*v1, rho*v3, allZero, rho*v1, allZero])
    df1_E = np.column_stack([allZero, (rho*E+P), allZero, allZero, allZero]) + v1.reshape([-1,1])*( np.column_stack([E, allZero, allZero, allZero, allZero]) + rho.reshape([-1,1])*dE + dP )
    
    
    df2_rho = np.column_stack([v2, allZero, rho, allZero, allZero])
    df2_v1 = np.column_stack([v1*v2, rho*v2, rho*v1, allZero, allZero ])
    df2_v2 = np.column_stack([v2*v2, allZero, 2*rho*v2, allZero, allZero ]) + dP
    df2_v3 = np.column_stack([v3*v2, allZero, rho*v3, rho*v2, allZero ])
    df2_E = np.column_stack([allZero, allZero, (rho*E+P), allZero, allZero]) + v2.reshape([-1,1])*( np.column_stack([E, allZero, allZero, allZero, allZero]) + rho.reshape([-1,1])*dE + dP ) 
    
    
    df3_rho = np.column_stack([v3, allZero, allZero, rho, allZero])
    df3_v1 = np.column_stack([v1*v3, rho*v3, allZero, rho*v1, allZero ])
    df3_v2 = np.column_stack([v2*v3, allZero, rho*v3, rho*v2, allZero ])
    df3_v3 = np.column_stack([v3*v3, allZero, allZero, 2*rho*v3, allZero ]) + dP
    df3_E = np.column_stack([allZero, allZero, allZero, (rho*E+P), allZero]) + v3.reshape([-1,1])*( np.column_stack([E, allZero, allZero, allZero, allZero]) + rho.reshape([-1,1])*dE + dP ) 
    
    

    return df1_rho,df1_v1,df1_v2,df1_v3,df1_E, df2_rho,df2_v1,df2_v2,df2_v3,df2_E, df3_rho,df3_v1,df3_v2,df3_v3,df3_E




def F123(rho,v1,v2,v3,en):

    f1_rho,f1_v1,f1_v2,f1_v3,f1_E,\
        f2_rho,f2_v1,f2_v2,f2_v3,f2_E,\
            f3_rho,f3_v1,f3_v2,f3_v3,f3_E = fAll(rho,v1,v2,v3,en)
    
#    F1 = np.array([f1_rho, f1_v1, f1_v2, f1_V3, f1_E]).reshape(-1,1)
#    F2 = np.array([f2_rho, f2_v1, f2_v2, f2_V3, f2_E]).reshape(-1,1)
#    F3 = np.array([f3_rho, f3_v1, f3_v2, f3_V3, f3_E]).reshape(-1,1)
    
    F1 = np.concatenate([f1_rho, f1_v1, f1_v2, f1_v3, f1_E])
    F2 = np.concatenate([f2_rho, f2_v1, f2_v2, f2_v3, f2_E])
    F3 = np.concatenate([f3_rho, f3_v1, f3_v2, f3_v3, f3_E])
    
    return F1, F2, F3



def fAll(rho,v1,v2,v3,en):
    
    P = pressure(rho,en)
    
    E = en + .5*( v1*v1 + v2*v2 + v3*v3 )

    f1_rho = rho*v1
    f1_v1 = rho*v1*v1 + P
    f1_v2 = rho*v2*v1
    f1_v3 = rho*v3*v1
    f1_E = (rho*E + P)*v1
    
    f2_rho = rho*v2
    f2_v1 = rho*v1*v2
    f2_v2 = rho*v2*v2 + P
    f2_v3 = rho*v3*v2
    f2_E = (rho*E + P)*v2
    
    f3_rho = rho*v3
    f3_v1 = rho*v1*v3
    f3_v2 = rho*v2*v3
    f3_v3 = rho*v3*v3 + P
    f3_E = (rho*E + P)*v3
    
    
    return f1_rho,f1_v1,f1_v2,f1_v3,f1_E, f2_rho,f2_v1,f2_v2,f2_v3,f2_E, f3_rho,f3_v1,f3_v2,f3_v3,f3_E




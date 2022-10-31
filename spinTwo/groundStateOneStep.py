# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 11:44:30 2022

@author: Nathan

groundStateOneStep.py

Performs one step of the gradient descent algorithm to find the ground state of 
a spin two BEC.
"""

"""Imports"""
import numpy as np
import scipy as sp
from scipy.sparse.linalg import spsolve



import groundStateTools as gst




"""Write a general function that solves one of the GPEs for one of the spin components"""
def solveEq(chi,c1,c2, phiB, L, h, tau):
    """Construct the matrix A and column vector b in the equation A*phi = b.
    The original equation (discretized) is of the form:
        chi*phi[i,j] -tau*phi[i,j-1] -tau*phi[i,j+1] -tau*phi[i-1,j] -tau*phi[i+1,j] 
            = 2*(h**2)*c1 - 2*(h**2)*tau*c2.
    
    c1 and c2 will be arguments in terms of some or all of the phi spin components at 
    the previous time step.
    
    phiB is a list of the boundary values of phi.
    
    Returns:
        phiCompTemp: Solves for phi in A*phi = b and stores this in phiCompTemp."""
    
    """Need to construct the matrix A which is part of the equation I wish to 
    solve (ie: A\phi = b). A has dimensions (L-1)^2 x (L-1)^2. Since these matrices
    could be quite large, I am going to use sparse matrices (csc matrices from scipy)."""
    index = 0 #Will use this to index the row,col and data lists for csc matrix
    
    """Make lists for row, col and data for csc matrix A. We can set the size of
    these in advance as I know how many non zero entries there will be in A in terms
    of L which is (L-1)^2 + 4(L-1)(L-2). To see where this came from, look at my 
    thesis for an explanation."""
    numElts = (L-1)**2 + 4*(L-1)*(L-2)
    
    row = np.zeros(numElts)
    col = np.zeros(numElts)
    data = np.zeros(numElts)
    
    """Now fill these lists with the coordinates of the nonzero entries of A along
    with what those nonzero entries are into the data list. Note, we are looping down 
    the main diagonal of A."""
    for i in range((L-1)**2):
        """Fill main diagonal of A with relevant chi value."""
        row[index] = i
        col[index] = i
        data[index] = chi[i]
        index+=1
        
        """Fill just off diagonal tau values."""
        if i != (L-1)**2 -1:
            if i % (L-1) != (L-2):
                row[index] = i+1
                col[index] = i
                data[index] = -tau
                index += 1
                
                row[index] = i
                col[index] = i+1
                data[index] = -tau
                index += 1
        
        """Fill further off diagonal tau values."""
        if i < (L-1)*(L-2):
            row[index] = i
            col[index] = i + (L-1)
            data[index] = -tau
            index += 1
            
            row[index] = i + (L-1)
            col[index] = i
            data[index] = -tau
            index += 1
            
    """Now make the sparse matrix A"""
    A = sp.sparse.csc_matrix((data, (row,col)), shape=((L-1)**2,(L-1)**2))
    
    """Need to construct the b column vector in the equation A\phi = b. b has 
    dimensions (L-1)^2 x 1"""
    b = np.zeros((L-1)**2,dtype='complex_')
    
    """Fill in values to b vector including boundary values. The boundary values
    are all zero at the moment, however they can be nonzero in other cases so I 
    am going to construct b generally here so that I can change the boundary 
    values later if I want to and not need to change b as it will already be set 
    up for nonzero boundary values."""
    
    
    
    """c1 and c2 are already 1D arrays but they are ordered in column major 
    order. This means that we just have to iterate through them normally to 
    put the correct c1 and c2 value in the correct place in the b vector.
    
    We just have to switch i and j in the getPhiBIndex call to make sure we 
    are referring to the correct lattice coordinate for the corresponding 
    entry in the b vector."""
    
    for i in range(2,L+1):
        """Going to L+1 as that is what 2 + (L-1) equals"""
        for j in range(2,L+1):
            m = (L-1)*(i-2) + (j-2)
            b[m] = 2*(h**2)*c1[m] + 2*(h**2)*tau*c2[m]
            if i-1 == 1:
                b[m] += tau*phiB[gst.getPhiBIndex(j, i-1, L)]
            if i+1 == L+1:
                b[m] += tau*phiB[gst.getPhiBIndex(j, i+1, L)]
            if j-1 == 1:
                b[m] += tau*phiB[gst.getPhiBIndex(j-1,i,L)]
            if j+1 == L+1:
                b[m] += tau*phiB[gst.getPhiBIndex(j+1,i,L)]
    
    """Now, solve for the relevant phi component and return this."""
    phiCompTemp = spsolve(A,b)
    
    return phiCompTemp

def groundState(phiTwoList,phiOneList,phiZeroList,phiMinusOneList,phiMinusTwoList,L,V,h,tau,p,q,b0,b1,b2,phiBTwo=0,phiBOne=0,phiBZero=0,phiBMinusOne=0,phiBMinusTwo=0):
    """Given initial conditions for the wavefunction phi. Do one step of the gradient 
    descent towards solving for the ground state of the system."""
    
    """Set up a list of boundary values for phi. At the moment, these will all just
    be set to zero, however in other cases they may be nonzero. See thesis for how the 
    order of this list is done out. I also have a function that will return
    the index in this list for an input of the coordinate of the point on the boundary 
    of the 2d grid. See groundStateTools for this function."""
    if (phiBTwo==0 and phiBOne==0 and phiBZero==0 and phiBMinusOne==0 and phiBMinusTwo==0):
        phiBTwo = np.zeros(4*L)
        phiBOne = np.zeros(4*L)
        phiBZero = np.zeros(4*L)
        phiBMinusOne = np.zeros(4*L)
        phiBMinusTwo = np.zeros(4*L)
    
    """If the user has set some value for the boundary lists then the above will be 
    skipped and the user's preference for the boundary lists will be used."""
    
    """Flattened arrays of the phi component lists passed in. Elements are stored 
    in column major order."""
    phiTwoReshaped = np.reshape(phiTwoList,(L-1)**2,order='F')
    phiOneReshaped = np.reshape(phiOneList,(L-1)**2,order='F')
    phiZeroReshaped = np.reshape(phiZeroList,(L-1)**2,order='F')
    phiMinusOneReshaped = np.reshape(phiMinusOneList,(L-1)**2,order='F')
    phiMinusTwoReshaped = np.reshape(phiMinusTwoList,(L-1)**2,order='F')
    
    
    
    """Solve first of the five GPE's: Solve for phiTwo"""
    def chiTwo(V):
        """Compute chiTwo for each value in our grid. These are the values that 
        get stored on the main diagonal of the A matrix. That is, chiTwo is everything 
        that gets multiplied by phiTwo at the next gradient descent step at each position 
        in our grid.
        
        NOTE: Passing values from 2D lists into chi in column major order."""
        chiList = np.zeros((L-1)**2)
        for i in range(L-1):
            for j in range(L-1):
                chiList[((L-1)*i) + j] = (2*h**2 + 4*tau + (2*(h**2)*tau)*(V[j,i] -2*p + 4*q
                                            + (b0+4*b1)*(abs(phiTwoList[j,i])**2)
                                            + (b0-4*b1)*(abs(phiMinusTwoList[j,i])**2)
                                            + (b0+2*b1)*(abs(phiOneList[j,i])**2)
                                            + (b0-2*b1)*(abs(phiMinusOneList[j,i])**2)
                                            + b0*(abs(phiZeroList[j,i])**2)
                                          
                                            + 2*b1*abs(phiOneList[j,i])**2
                                            + (2*b2/5)*abs(phiMinusTwoList[j,i])**2))
        return chiList
    
    chi = chiTwo(V)
    
    """Define the right hand side of the component equation."""
    c1 = np.array(phiTwoReshaped)
    c2 = -np.array((2*np.conjugate(phiMinusTwoReshaped)*phiMinusOneReshaped 
                   + np.sqrt(6)*(np.conjugate(phiZeroReshaped)*phiOneReshaped 
                                 + np.conjugate(phiMinusOneReshaped)*phiZeroReshaped))*b1*phiOneReshaped
                  
                  + (b2/5)*(-2*phiOneReshaped*phiMinusOneReshaped + phiZeroReshaped**2)*np.conjugate(phiMinusTwoReshaped))
    
    """Find the new value phiTwo after one gradient descent step."""
    phiTwoTemp = solveEq(chi, c1, c2, phiBTwo, L, h, tau)
    
    """PHITWO EQUATION COMPLETE"""
    
    """Solve second of five GPE's: Solve for phiOne"""
    def chiOne(V):
        """Compute chiOne for each value in our grid. These are the values that 
        get stored on the main diagonal of the A matrix. That is, chiOne is everything 
        that gets multiplied by phiOne at the next gradient descent step at each position 
        in our grid.
        
        NOTE: Passing values from 2D lists into chi in column major order."""
        chiList = np.zeros((L-1)**2)
        for i in range(L-1):
            for j in range(L-1):
                chiList[((L-1)*i) + j] = (2*h**2 + 4*tau + (2*(h**2)*tau)*(V[j,i] -p + q
                                            + (b0+2*b1)*(abs(phiTwoList[j,i])**2)
                                            + (b0-2*b1)*(abs(phiMinusTwoList[j,i])**2)
                                            + (b0+b1)*(abs(phiOneList[j,i])**2)
                                            + (b0-b1)*(abs(phiMinusOneList[j,i])**2)
                                            + b0*(abs(phiZeroList[j,i])**2)
                                            
                                            + 3*b1*abs(phiZeroList[j,i])**2
                                            + 2*b1*abs(phiTwoList[j,i])**2
                                            + (2*b2/5)*abs(phiMinusOneList[j,i])**2))
        return chiList
    
    chi = chiOne(V)
    
    """Define the right hand side of the component equation."""
    c1 = np.array(phiOneReshaped)
    c2 = np.array(-b1*(phiZeroReshaped*(np.sqrt(6)*(np.conjugate(phiOneReshaped)*phiTwoReshaped 
                                                + np.conjugate(phiMinusTwoReshaped)*phiMinusOneReshaped) 
                                    + 3*(np.conjugate(phiMinusOneReshaped)*phiZeroReshaped)) 
                       + phiTwoReshaped*(2*(np.conjugate(phiMinusOneReshaped)*phiMinusTwoReshaped) 
                                     + np.sqrt(6)*(np.conjugate(phiOneReshaped)*phiZeroReshaped 
                                                   + np.conjugate(phiZeroReshaped)*phiMinusOneReshaped))) 
                  +(b2/5)*np.conjugate(phiMinusOneReshaped)*(2*phiTwoReshaped*phiMinusTwoReshaped 
                                                         + phiZeroReshaped**2))
    
    
    """Find the new value phiOne after one gradient descent step."""
    phiOneTemp = solveEq(chi, c1, c2, phiBOne, L, h, tau)
    
    """CHIONE EQUATION IS DONE"""
    
    """Solve second of three GPE's: Solve for phiZero"""
    def chiZero(V):
        """Compute chiZero for each value in our grid. These are the values that 
        get stored on the main diagonal of the A matrix. That is, chiZero is everything 
        that gets multiplied by phiZero at the next gradient descent step at each position 
        in our grid.
        
        NOTE: Passing values from 2D lists into chi in column major order."""
        chiList = np.zeros((L-1)**2)
        for i in range(L-1):
            for j in range(L-1):
                chiList[((L-1)*i) + j] = (2*h**2 + 4*tau + (2*(h**2)*tau)*(V[j,i]
                                            + b0*(abs(phiTwoList[j,i])**2 + abs(phiOneList[j,i])**2 
                                                  + abs(phiZeroList[j,i])**2 + abs(phiMinusOneList[j,i])**2 
                                                  + abs(phiMinusTwoList[j,i])**2)
                                            
                                            + 3*b1*(abs(phiOneList[j,i])**2 + abs(phiMinusOneList[j,i])**2)
                                            + (b2/5)*abs(phiZeroList[j,i])**2))
        return chiList
    
    chi = chiZero(V)
    
    """Define the right hand side of the component equation."""
    c1 = np.array(phiZeroReshaped)
    c2 = -np.array(b1*(phiOneReshaped*(np.sqrt(6)*(np.conjugate(phiTwoReshaped)*phiOneReshaped 
                                               + np.conjugate(phiMinusOneReshaped)*phiMinusTwoReshaped) 
                                   + 3*(np.conjugate(phiZeroReshaped)*phiMinusOneReshaped)) 
                       + phiMinusOneReshaped*(np.sqrt(6)*(np.conjugate(phiOneReshaped)*phiTwoReshaped 
                                                      + np.conjugate(phiMinusTwoReshaped)*phiMinusOneReshaped) 
                                          + 3*(np.conjugate(phiZeroReshaped)*phiOneReshaped))) 
                   + (b2/5)*np.conjugate(phiZeroReshaped)*(2*phiTwoReshaped*phiMinusTwoReshaped 
                                                       - 2*phiOneReshaped*phiMinusOneReshaped))
    
    
    """Find the new value phiZero after one gradient descent step."""
    phiZeroTemp = solveEq(chi, c1, c2, phiBZero, L, h, tau)
    
    """CHIZERO EQUATION IS DONE"""
    
    """Solve fourth of five GPE's: Solve for phiMinusOne"""
    def chiMinusOne(V):
        """Compute chiMinusOne for each value in our grid. These are the values that 
        get stored on the main diagonal of the A matrix. That is, chiMinusOne is everything 
        that gets multiplied by phiMinusOne at the next gradient descent step at each position 
        in our grid.
        
        NOTE: Passing values from 2D lists into chi in column major order."""
        chiList = np.zeros((L-1)**2)
        for i in range(L-1):
            for j in range(L-1):
                chiList[((L-1)*i) + j] = (2*h**2 + 4*tau + (2*(h**2)*tau)*(V[j,i] +p + q
                                            + (b0-2*b1)*(abs(phiTwoList[j,i])**2)
                                            + (b0+2*b1)*(abs(phiMinusTwoList[j,i])**2)
                                            + (b0-b1)*(abs(phiOneList[j,i])**2)
                                            + (b0+b1)*(abs(phiMinusOneList[j,i])**2)
                                            + b0*(abs(phiZeroList[j,i])**2)
                                            
                                            + 3*b1*abs(phiZeroList[j,i])**2
                                            + 2*b1*abs(phiMinusTwoList[j,i])**2
                                            + (2*b2/5)*abs(phiOneList[j,i])**2))
        return chiList
    
    chi = chiMinusOne(V)
    
    """Define the right hand side of the component equation."""
    c1 = np.array(phiMinusOneReshaped)
    c2 = np.array(-b1*(phiZeroReshaped*(np.sqrt(6)*(np.conjugate(phiMinusOneReshaped)*phiMinusTwoReshaped 
                                                + np.conjugate(phiTwoReshaped)*phiOneReshaped) 
                                    + 3*(np.conjugate(phiOneReshaped)*phiZeroReshaped)) 
                       + phiMinusTwoReshaped*(2*(np.conjugate(phiOneReshaped)*phiTwoReshaped) 
                                     + np.sqrt(6)*(np.conjugate(phiMinusOneReshaped)*phiZeroReshaped 
                                                   + np.conjugate(phiZeroReshaped)*phiOneReshaped))) 
                  +(b2/5)*np.conjugate(phiOneReshaped)*(2*phiTwoReshaped*phiMinusTwoReshaped 
                                                         + phiZeroReshaped**2))
    
    
    """Find the new value phiMinusOne after one gradient descent step."""
    phiMinusOneTemp = solveEq(chi, c1, c2, phiBMinusOne, L, h, tau)
    
    """CHIMINUSONE EQUATION IS DONE"""
    
    
    """Solve fifth of the five GPE's: Solve for phiMinusTwo"""
    def chiMinusTwo(V):
        """Compute chiMinusTwo for each value in our grid. These are the values that 
        get stored on the main diagonal of the A matrix. That is, chiMinusTwo is everything 
        that gets multiplied by phiMinusTwo at the next gradient descent step at each position 
        in our grid.
        
        NOTE: Passing values from 2D lists into chi in column major order."""
        chiList = np.zeros((L-1)**2)
        for i in range(L-1):
            for j in range(L-1):
                chiList[((L-1)*i) + j] = (2*h**2 + 4*tau + (2*(h**2)*tau)*(V[j,i] +2*p + 4*q
                                            + (b0-4*b1)*(abs(phiTwoList[j,i])**2)
                                            + (b0+4*b1)*(abs(phiMinusTwoList[j,i])**2)
                                            + (b0-2*b1)*(abs(phiOneList[j,i])**2)
                                            + (b0+2*b1)*(abs(phiMinusOneList[j,i])**2)
                                            + b0*(abs(phiZeroList[j,i])**2)
                                          
                                            + 2*b1*abs(phiMinusOneList[j,i])**2
                                            + (2*b2/5)*abs(phiTwoList[j,i])**2))
        return chiList
    
    chi = chiMinusTwo(V)
    
    """Define the right hand side of the component equation."""
    c1 = np.array(phiMinusTwoReshaped)
    c2 = -np.array((2*np.conjugate(phiTwoReshaped)*phiOneReshaped 
                   + np.sqrt(6)*(np.conjugate(phiZeroReshaped)*phiMinusOneReshaped 
                                 + np.conjugate(phiOneReshaped)*phiZeroReshaped))*b1*phiMinusOneReshaped
                  
                  + (b2/5)*(-2*phiOneReshaped*phiMinusOneReshaped + phiZeroReshaped**2)*np.conjugate(phiTwoReshaped))
    
    
    """Find the new value phiMinusTwo after one gradient descent step."""
    phiMinusTwoTemp = solveEq(chi, c1, c2, phiBMinusTwo, L, h, tau)
    
    """CHIMINUSTWO EQUATION COMPLETE"""
    
    
    """Renormalize the wavefunctions found above by dividing by the norm and hence, 
    conserve particle number."""
    sigmaNorm = gst.getPhiNorm(phiTwoTemp,phiOneTemp,phiZeroTemp,phiMinusOneTemp,phiMinusTwoTemp,h)
    sigmaNorm = 1/sigmaNorm
    
    phiZeroTemp *= sigmaNorm
    phiOneTemp *= sigmaNorm
    phiMinusOneTemp *= sigmaNorm
    phiTwoTemp *= sigmaNorm
    phiMinusTwoTemp *= sigmaNorm
    
    """Setup 2D arrays to store the phi components in."""
    phiTwo = np.zeros((len(phiTwoList),len(phiTwoList)),dtype='complex_')
    phiOne = np.zeros((len(phiOneList),len(phiOneList)),dtype='complex_')
    phiZero = np.zeros((len(phiZeroList),len(phiZeroList)),dtype='complex_')
    phiMinusOne = np.zeros((len(phiMinusOneList),len(phiMinusOneList)),dtype='complex_')
    phiMinusTwo = np.zeros((len(phiMinusTwoList),len(phiMinusTwoList)),dtype='complex_')
    
    
    """Fill in the most up to date answer into the 2d phi component arrays.
    Making sure to keep in line with column major order that we have been using."""
    for i in range(L-1):
        for j in range(L-1):
            phiTwo[j,i] = phiTwoTemp[(L-1)*i + j]
            phiOne[j,i] = phiOneTemp[(L-1)*i + j]
            phiZero[j,i] = phiZeroTemp[(L-1)*i + j]
            phiMinusOne[j,i] = phiMinusOneTemp[(L-1)*i + j]
            phiMinusTwo[j,i] = phiMinusTwoTemp[(L-1)*i + j]
            
    
    return phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo

    









































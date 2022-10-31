# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 14:18:11 2021

@author: Nathan

bogoliubov.py

I am solving the bogoliubov equations for a spin one condensate in order to look 
at excitations of the ground state.
"""

"""Imports"""
import numpy as np
from configparser import ConfigParser
import pickle
import scipy as sp
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
import groundStateTools as gst
import time


def computeMu(phiOne,phiZero,phiMinusOne,x,y,V,b0,b1,p,q):
    """Calculating the chemical potential, mu, for a wavefunction in 2 dimensions.
    
    Inputs:
        phiOne,phiZero,phiMinusOne: The m=1,0,-1 components of the condensate,
        x,y: x and y lists of x and y coordinates in lattice,
        V: 2D array of potential,
        b0,b1: Interaction parameters,
        p,q: Zeeman terms.
    
    Returns:
        mu: The chemical potential."""
    
    def derivAlt(phi):
        """Returns the abs value squared of the gradient of phi in 2-dimensions.
        Here, phi is a 2d array."""
        
        
        dx = x[1]-x[0]
        
        
        dy = y[1]-y[0]
        
        grad = np.gradient(phi,dx,dy)
        conjGrad = np.gradient(np.conjugate(phi),dx,dy)
        
        absDerivSq = np.array(grad[0]*conjGrad[0] + grad[1]*conjGrad[1])
        
        return absDerivSq
    
    
    def phiFour(phiOne,phiZero,phiMinusOne):
        pOne = abs(phiOne)**2
        pZero = abs(phiZero)**2
        pMinusOne = abs(phiMinusOne)**2
        
        phiFour = (pOne**2 + pZero**2 + pMinusOne**2 + 2*pOne*pMinusOne + 2*pOne*pZero
                       + 2*pMinusOne*pZero)
        
        return phiFour
    
    def fFour(phiOne,phiZero,phiMinusOne):
        pOne = abs(phiOne)**2
        pZero = abs(phiZero)**2
        pMinusOne = abs(phiMinusOne)**2
        
        fFour = (pOne**2 + pMinusOne**2 + 2*pOne*pZero + 2*pZero*pMinusOne 
                     - 2*pOne*pMinusOne + 2*(phiZero**2)*(np.conjugate(phiOne))*(np.conjugate(phiMinusOne))
                     + 2*(np.conjugate(phiZero**2))*(phiOne)*(phiMinusOne))
        
        return fFour
    
    phi4 = phiFour(phiOne,phiZero,phiMinusOne)
    f4 = fFour(phiOne,phiZero,phiMinusOne)
        
    
    
    
    
    integrand = ((1/2)*(derivAlt(phiOne)) + (1/2)*(derivAlt(phiZero)) + (1/2)*(derivAlt(phiMinusOne))
                 + V*abs(phiOne)**2 + V*abs(phiZero)**2 + V*abs(phiMinusOne)**2 
                 + (-p+q)*abs(phiOne)**2 + (p+q)*abs(phiMinusOne) 
                 + (b0)*(phi4) + (b1)*(f4))
    
    
    integrand = np.vstack((integrand,integrand[0]))
    integrandCol = np.zeros(len(integrand))
    for i in range(len(integrand)):
        integrandCol[i] = integrand[i,0]
    
    
    integrand = np.concatenate((integrand,np.array([integrandCol]).T),axis=1)
    
    xInt = np.zeros(len(integrand),dtype='complex_')
    
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    
    
    for i in range(len(integrand)):
        xInt[i] = np.trapz(integrand[i],dx=dx)
    
    
    mu = np.trapz(xInt,dx=dy)
    
    return mu


def constructSubMat(term,L,h,deriv=False):
    """This function constructs a subMat of the matrix of linear equations that 
    are the Bogoliubov equations. term is the phi terms or otherwise needed to be 
    multiplied together to produce the term that fills the current submat.
    deriv tells us whether we are constructing an Li term in which case we need to handle 
    the derivative term as well."""
    
    index = 0 #Variable to index the row, col and data lists.
    
    if (deriv):
        numEntries = (L-1)**2 + 4*(L-1)*(L-2) + 4*(L-1)
    else:
        numEntries = (L-1)**2
        
    row = np.zeros(numEntries)
    col = np.zeros(numEntries)
    data = np.zeros(numEntries,dtype='complex_')
        
    for i in range((L-1)**2):
        row[index] = i
        col[index] = i
        data[index] = term[int(i%(L-1)),int(i/(L-1))]
        
        if (deriv):
            data[index] += 2/(h**2)
            index += 1
            
            
            if i % (L-1) != (L-2):
                row[index] = i+1
                col[index] = i
                data[index] = -1/(2*h**2)
                index += 1
                
                row[index] = i
                col[index] = i+1
                data[index] = -1/(2*h**2)
                index += 1
            else:
                """Implement y periodic boundary conditions"""
                row[index] = i - (L-2)
                col[index] = i
                data[index] = -1/(2*h**2)
                index += 1
                
                row[index] = i
                col[index] = i - (L-2)
                data[index] = -1/(2*h**2)
                index += 1
            
            if i < (L-1)*(L-2):
                row[index] = i
                col[index] = i + (L-1)
                data[index] = -1/(2*h**2)
                index += 1
                
                row[index] = i + (L-1)
                col[index] = i
                data[index] = -1/(2*h**2)
                index += 1
            
            """Adding in bit to make x periodic boundary conditions"""
            if i < (L-1):
                row[index] = i
                col[index] = i + (L-1)*(L-2)
                data[index] = -1/(2*h**2)
                index += 1
                
                row[index] = i + (L-1)*(L-2)
                col[index] = i
                data[index] = -1/(2*h**2)
                index += 1
        else:
            index += 1
            
     
    subMat = sp.sparse.csc_matrix((data,(row,col)),shape=((L-1)**2,(L-1)**2))
    
    return subMat

def bogoliubov(phiOneList,phiZeroList,phiMinusOneList,L,a,b,x,y,V,b0,b1,p,q,mainIter=0,directory='',numVecs=100):
    """Given the background order parameter (phiOneList,phiZeroList,phiMinusOneList), solve 
    the Bogoliubov equations and return the eigenvectors and eigenvalues.
    
    Inputs:
        phiOneList: m=1 component of background,
        phiZeroList: m=0 component of background,
        phiMinusOneList: m=-1 component of background,
        L: Number of lattice spacings,
        a,b: Boundaries of our dicretized grid,
        x,y: Lists of x and y values on our grid,
        V: 2D array of the potential for the system,
        b0,b1: Interaction parameters,
        p,q: Linear and quadratic Zeeman terms,
        mainIter: Number of run to append onto the saved eigVecs and eigVals data,
        directory: directory to save results in. Eg: to save in results, set directory = 'results/'
        numVecs: The number of eigenvectors to obtain, default is 100.
        
    This file does not return anything and instead saves the eigVecs and eigVals 
    as eigVecs[mainIter] and eigVals[mainIter] that can be loaded back in to 
    inspect after.
        """
    
    t1 = time.time()
    
    h = (b-a)/(L-1)
    
    """Set up meshgrids to have arrays listing all coordinates in our 2d grid"""
    xx, yy = np.meshgrid(x,y)
    
    
    muG = computeMu(phiOneList,phiZeroList,phiMinusOneList,x,y,V,b0,b1,p,q)
    print('muG = ' + str(muG))
    

    lOneTerm = V + q - p + 2*(b0+b1)*(abs(phiOneList)**2) + (b0-b1)*(abs(phiMinusOneList)**2) + (b0+b1)*(abs(phiZeroList)**2) - muG
    L1 = constructSubMat(lOneTerm,L,h,True)
    
    lTwoTerm = V + 2*b0*abs(phiZeroList)**2 + (b0+b1)*(abs(phiMinusOneList)**2 + abs(phiOneList)**2) - muG
    L2 = constructSubMat(lTwoTerm,L,h,True)
    
    lThreeTerm = V + q + p + 2*(b0+b1)*(abs(phiMinusOneList)**2) + (b0-b1)*(abs(phiOneList)**2) + (b0+b1)*(abs(phiZeroList)**2) - muG
    L3 = constructSubMat(lThreeTerm,L,h,True)
    
    H12 = constructSubMat((b0+b1)*(phiOneList)**2,L,h)
    H13 = constructSubMat((b0+b1)*(np.conjugate(phiZeroList)*phiOneList) + 2*b1*(np.conjugate(phiMinusOneList)*phiZeroList),L,h)
    H14 = constructSubMat((b0+b1)*(phiZeroList*phiOneList),L,h)
    H15 = constructSubMat((b0-b1)*(np.conjugate(phiMinusOneList)*phiOneList),L,h)
    H16 = constructSubMat((b0-b1)*(phiMinusOneList*phiOneList) + b1*(phiZeroList)**2,L,h)
    H21 = -np.conjugate(H12)
    H23 = constructSubMat(-(b0+b1)*(np.conjugate(phiZeroList*phiOneList)),L,h)
    H24 = constructSubMat(-(b0+b1)*(phiZeroList*np.conjugate(phiOneList)) -2*b1*(np.conjugate(phiZeroList)*phiMinusOneList),L,h)
    H25 = constructSubMat(-(b0-b1)*(np.conjugate(phiMinusOneList*phiOneList)) -b1*(np.conjugate(phiZeroList))**2,L,h)
    H26 = constructSubMat(-(b0-b1)*(phiMinusOneList*np.conjugate(phiOneList)),L,h)
    H31 = np.conjugate(H13)
    H32 = -np.conjugate(H23)
    H34 = constructSubMat(b0*(phiZeroList)**2 + 2*b1*(phiMinusOneList*phiOneList),L,h)
    H35 = constructSubMat((b0+b1)*(np.conjugate(phiMinusOneList)*phiZeroList) + 2*b1*(np.conjugate(phiZeroList)*phiOneList),L,h)
    H36 = constructSubMat((b0+b1)*(phiMinusOneList*phiZeroList),L,h)
    H41 = -np.conjugate(H14)
    H42 = np.conjugate(H24)
    H43 = -np.conjugate(H34)
    H45 = constructSubMat(-(b0+b1)*(np.conjugate(phiMinusOneList*phiZeroList)),L,h)
    H46 = constructSubMat(-(b0+b1)*(phiMinusOneList*np.conjugate(phiZeroList)) -2*b1*(phiZeroList*np.conjugate(phiOneList)),L,h)
    H51 = np.conjugate(H15)
    H52 = -np.conjugate(H25)
    H53 = np.conjugate(H35)
    H54 = -np.conjugate(H45)
    H56 = constructSubMat((b0+b1)*(phiMinusOneList)**2,L,h)
    H61 = -np.conjugate(H16)
    H62 = np.conjugate(H26)
    H63 = -np.conjugate(H36)
    H64 = np.conjugate(H46)
    H65 = -np.conjugate(H56)
    
    print('Submats made')
    t2 = time.time()
    print('Time elapsed = ' + str(t2-t1))
    
    
    """Now I need to put these submats together into the large matrix that I will 
    find it's eigvecs and eigvals to solve the Bogoliubov equations."""
    H1 = sp.sparse.csc_matrix(sp.sparse.vstack(((L1,H21,H31,H41,H51,H61))))
    H2 = sp.sparse.csc_matrix(sp.sparse.vstack(((H12,-L1,H32,H42,H52,H62))))
    H3 = sp.sparse.csc_matrix(sp.sparse.vstack(((H13,H23,L2,H43,H53,H63))))
    H4 = sp.sparse.csc_matrix(sp.sparse.vstack(((H14,H24,H34,-L2,H54,H64))))
    H5 = sp.sparse.csc_matrix(sp.sparse.vstack(((H15,H25,H35,H45,L3,H65))))
    H6 = sp.sparse.csc_matrix(sp.sparse.vstack(((H16,H26,H36,H46,H56,-L3))))
    
    print('First stack done')
    t3 = time.time()
    print('Time elapsed = ' + str(t3-t1))
    
    
    H = sp.sparse.csc_matrix(sp.sparse.hstack(((H1,H2,H3,H4,H5,H6))))
    
    print('Main stack done')
    t4 = time.time()
    print('Time elapsed = ' + str(t4-t1))
    
    
    
    eigVals,eigVecs = eigs(H,numVecs,sigma=0,which='LM')
    
    print('eigVals found')
    t5 = time.time()
    print('Time elapsed = ' + str(t5-t1))
    
    
    """Order the eigenvalues and eigenvectors in terms of increasing abs value of eigvals"""
    idx = abs(eigVals).argsort()   
    eigVals = eigVals[idx]
    eigVecs = eigVecs[:,idx]
    
    eigVecs = np.transpose(eigVecs) #Reformat to be compatible with the way I'm looping in plotModes function.
    
    
    print('Returning eigVecs and eigVals')
    t6 = time.time()
    print('Time elapsed = ' + str(t6-t1))
    
    gst.saveResults(name=mainIter,directory=directory,eigVecs=eigVecs)
    gst.saveResults(name=mainIter,directory=directory,eigVals=eigVals)








    





























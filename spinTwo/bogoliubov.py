# -*- coding: utf-8 -*-
"""
Created on Wed May 18 17:13:50 2022

@author: bogoliubov.py


Solving the Bogoliubov equations for a spin two condensate in order to look at 
excitations of the ground state.
"""

"""Imports"""
import numpy as np
from configparser import ConfigParser
import scipy as sp
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
#import modularGroundStateCopy as mgsc
import groundStateTools as gst
import time


def computeMu(phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo,x,y,V,b0,b1,b2,p,q):
    """Calculate the chemical potential, mu, for a spin two wavefunction in 2 dimensions.
    
    Inputs:
        phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo: The m=2,1,0,-1,-2 components of the condensate,
        x,y: x and y lists of the x and y coordinates in lattice,
        V: 2D array of potential,
        b0,b1,b2: Interaction parameters,
        p,q: Zeeman terms.
        
    Returns:
        mu: The chemical potential."""
    
    def deriv(phi):
        """Returns the absolute value squared of the gradient of phi in 2 dimensions.
        Here, phi is a 2d array."""
        
        dx = x[1]-x[0]
        dy = y[1]-y[0]
        
        grad = np.gradient(phi,dx,dy)
        conjGrad = np.gradient(np.conjugate(phi),dx,dy)
        
        absDerivSq = np.array(grad[0]*conjGrad[0] + grad[1]*conjGrad[1])
        
        return absDerivSq
    
    def phiFour(phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo):
        pTwo = abs(phiTwo)**2
        pOne = abs(phiOne)**2
        pZero = abs(phiZero)**2
        pMinusOne = abs(phiMinusOne)**2
        pMinusTwo = abs(phiMinusTwo)**2
        
        phiFour = (pTwo+pOne+pZero+pMinusOne+pMinusTwo)**2
        
        return phiFour
    
    def fFour(phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo):
        
        pTwo = abs(phiTwo)**2
        pOne = abs(phiOne)**2
        pMinusOne = abs(phiMinusOne)**2
        pMinusTwo = abs(phiMinusTwo)**2
        
        fx = (np.conjugate(phiTwo)*phiOne + np.conjugate(phiOne)*phiTwo
              + np.sqrt(3/2)*np.conjugate(phiOne)*phiZero 
              + np.sqrt(3/2)*np.conjugate(phiZero)*phiOne 
              + np.sqrt(3/2)*np.conjugate(phiZero)*phiMinusOne
              + np.sqrt(3/2)*np.conjugate(phiMinusOne)*phiZero
              + np.conjugate(phiMinusOne)*phiMinusTwo
              + np.conjugate(phiMinusTwo)*phiMinusOne)
        
        fy = 1j*(-np.conjugate(phiTwo)*phiOne + np.conjugate(phiOne)*phiTwo 
                 -np.sqrt(3/2)*np.conjugate(phiOne)*phiZero 
                 +np.sqrt(3/2)*np.conjugate(phiZero)*phiOne 
                 -np.sqrt(3/2)*np.conjugate(phiZero)*phiMinusOne 
                 +np.sqrt(3/2)*np.conjugate(phiMinusOne)*phiZero 
                 -np.conjugate(phiMinusOne)*phiMinusTwo 
                 +np.conjugate(phiMinusTwo)*phiMinusOne)
        
        fz = 2*pTwo + pOne - pMinusOne - 2*pMinusTwo
        
        fFour = abs(fx)**2 + abs(fy)**2 + abs(fz)**2
        
        return fFour
    
    def aZeroSq(phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo):
        aZero = np.sqrt(1/5)*(2*phiTwo*phiMinusTwo - 2*phiOne*phiMinusOne + phiZero**2)
        aZero = abs(aZero)**2
        return aZero
    
    phi4 = phiFour(phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo)
    f4 = fFour(phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo)
    a0Sq = aZeroSq(phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo)
    
    integrand = ((1/2)*(deriv(phiTwo)) + (1/2)*(deriv(phiOne)) + (1/2)*(deriv(phiZero))
                + (1/2)*(deriv(phiMinusOne)) + (1/2)*(deriv(phiMinusTwo)) 
                + V*(abs(phiTwo)**2 + abs(phiOne)**2 + abs(phiZero)**2 + abs(phiMinusOne)**2 + abs(phiMinusTwo)**2) 
                + (-2*p + 4*q)*abs(phiTwo)**2 + (-p+q)*abs(phiOne)**2 
                + (p+q)*abs(phiMinusOne)**2 + (2*p + 4*q)*abs(phiMinusTwo)**2 
                + b0*phi4 + b1*f4 + b2*a0Sq)
    
    """This bit is only used for periodic boundary conditions"""
    integrand = np.vstack((integrand,integrand[0]))
    integrandCol = np.zeros(len(integrand),dtype='complex_')
    for i in range(len(integrand)):
        integrandCol[i] = integrand[i,0]
    
    integrand = np.concatenate((integrand,np.array([integrandCol]).T),axis=1)
    
    xInt = np.zeros(len(integrand),dtype='complex_')
    dx = x[1]-x[0]
    dy = x[1]-x[0]
    
    for i in range(len(integrand)):
        xInt[i] = np.trapz(integrand[i],dx=dx)
    
    mu = np.trapz(xInt,dx=dy)
    
    return mu

def constructSubMat(term,L,h,deriv=False):
    """This function constructs a subMat of the matrix of linear equations that 
    are the Bogoliubov equations. term is the phi terms or otherwise needed to be 
    multiplied together to produce the term that fills the current submat.
    deriv tells us whether we are constructing an Li term in which case we need to handle 
    the derivative term aswell."""
    
    index = 0 #Variable to index the row, col and data lists.
    
    if (deriv):
        numEntries = (L)**2 + 4*(L)*(L-1) + 4*(L)
    else:
        numEntries = (L)**2
        
    row = np.zeros(numEntries)
    col = np.zeros(numEntries)
    data = np.zeros(numEntries,dtype='complex_')
        
    for i in range((L)**2):
        row[index] = i
        col[index] = i
        data[index] = term[int(i%(L)),int(i/(L))]
        
        if (deriv):
            data[index] += 2/(h**2)
            index += 1
            
            
            if i % (L) != (L-1):
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
                row[index] = i - (L-1)
                col[index] = i
                data[index] = -1/(2*h**2)
                index += 1
                
                row[index] = i
                col[index] = i - (L-1)
                data[index] = -1/(2*h**2)
                index += 1
            
            if i < (L)*(L-1):
                row[index] = i
                col[index] = i + (L)
                data[index] = -1/(2*h**2)
                index += 1
                
                row[index] = i + (L)
                col[index] = i
                data[index] = -1/(2*h**2)
                index += 1
            
            """Adding in bit to make x periodic boundary conditions"""
            if i < (L):
                row[index] = i
                col[index] = i + (L)*(L-1)
                data[index] = -1/(2*h**2)
                index += 1
                
                row[index] = i + (L)*(L-1)
                col[index] = i
                data[index] = -1/(2*h**2)
                index += 1
        else:
            index += 1
            
     
    subMat = sp.sparse.csc_matrix((data,(row,col)),shape=((L)**2,(L)**2))
    
    return subMat


def bogoliubov(phiTwoList,phiOneList,phiZeroList,phiMinusOneList,phiMinusTwoList,L,a,b,x,y,V,b0,b1,b2,p,q,directory='',numVecs=100):
    """Given the background order parameter (phiOneList,phiZeroList,phiMinusOneList), solve 
    the Bogoliubov equations and return the eigenvectors and eigenvalues.
    
    Inputs:
        phiTwoList: m=2 component of background,
        phiOneList: m=1 component of background,
        phiZeroList: m=0 component of background,
        phiMinusOneList: m=-1 component of background,
        phiMinusTwoList: m=-2 component of background,
        L: Number of lattice spacings,
        a,b: Boundaries of our dicretized grid,
        x,y: Lists of x and y values on our grid,
        V: 2D array of the potential for the system,
        b0,b1,b2: Interaction parameters,
        p,q: Linear and quadratic Zeeman terms,
        directory: directory to save results in. Eg: to save in results, set directory = 'results/'
        numVecs: The number of eigenvectors to obtain, default is 100.
        
    This file does not return anything and instead saves the eigVecs and eigVals 
    as eigVecs[mainIter] and eigVals[mainIter] that can be loaded back in to 
    inspect after.
        """
    
    
    
    t1 = time.time()
    
    h = (b-a)/(L)
    
    muG = computeMu(phiTwoList,phiOneList,phiZeroList,phiMinusOneList,phiMinusTwoList,x,y,V,b0,b1,b2,p,q)
    
    lOneTerm = (V -2*p + 4*q + 2*(4*b1 + b0)*abs(phiTwoList)**2 + (4*b1 + b0)*abs(phiOneList)**2 
                + b0*abs(phiZeroList)**2 + (b0-2*b1)*abs(phiMinusOneList)**2 
                + (b0 - 4*b1 + (2/5)*b2)*abs(phiMinusTwoList)**2 - muG)
    
    L1 = constructSubMat(lOneTerm,L,h,True)
    
    lTwoTerm = (V - p + q + (b0 + 4*b1)*abs(phiTwoList)**2 + 2*(b0 + b1)*abs(phiOneList)**2 
                + (b0 + 3*b1)*abs(phiZeroList)**2 + (b0 - b1 + (2/5)*b2)*abs(phiMinusOneList)**2 
                + (b0 - 2*b1)*abs(phiMinusTwoList)**2 - muG)
    
    L2 = constructSubMat(lTwoTerm,L,h,True)
    
    lThreeTerm = (V + b0*(abs(phiTwoList)**2 + abs(phiOneList)**2 + 2*abs(phiZeroList)**2 + abs(phiMinusOneList)**2 + abs(phiMinusTwoList)**2) 
                  + 3*b1*(abs(phiOneList)**2 + abs(phiMinusOneList)**2) + (2/5)*b2*abs(phiZeroList)**2 - muG + 5)
    
    L3 = constructSubMat(lThreeTerm,L,h,True)
    
    lFourTerm = (V + p + q + (b0 - 2*b1)*abs(phiTwoList)**2 + (b0 - b1 + (2/5)*b2)*abs(phiOneList)**2 
                 + (b0 + 3*b1)*abs(phiZeroList)**2 + 2*(b0+b1)*abs(phiMinusOneList)**2 
                 + (b0 + 4*b1)*abs(phiMinusTwoList)**2 - muG)
    
    L4 = constructSubMat(lFourTerm,L,h,True)
    
    lFiveTerm = (V + 2*p + 4*q + (b0 - 4*b1 + (2/5)*b2)*abs(phiTwoList)**2 + (b0 - 2*b1)*abs(phiOneList)**2 
                 + b0*abs(phiZeroList)**2 + (b0 + 4*b1)*abs(phiMinusOneList)**2 
                 + 2*(b0 + 4*b1)*abs(phiMinusTwoList)**2 - muG)
    
    L5 = constructSubMat(lFiveTerm,L,h,True)
    
    H12 = constructSubMat((4*b1 + b0)*(phiTwoList)**2,L,h)
    H13 = constructSubMat(b0*np.conjugate(phiOneList)*phiTwoList + b1*(4*np.conjugate(phiOneList)*phiTwoList 
                                                                       + 2*np.sqrt(6)*np.conjugate(phiZeroList)*phiOneList 
                                                                       + np.sqrt(6)*np.conjugate(phiMinusOneList)*phiZeroList 
                                                                       + 2*np.conjugate(phiMinusTwoList)*phiMinusOneList) 
                          - (2/5)*b2*np.conjugate(phiMinusTwoList)*phiMinusOneList,L,h)
    
    H14 = constructSubMat((4*b1 + b0)*phiOneList*phiTwoList,L,h)
    H15 = constructSubMat(b0*np.conjugate(phiZeroList)*phiTwoList + np.sqrt(6)*b1*np.conjugate(phiMinusOneList)*phiOneList 
                          + (2/5)*b2*np.conjugate(phiMinusTwoList)*phiZeroList,L,h)
    
    H16 = constructSubMat(b0*phiZeroList*phiTwoList + np.sqrt(6)*b1*phiOneList**2,L,h)
    H17 = constructSubMat((b0 - 2*b1)*np.conjugate(phiMinusOneList)*phiTwoList + 2*b1*np.conjugate(phiMinusTwoList)*phiOneList 
                          - (2/5)*b2*np.conjugate(phiMinusTwoList)*phiOneList,L,h)
    H18 = constructSubMat((b0 - 2*b1)*phiMinusOneList*phiTwoList + np.sqrt(6)*b1*phiZeroList*phiOneList,L,h)
    H19 = constructSubMat((b0 - 4*b1)*np.conjugate(phiMinusTwoList)*phiTwoList + (2/5)*b2*np.conjugate(phiMinusTwoList)*phiTwoList,L,h)
    H110 = constructSubMat((b0 - 4*b1 + (2/5)*b2)*phiMinusTwoList*phiTwoList + (2*b1 - (2/5)*b2)*phiOneList*phiMinusOneList + (1/5)*b2*phiZeroList**2,L,h)
    
    H21 = -np.conjugate(H12)
    H23 = constructSubMat(-(4*b1 + b0)*np.conjugate(phiOneList*phiTwoList),L,h)
    H24 = constructSubMat(-(b0*phiOneList*np.conjugate(phiTwoList) + b1*(4*phiOneList*np.conjugate(phiTwoList) 
                                                                         + 2*np.sqrt(6)*phiZeroList*np.conjugate(phiOneList) 
                                                                         + np.sqrt(6)*phiMinusOneList*np.conjugate(phiZeroList) 
                                                                         + 2*phiMinusTwoList*np.conjugate(phiMinusOneList)) 
                            - (2/5)*b2*phiMinusTwoList*np.conjugate(phiMinusOneList)),L,h)
    
    H25 = constructSubMat(-(b0*np.conjugate(phiZeroList*phiTwoList) + np.sqrt(6)*b1*np.conjugate(phiOneList)**2),L,h)
    H26 = constructSubMat(-(b0*phiZeroList*np.conjugate(phiTwoList) + np.sqrt(6)*b1*phiMinusOneList*np.conjugate(phiOneList) + (2/5)*b2*phiMinusTwoList*np.conjugate(phiZeroList)),L,h)
    H27 = constructSubMat(-((b0 - 2*b1)*np.conjugate(phiMinusOneList*phiTwoList) + np.sqrt(6)*b1*np.conjugate(phiZeroList*phiOneList)),L,h)
    H28 = constructSubMat(-((b0 - 2*b1)*phiMinusOneList*np.conjugate(phiTwoList) + 2*b1*phiMinusTwoList*np.conjugate(phiOneList) - (2/5)*b2*phiMinusTwoList*np.conjugate(phiOneList)),L,h)
    H29 = constructSubMat(-((b0 - 4*b1 + (2/5)*b2)*np.conjugate(phiMinusTwoList*phiTwoList) + (2*b1 - (2/5)*b2)*np.conjugate(phiMinusOneList*phiOneList) + (1/5)*b2*np.conjugate(phiZeroList)**2),L,h)
    H210 = constructSubMat(-((b0 - 4*b1 + (2/5)*b2)*phiMinusTwoList*np.conjugate(phiTwoList)),L,h)
    
    H31 = np.conjugate(H13)
    H32 = -np.conjugate(H23)
    H34 = constructSubMat((b0 + b1)*phiOneList**2 + 2*np.sqrt(6)*b1*phiTwoList*phiZeroList,L,h)
    H35 = constructSubMat((b0 + 3*b1)*np.conjugate(phiZeroList)*phiOneList + 2*np.sqrt(6)*b1*np.conjugate(phiOneList)*phiTwoList 
                          + np.sqrt(6)*b1*np.conjugate(phiMinusTwoList)*phiMinusOneList + (6*b1 - (2/5)*b2)*np.conjugate(phiMinusOneList)*phiZeroList,L,h)
    
    H36 = constructSubMat((b0 + 3*b1)*phiZeroList*phiOneList + np.sqrt(6)*b1*phiMinusOneList*phiTwoList,L,h)
    H37 = constructSubMat((b0 - b1 + (2/5)*b2)*np.conjugate(phiMinusOneList)*phiOneList + np.sqrt(6)*b1*(np.conjugate(phiMinusTwoList)*phiZeroList + np.conjugate(phiZeroList)*phiTwoList),L,h)
    H38 = constructSubMat((b0 - b1 + (2/5)*b2)*phiOneList*phiMinusOneList + (2*b1 - (2/5)*b2)*phiMinusTwoList*phiTwoList + (3*b1 - (1/5)*b2)*phiZeroList**2,L,h)
    H39 = constructSubMat((b0 - 2*b1)*np.conjugate(phiMinusTwoList)*phiOneList + (2*b1 - (2/5)*b2)*np.conjugate(phiMinusOneList)*phiTwoList,L,h)
    H310 = constructSubMat((b0 - 2*b1)*phiMinusTwoList*phiOneList + np.sqrt(6)*b1*phiMinusOneList*phiZeroList,L,h)
    
    H41 = -np.conjugate(H14)
    H42 = np.conjugate(H24)
    H43 = -np.conjugate(H34)
    H45 = constructSubMat(-((b0 + 3*b1)*np.conjugate(phiZeroList*phiOneList) + np.sqrt(6)*b1*np.conjugate(phiMinusOneList*phiTwoList)),L,h)
    H46 = constructSubMat(-((b0 + 3*b1)*phiZeroList*np.conjugate(phiOneList) + 2*np.sqrt(6)*b1*phiOneList*np.conjugate(phiTwoList)
                            + np.sqrt(6)*b1*phiMinusTwoList*np.conjugate(phiMinusOneList) + (6*b1 - (2/5)*b2)*phiMinusOneList*np.conjugate(phiZeroList)),L,h)
    
    H47 = constructSubMat(-((b0 - b1 + (2/5)*b2)*np.conjugate(phiOneList*phiMinusOneList) + (2*b1 - (2/5)*b2)*np.conjugate(phiMinusTwoList*phiTwoList) 
                            + (3*b1 - (1/5)*b2)*np.conjugate(phiZeroList)**2),L,h)
    
    H48 = constructSubMat(-((b0 - b1 + (2/5)*b2)*phiMinusOneList*np.conjugate(phiOneList) + np.sqrt(6)*b1*(phiMinusTwoList*np.conjugate(phiZeroList) + phiZeroList*np.conjugate(phiTwoList))),L,h)
    H49 = constructSubMat(-((b0 - 2*b1)*np.conjugate(phiMinusTwoList*phiOneList) + np.sqrt(6)*b1*np.conjugate(phiMinusOneList*phiZeroList)),L,h)
    H410 = constructSubMat(-((b0 - 2*b1)*phiMinusTwoList*np.conjugate(phiOneList) + (2*b1 - (2/5)*b2)*phiMinusOneList*np.conjugate(phiTwoList)),L,h)
    
    H51 = np.conjugate(H15)
    H52 = -np.conjugate(H25)
    H53 = np.conjugate(H35)
    H54 = -np.conjugate(H45)
    H56 = constructSubMat((b0 + (1/5)*b2)*phiZeroList**2 + (6*b1 - (2/5)*b2)*phiOneList*phiMinusOneList + (2/5)*b2*phiMinusTwoList*phiTwoList,L,h)
    H57 = constructSubMat((b0 + 3*b1)*np.conjugate(phiMinusOneList)*phiZeroList + (6*b1 - (2/5)*b2)*np.conjugate(phiZeroList)*phiOneList 
                          + np.sqrt(6)*b1*np.conjugate(phiOneList)*phiTwoList + 2*np.sqrt(6)*b1*np.conjugate(phiMinusTwoList)*phiMinusOneList,L,h)
    
    H58 = constructSubMat((b0 + 3*b1)*phiMinusOneList*phiZeroList + np.sqrt(6)*b1*phiMinusTwoList*phiOneList,L,h)
    H59 = constructSubMat(b0*np.conjugate(phiMinusTwoList)*phiZeroList + np.sqrt(6)*b1*np.conjugate(phiMinusOneList)*phiOneList + (2/5)*b2*np.conjugate(phiZeroList)*phiTwoList,L,h)
    H510 = constructSubMat(b0*phiMinusTwoList*phiZeroList + np.sqrt(6)*b1*phiMinusOneList**2,L,h)
    
    H61 = -np.conjugate(H16)
    H62 = np.conjugate(H26)
    H63 = -np.conjugate(H36)
    H64 = np.conjugate(H46)
    H65 = -np.conjugate(H56)
    H67 = constructSubMat(-((b0 + 3*b1)*np.conjugate(phiMinusOneList*phiZeroList) + np.sqrt(6)*b1*np.conjugate(phiMinusTwoList*phiOneList)),L,h)
    H68 = constructSubMat(-((b0 + 3*b1)*phiMinusOneList*np.conjugate(phiZeroList) + (6*b1 - (2/5)*b2)*phiZeroList*np.conjugate(phiOneList) 
                            + np.sqrt(6)*b1*phiOneList*np.conjugate(phiTwoList) + 2*np.sqrt(6)*b1*phiMinusTwoList*np.conjugate(phiMinusOneList)),L,h)
    
    H69 = constructSubMat(-(b0*np.conjugate(phiMinusTwoList*phiZeroList) + np.sqrt(6)*b1*np.conjugate(phiMinusOneList)**2),L,h)
    H610 = constructSubMat(-(b0*phiMinusTwoList*np.conjugate(phiZeroList) + np.sqrt(6)*b1*phiMinusOneList*np.conjugate(phiOneList) + (2/5)*b2*phiZeroList*np.conjugate(phiTwoList)),L,h)
    
    H71 = np.conjugate(H17)
    H72 = -np.conjugate(H27)
    H73 = np.conjugate(H37)
    H74 = -np.conjugate(H47)
    H75 = np.conjugate(H57)
    H76 = -np.conjugate(H67)
    H78 = constructSubMat((b0 + b1)*phiMinusOneList**2 + 2*np.sqrt(6)*b1*phiZeroList*phiMinusTwoList,L,h)
    H79 = constructSubMat((b0 + 4*b1)*np.conjugate(phiMinusTwoList)*phiMinusOneList + 2*np.sqrt(6)*b1*np.conjugate(phiMinusOneList)*phiZeroList 
                          + np.sqrt(6)*b1*np.conjugate(phiZeroList)*phiOneList + (2*b1 - (2/5)*b2)*np.conjugate(phiOneList)*phiTwoList,L,h)
    
    H710 = constructSubMat((b0 + 4*b1)*phiMinusTwoList*phiMinusOneList,L,h)
    
    H81 = -np.conjugate(H18)
    H82 = np.conjugate(H28)
    H83 = -np.conjugate(H38)
    H84 = np.conjugate(H48)
    H85 = -np.conjugate(H58)
    H86 = np.conjugate(H68)
    H87 = -np.conjugate(H78)
    H89 = constructSubMat(-((b0 + 4*b1)*np.conjugate(phiMinusTwoList*phiMinusOneList)),L,h)
    H810 = constructSubMat(-((b0 + 4*b1)*phiMinusTwoList*np.conjugate(phiMinusOneList) + 2*np.sqrt(6)*b1*phiMinusOneList*np.conjugate(phiZeroList) 
                             + (2*b1 - (2/5)*b2)*phiOneList*np.conjugate(phiTwoList) + np.sqrt(6)*b1*phiZeroList*np.conjugate(phiOneList)),L,h)
    
    H91 = np.conjugate(H19)
    H92 = -np.conjugate(H29)
    H93 = np.conjugate(H39)
    H94 = -np.conjugate(H49)
    H95 = np.conjugate(H59)
    H96 = -np.conjugate(H69)
    H97 = np.conjugate(H79)
    H98 = -np.conjugate(H89)
    H910 = constructSubMat((b0 + 4*b1)*phiMinusTwoList**2,L,h)
    
    H101 = -np.conjugate(H110)
    H102 = np.conjugate(H210)
    H103 = -np.conjugate(H310)
    H104 = np.conjugate(H410)
    H105 = -np.conjugate(H510)
    H106 = np.conjugate(H610)
    H107 = -np.conjugate(H710)
    H108 = np.conjugate(H810)
    H109 = -np.conjugate(H910)
    
    print('Constructed submats')
    t2 = time.time()
    print('Time elapsed = ' + str(t2-t1))
    
    
    """Put these submats together into the large matrix whose eigenvecs and 
    eigvals will be found to solve the Bogoliubov equations."""
    H1 = sp.sparse.csc_matrix(sp.sparse.vstack(((L1,H21,H31,H41,H51,H61,H71,H81,H91,H101))))
    del H21,H31,H41,H51,H61,H71,H81,H91,H101
    H2 = sp.sparse.csc_matrix(sp.sparse.vstack(((H12,-L1,H32,H42,H52,H62,H72,H82,H92,H102))))
    del H12,L1,H32,H42,H52,H62,H72,H82,H92,H102
    H3 = sp.sparse.csc_matrix(sp.sparse.vstack(((H13,H23,L2,H43,H53,H63,H73,H83,H93,H103))))
    del H13,H23,H43,H53,H63,H73,H83,H93,H103
    H4 = sp.sparse.csc_matrix(sp.sparse.vstack(((H14,H24,H34,-L2,H54,H64,H74,H84,H94,H104))))
    del H14,H24,H34,L2,H54,H64,H74,H84,H94,H104
    H5 = sp.sparse.csc_matrix(sp.sparse.vstack(((H15,H25,H35,H45,L3,H65,H75,H85,H95,H105))))
    del H15,H25,H35,H45,H65,H75,H85,H95,H105
    H6 = sp.sparse.csc_matrix(sp.sparse.vstack(((H16,H26,H36,H46,H56,-L3,H76,H86,H96,H106))))
    del H16,H26,H36,H46,H56,L3,H76,H86,H96,H106
    H7 = sp.sparse.csc_matrix(sp.sparse.vstack(((H17,H27,H37,H47,H57,H67,L4,H87,H97,H107))))
    del H17,H27,H37,H47,H57,H67,H87,H97,H107
    H8 = sp.sparse.csc_matrix(sp.sparse.vstack(((H18,H28,H38,H48,H58,H68,H78,-L4,H98,H108))))
    del H18,H28,H38,H48,H58,H68,H78,L4,H98,H108
    H9 = sp.sparse.csc_matrix(sp.sparse.vstack(((H19,H29,H39,H49,H59,H69,H79,H89,L5,H109))))
    del H19,H29,H39,H49,H59,H69,H79,H89,H109
    H10 = sp.sparse.csc_matrix(sp.sparse.vstack(((H110,H210,H310,H410,H510,H610,H710,H810,H910,-L5))))
    del H110,H210,H310,H410,H510,H610,H710,H810,H910,L5
    
    print('First stacking done')
    t3 = time.time()
    print('Time elapsed = ' + str(t3-t1))
    
    
    H = sp.sparse.csc_matrix(sp.sparse.hstack(((H1,H2,H3,H4,H5,H6,H7,H8,H9,H10))))
    del H1,H2,H3,H4,H5,H6,H7,H8,H9,H10
    
    print('Second stacking done')
    t4 = time.time()
    print('Time elapsed = ' + str(t4-t1))
    
    
    
    eigVals,eigVecs = eigs(H,numVecs,sigma=0,which='LM')
    
    print('Eigvals and eigvecs found')
    t5 = time.time()
    print('Time elapsed = ' + str(t5-t1))
    
    
    """Order the eigenvalues and eigenvectors in terms of increasing abs value of eigvals."""
    idx = abs(eigVals).argsort()
    eigVals = eigVals[idx]
    eigVecs = eigVecs[:,idx]
    
    eigVecs = np.transpose(eigVecs) #Reformat to be compatible with the way I'm looping below.
    
    print('Returning eigvecs and eigvals')
    t6 = time.time()
    print('Time elapsed = ' + str(t6-t1))
    
    gst.saveResults(directory=directory,eigVecs=eigVecs)
    gst.saveResults(directory=directory,eigVals=eigVals)







    
    
    
            
    
    
    
    
    


    
    
    


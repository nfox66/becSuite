# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 12:35:36 2022

@author: 16383526
"""

import numpy as np
import pickle
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import loop
import graphicalBEC as gbec
import graphicalVortex as gv


def getPhiNorm(phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo,h):
    """Takes in components of ground state phi and lattice spacing h and returns 
    the norm.
    
    Inputs:
        phiTwo: m = 2 component of phi,
        phiOne: m = 1 component of phi,
        phiZero: m = 0 component of phi,
        phiMinusOne: m = -1 component of phi,
        phiMinusTwo: m = -2 component of phi,
        h: Lattice spacing (same for x and y direction).
    
    Returns:
        norm: The norm of phi."""
    
    """Make sure the phi lists are flattened, this way I can now compute the norm 
    even if say a 2D array is passed in"""
    
    def flattenArray(phi):
        if (len(np.shape(phi)) > 1):
            phi = np.reshape(phi,len(phi)**2,order='F')
        return phi
    
    phiTwo = flattenArray(phiTwo)
    phiOne = flattenArray(phiOne)
    phiZero = flattenArray(phiZero)
    phiMinusOne = flattenArray(phiMinusOne)
    phiMinusTwo = flattenArray(phiMinusTwo)
    
    
    norm = 0
    for i in range(len(phiOne)):
        norm += (abs(phiTwo[i])**2 + abs(phiOne[i])**2 + abs(phiZero[i])**2 + abs(phiMinusOne[i])**2 + abs(phiMinusTwo[i])**2)
    norm *= h**2
    norm = np.sqrt(norm)
    return norm

def phiFour(phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo):
    """Return density term in energy functional.
        
        Inputs:
            phiTwo: m = 2 component of phi,
            phiOne: m = 1 component of phi,
            phiZero: m = 0  component of phi,
            phiMinusOne: m = -1 component of phi,
            phiMinusTwo: m = -2 component of phi.
            
        Returns:
            phiFour: The density term in the energy functional."""
    pTwo = abs(phiTwo)**2
    pOne = abs(phiOne)**2
    pZero = abs(phiZero)**2
    pMinusOne = abs(phiMinusOne)**2
    pMinusTwo = abs(phiMinusTwo)**2
    
    phiFour = (pTwo+pOne+pZero+pMinusOne+pMinusTwo)**2
    
    return phiFour

def fFour(phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo):
    """Return the spin density term in the energy functional.
        
        Inputs:
            phiTwo: m = 2 component of phi,
            phiOne: m = 1 component of phi,
            phiZero: m = 0 component of phi,
            phiMinusOne: m = -1 component of phi,
            phiMinusTwo: m = -2 component of phi.
            
        Returns:
            fFour: The spin density term in the energy functional."""
    
    pTwo = abs(phiTwo)**2
    pOne = abs(phiOne)**2
    pZero = abs(phiZero)**2
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
    """Return the spin singlet density term in the energy functional.
    
    Inputs:
        phiTwo: m = 2 component of phi,
        phiOne, m = 1 component of phi,
        phiZero: m = 0 component of phi,
        phiMinusOne: m = -1 component of phi,
        phiMinusTwo: m = -2 component of phi.
        
    Returns:
        aZero: The spin singlet density term in the energy functional."""
    
    
    aZero = np.sqrt(1/5)*(2*phiTwo*phiMinusTwo - 2*phiOne*phiMinusOne + phiZero**2)
    aZero = abs(aZero)**2
    return aZero

"""Need a function to take the derivative of phi"""
def deriv(phi,h):
    """Returns the abs value squared of the gradient of phi in 2-dimensions.
    
    Inputs:
        phi: A 2D array storing the value of phi at all positions in our grid.
        
    Returns:
        absDerivSq: The absolute value squared of the gradient of phi."""
    
    
    grad = np.gradient(phi,h,h)
    conjGrad = np.gradient(np.conjugate(phi),h,h)
    
    absDerivSq = np.array(grad[0]*conjGrad[0] + grad[1]*conjGrad[1])
    
    return absDerivSq



def computeEnergy(phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo,L,h,V,b0,b1,b2,p,q,periodic=False):
    """Takes in components of ground state phi along with certain parameters of 
    the relevant system and returns the energy of the system in the given ground 
    state.
    
    Inputs:
        phiTwo: m = 2 component of phi,
        phiOne: m = 1 component of phi,
        phiZero: m = 0 component of phi,
        phiMinusOne: m = -1 component of phi,
        phiMinusTwo: m = -2 component of phi,
        L: The number of lattice spacings,
        h: Lattice spacing (same for x and y direction),
        V: The 2D array storing the potential the sytem is in,
        b0: The dimensionless density interaction parameter,
        b1: The dimensionless spin density interaction parameter,
        p: The linear Zeeman term,
        q: The quadratic Zeeman term.
        
    Returns:
        energy: The value of the energy functional of the current state of the system."""
    
    
    phi4 = phiFour(phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo)
    f4 = fFour(phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo)
    a0Sq = aZeroSq(phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo)
    
    # print('phi4 = ' + str(phi4))
    # print('f4 = ' + str(f4))
    # print('a0Sq = ' + str(a0Sq))
    
    # print('derivTwo = ' + str(deriv(phiTwo,h)))
    # print('derivOne = ' + str(deriv(phiOne,h)))
    # print('derivZero = ' + str(deriv(phiZero,h)))
    # print('derivMinusOne = ' + str(deriv(phiMinusOne,h)))
    # print('derivMinusTwo = ' + str(deriv(phiMinusTwo,h)))
    
    """Compute the integrand of the energy functional."""
    integrand = ((1/2)*(deriv(phiTwo,h)) + (1/2)*(deriv(phiOne,h)) + (1/2)*(deriv(phiZero,h)) 
                 + (1/2)*(deriv(phiMinusOne,h)) + (1/2)*(deriv(phiMinusTwo,h))
                 + V*abs(phiTwo)**2 + V*abs(phiOne)**2 + V*abs(phiZero)**2 
                 + V*abs(phiMinusOne)**2 + V*abs(phiMinusTwo)**2
                 + (-2*p + 4*q)*abs(phiTwo) + (-p+q)*abs(phiOne)**2 
                 + (p+q)*abs(phiMinusOne) + (2*p + 4*q)*abs(phiMinusTwo) 
                 + (b0/2)*(phi4) + (b1/2)*(f4) + (b2/2)*a0Sq)
    
    
    integrandOne = ((1/2)*(deriv(phiTwo,h)) + (1/2)*(deriv(phiOne,h)) + (1/2)*(deriv(phiZero,h)) 
                 + (1/2)*(deriv(phiMinusOne,h)) + (1/2)*(deriv(phiMinusTwo,h)))
    
    integrandTwo = (b0/2)*(phi4) + (b1/2)*(f4) + (b2/2)*a0Sq
    
    """Need to add this for periodic boundary conditions"""
    
    if (periodic == True):
        integrand = np.vstack((integrand,integrand[0]))
        integrandCol = np.zeros(len(integrand),dtype='complex_')
        for i in range(len(integrand)):
            integrandCol[i] = integrand[i,0]
        integrand = np.concatenate((integrand,np.array([integrandCol]).T),axis=1)
    
    
    # if (periodic == True):
    #     integrandOne = np.vstack((integrandOne,integrandOne[0]))
    #     integrandColOne = np.zeros(len(integrandOne),dtype='complex_')
    #     for i in range(len(integrandOne)):
    #         integrandColOne[i] = integrandOne[i,0]
    #     integrandOne = np.concatenate((integrandOne,np.array([integrandColOne]).T),axis=1)
        
    # if (periodic == True):
    #     integrandTwo = np.vstack((integrandTwo,integrandTwo[0]))
    #     integrandColTwo = np.zeros(len(integrandTwo),dtype='complex_')
    #     for i in range(len(integrandTwo)):
    #         integrandColTwo[i] = integrandTwo[i,0]
    #     integrandTwo = np.concatenate((integrandTwo,np.array([integrandColTwo]).T),axis=1)
    
    """Take the integral to compute the energy."""
    xInt = np.zeros(len(integrand),dtype='complex_')
    for i in range(len(integrand)):
        xInt[i] = np.trapz(integrand[i],dx=h)
    energy = np.trapz(xInt,dx=h)
    
    
    # xIntOne = np.zeros(len(integrandOne),dtype='complex_')
    # for i in range(len(integrandOne)):
    #     xIntOne[i] = np.trapz(integrandOne[i],dx=h)
    # energyOne = np.trapz(xIntOne,dx=h)
    
    # xIntTwo = np.zeros(len(integrandTwo),dtype='complex_')
    # for i in range(len(integrandTwo)):
    #     xIntTwo[i] = np.trapz(integrandTwo[i],dx=h)
    # energyTwo = np.trapz(xIntTwo,dx=h)
    
    # return energyOne,energyTwo
    return energy

def getDensityEnergy(phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo,L,h,V,b0,b1,b2,p,q,periodic=False):
    """Takes in components of ground state phi along with certain parameters of 
    the relevant system and returns the density energy contribution of the system in the given ground 
    state.
    
    Inputs:
        phiTwo: m = 2 component of phi,
        phiOne: m = 1 component of phi,
        phiZero: m = 0 component of phi,
        phiMinusOne: m = -1 component of phi,
        phiMinusTwo: m = -2 component of phi,
        L: The number of lattice spacings,
        h: Lattice spacing (same for x and y direction),
        V: The 2D array storing the potential the sytem is in,
        b0: The dimensionless density interaction parameter,
        b1: The dimensionless spin density interaction parameter,
        p: The linear Zeeman term,
        q: The quadratic Zeeman term.
        
    Returns:
        densityEnergy: The value of the density energy contribution of the current state of the system."""
        
    phi4 = phiFour(phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo)
    integrand = (b0/2)*(phi4)
    
    """Need to add this for periodic boundary conditions"""
    
    if (periodic == True):
        integrand = np.vstack((integrand,integrand[0]))
        integrandCol = np.zeros(len(integrand),dtype='complex_')
        for i in range(len(integrand)):
            integrandCol[i] = integrand[i,0]
        integrand = np.concatenate((integrand,np.array([integrandCol]).T),axis=1)
        
    """Take the integral to compute the energy."""
    xInt = np.zeros(len(integrand),dtype='complex_')
    for i in range(len(integrand)):
        xInt[i] = np.trapz(integrand[i],dx=h)
    densityEnergy = np.trapz(xInt,dx=h)
    
    return densityEnergy


def getSpinDensityEnergy(phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo,L,h,V,b0,b1,b2,p,q,periodic=False):
    """Takes in components of ground state phi along with certain parameters of 
    the relevant system and returns the spin density energy contribution of the system in the given ground 
    state.
    
    Inputs:
        phiTwo: m = 2 component of phi,
        phiOne: m = 1 component of phi,
        phiZero: m = 0 component of phi,
        phiMinusOne: m = -1 component of phi,
        phiMinusTwo: m = -2 component of phi,
        L: The number of lattice spacings,
        h: Lattice spacing (same for x and y direction),
        V: The 2D array storing the potential the sytem is in,
        b0: The dimensionless density interaction parameter,
        b1: The dimensionless spin density interaction parameter,
        p: The linear Zeeman term,
        q: The quadratic Zeeman term.
        
    Returns:
        spinDensityEnergy: The value of the spin density energy contribution of the current state of the system."""
        
    f4 = fFour(phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo)
    integrand = (b1/2)*(f4)
    
    """Need to add this for periodic boundary conditions"""
    
    if (periodic == True):
        integrand = np.vstack((integrand,integrand[0]))
        integrandCol = np.zeros(len(integrand),dtype='complex_')
        for i in range(len(integrand)):
            integrandCol[i] = integrand[i,0]
        integrand = np.concatenate((integrand,np.array([integrandCol]).T),axis=1)
        
    """Take the integral to compute the energy."""
    xInt = np.zeros(len(integrand),dtype='complex_')
    for i in range(len(integrand)):
        xInt[i] = np.trapz(integrand[i],dx=h)
    spinDensityEnergy = np.trapz(xInt,dx=h)
    
    return spinDensityEnergy

def getSpinSingletEnergy(phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo,L,h,V,b0,b1,b2,p,q,periodic=False):
    """Takes in components of ground state phi along with certain parameters of 
    the relevant system and returns the spin singlet energy contribution of the system in the given ground 
    state.
    
    Inputs:
        phiTwo: m = 2 component of phi,
        phiOne: m = 1 component of phi,
        phiZero: m = 0 component of phi,
        phiMinusOne: m = -1 component of phi,
        phiMinusTwo: m = -2 component of phi,
        L: The number of lattice spacings,
        h: Lattice spacing (same for x and y direction),
        V: The 2D array storing the potential the sytem is in,
        b0: The dimensionless density interaction parameter,
        b1: The dimensionless spin density interaction parameter,
        p: The linear Zeeman term,
        q: The quadratic Zeeman term.
        
    Returns:
        spinSingletEnergy: The value of the spin density energy contribution of the current state of the system."""
        
    a0Sq = aZeroSq(phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo)
    integrand = (b2/2)*(a0Sq)
    
    """Need to add this for periodic boundary conditions"""
    
    if (periodic == True):
        integrand = np.vstack((integrand,integrand[0]))
        integrandCol = np.zeros(len(integrand),dtype='complex_')
        for i in range(len(integrand)):
            integrandCol[i] = integrand[i,0]
        integrand = np.concatenate((integrand,np.array([integrandCol]).T),axis=1)
        
    """Take the integral to compute the energy."""
    xInt = np.zeros(len(integrand),dtype='complex_')
    for i in range(len(integrand)):
        xInt[i] = np.trapz(integrand[i],dx=h)
    spinSingletEnergy = np.trapz(xInt,dx=h)
    
    return spinSingletEnergy

def getRotMat(alpha,beta,gamma):
    """Returns the 5 dim rep of SO(3) rotation matrix for a spin two condensate given the 
    Euler angles.
    
    Inputs:
        alpha, beta, gamma: The Euler angles for ZYZ format.
        
    Returns: 
        mat: The 5 dimensional representation of the SO(3) matrix corresponding 
        to the given Euler angles."""
    
    c = np.cos(beta/2)
    s = np.sin(beta/2)
    
    mat = np.zeros((5,5),dtype='object')
    
    mat[0][0] = np.exp(-2j*(alpha+gamma))*c**4
    mat[0][1] = -2*np.exp(-1j*(2*alpha + gamma))*(c**3)*(s)
    mat[0][2] = np.sqrt(6)*np.exp(-2j*alpha)*(c**2)*(s**2)
    mat[0][3] = -2*np.exp(-1j*(2*alpha - gamma))*(c)*(s**3)
    mat[0][4] = np.exp(-2j*(alpha - gamma))*s**4
    
    mat[1][0] = 2*np.exp(-1j*(alpha + 2*gamma))*(c**3)*(s)
    mat[1][1] = np.exp(-1j*(alpha + gamma))*(c**2)*(c**2 - 3*s**2)
    mat[1][2] = -np.sqrt(3/8)*np.exp(-1j*alpha)*np.sin(2*beta)
    mat[1][3] = -np.exp(-1j*(alpha - gamma))*(s**2)*(s**2 - 3*c**2)
    mat[1][4] = -2*np.exp(-1j*(alpha - 2*gamma))*(c)*(s**3)
    
    mat[2][0] = np.sqrt(6)*np.exp(-2j*gamma)*(c**2)*(s**2)
    mat[2][1] = np.sqrt(3/8)*np.exp(-1j*gamma)*np.sin(2*beta)
    mat[2][2] = (1/4)*(1 + 3*np.cos(2*beta))
    mat[2][3] = -np.sqrt(3/8)*np.exp(1j*gamma)*np.sin(2*beta)
    mat[2][4] = np.sqrt(6)*np.exp(2j*gamma)*(c**2)*(s**2)
    
    mat[3][0] = 2*np.exp(1j*(alpha - 2*gamma))*(c)*(s**3)
    mat[3][1] = -np.exp(1j*(alpha - gamma))*(s**2)*(s**2 - 3*c**2)
    mat[3][2] = np.sqrt(3/8)*np.exp(1j*alpha)*np.sin(2*beta)
    mat[3][3] = np.exp(1j*(alpha + gamma))*(c**2)*(c**2 - 3*s**2)
    mat[3][4] = -2*np.exp(1j*(alpha + 2*gamma))*(c**3)*(s)
    
    mat[4][0] = np.exp(2j*(alpha - gamma))*s**4
    mat[4][1] = 2*np.exp(1j*(2*alpha - gamma))*(c)*(s**3)
    mat[4][2] = np.sqrt(6)*np.exp(2j*alpha)*(c**2)*(s**2)
    mat[4][3] = 2*np.exp(1j*(2*alpha + gamma))*(c**3)*(s)
    mat[4][4] = np.exp(2j*(alpha + gamma))*(c**4)
    
    
    return mat

def getPhaseOrderParameter(alpha,beta,gamma,repOrderParam):
    """Multiply the 5D SO(3) matrix by the representative order parameter to 
    get the order parameter for whatever phase of the BEC we're looking at.
    
    Inputs:
        alpha,beta,gamma: The Euler angles for ZYZ format,
        repOrderParam: The representative order parameter for the phase of the 
                        BEC we are looking at.
    
    Returns: 
        phaseOrderParameter[i]: The ith component of the rotated order parameter."""
    
    mat = getRotMat(alpha,beta,gamma)
    
    print('Shape of matrix = ' + str(np.shape(mat)))
    print(mat)
    
    phaseOrderParameter = np.matmul(mat,repOrderParam)
    
    return phaseOrderParameter[0],phaseOrderParameter[1],phaseOrderParameter[2],phaseOrderParameter[3],phaseOrderParameter[4]

    

def saveResults(name='',directory='',**kwargs):
    """Pass in arrays from other files as the kwargs to be saved and this 
    code will save them in the optional directory specified and with an optional 
    name appended onto the end of the filename.
    
    Inputs:
        name: Optional string to append onto the end of the filename,
        directory: Optional path to save the files at,
        kwargs: List of arrays to save:
            Eg: If we pass in an array called 'list' to kwargs, it will save list 
            with the filename 'list' or 'list[name]' if name is 
            specified.
    """
    for i in kwargs:
        f = open(str(directory)+str(i)+str(name),'wb')
        pickle.dump(kwargs[i],f)
        f.close()
        
def loadResults(name,tag='',directory=''):
    """Load in results that have previously been saved as pickle files.
    
    Inputs:
        name: The name of the pickle file to load in as an array,
        tag: Optional string to append onto the end of the filename,
        directory: directory which the files are in (make sure to include / at the end).
    
    Returns:
        result: The loaded pickle file into the relevant object, eg: an array."""
    f = open(str(directory) + str(name) + str(tag),'rb')
    result = pickle.load(f)
    f.close()
    return result

def loadFullResults(name,folder=''):
    """Shortcut function to load saved components of a generated ground state phi.
    
    Inputs:
        name: The optional name appended onto the end of the pickle files for the 
        relevant ground state components.
        folder: Allows the user to specify what folder the files to load in are in. 
                Must include / at the end of the filepath.
        
    Returns:
        phiTwo: An array with the value of the m = 2 component of the ground state
                at each gradient descent step,
        phiOne: An array with the value of the m = 1 component of the ground state 
                at each gradient descent step,
        phiZero: An array with the value of the m = 0 component of the ground state 
                    at each gradient descent step,
        phiMinusOne: An array with the value of the m = -1 component of the ground state 
                        at each gradient descent step,
        phiMinusTwo: An array with the value of the m = -2 component of the ground state 
                        at each gradient descent step,
          
        phiTwoList: The ground state returned by the code for the m = 2 component,
        phiOneList: The ground state returned by the code for the m = 1 component,
        phiZeroList: The ground state returned by the code for the m = 0 component,
        phiMinusOneList: The ground state returned by the code for the m = -1 component,
        phiMinusTwoList: The ground state returned by the code for the m = -2 component."""
    phiTwo = loadResults(str(folder) + 'phiTwo' + str(name))
    phiOne = loadResults(str(folder) + 'phiOne' + str(name))
    phiZero = loadResults(str(folder) + 'phiZero' + str(name))
    phiMinusOne = loadResults(str(folder) + 'phiMinusOne' + str(name))
    phiMinusTwo = loadResults(str(folder) + 'phiMinusTwo' + str(name))
    
    phiTwoList = phiTwo[-1]
    phiOneList = phiOne[-1]
    phiZeroList = phiZero[-1]
    phiMinusOneList = phiMinusOne[-1]
    phiMinusTwoList = phiMinusTwo[-1]
    
    return phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo,phiTwoList,phiOneList,phiZeroList,phiMinusOneList,phiMinusTwoList

    
        
      
        
def animate(frameList,a,b,phase=False):
    """Given a list of frames, animate these frames. The frames will either be the 
    density at each gradient descent step or the phase at each gradient descent 
    step.
    
    Inputs:
        frameList: List of frames to animate,
        a: The lower bound of our grid (in both the x and y direction),
        b: The upper bound of our grid (in both the x and y direction),
        phase: Specifies whether we are looking at frames that are describing 
                the phase of the wavefunction or not."""
    
    fig1,ax1 = plt.subplots()
    
    if phase:
    
        def animatePhase(i):
            # _min = 0
            # _max = 0.01
            ax1.cla()
            img = ax1.imshow(frameList[i],vmin=0,vmax=2*np.pi,extent=[a,b,a,b],cmap=plt.get_cmap('twilight_shifted'))
            ax1.set_title('Frame = ' + str(i))
            return img
            
        plt.colorbar(animatePhase(0))
        
        anim = animation.FuncAnimation(plt.gcf(),animatePhase,frames=len(frameList),interval=0.1)
        
    else:
        
        minDensity = 0.0
        maxDensity = 0.0
        for i in range(len(frameList[0])):
            currentMax = max(frameList[0][i])
            if currentMax > maxDensity:
                maxDensity = currentMax
        
        minDensity = maxDensity
        for i in range(len(frameList[0])):
            currentMin = min(frameList[0][i])
            if currentMin < minDensity:
                minDensity = currentMin
        
        
        def animateDensity(i):
            # _min = 0
            # _max = 0.01
            ax1.cla()
            img = ax1.imshow(frameList[i],vmin=minDensity,vmax=1e-05,extent=[a,b,a,b],cmap=plt.get_cmap('viridis'))
            ax1.set_title('Frame = ' + str(i))
            return img
            
        plt.colorbar(animateDensity(0))
        anim = animation.FuncAnimation(plt.gcf(),animateDensity,frames=len(frameList),interval=300)
    
    return anim

def plotPhi(phi,x,y,L):
    """Plots a 3d plot of any component of the wavefunction phi that we pass in.
    Also plots a cross section of the same wavefunction component.
    
    Inputs:
        phi: Component of the wavefunction to be plotted,
        x: List of xpts for our grid,
        y: List of ypts for our grid,
        L: The number of lattice spacing in the x and y direction."""
    
    xx,yy = np.meshgrid(x,y)
    
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.plot_surface(xx,yy,phi,cmap=plt.get_cmap('viridis'))
    plt.show()
    
    """Plot a cross-section of phiDown when y = 0"""
    fig1,ax = plt.subplots()
    ax.scatter(x,phi[int((L-2)/2)])
    
def getSingleDensity(phi):
    """Given a wavefunction, return it's density"""
    return abs(phi)**2

def plotDensity(density,a,b):
    """Plot the density of a wavefunction."""
    fig,ax = plt.subplots()
    img = ax.imshow(density,extent=[a,b,a,b],cmap=plt.get_cmap('viridis'))
    plt.colorbar(img)
    


def getDensities(phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo):
    """Given a list of wavefunctions, return of list of corresponding densities"""
    densities = []
    for i in range(len(phiTwo)):
        densities.append(abs(phiTwo[i])**2 + abs(phiOne[i])**2 + abs(phiZero[i])**2 + abs(phiMinusOne[i])**2 + abs(phiMinusTwo[i])**2)
    return densities

def getPointPolarAngle(x1,y1):
    """Return the corresponding polar angle of a point (x1,y1)"""
    polarAngle = np.arctan2(y1,x1)
    if (polarAngle < 0):
        polarAngle += 2*np.pi
    return polarAngle

def getSinglePhase(phi,L):
    """Given a wavefunction, return it's phase"""
    phase = np.zeros((L-1,L-1))
    for i in range(L-1):
        for j in range(L-1):
            phase[i,j] = np.arctan2(phi[i,j].imag,phi[i,j].real)
            if (phase[i,j] < 0):
                phase[i,j] += 2*np.pi

    return phase

def plotPhase(phase,a,b):
    """Plot the phase of a wavefunction"""
    fig,ax = plt.subplots()
    ax.imshow(phase,vmin=0,vmax=2*np.pi,extent=[a,b,a,b],cmap=plt.get_cmap('twilight_shifted'))

def getPhases(phi):
    """Given a list of wavefunctions, return a list of corresponding phases."""
    phaseList = []
    for i in range(len(phi)):
        phaseList.append(getSinglePhase(phi[i]))
    return phaseList

def getSingleSpinSingletDensity(phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo):
    """Return the spin singlet density of a wavefunction."""
    return aZeroSq(phiTwo, phiOne, phiZero, phiMinusOne, phiMinusTwo)

def getSpinSingletDensities(phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo):
    """Given a list of wavefunctions, return a corresponding list of spin singlet densities."""
    singletList = []
    for i in range(len(phiTwo)):
        singletList.append(getSingleSpinSingletDensity(phiTwo[i],phiOne[i],phiZero[i],phiMinusOne[i],phiMinusTwo[i]))
    return singletList

def getSingleSpinDensity(phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo):
    """Return the spin density of a wavefunction."""
    return fFour(phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo)

def getSpinDensities(phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo):
    """Given a list of wavefuncions, return a corresponding list of spin densities."""
    spinDensityList = []
    for i in range(len(phiTwo)):
        spinDensityList.append(getSingleSpinDensity(phiTwo[i],phiOne[i],phiZero[i],phiMinusOne[i],phiMinusTwo[i]))
    return spinDensityList

def calculateHealingLength(a,b,L,beta):
    """Calculate the healing lenght associated with an interaction parameter beta.
    Returns the healing length in units of lattice spacings."""
    return (L/(b-a))*(1/np.sqrt(2*beta))

def getPhiMax(phi):
    """Given a 2D phi list, return its max value"""
    maxVal = 0
    for i in range(len(phi)):
        for j in range(len(phi)):
            if (phi[i,j] > maxVal):
                maxVal = phi[i,j]
    return maxVal

def visualizeVortex(x,y,r,loopPts,phiTwoList,phiOneList,phiZeroList,phiMinusOneList,phiMinusTwoList,periodic=False,colourbar=False,axis=False,grid=False):
    """Given an order parameter that contains a vortex in it. Produce a visual 
    representation of that vortex using spherical harmonics.
    
    Inputs:
        x,y: The x and y lists for our grid.
        r: For periodic boundary conditions, we put the vortex in the x direction.
                So specify the y point at which we wish to view the vortex.
           For non-periodic boundary conditions, r is the radius at which to view
                the vortex.             
        loopPts: The number of pts in the condensate at which to view the vortex.
        phiTwoList,phiOneList,phiZeroList,phiMinusOneList,phiMinusTwoList:
            The components of the order parameter.
        periodic: Specify whether we are looking at a vortex on a torus or not.
    
    This function does not return anything. Instead it produces plots visualizing the 
    vortex within the condensate."""
    
    
    path = [] #Hold spherical harmonic images
    if (periodic):
        psi = loop.getPeriodicCirclePhis(np.array(x),np.array(y),r,loopPts,phiTwoList,phiOneList,phiZeroList,phiMinusOneList,phiMinusTwoList)
    else:
        psi = loop.getCirclePhis(r,loopPts,np.array(x),np.array(y),phiTwoList,phiOneList,phiZeroList,phiMinusOneList,phiMinusTwoList)
    
    """Produce spherical harmonic representation for each point on loop."""
    for i in range(loopPts):
        gbec.sphericalHarmonicRep(psi[i],2,i,colourbar=colourbar,axis=axis,grid=grid)
        path.append('sphericalHarmonic'+str(i)+'.png')
    
    """Display graphical representation of vortex."""
    phiAngleList = np.linspace(0,2*np.pi,loopPts,endpoint=False)
    gv.graphicalVortex(path,phiAngleList)
    
def getPhiBIndex(i,j,L):
    """Given the coordinate (i,j) of a point on the boundary of the phi lattice:
        Return the index of this point in the phiB list. L is the number of lattice spacings"""
    s = i+j
    p = 0
    if (s > L+1):
        p = 1
    if (i < j):
        index = 2*(s-(3+p))
    else:
        index = 2*(s-(3+p)) + 1
    return index

def calcRotOrderParam(alpha,beta,gamma):
    c = np.cos(beta/2)
    s = np.sin(beta/2)
    term1 = (1/2)*np.exp(-2j*(alpha+gamma))*c**4 + 1j*np.sqrt(3)*np.exp(-2j*alpha)*(c**2)*(s**2) + (1/2)*np.exp(-2j*(alpha-gamma))*s**4
    term2 = np.exp(-1j*(alpha+2*gamma))*(c**3)*(s) - 1j*np.sqrt(3/16)*np.exp(-1j*alpha)*np.sin(2*beta) - np.exp(-1j*(alpha-2*gamma))*c*(s**3)
    term3 = (np.sqrt(6)/2)*np.exp(-2j*gamma)*(c**2)*(s**2) + (1j/(4*np.sqrt(2)))*(1 + 3*np.cos(2*beta)) + (np.sqrt(6)/2)*np.exp(2j*gamma)*(c**2)*(s**2)
    term4 = np.exp(1j*(alpha-2*gamma))*c*(s**3) + 1j*np.sqrt(3/16)*np.exp(1j*alpha)*np.sin(2*beta) - np.exp(1j*(alpha+2*gamma))*(c**3)*s
    term5 = (1/2)*np.exp(2j*(alpha-gamma))*(s**4) + 1j*np.sqrt(3)*np.exp(2j*alpha)*(c**2)*(s**2) + (1/2)*np.exp(2j*(alpha+gamma))*c**4
    rotOrderParam = np.zeros(5,dtype='complex_')
    rotOrderParam[0] = term1
    rotOrderParam[1] = term2
    rotOrderParam[2] = term3
    rotOrderParam[3] = term4
    rotOrderParam[4] = term5
    return rotOrderParam

def roundPhi(phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo):
    phiTwo = np.round(phiTwo,5)
    phiOne = np.round(phiOne,5)
    phiZero = np.round(phiZero,5)
    phiMinusOne = np.round(phiMinusOne,5)
    phiMinusTwo = np.round(phiMinusTwo,5)
    return phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo

def convertPhiBoundary(phi):
    """Given a 2d phi array. Pad in the correct way to convert a phiList made using 
    zero boundary conditions into one that is compatible with a code using periodic 
    boundary conditions."""
    phiNew = np.pad(phi,1,'constant',constant_values=0)
    phiNew = np.delete(phiNew,-1,axis=1)
    phiNew = np.delete(phiNew,-1,axis=0)
    return phiNew


def generateAngle(userInput):
    """Given a user's input of a fraction of pi radians. Generate this numerical value."""
    pi = np.pi
    numbers = np.zeros(2)
    numbersString = ['','']
    i = 0
    j = 0
    while i < len(userInput):
        if (userInput[i] != ' '):
            numbersString[j] += userInput[i]
        else:
            j += 1
            numbersString[j] += userInput[i]
        i+=1
    numbers[0] = int(numbersString[0])
    numbers[1] = int(numbersString[1])
    angle = numbers[0]*pi/numbers[1]
    return angle


def getBogMode(eigVecs,modeNumber,x,y,L):
    """Given a list of all eigVecs returned by the Bogoliubov code and the eigVec
    number to look at, return the u and v of each spin component of this eigVec.
    
    Inputs:
        eigVecs: The list of eigVecs returned by the Bogoliubov code,
        modeNumber: The particular eigVec we wish to look at,
        x: List of x values on our grid,
        y: List of y values on our grid,
        L: Number of lattice spacings.
    
    Returns:
        uVecTwo: The m=2 component of u for the given eigVec,
        vVecTwo: The m=2 component of v for the given eigVec,
        uVecOne: The m=1 component of u for the given eigVec,
        vVecOne: The m=1 component of v for the given eigVec,
        uVecZero: The m=0 component of u for the given eigVec,
        vVecZero: The m=0 component of v for the given eigVec,
        uVecMinusOne: The m=-1 component of u for the given eigVec,
        vVecMinusOne: The m=-1 component of v for the given eigVec,
        uVecMinusTwo: The m=-2 component of u for the given eigVec,
        vVecMinusTwo: The m=-2 component of v for the given eigVec.
    """
    
    xx,yy = np.meshgrid(x,y)
    yy *= -1
    
    """Sort each eigenvector into uUp, vUp, uDown and vDown"""
    uVecTwo = np.zeros((L)**2,dtype='complex_')
    vVecTwo = np.zeros((L)**2,dtype='complex_')
    uVecOne = np.zeros((L)**2,dtype='complex_')
    vVecOne = np.zeros((L)**2,dtype='complex_')
    uVecZero = np.zeros((L)**2,dtype='complex_')
    vVecZero = np.zeros((L)**2,dtype='complex_')
    uVecMinusOne = np.zeros((L)**2,dtype='complex_')
    vVecMinusOne = np.zeros((L)**2,dtype='complex_')
    uVecMinusTwo = np.zeros((L)**2,dtype='complex_')
    vVecMinusTwo = np.zeros((L)**2,dtype='complex_')
    
    
    
    for i in range(10*(L)**2):
        if i < (L)**2:
            uVecTwo[i] = eigVecs[modeNumber,i]
        elif i < 2*(L)**2:
            vVecTwo[int(i%((L)**2))] = eigVecs[modeNumber,i]
        elif i < 3*(L)**2:
            uVecOne[int(i%(2*(L)**2))] = eigVecs[modeNumber,i]
        elif i < 4*(L)**2:
            vVecOne[int(i%(3*(L)**2))] = eigVecs[modeNumber,i]
        elif i < 5*(L)**2:
            uVecZero[int(i%(4*(L)**2))] = eigVecs[modeNumber,i]
        elif i < 6*(L)**2:
            vVecZero[int(i%(5*(L)**2))] = eigVecs[modeNumber,i]
        elif i < 7*(L)**2:
            uVecMinusOne[int(i%(6*(L)**2))] = eigVecs[modeNumber,i]
        elif i < 8*(L)**2:
            vVecMinusOne[int(i%(7*(L)**2))] = eigVecs[modeNumber,i]
        elif i < 9*(L)**2:
            uVecMinusTwo[int(i%(8*(L)**2))] = eigVecs[modeNumber,i]
        else:
            vVecMinusTwo[int(i%(9*(L)**2))] = eigVecs[modeNumber,i]
            
    """Now we have the kth eigenvector (ie the u's and v's in flat 1d format). 
    I need to reformat them as 2D arrays for plotting."""
    
    uVecTwo = np.reshape(uVecTwo,((L),(L)),order='F')
    vVecTwo = np.reshape(vVecTwo,((L),(L)),order='F')
    uVecOne = np.reshape(uVecOne,((L),(L)),order='F')
    vVecOne = np.reshape(vVecOne,((L),(L)),order='F')
    uVecZero = np.reshape(uVecZero,((L),(L)),order='F')
    vVecZero = np.reshape(vVecZero,((L),(L)),order='F')
    uVecMinusOne = np.reshape(uVecMinusOne,((L),(L)),order='F')
    vVecMinusOne = np.reshape(vVecMinusOne,((L),(L)),order='F')
    uVecMinusTwo = np.reshape(uVecMinusTwo,((L),(L)),order='F')
    vVecMinusTwo = np.reshape(vVecMinusTwo,((L),(L)),order='F')
    
    return uVecTwo,vVecTwo,uVecOne,vVecOne,uVecZero,vVecZero,uVecMinusOne,vVecMinusOne,uVecMinusTwo,vVecMinusTwo


def plotModes(eigVecs,n,x,y,L):
    
    xx,yy = np.meshgrid(x,y)
    
    #n = 28
    
    """Sort each eigenvector into u_{2},v_{2},u_{1},v_{1} ..."""
    uVecTwo = np.zeros((L)**2,dtype='complex_')
    vVecTwo = np.zeros((L)**2,dtype='complex_')
    uVecOne = np.zeros((L)**2,dtype='complex_')
    vVecOne = np.zeros((L)**2,dtype='complex_')
    uVecZero = np.zeros((L)**2,dtype='complex_')
    vVecZero = np.zeros((L)**2,dtype='complex_')
    uVecMinusOne = np.zeros((L)**2,dtype='complex_')
    vVecMinusOne = np.zeros((L)**2,dtype='complex_')
    uVecMinusTwo = np.zeros((L)**2,dtype='complex_')
    vVecMinusTwo = np.zeros((L)**2,dtype='complex_')
    
    for i in range(10*(L)**2):
        if i < (L)**2:
            uVecTwo[i] = eigVecs[n,i]
        elif i < 2*(L)**2:
            vVecTwo[int(i%((L)**2))] = eigVecs[n,i]
        elif i < 3*(L)**2:
            uVecOne[int(i%(2*(L)**2))] = eigVecs[n,i]
        elif i < 4*(L)**2:
            vVecOne[int(i%(3*(L)**2))] = eigVecs[n,i]
        elif i < 5*(L)**2:
            uVecZero[int(i%(4*(L)**2))] = eigVecs[n,i]
        elif i < 6*(L)**2:
            vVecZero[int(i%(5*(L)**2))] = eigVecs[n,i]
        elif i < 7*(L)**2:
            uVecMinusOne[int(i%(6*(L)**2))] = eigVecs[n,i]
        elif i < 8*(L)**2:
            vVecMinusOne[int(i%(7*(L)**2))] = eigVecs[n,i]
        elif i < 9*(L)**2:
            uVecMinusTwo[int(i%(8*(L)**2))] = eigVecs[n,i]
        else:
            vVecMinusTwo[int(i%(9*(L)**2))] = eigVecs[n,i]
    
    """Now we have the kth eigenvector (ie: the u's and v's in flat 1d format), 
    I need to reformat them as 2D arrays for plotting."""
    
    uVecTwo = np.reshape(uVecTwo,((L),(L)),order='F')
    vVecTwo = np.reshape(vVecTwo,((L),(L)),order='F')
    uVecOne = np.reshape(uVecOne,((L),(L)),order='F')
    vVecOne = np.reshape(vVecOne,((L),(L)),order='F')
    uVecZero = np.reshape(uVecZero,((L),(L)),order='F')
    vVecZero = np.reshape(vVecZero,((L),(L)),order='F')
    uVecMinusOne = np.reshape(uVecMinusOne,((L),(L)),order='F')
    vVecMinusOne = np.reshape(vVecMinusOne,((L),(L)),order='F')
    uVecMinusTwo = np.reshape(uVecMinusTwo,((L),(L)),order='F')
    vVecMinusTwo = np.reshape(vVecMinusTwo,((L),(L)),order='F')
    
    
    
    
    
    """Plot the eigenvectors."""
    fig1, ax1 = plt.subplots(subplot_kw=dict(projection='3d'))
    ax1.plot_surface(xx,yy,abs(uVecTwo)**2,cmap=plt.get_cmap('jet'))
    ax1.set_zlabel(r'|$u_{2}|^{2}$')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.show()
    
    fig2, ax2 = plt.subplots(subplot_kw=dict(projection='3d'))
    ax2.plot_surface(xx,yy,abs(vVecTwo)**2,cmap=plt.get_cmap('jet'))
    ax2.set_zlabel(r'|$v_{2}|^{2}$')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.show()
    
    fig3, ax3 = plt.subplots(subplot_kw=dict(projection='3d'))
    ax3.plot_surface(xx,yy,abs(uVecOne)**2,cmap=plt.get_cmap('jet'))
    ax3.set_zlabel(r'|$u_{1}|^{2}$')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    plt.show()
    
    fig4, ax4 = plt.subplots(subplot_kw=dict(projection='3d'))
    ax4.plot_surface(xx,yy,abs(vVecOne)**2,cmap=plt.get_cmap('jet'))
    ax4.set_zlabel(r'|$v_{1}|^{2}$')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    plt.show()
    
    fig5, ax5 = plt.subplots(subplot_kw=dict(projection='3d'))
    ax5.plot_surface(xx,yy,abs(uVecZero)**2,cmap=plt.get_cmap('jet'))
    ax5.set_zlabel(r'|$u_{0}|^{2}$')
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    plt.show()
    
    fig6, ax6 = plt.subplots(subplot_kw=dict(projection='3d'))
    ax6.plot_surface(xx,yy,abs(vVecZero)**2,cmap=plt.get_cmap('jet'))
    ax6.set_zlabel(r'|$v_{0}|^{2}$')
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    plt.show()
    
    fig7, ax7 = plt.subplots(subplot_kw=dict(projection='3d'))
    ax7.plot_surface(xx,yy,abs(uVecMinusOne)**2,cmap=plt.get_cmap('jet'))
    ax7.set_zlabel(r'|$u_{-1}|^{2}$')
    ax7.set_xlabel('x')
    ax7.set_ylabel('y')
    plt.show()
    
    fig8, ax8 = plt.subplots(subplot_kw=dict(projection='3d'))
    ax8.plot_surface(xx,yy,abs(vVecMinusOne)**2,cmap=plt.get_cmap('jet'))
    ax8.set_zlabel(r'|$v_{-1}|^{2}$')
    ax8.set_xlabel('x')
    ax8.set_ylabel('y')
    plt.show()
    
    fig9, ax9 = plt.subplots(subplot_kw=dict(projection='3d'))
    ax9.plot_surface(xx,yy,abs(uVecMinusTwo)**2,cmap=plt.get_cmap('jet'))
    ax9.set_zlabel(r'|$u_{-2}|^{2}$')
    ax9.set_xlabel('x')
    ax9.set_ylabel('y')
    plt.show()
    
    fig10, ax10 = plt.subplots(subplot_kw=dict(projection='3d'))
    ax10.plot_surface(xx,yy,abs(vVecMinusTwo)**2,cmap=plt.get_cmap('jet'))
    ax10.set_zlabel(r'|$v_{-2}|^{2}$')
    ax10.set_xlabel('x')
    ax10.set_ylabel('y')
    plt.show()
    
    plt.figure()
    plt.plot(x,uVecZero[int(len(uVecTwo)/2 -1)])
    plt.plot(x,-1j*uVecZero[int(len(uVecTwo)/2 - 1)],'r')
        
        

        
    

    
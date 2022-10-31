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
from mpl_toolkits.axes_grid.inset_locator import(inset_axes,InsetPosition)


def getPhiNorm(phiOne,phiZero,phiMinusOne,h):
    """Takes in components of ground state phi and lattice spacing h and returns 
    the norm.
    
    Inputs:
        phiOne: m = 1 component of phi,
        phiZero: m = 0 component of phi,
        phiMinusOne: m = -1 component of phi,
        h: Lattice spacing (same for x and y direction).
    
    Returns:
        norm: The norm of phi."""
    
    """Make sure the phi lists are flattened, this way I can now compute the norm 
    even if say a 2D array is passed in"""
    
    def flattenArray(phi):
        if (len(np.shape(phi)) > 1):
            phi = np.reshape(phi,len(phi)**2,order='F')
        return phi
    
    phiOne = flattenArray(phiOne)
    phiZero = flattenArray(phiZero)
    phiMinusOne = flattenArray(phiMinusOne)
    
    
    norm = 0
    for i in range(len(phiOne)):
        norm += (abs(phiOne[i])**2 + abs(phiZero[i])**2 + abs(phiMinusOne[i])**2)
    norm *= h**2
    norm = np.sqrt(norm)
    return norm

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

def phiFour(phiOne,phiZero,phiMinusOne):
    """Return density term in energy functional.
    
    Inputs:
        phiOne: m = 1 component of phi,
        phiZero: m = 0  component of phi,
        phiMinusOne: m = -1 component of phi.
        
    Returns:
        phiFour: The density term in the energy functional."""
    pOne = abs(phiOne)**2
    pZero = abs(phiZero)**2
    pMinusOne = abs(phiMinusOne)**2
    
    phiFour = (pOne**2 + pZero**2 + pMinusOne**2 + 2*pOne*pMinusOne + 2*pOne*pZero
                   + 2*pMinusOne*pZero)
    
    return phiFour

def fFour(phiOne,phiZero,phiMinusOne):
    """Return the spin density term in the energy functional.
    
    Inputs:
        phiOne: m = 1 component of phi,
        phiZero: m = 0 component of phi,
        phiMinusOne: m = -1 component of phi.
        
    Returns:
        fFour: The spin density term in the energy functional."""
    pOne = abs(phiOne)**2
    pZero = abs(phiZero)**2
    pMinusOne = abs(phiMinusOne)**2
    
    fFour = (pOne**2 + pMinusOne**2 + 2*pOne*pZero + 2*pZero*pMinusOne 
                 - 2*pOne*pMinusOne + 2*(phiZero**2)*(np.conjugate(phiOne))*(np.conjugate(phiMinusOne))
                 + 2*(np.conjugate(phiZero**2))*(phiOne)*(phiMinusOne))
    
    return fFour


def computeEnergy(phiOne,phiZero,phiMinusOne,L,h,V,b0,b1,p,q):
    """Takes in components of ground state phi along with certain parameters of 
    the relevant system and returns the energy of the system in the given ground 
    state.
    
    Inputs:
        phiOne: m = 1 component of phi,
        phiZero: m = 0 component of phi,
        phiMinusOne: m = -1 component of phi,
        L: The number of lattice spacings,
        h: Lattice spacing (same for x and y direction),
        V: The 2D array storing the potential the sytem is in,
        b0: The dimensionless density interaction parameter,
        b1: The dimensionless spin density interaction parameter,
        p: The linear Zeeman term,
        q: The quadratic Zeeman term.
        
    Returns:
        energy: The value of the energy functional of the current state of the system."""
    
    integrand = getEnergyDensity(phiOne,phiZero,phiMinusOne,L,h,V,b0,b1,p,q)
    
    """Take the integral to compute the energy."""
    xInt = np.zeros(L-1,dtype='complex_')
    for i in range(L-1):
        xInt[i] = np.trapz(integrand[i],dx=h)
    energy = np.trapz(xInt,dx=h)
    
    return energy

def getEnergyDensity(phiOne,phiZero,phiMinusOne,L,h,V,b0,b1,p,q):
    """Takes in components of ground state phi along with certain parameters of 
    the relevant system and returns the energy density of the system in the given ground 
    state.
    
    Inputs:
        phiOne: m = 1 component of phi,
        phiZero: m = 0 component of phi,
        phiMinusOne: m = -1 component of phi,
        L: The number of lattice spacings,
        h: Lattice spacing (same for x and y direction),
        V: The 2D array storing the potential the sytem is in,
        b0: The dimensionless density interaction parameter,
        b1: The dimensionless spin density interaction parameter,
        p: The linear Zeeman term,
        q: The quadratic Zeeman term.
        
    Returns:
        energy: The value of the energy density of the current state of the system."""
        
    
    
    phi4 = phiFour(phiOne,phiZero,phiMinusOne)
    f4 = fFour(phiOne,phiZero,phiMinusOne)
        
    
    
    
    """Compute the integrand of the energy functional."""
    energyDensity = ((1/2)*(deriv(phiOne,h)) + (1/2)*(deriv(phiZero,h)) + (1/2)*(deriv(phiMinusOne,h))
                 + V*abs(phiOne)**2 + V*abs(phiZero)**2 + V*abs(phiMinusOne)**2 
                 + (-p+q)*abs(phiOne)**2 + (p+q)*abs(phiMinusOne) 
                 + (b0/2)*(phi4) + (b1/2)*(f4))
    
    return energyDensity.real

def getEnergyDensities(phiOne,phiZero,phiMinusOne,L,h,V,b0,b1,p,q):
    """Takes in list of components of ground state phi along with certain parameters of 
    the relevant system and returns a list of energy densities of the system for the given ground 
    states.
    
    Inputs:
        phiOne: List of  m = 1 components of phi,
        phiZero: List of m = 0 components of phi,
        phiMinusOne: List of m = -1 components of phi,
        L: The number of lattice spacings,
        h: Lattice spacing (same for x and y direction),
        V: The 2D array storing the potential the sytem is in,
        b0: The dimensionless density interaction parameter,
        b1: The dimensionless spin density interaction parameter,
        p: The linear Zeeman term,
        q: The quadratic Zeeman term.
        
    Returns:
        energy: The value of the energy density of the current state of the system."""
    
    energyDensityList = np.zeros(len(phiOne),dtype='object')
    for i in range(len(energyDensityList)):
        energyDensityList[i] = getEnergyDensity(phiOne[i], phiZero[i], phiMinusOne[i], L, h, V, b0, b1, p, q)
    
    return energyDensityList

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
        name: The name of the pickle file to load in as an array.
        tag: Optional string to append onto the end of the filename.
        directory: directory the files to load are in. Must include / at end of folder.
    
    Returns:
        result: The loaded pickle file into the relevant object, eg: an array."""
    f = open(str(directory)+str(name)+str(tag),'rb')
    result = pickle.load(f)
    f.close()
    return result
    
def loadFullResults(tag='',directory=''):
    """Shortcut function to load saved components of a generated ground state phi.
    
    Inputs:
        tag: Optional string to append onto the end of the filename.
        directory: directory the files to load are in. Must include / at end of folder.
        
    Returns:
        phiOne: An array with the value of the m = 1 component of the ground state 
                at each gradient descent step,
        phiZero: An array with the value of the m = 0 component of the ground state 
                    at each gradient descent step,
        phiMinusOne: An array with the value of the m = 1 component of the ground state 
                        at each gradient descent step,
                        
        phiOneList: The ground state returned by the code for the m = 1 component,
        phiZeroList: The ground state returned by the code for the m = 0 component,
        phiMinusOneList: The ground state returned by the code for the m = -1 component"""
    phiOne = loadResults(str(directory)+'phiOne' + str(tag))
    phiZero = loadResults(str(directory)+'phiZero' + str(tag))
    phiMinusOne = loadResults(str(directory)+'phiMinusOne' + str(tag))
    
    phiOneList = phiOne[-1]
    phiZeroList = phiZero[-1]
    phiMinusOneList = phiMinusOne[-1]
    
    return phiOne,phiZero,phiMinusOne,phiOneList,phiZeroList,phiMinusOneList       
      
        
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
            
        
        cbar = fig1.colorbar(animatePhase(0), ticks=[0, 3.14, 6.28])
        cbar.ax.set_yticklabels(['0', r'$\pi$', r'$2\pi$'])
        
        
        anim = animation.FuncAnimation(plt.gcf(),animatePhase,frames=len(frameList),interval=0.1)
        
    else:
        
        minDensity = 0.0
        maxDensity = 0.0
        for i in range(len(frameList[0])):
            currentMax = max(frameList[0][i])
            if currentMax > maxDensity:
                maxDensity = currentMax
        
        
        def animateDensity(i):
            # _min = 0
            # _max = 0.01
            ax1.cla()
            img = ax1.imshow(frameList[i],vmin=minDensity,vmax=0.1,extent=[a,b,a,b],cmap=plt.get_cmap('viridis'))
            ax1.set_title('Frame = ' + str(i))
            return img
            
        plt.colorbar(animateDensity(0))
        anim = animation.FuncAnimation(plt.gcf(),animateDensity,frames=len(frameList),interval=0.1)
    
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
    yy *= -1
    
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.plot_surface(xx,yy,phi,cmap=plt.get_cmap('jet'))
    plt.show()
    
    """Plot a cross-section of phi when y = 0"""
    fig1,ax = plt.subplots()
    ax.plot(x,phi[int((L-2)/2)])

def plotDensity(density,a,b):
    """Plot the density of a wavefunction."""
    fig,ax = plt.subplots()
    img = ax.imshow(density,vmin=0,extent=[a,b,a,b],cmap=plt.get_cmap('viridis'))
    plt.colorbar(img)    
    
def plotDensityInset(density,a,b,x,y,L):
    fig1,ax1 = plt.subplots()
    img = ax1.imshow(density,vmin=0,extent=[a,b,a,b],cmap=plt.get_cmap('viridis'))
    plt.colorbar(img)
    ax2 = plt.axes([0,0,1,1])
    ip = InsetPosition(ax1, [0.6,0.15,0.35,0.35])
    ax2.set_axes_locator(ip)
    ax2.xaxis.label.set_color('white')
    ax2.tick_params(axis='x',colors='white')
    ax2.yaxis.label.set_color('white')
    ax2.tick_params(axis='y',colors='white')
    ax2.set_xlabel('x')
    ax2.set_ylabel(r'$|\phi|^{2}$')
    ax2.patch.set_alpha(0.0)
    ax2.plot(x,density[int((L-2)/2)],'white')
    
    
    
def plotPhase(phase,a,b):
    """Plot the phase of a wavefunction"""
    fig,ax = plt.subplots()
    img = ax.imshow(phase,vmin=0,vmax=2*np.pi,extent=[a,b,a,b],cmap=plt.get_cmap('twilight_shifted'))
    cbar = fig.colorbar(img, ticks=[0, 3.14, 6.28])
    cbar.ax.set_yticklabels(['0', r'$\pi$', r'$2\pi$'])

def getSingleDensity(phi):
    """Given a wavefunction, return it's density"""
    return abs(phi)**2

def getDensities(phi):
    """Given a list of wavefunctions, return of list of corresponding densities"""
    densities = []
    for i in range(len(phi)):
        densities.append(abs(phi[i])**2)
    return densities

def getSinglePhase(phi,L):
    """Given a wavefunction, return it's phase"""
    phase = np.zeros((L-1,L-1))
    for i in range(L-1):
        for j in range(L-1):
            phase[i,j] = np.arctan2(phi[i,j].imag,phi[i,j].real)
            if (phase[i,j] < 0):
                phase[i,j] += 2*np.pi

    return phase

def getPhases(phi,L):
    """Given a list of wavefunctions, return a list of corresponding phases."""
    phaseList = []
    for i in range(len(phi)):
        phaseList.append(getSinglePhase(phi[i],L))
    return phaseList

def calculateHealingLength(a,b,L,beta):
    """Calculate the healing lenght associated with an interaction parameter beta.
    Returns the healing length in units of lattice spacings."""
    return (L/(b-a))*(1/np.sqrt(2*beta))

def visualizeVortex(x,y,r,loopPts,phiOneList,phiZeroList,phiMinusOneList,periodic=False,colourbar=False,axis=False,grid=False):
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
        psi = loop.getPeriodicCirclePhis(np.array(x),np.array(y),r,loopPts,phiOneList,phiZeroList,phiMinusOneList)
    else:
        psi = loop.getCirclePhis(r,loopPts,np.array(x),np.array(y),phiOneList,phiZeroList,phiMinusOneList)
    
    """Produce spherical harmonic representation for each point on loop."""
    for i in range(loopPts):
        gbec.sphericalHarmonicRep(psi[i],1,i,colourbar=colourbar,axis=axis,grid=grid)
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
        uVecOne: The m=1 component of u for the given eigVec,
        vVecOne: The m=1 component of v for the given eigVec,
        uVecZero: The m=0 component of u for the given eigVec,
        vVecZero: The m=0 component of v for the given eigVec,
        uVecMinusOne: The m=-1 component of u for the given eigVec,
        vVecMinusOne: The m=-1 component of v for the given eigVec.
    """
    
    xx,yy = np.meshgrid(x,y)
    yy *= -1
    
    
    """Sort each eigenvector into uUp, vUp, uDown and vDown"""
    uVecOne = np.zeros((L-1)**2,dtype='complex_')
    vVecOne = np.zeros((L-1)**2,dtype='complex_')
    uVecZero = np.zeros((L-1)**2,dtype='complex_')
    vVecZero = np.zeros((L-1)**2,dtype='complex_')
    uVecMinusOne = np.zeros((L-1)**2,dtype='complex_')
    vVecMinusOne = np.zeros((L-1)**2,dtype='complex_')
    
    
    
    for i in range(6*(L-1)**2):
        if i < (L-1)**2:
            uVecOne[i] = eigVecs[modeNumber,i]
        elif i < 2*(L-1)**2:
            vVecOne[int(i%((L-1)**2))] = eigVecs[modeNumber,i]
        elif i < 3*(L-1)**2:
            uVecZero[int(i%(2*(L-1)**2))] = eigVecs[modeNumber,i]
        elif i < 4*(L-1)**2:
            vVecZero[int(i%(3*(L-1)**2))] = eigVecs[modeNumber,i]
        elif i < 5*(L-1)**2:
            uVecMinusOne[int(i%(4*(L-1)**2))] = eigVecs[modeNumber,i]
        else:
            vVecMinusOne[int(i%(5*(L-1)**2))] = eigVecs[modeNumber,i]
    
    """Now we have the kth eigenvector (ie the u's and v's in flat 1d format). 
    I need to reformat them as 2D arrays for plotting."""
    
    uVecOne = np.reshape(uVecOne,((L-1),(L-1)),order='F')
    vVecOne = np.reshape(vVecOne,((L-1),(L-1)),order='F')
    uVecZero = np.reshape(uVecZero,((L-1),(L-1)),order='F')
    vVecZero = np.reshape(vVecZero,((L-1),(L-1)),order='F')
    uVecMinusOne = np.reshape(uVecMinusOne,((L-1),(L-1)),order='F')
    vVecMinusOne = np.reshape(vVecMinusOne,((L-1),(L-1)),order='F')
    
    return uVecOne,vVecOne,uVecZero,vVecZero,uVecMinusOne,vVecMinusOne




        
        

        
    

    
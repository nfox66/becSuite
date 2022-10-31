# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 17:19:30 2022

@author: Nathan

graphicalBec.py

Plots the spherical harmonic representation of the condensate given the order parameter 
of the condensate at some point.
"""

"""Imports"""
from scipy import special
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import groundStateTools as gst

"""Extra bit just for a test"""
import loop
import graphicalVortex as gv
from configparser import ConfigParser


def sphericalHarmonicRep(psi,f,j=0,colourbar=False,axis=False,grid=False):
    """Given the order parameter of the condensate at a certain point in space, 
    produce a plot of its spherical harmonic representation.
    
    Inputs:
        psi: The order parameter:
                For spin one: psi = (psiOne,psiZero,psiMinusOne)
                For spin two: psi = (psiTwo,psiOne,psiZero,psiMinusOne,psiMinusTwo)
        
        f: Tells us whether we are looking at a spin f=1 or spin f=2 condensate,
        j: Just a counter to append to the end of the saved spherical harmonic plot.
        colourbar: Specify whether we want a colourbar included in the plot.
        ax: Specify whether we want an axis included in the plot.
        grid: Specify whether we want a grid included in the plot.
        
    This file does not return anything and instead saves the spherical harmonic plot
    as 'sphericalHarmonic[j].png"""
    
    
    n = 100 #Number of pts
    
    """Setting up spherical coordinates theta and phi"""
    theta = np.linspace(0,2*np.pi,n)
    phi = np.linspace(0,np.pi,n)
    
    
    THETA,PHI = np.meshgrid(theta,phi)
    
    sh = 0
    
    """Equation for producing the spherical harmonic representation for the 
    corresponding order parameter of the condensate."""
    for i in range(len(psi)):
        sh += psi[i]*special.sph_harm(f-i,f,THETA,PHI)
        
    
    
    
    
    
    """Find the phase of the spherical harmonic representation to plot on top 
    of it."""
    phase = gst.getSinglePhase(sh, len(sh)+1)
    
    
    
    
    
    viridis = cm.get_cmap('twilight_shifted',12)
    
    """Renormalize phase for the colourplot"""
    phase/= 2*np.pi
    
    
    
    """Create phaseColours list"""
    phaseColours = viridis(phase)
    
     
    """Last step for producing spherical harmonic rep is to take abs value squared."""
    absSh = abs(sh)**2
    
    """Convert from spherical to cartesian in order to plot"""
    x = absSh*np.sin(PHI)*np.cos(THETA)
    y = absSh*np.sin(PHI)*np.sin(THETA)
    z = absSh*np.cos(PHI)
    
    
    
    
    
    xFlat = np.reshape(x,len(x)**2)
    yFlat = np.reshape(y,len(y)**2)
    zFlat = np.reshape(z,len(z)**2)
    lim = max(xFlat)
    if (max(xFlat)<max(yFlat)):
        lim = max(yFlat)
        if (max(yFlat)<max(zFlat)):
            lim = max(zFlat)
    elif(max(xFlat)<max(zFlat)):
        lim = max(zFlat)
        
    
    
    
    
    
        
    
    
    
    lim += lim
    
    """Produce the spherical harmonic plot"""
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim3d(-lim,lim)
    ax.set_ylim3d(-lim,lim)
    ax.set_zlim3d(-lim,lim)
    ax.plot_surface(x,y,z,facecolors=phaseColours)
    cmap = mpl.cm.twilight_shifted
    norm = mpl.colors.Normalize(vmin=0,vmax=1)
    sm = mpl.cm.ScalarMappable(norm=norm,cmap=cmap)
    if (colourbar):
        cbar = plt.colorbar(sm,fraction=0.02)
        cbar.ax.set_ylabel(r'Phase/2$\pi$',rotation=90)
    if (grid == False):
        plt.grid(False)
    if (axis == False):
        plt.axis('off')
    plt.savefig('sphericalHarmonic' + str(j), transparent=True)
    
    return phase
    
    
    
    
    
    











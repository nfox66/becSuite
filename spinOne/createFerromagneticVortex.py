# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 12:29:27 2022

@author: 16383526
"""

import groundStateTools as gst
from configparser import ConfigParser
import numpy as np


def createFerromagneticVortex(winding,sigma,inputFile='spinOneInput.ini'):
    """Read in inputs from the ini input file"""
    cp = ConfigParser()
    cp.read(inputFile)
    
    
    
    
    
    """ie the number of grid pts in the x and y dimension so that we have a grid
    of (L-1)x(L-1) lattice pts as we will knock off the boundary points of the grid"""
    L = int(cp.get('latticePts','L'))
    a = float(cp.get('latticePts','a'))
    b = float(cp.get('latticePts','b'))
    
    
    """Set up an x and y list which has L number of pts"""
    x = np.linspace(a,b,L+1)
    y = np.linspace(a,b,L+1)
    
    """Get rid of the boundary pts of x and y"""
    x = np.delete(x,0)
    x = np.delete(x,-1)
    y = np.delete(y,0)
    y = np.delete(y,-1)
    
    """Set up meshgrids to have arrays listing all coordinates in our 2d grid"""
    xx, yy = np.meshgrid(x,y)
    yy*=-1
    
    r = np.sqrt(xx**2 + yy**2)
    phi = np.zeros((len(xx),len(xx)))
    
    for i in range(len(xx)):
        for j in range(len(xx)):
            phi[i,j] = np.arctan2(yy[i,j],xx[i,j])
            if (phi[i,j] < 0):
                phi[i,j] += 2*np.pi
    
    
    if winding%2 == 0:
        n_a = winding/2
        phib = (n_a)*phi
    else:
        n_a = (winding-1)/2
        phib = (n_a+1)*phi
    
    alpha = -n_a*phi
    beta  = np.pi/64
    
    """For now, try set up the wavefunction in the ferromagnetic phase"""
    phiOneList = (r**5)*(np.exp(-abs((xx**2 + yy**2)/sigma)))*np.exp(1j*phib)*np.exp(-1j*alpha)*np.cos(beta/2)**2
    phiZeroList = (r**5)*(np.exp(-abs((xx**2 + yy**2)/sigma)))*(1/np.sqrt(2))*np.sin(beta)
    phiMinusOneList = (r**5)*(np.exp(-abs((xx**2 + yy**2)/sigma)))*np.exp(1j*phib)*np.exp(1j*alpha)*np.sin(beta/2)**2
    
    return phiOneList,phiZeroList,phiMinusOneList
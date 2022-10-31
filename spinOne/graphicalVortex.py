# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 14:15:48 2022

@author: 16383526
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def graphicalVortex(path,phi,j=0):
    """Given a list of spherical harmonics, this function plots them in an ellipse 
    corresponding to the polar coordinate phi list. r just gives the radius of the loop 
    that we took in the condensate.
    
    Inputs:
        path: List containing the spherical harmonic plots corresponding to points 
                in the condensate,
        phi: A list of phi values to plot these images at in order to plot them 
                in a loop.
        j: Number to append onto the saved image file.
            
    This function does not return anything, instead it displays and saves a plot of the spherical 
    harmonics along the loop in the condensate."""

    def getImage(path, zoom=0.5):
        return OffsetImage(plt.imread(path),zoom=zoom)
    
    r = 2 #Just give some radius to transform circle to ellipse, this value does not really matter.
    
    x = (1/2)*(r + 1/r)*np.cos(phi)
    y = (1/2)*(r - 1/r)*np.sin(phi)
    
    fig,ax = plt.subplots()
    ax.scatter(x,y)
    
    plt.axis('off')
    
    
    
    for x0,y0,p in zip(x,y,path):
        ab = AnnotationBbox(getImage(p),(x0,y0),frameon = False)
        ax.add_artist(ab)
    
    plt.savefig('vortexImage' + str(j) + '.png')
        
    
    

    
    




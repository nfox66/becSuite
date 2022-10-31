# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 10:55:42 2022

@author: Nathan

createVortex.py

This code sets up a vortex as an initial condition to pass into modular ground state.
"""

"""Imports"""
import numpy as np
from configparser import ConfigParser

import groundStateTools as gst
import axisAngleToEuler as euler


def createInitialVortex(aox,aoy,aoz,angle,gaugePhi,repOrderParam,smoothingDensity=0,inputFile='spinTwoInput.ini',rotated=False):
    """Given an axis of rotation and angle of rotation specifying a vortex.
    Produce the initial condition of that vortex in the condensate.
    
    Inputs:
        aox,aoy,aoz = The x,y and z axes of rotation.
        angle: The anlge of rotation.
        gaugePhi: The gauge angle for the given vortex.
        repOrderParam: Representative order parameter for a particular phase of the 
                        condensate. [1/2,0,i/np.sqrt(2),0,1/2] for cyclic phase,
                                    [1/np.sqrt(2),0,0,0,1/np.sqrt(2)] for biaxial nematic phase.
        inputFile: ini file containing constants and relevant parameters to read in.
        rotated: Specifying whether to rotate the axis of rotation onto the z-axis or not.
    
    Returns:
        phiTwoList,phiOneList,phiZeroList,phiMinusOneList,phiMinusTwoList:
            The components of the condensate with the vortex imprinted in them."""
    
    """Read in inputs from the ini input file"""
    cp = ConfigParser()
    cp.read(inputFile)
    
    
    """Number of lattice pts in each direction of the 2D grid along with the 
    boundary pts of the grid."""
    L = int(cp.get('latticePts','L'))
    a = float(cp.get('latticePts','a'))
    b = float(cp.get('latticePts','b'))
    
    """Lattice spacing"""
    h = (b-a)/(L)    
        
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
    invYY = -np.array(yy)
    
    """Set up polar angle phi to imprint vortex in."""
    r = np.sqrt(xx**2 + yy**2)
    phi = np.zeros((len(xx),len(xx)))
    
    for i in range(len(xx)):
        for j in range(len(xx)):
            phi[i,j] = np.arctan2(invYY[i,j],xx[i,j])
            if (phi[i,j] < 0):
                phi[i,j] += 2*np.pi
    
    if (rotated):
        """If True, we want to rotate the axis of rotation of the vortex onto the 
        z-axis."""
        ax = 0
        ay = 0
        az = 1
        
        """Find the Euler angles for rotation of angle about z-axis."""
        alpha,beta,gamma = euler.axisAngleToEuler(ax,ay,az,angle)
        
        """Vary alpha,beta and gamma smoothly from 0 to their final values along a closed 
        loop in the condensate. Here we can linearly vary the actual Euler angles as we are 
        only really varying one angle gamma in the instance where we are rotating about the 
        z-axis."""
        alpha = alpha*phi/(2*np.pi)
        beta = beta*phi/(2*np.pi)
        gamma = gamma*phi/(2*np.pi)
        
        """For the original axis of rotation, we need to find the the 5x5 rotation matrix
        that brings this onto the z-axis. The best way to do this is to find the angle and 
        axis of rotation needed to do this. We can find these using the cross product and 
        dot product between this axis of rotation and the z-axis."""
        axis = np.array([aox,aoy,aoz])
        zAxis = np.array([ax,ay,az])
        
        transformAxis = np.cross(axis,zAxis)
        transformAngle = np.arccos(np.dot(axis,zAxis)/(np.linalg.norm(axis)*np.linalg.norm(zAxis)))
        
        alphaTransform,betaTransform,gammaTransform = euler.axisAngleToEuler(transformAxis[0],transformAxis[1],transformAxis[2],transformAngle)
        transformMatrix = gst.getRotMat(alphaTransform,betaTransform,gammaTransform)
        
        """Get order parameter for the vortex rotated up to the z-axis."""
        phiTwoList,phiOneList,phiZeroList,phiMinusOneList,phiMinusTwoList = gst.getPhaseOrderParameter(alpha,beta,gamma,np.matmul(transformMatrix,repOrderParam))
        
    else:
        """Otherwise we are not rotating the vortex onto the z-axis first and so we set up 
        the vortex differently"""
        
        """Find the amount that angle has rotated by after going a certain amount of the 
        way around the torus. We have to do it this way as all three Euler angles may be 
        varying in this case and as a result varying them linearly may not go through the 
        correct path for the vortex."""
        angle = angle*phi/(2*np.pi)
        
        alpha = np.zeros((len(xx),len(xx)))
        beta = np.zeros((len(xx),len(xx)))
        gamma = np.zeros((len(xx),len(xx)))
        
        """For each of these angles, find the corresponding Euler angles"""
        for i in range(len(angle)):
            for j in range(len(angle)):
                alphaTerm,betaTerm,gammaTerm = euler.axisAngleToEuler(aox,aoy,aoz,angle[i,j])
                alpha[i,j] = alphaTerm
                beta[i,j] = betaTerm
                gamma[i,j] = gammaTerm
        
        """Get order parameter for this vortex."""
        phiTwoList,phiOneList,phiZeroList,phiMinusOneList,phiMinusTwoList = gst.getPhaseOrderParameter(alpha,beta,gamma,repOrderParam)
        
    """Factor in the gauge angle that describes this vortex."""
    gaugePhi = gaugePhi*phi/(2*np.pi)
    
    if (len(smoothingDensity) == 0):
        sigma = 1.75
        prefactor = np.exp(1j*gaugePhi)*(r**2)*(np.exp(-abs((r**2)/sigma**2)))
    else:
        prefactor = np.exp(1j*gaugePhi)*smoothingDensity
    
    
    phiTwoList *= prefactor
    phiOneList *= prefactor
    phiZeroList *= prefactor
    phiMinusOneList *= prefactor
    phiMinusTwoList *= prefactor
    
    """Get the norm of this order parameter and use this to renormalize."""
    norm = gst.getPhiNorm(phiTwoList,phiOneList,phiZeroList,phiMinusOneList,phiMinusTwoList,h)
    
    phiTwoList /= norm
    phiOneList /= norm
    phiZeroList /= norm
    phiMinusOneList /= norm
    phiMinusTwoList /= norm

    return phiTwoList,phiOneList,phiZeroList,phiMinusOneList,phiMinusTwoList


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

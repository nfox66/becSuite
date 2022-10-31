# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 12:57:19 2022

@author: Nathan

loop.py

This file has functions to help set up loops in the condensate for both periodic 
and dirichlet boundary conditions and returns a list of values of the condensate
at a finite number of points on the circle.
"""

import numpy as np


"""These first few functions are just helper functions for the main functions that 
set up the loop within the condensate and return values of the condensate at a finite 
number of points on that circle."""

"""Find the index (or indices) of the closest grid points to each xr and yr 
    point."""
    
def findValueXIndex(dirList,val):
    """Find the index (or indices) of the closest points in dirList to val."""
    startIndex = 0
    endIndex = len(dirList)-1
    indexList = []
    while(abs(endIndex-startIndex)>1):
        midIndex = int((endIndex+startIndex)/2)
        if (val == dirList[midIndex]):
            indexList.append(midIndex)
            return indexList
        if (val > dirList[midIndex]):
            startIndex = midIndex
        else:
            endIndex = midIndex
    if (dirList[startIndex] == val):
        indexList.append(startIndex)
        return indexList
    elif(dirList[endIndex] == val):
        indexList.append(endIndex)
        return indexList
    indexList.append(startIndex)
    indexList.append(endIndex)
    return indexList

def findValueYIndex(dirList,val):
    """Find the index (or indices) of the closest points in dirList to val."""
    startIndex = 0
    endIndex = len(dirList)-1
    indexList = []
    while(abs(endIndex-startIndex)>1):
        midIndex = int((endIndex+startIndex)/2)
        if (val == dirList[midIndex]):
            indexList.append(midIndex)
            return indexList
        if (val > dirList[midIndex]):
            endIndex = midIndex
        else:
            startIndex = midIndex
    if (dirList[startIndex] == val):
        indexList.append(startIndex)
        return indexList
    elif(dirList[endIndex] == val):
        indexList.append(endIndex)
        return indexList
    indexList.append(startIndex)
    indexList.append(endIndex)
    return indexList

def setPhi(phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo,ix,iy):
    """A shorthand to set the components of phi and store them in an array.
    
    Inputs:
        phiTwo: The m = 2 component of the condensate wavefunction,
        phiOne: The m = 1 component of the condensate wavefunction,
        phiZero: The m = 0 component of the condensate wavefunction,
        phiMinusOne: The m = -1 component of the condensate wavefunction,
        phiMinusTwo: The m = -2 component of the condensate wavefunction.
        ix: The x index to take the phi values from,
        iy: The y index to take the phi values from.
        
    Returns:
        phiList: An array storing the values of each component of phi 
                    at the given position."""
    
    """Note: ix and iy are swapped because the j index (2nd index usually) 
    describes the x direction and the i index describes the y direction."""
    
    
    phiList = np.zeros(5,dtype='complex_')
    phiList[0] = phiTwo[iy,ix]
    phiList[1] = phiOne[iy,ix]
    phiList[2] = phiZero[iy,ix]
    phiList[3] = phiMinusOne[iy,ix]
    phiList[4] = phiMinusTwo[iy,ix]
    
    return phiList


"""The following functions are the main functions that are called in other files 
in order to set up loops within the condensate."""



def getPeriodicCirclePhis(x,y,yPoint,n,phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo):
    """For a condensate with periodic boundary conditions, return values of phi 
    at finite points along a closed loop in the condensate.
    
    Inputs:
        x: A list of x points along the x-side of our lattice,
        y: A list of y points along the y-side of our lattice,
        yPoint: The y point to take our closed loop at,
                    Note: We will take our closed loop in the x direction 
                    and stay at a constant y value which is called yPoint here.
        n: The number of points to look at along our closed loop,
        phiTwo: The m = 2 component of the condensate wavefunction,
            phiOne: The m = 1 component of the condensate wavefunction,
            phiZero: The m = 0 component of the condensate wavefunction,
            phiMinusOne: The m = -1 component of the condensate wavefunction,
            phiMinusTwo: The m = -2 component of the condensate wavefunction.
    
    Returns:
        phiList: A list of condensate order parameters describing the value of the 
        condensate at each point on our closed loop.
    """
    
    
    """Setup indices for closed loop points."""
    xr = np.linspace(0,len(x)-1,n,endpoint=False)
    xr = xr.astype(np.int32)
       
    """Change sign of y list to bring in line with x-y coordinate plane."""
    y *= -1
    
    
    """Find the closest index in the y list to our yPoint."""
    def findOneValueYIndex(dirList,val):
        """Find the index of one of the closest points in dirList to val."""
        indexList = findValueYIndex(dirList, val)
        return indexList[0]
    
    yIndex = findOneValueYIndex(y,yPoint)
    
    """Find the value of the condensate at each point on the closed loop."""
    phiList = np.zeros(n,dtype='object')
    for i in range(n):
        phiList[i] = setPhi(phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo,xr[i],yIndex)
    
    return phiList

        
        
    
def getCirclePhis(r,loopPts,xpts,ypts,phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo):
    """Given a radius r of a loop to take in the condensate, compute n equally 
    spaced points on that circle. Then approximate the value of the condensate, 
    phi, at these points as they may not be points in our grid. Then return a list 
    of these condensate values.
    
    Inputs:
        r: Radius of circle to loop around in condensate,
        loopPts: Number of points to take on circle,
        x: List of xpts on the x side of our lattice,
        y: List of ypts on the y side of our lattice.
        
    Returns:
        phiList: A list of condensate order parameters describing the value of 
        the condensate at each point on the loop."""
    
    
    """Produce lists storing the x and y coordinates of the circle of radius r."""
    theta = np.linspace(0,2*np.pi,loopPts+1)
    theta = np.delete(theta,-1)
    xr = r*np.cos(theta)
    yr = r*np.sin(theta)
    
    """Reverse the sign of y to be in line with the x-y coordinate plane."""
    ypts *= -1
    
    
    def approximatePhi(xr,yr,xIndex,yIndex,phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo):
        """Given lists detailing the points closest to the point on the circle 
        we are looking at, find an approximate value of phi.
        
        Inputs:
            xIndex: List of x indices of lattice coordinates close to current circle point,
            yIndex: List of y indices of lattice coordinates close to current circle point,
            phiTwo: The m = 2 component of the condensate wavefunction,
            phiOne: The m = 1 component of the condensate wavefunction,
            phiZero: The m = 0 component of the condensate wavefunction,
            phiMinusOne: The m = -1 component of the condensate wavefunction,
            phiMinusTwo: The m = -2 component of the condensate wavefunction.
            
        Returns:
            An approximate value of phi, the value of the condensate at our circle point."""
            
        
        
        def dist(x1,y1,x2,y2):
            """Standard distance function taking points (x1,y1) and (x2,y2).
            Returns the distance between these points."""
            return (np.sqrt((x2-x1)**2 + (y2-y1)**2))
        
        phiList = np.zeros(5)    
        
        """Depending on how many surrounding grid points are close to our point 
        on the loop, approximate phi with weighted averages of these grid points."""
        
        if (len(xIndex) == 1 and len(yIndex) == 1):
            """Point on loop coincides with grid point, return value of phi at that grid point."""
            phiList = setPhi(phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo,xIndex[0],yIndex[0])
        elif(len(xIndex) == 1):
            """Loop point between two grid points, return weighted average of condensate values."""
            phiA = setPhi(phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo,xIndex[0],yIndex[0])
            phiB = setPhi(phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo,xIndex[0],yIndex[1])
            
            phiList = ((abs(yr-ypts[yIndex[0]])/abs(ypts[yIndex[0]]-ypts[yIndex[1]]))*phiB 
                          + (abs(yr-ypts[yIndex[1]])/abs(ypts[yIndex[0]]-ypts[yIndex[1]]))*phiA)
        elif(len(yIndex) == 1):
            phiA = setPhi(phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo,xIndex[0],yIndex[0])
            phiB = setPhi(phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo,xIndex[1],yIndex[0])
            
            phiList = ((abs(xr-xpts[xIndex[0]])/abs(xpts[xIndex[0]]-xpts[xIndex[1]]))*phiB
                          + (abs(xr-xpts[xIndex[1]])/abs(xpts[xIndex[0]]-xpts[xIndex[1]]))*phiA)
        
        else:
            """Loop point between four grid points. Return weighted average."""
            phiA = setPhi(phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo,xIndex[0],yIndex[0])
            phiB = setPhi(phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo,xIndex[0],yIndex[1])
            phiC = setPhi(phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo,xIndex[1],yIndex[0])
            phiD = setPhi(phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo,xIndex[1],yIndex[1])
            
            distA = dist(xr,yr,xIndex[0],yIndex[0])
            distB = dist(xr,yr,xIndex[0],yIndex[1])
            distC = dist(xr,yr,xIndex[1],yIndex[0])
            distD = dist(xr,yr,xIndex[1],yIndex[1])
            
            distTot = distA + distB + distC + distD
            
            phiList = (1-distA/distTot)*phiA + (1-distB/distTot)*phiB + (1-distC/distTot)*phiC + (1-distD/distTot)*phiD
            totWeight = (1-distA/distTot) + (1-distB/distTot) + (1-distC/distTot) + (1-distD/distTot)
            phiList/=totWeight
        
        return phiList
    
    
    """Approximate the condensate wavefunction for each point on the loop and 
    return this list."""
    phiList = np.zeros(loopPts,dtype='object')
    for i in range(loopPts):
        xIndex = findValueXIndex(xpts,xr[i])
        yIndex = findValueYIndex(ypts,yr[i])
        
        print('xIndex = ' + str(xIndex))
        print('yIndex = ' + str(yIndex))
        
        phiList[i] = approximatePhi(xr[i], yr[i], xIndex, yIndex, phiTwo, phiOne, phiZero, phiMinusOne, phiMinusTwo)
    
    return phiList





































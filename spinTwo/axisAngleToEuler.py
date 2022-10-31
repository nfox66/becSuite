# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 11:07:32 2022

@author: Nathan


This file takes an axis-angle representation for a rotation and converts it to 
Euler angles in ZYZ format.
"""

"""Imports"""
import numpy as np


def rotMatAxes(ax,ay,az,angle):
    """Given an axis of rotation and an angle of rotation, construct the 3x3 
    rotation matrix and return the vectors that the x, y and z axes get transformed 
    to by this rotation, which are the columns of the 3x3 rotation matrix.
    
    Inputs:
        ax: The x component of the axis of rotation,
        ay: The y component of the axis of rotation,
        az: The z component of the axis of rotation,
        angle: The angle of rotation.
        
    Returns:
        xNew: The vector where the x-axis ends up after the rotation,
        yNew: The vector where the y-axis ends up after the rotation,
        zNew: The vector where the z-axis ends up after the rotation."""
    
    """Make sure ax, ay and az are normalized first"""
    norm = np.sqrt(ax**2 + ay**2 + az**2)
    ax /= norm
    ay /= norm
    az /= norm
    
    mat = np.zeros((3,3))
    
    cos = np.cos(angle)
    sin = np.sin(angle)
    
    """Construct rotation matrix"""
    mat[0][0] = ax**2 + (ay**2 + az**2)*cos
    mat[0][1] = ax*ay*(1-cos) - az*sin
    mat[0][2] = ax*az*(1-cos) + ay*sin
    
    mat[1][0] = ax*ay*(1-cos) + az*sin
    mat[1][1] = ay**2 + (ax**2 + az**2)*cos
    mat[1][2] = ay*az*(1-cos) - ax*sin
    
    mat[2][0] = ax*az*(1-cos) - ay*sin
    mat[2][1] = ay*az*(1-cos) + ax*sin
    mat[2][2] = az**2 + (ax**2 + ay**2)*cos
    
    
    """Find where the x,y and z axes end up after the rotation."""
    xNew = np.array([mat[0][0],mat[1][0],mat[2][0]])
    yNew = np.array([mat[0][1],mat[1][1],mat[2][1]])
    zNew = np.array([mat[0][2],mat[1][2],mat[2][2]])
    
    return xNew,yNew,zNew



def matToEuler(xVec,yVec,zVec):
    """Given the new vectors that the x, y and z axis end up at after a rotation, 
    find the euler angles: alpha, beta and gamma (in zyz format) needed to do 
    this rotation.
    
    Inputs:
        xNew: The vector where the x-axis ends up after the rotation,
        yNew: The vector where the y-axis ends up after the rotation,
        zNew: The vector where the z-axis ends up after the rotation.
    
    Returns:
        alpha,beta,gamma: The Euler angles for this rotation."""
    
    
    def checkrot(oldAxis,angle,newAxis,rotAxis):
        """Check if the Euler angle, called angle, correctly rotates the oldAxis 
        about the rotAxis (either 'z' or 'y') onto the newAxis.
        If so: return angle
        If not: return 2*pi - angle
        
        To do this, take the cross product between the two vectors that we are 
        finding the euler angle between, then since we know what the rotAxis is, 
        we can see if the cross product gave that axis or not, if it did then we 
        have the correct orientation for the Euler angle called angle, and if not 
        then we have to reverse the orientation using angle = 2pi - angle."""
        
        crossVec = np.cross(oldAxis,newAxis)
        dotVal = np.dot(crossVec,rotAxis)
        
        
        if (dotVal > 0):
            if (angle < 0):
                angle +=  2*np.pi
            return angle
        else:
            if (angle < np.pi and angle != 0):
                angle = 2*np.pi - angle
            return (angle)
    
    pi = np.pi
    
    """Find alpha: The angle for the first rotation about the z-axis. Put another 
    way, the angle to bring the original x axis in line with the projection of the 
    new z axis into the original xy-plane"""
    
    alpha = np.arctan2(zVec[1],zVec[0])
    if (alpha < 0):
        alpha += 2*pi
    
    alpha = checkrot(np.array([1,0,0]),alpha,np.array([zVec[0],zVec[1],0]),np.array([0,0,1]))
        
    
    """Now rotate by beta about the 'new' y-axis (can actually just do it about the 
    old, it still works out the same) to bring the old z-axis onto the new z-axis."""
    zOld = np.array([0,0,1])
    beta = np.arccos(np.dot(zVec,zOld)/(np.linalg.norm(zVec)*np.linalg.norm(zOld)))
    
    """Need yPrime, where the y axis goes to after the alpha rotation."""
    yPrime = np.array([-np.sin(alpha),np.cos(alpha),0])
    beta = checkrot(np.array([0,0,1]),beta,zVec,yPrime)
    
    
    """Finally, find gamma, the angle needed to rotate about the new z axis to bring 
    the old x and y axes onto the new ones."""
    gamma = np.arccos(np.dot(yVec,yPrime)/(np.linalg.norm(yVec)*np.linalg.norm(yPrime)))
    gamma = checkrot(yPrime,gamma,yVec,zVec)
    
    
    return alpha,beta,gamma

def axisAngleToEuler(ax,ay,az,angle):
    """Given an axis of rotation and an angle of rotation, find the corresponding 
    Euler angles for the same rotation in ZYZ format."""
    
    xNew, yNew, zNew = rotMatAxes(ax, ay, az, angle)
    alpha,beta,gamma = matToEuler(xNew,yNew,zNew)
    
    return alpha,beta,gamma
    


    


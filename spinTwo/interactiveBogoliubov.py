# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 11:41:15 2022

@author: Nathan

interactiveBogoliubov.py
"""

import bogoliubov as bog
import numpy as np
import groundStateTools as gst

from configparser import ConfigParser

import sys
import os
import glob

loadChoice = input('Do you want to load or run a Bogoliubov configuration? (load or run)\n')

if (loadChoice != 'load' and loadChoice != 'run'):
    print('Please choose a valid option.')
    sys.exit()

currentDirectory = input('Please enter the directory you wish to work in (please include / at the end of the path name).\n')
os.chdir(currentDirectory)
inputFile = glob.glob('*.ini')[0]
cp = ConfigParser()
cp.read(inputFile)
os.chdir('../')

"""Load in all relevant parameters from input file."""


b0 = float(cp.get('interaction','b0'))
b1 = float(cp.get('interaction','b1'))
b2 = float(cp.get('interaction','b2'))

p = float(cp.get('zeeman','p'))
q = float(cp.get('zeeman','q'))

"""Number of lattice spacings in each direction of the 2D grid along with the 
boundary pts of the grid."""
L = int(cp.get('latticePts','L'))
a = float(cp.get('latticePts','a'))
b = float(cp.get('latticePts','b'))

"""h changes slightly as we have one less lattice spacing and point on the 
torus since we are identifying both ends."""
h = (b-a)/(L)

tau = float(cp.get('time','tau'))

"""Set up an x and y list for our grid. Slightly different from fixed boundary 
conditions again to account for one ends of the lattice being identifying."""
x = np.linspace(a,b,L+1)
y = np.linspace(a,b,L+1)

x = np.delete(x,-1)
y = np.delete(y,-1)
    
xx,yy = np.meshgrid(x,y)

yy *= -1


V = (1/2)*(xx**2) + (1/2)*(yy**2)

for i in range(len(V)):
        for j in range(len(V)):
            if (np.sqrt((xx[i,j])**2 + (yy[i,j])**2) < 1):
                V[i,j] += 500
                

if (loadChoice == 'load'):
    eigVals = gst.loadResults('eigVals',directory=currentDirectory)
    eigVecs = gst.loadResults('eigVecs',directory=currentDirectory)
    sys.exit()
    
elif(loadChoice == 'run'):
    loadNumber = input('Please provide the number of wavefunction you wish to run the Bogoliubov code for.\n')
    numVecs = int(input('Please choose how many eigenvectors you wish to generate.\n'))
    phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo,phiTwoList,phiOneList,phiZeroList,phiMinusOneList,phiMinusTwoList = gst.loadFullResults(loadNumber,currentDirectory)
    phiTwoList = gst.convertPhiBoundary(phiTwoList)
    phiOneList = gst.convertPhiBoundary(phiOneList)
    phiZeroList = gst.convertPhiBoundary(phiZeroList)
    phiMinusOneList = gst.convertPhiBoundary(phiMinusOneList)
    phiMinusTwoList = gst.convertPhiBoundary(phiMinusTwoList)
    bog.bogoliubov(phiTwoList, phiOneList, phiZeroList, phiMinusOneList, phiMinusTwoList, L, a, b, x, y, V, b0, b1, b2, p, q, directory=currentDirectory, numVecs=numVecs)
    












































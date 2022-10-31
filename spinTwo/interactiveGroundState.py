# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 16:36:45 2022

@author: Nathan

interactiveGroundState.py
"""

import numpy as np
import createVortex as cv
import modularGroundState as mgs
from configparser import ConfigParser
import groundStateTools as gst

import sys
import os
import shutil
import time
import glob
import re



f = open('userInputs.txt','w')

loadChoice = input('Do you want to load or run a configuration? (load or run)\n')

if (loadChoice == 'load'):
    loadDirectory = input('Please enter the directory you wish to load from. (Please include / at the end of path)\n')
    loadNumber = input('Please enter the number of the configuration you wish to load\n')
    phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo,phiTwoList,phiOneList,phiZeroList,phiMinusOneList,phiMinusTwoList = gst.loadFullResults(loadNumber,loadDirectory)
    os.chdir(loadDirectory)
    inputFile = glob.glob('*.ini')[0]
    cp = ConfigParser()
    cp.read(inputFile)
    os.chdir('../')
    
    
if (loadChoice == 'run'):
    repOrderParamChoice = input('What phase would you like to look at? (cyclic or binem)\n')
    checkOrderParam = re.search("cyclic|binem",repOrderParamChoice)
    if not checkOrderParam:
        print('Please choose a valid option.')
        sys.exit()
    f.write('What phase would you like to look at? (cyclic or binem)\n')
    f.write(repOrderParamChoice+'\n')
    f.write('\n')
    if (repOrderParamChoice == 'cyclic'):
        repOrderParam = np.array([1/2,0,1j/np.sqrt(2),0,1/2])
        inputFile = 'spinTwoInputCyclic.ini'
    elif(repOrderParamChoice == 'binem'):
        repOrderParam = np.array([1/np.sqrt(2),0,0,0,1/np.sqrt(2)])
        inputFile = 'spinTwoInputBinem.ini'
    cp = ConfigParser()
    cp.read(inputFile)
        
if (loadChoice != 'load' and loadChoice != 'run'):
    print('Please choose a valid option.')
    sys.exit()
    
    


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

x = np.delete(x,0)
x = np.delete(x,-1)
y = np.delete(y,0)
y = np.delete(y,-1)
    
xx,yy = np.meshgrid(x,y)

yy *= -1


V = (1/2)*(xx**2) + (1/2)*(yy**2)

for i in range(len(V)):
        for j in range(len(V)):
            if (np.sqrt((xx[i,j])**2 + (yy[i,j])**2) < 1):
                V[i,j] += 500

    
    
if (loadChoice == 'load'):
    sys.exit()
    


ax = int(input('Please enter the x-axis of the vortex\n'))
f.write('Please enter the x-axis of the vortex\n')
f.write(str(ax)+'\n')
f.write('\n')
ay = int(input('Please enter the y-axis of the vortex\n'))
f.write('Please enter the y-axis of the vortex\n')
f.write(str(ay)+'\n')
f.write('\n')
az = int(input('Please enter the z-axis of the vortex\n'))
f.write('Please enter the z-axis of the vortex\n')
f.write(str(az)+'\n')
f.write('\n')
rotAngle = input('Please enter the angle of rotation of the vortex(as a fraction of pi radians, eg for 4*pi/3 enter 4 followed by a space and then 3\n')
f.write('Please enter the angle of rotation of the vortex (as a fraction of pi radians, eg for 4*pi/3 enter 4 followed by a space and then 3).\n')
f.write(rotAngle+'\n')
f.write('\n')
checkRotAngle = re.search("\d+ \d+",rotAngle)
if not checkRotAngle:
    print('Please input in the correct format.')
    sys.exit()
rotAngle = gst.generateAngle(rotAngle)
gaugeAngle = input('Please enter the value for the gaauge angle of the vortex. In the same format as that entered for the angle of rotation.\n')
f.write('Please enter the value for the gauge angle of the vortex (in the same format as that entered for the angle of rotation).\n')
f.write(gaugeAngle+'\n')
f.write('\n')
checkGaugeAngle = re.search("\d+ \d+",gaugeAngle)
if not checkGaugeAngle:
    print('Please input in the correct format.')
    sys.exit()
gaugeAngle = gst.generateAngle(gaugeAngle)

if (ax != 0 or ay != 0):
    rotated = True

"""Load in the wavefunction that will be used to produce the smoothing density."""
generateSmoothing = input('Do you want to generate a smoothing density or load one in? (generate or load)\n')
f.write('Do you want to generate a smoothing function or load one in? (generate or load)\n')
f.write(generateSmoothing+'\n')
f.write('\n')

if (generateSmoothing != 'load' and generateSmoothing != 'generate'):
    print('Please choose a valid option.')
    sys.exit()
    
elif(generateSmoothing == 'load'):
    loadSmoothingDirectory = input('Please enter the directory where the smoothing density you wish to load is. (include / at the end of the path)\n')
    f.write('Please enter the directory where the smoothing density you wish to load is. (include / at the end of the path)\n')
    f.write(loadSmoothingDirectory+'\n')
    f.write('\n')
    loadSmoothing = input('Please enter the smoothing density wavefunction you wish to load in (Note you must be in the directory where the smoothing density wavefunction is and this file must have the format for example phiTwoNum). Please provide Num.\n')
    f.write('Please enter the smoothing density wavefunction you wish to load in (Note you must be in the directory where the smoothing density wavefunction is and this file must have the format for example phiTwoNum). Please provide Num.\n')
    f.write(loadSmoothing+'\n')
    f.write('\n')
    phiTwoSmooth,phiOneSmooth,phiZeroSmooth,phiMinusOneSmooth,phiMinusTwoSmooth,phiTwoSmoothList,phiOneSmoothList,phiZeroSmoothList,phiMinusOneSmoothList,phiMinusTwoSmoothList = gst.loadFullResults(loadSmoothing,folder=loadSmoothingDirectory)
    smoothingDensity = abs(phiTwoSmoothList)**2 + abs(phiOneSmoothList)**2 + abs(phiZeroSmoothList)**2 + abs(phiMinusOneSmoothList)**2 + abs(phiMinusTwoSmoothList)**2

    
saveNumber = input('What number would you like to give to this run?\n')
f.write('What number would you like to give to this run?\n')
f.write(saveNumber+'\n')
f.write('\n')
os.makedirs('run'+str(saveNumber))
time.sleep(1)
saveDirectory = 'run'+str(saveNumber)+'/'
shutil.copy(inputFile,saveDirectory)
if (generateSmoothing != 'generate'):
    f.close()
    shutil.move('userInputs.txt',saveDirectory)
    
if (generateSmoothing == 'generate'):
    pi = np.pi
    
    numGen = int(input('Please enter how many generations you wish to run before computing the smoothing density.\n'))
    f.write('Please enter how many generations you wish to run before computing the smoothing density.\n')
    f.write(str(numGen)+'\n')
    f.write('\n')
    
    saveGens = input('Would you like to save all generations? (y or n)\n')
    f.write('Would you like to save all generations? (y or n)\n')
    f.write(saveGens + '\n')
    f.write('\n')
    
    f.close()
    shutil.move('userInputs.txt',saveDirectory)
    
    b0List = np.linspace(0,b0,numGen)
    b1List = np.linspace(0,b1,numGen)
    b2List = np.linspace(0,b2,numGen)
    mainIterCount = 90
    
    
    """First get the ground state of the 500 pinning potential with no interaction."""
    phiTwoList = repOrderParam[0]*np.ones((L-1,L-1),dtype='complex')
    phiOneList = repOrderParam[1]*np.ones((L-1,L-1),dtype='complex')
    phiZeroList = repOrderParam[2]*np.ones((L-1,L-1),dtype='complex')
    phiMinusOneList = repOrderParam[3]*np.ones((L-1,L-1),dtype='complex')
    phiMinusTwoList = repOrderParam[4]*np.ones((L-1,L-1),dtype='complex')
    
    r = np.sqrt(xx**2 + yy**2)
    sigma = 1.75
    prefactor = (np.exp(-abs((r**2)/sigma**2)))
        
    phiTwoList *= prefactor
    phiOneList *= prefactor
    phiZeroList *= prefactor
    phiMinusOneList *= prefactor
    phiMinusTwoList *= prefactor
    
    norm = gst.getPhiNorm(phiTwoList,phiOneList,phiZeroList,phiMinusOneList,phiMinusTwoList,h)
    
    phiTwoList /= norm
    phiOneList /= norm
    phiZeroList /= norm
    phiMinusOneList /= norm
    phiMinusTwoList /= norm
    
    for i in range(len(b0List)):
        mgs.groundState(phiTwoList, phiOneList, phiZeroList, phiMinusOneList, phiMinusTwoList, V, b0List[i], b1List[i], b2List[i], inputFile=inputFile, directory=saveDirectory, mainIter=mainIterCount)
        
        """Now get ground state with 500 pinning potential with interaction turned on using 
        the above as the initial condition."""
        phiTwo,phiOne,phiZero,phiMinusOne,phiMinusTwo,phiTwoList,phiOneList,phiZeroList,phiMinusOneList,phiMinusTwoList = gst.loadFullResults(mainIterCount,saveDirectory)
        
        if (saveGens == 'y'):
            mainIterCount += 1
    
    
    """Now use the density of this state as the smoothing function for the initial condition 
    of the vortex states."""
    smoothingDensity = abs(phiTwoList)**2 + abs(phiOneList)**2 + abs(phiZeroList)**2 + abs(phiMinusOneList)**2 + abs(phiMinusTwoList)**2



    

phiTwoList,phiOneList,phiZeroList,phiMinusOneList,phiMinusTwoList = cv.createInitialVortex(ax,ay,az,rotAngle,gaugeAngle,repOrderParam,smoothingDensity=smoothingDensity,inputFile=inputFile,rotated=True)
mgs.groundState(phiTwoList,phiOneList,phiZeroList,phiMinusOneList,phiMinusTwoList,V,b0,b1,b2,inputFile=inputFile,mainIter=saveNumber,directory=saveDirectory)


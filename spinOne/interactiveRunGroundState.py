# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 11:24:21 2022

@author: 16383526
"""

import modularGroundState as mgs
import groundStateTools as gst
from configparser import ConfigParser
import numpy as np
import createFerromagneticVortex as cfv
import sys
import os
import shutil
import time

f = open('userInputs.txt','w')

inputType = input('Please choose an input file (options: ferromagnetic, polar, custom) Note that we are only providing vortices in the ferromagnetic phase currently\n')
f.write('Please choose an input file (options: ferromagnetic, polar, override, custom) Note that we are only providing vortices in the ferromagnetic phase currently\n')
f.write(inputType+'\n')
f.write('\n')
if(inputType == 'ferromagnetic'):
    inputFile = 'spinOneInputFM.ini'
elif(inputType == 'polar'):
    inputFile = 'spinOneInputPolar.ini'
elif(inputType == 'override'):
    inputFile = 'spinOneInput.ini'
elif(inputType == 'custom'):
    directory = input('Please specify the folder you wish to load from (including / at the end)\n')
    f.write('Please specify the folder you wish to load from (including / at the end)\n')
    f.write(directory+'\n')
    f.write('\n')
    inputFileName = input('Please type the exact name of the input file (including extension)\n')
    inputFile = str(directory)+str(inputFileName)
    f.write('Please type the exact name of the input file (including extension)\n')
    f.write(inputFileName + '\n')
    f.write('\n')
else:
    print('Please choose a valid option')
    sys.exit()
    
cp = ConfigParser()
cp.read(inputFile)    
    
b0 = float(cp.get('interaction','b0'))
b1 = float(cp.get('interaction','b1'))
    
L = int(cp.get('latticePts','L'))
a = float(cp.get('latticePts','a'))
b = float(cp.get('latticePts','b'))

h = (b-a)/L

p = float(cp.get('zeeman','p'))
q = float(cp.get('zeeman','q'))

x = np.linspace(a,b,L+1)
y = np.linspace(a,b,L+1)

x = np.delete(x,0)
x = np.delete(x,-1)
y = np.delete(y,0)
y = np.delete(y,-1)

xx,yy = np.meshgrid(x,y)
yy *= -1

V = (1/2)*(xx)**2 + (1/2)*(yy)**2


initType = input('Please choose an initial condition (options: generic, vortex, contd)\n')
f.write('Please choose an initial condition (options: generic, vortex, contd)\n')
f.write(initType+'\n')
f.write('\n')
if (initType == 'generic'):
    if (inputType == 'ferromagnetic'):
        phiOneList = np.ones((L-1,L-1))
        phiZeroList = (0.01)*np.ones((L-1,L-1))
        phiMinusOneList = (0.01)*np.ones((L-1,L-1))
    elif(inputType == 'polar'):
        phiOneList = (0.01)*np.ones((L-1,L-1))
        phiZeroList = np.ones((L-1,L-1))
        phiMinusOneList = (0.01)*np.ones((L-1,L-1))
elif(initType == 'vortex'):
    print('Note that vortices are only set up for the ferromagnetic phase.')
    winding = int(input('Please choose a winding for the vortex\n'))
    f.write('Please choose a winding for the vortex\n')
    f.write(str(winding)+'\n')
    f.write('\n')
    sigma = float(input ('Please choose a value of sigma for the width of the Gaussian\n'))
    f.write('Please choose a value of sigma for the width of the Gaussian\n')
    f.write(str(sigma)+'\n')
    f.write('\n')
    phiOneList,phiZeroList,phiMinusOneList = cfv.createFerromagneticVortex(winding, sigma, inputFile)
    for i in range(len(V)):
        for j in range(len(V)):
            if (np.sqrt((xx[i,j])**2 + (yy[i,j])**2) < 1):
                V[i,j] += 500
elif(initType == 'contd'):
    mainIterLoad = input('Please provide the number of run that you want to load in\n')
    f.write('Please provide the number of run that you want to load in\n')
    f.write(mainIterLoad+'\n')
    f.write('\n')
    vortex = input('Does this run have a pinning (if this run was run as a vortex and is not being passed into ground state again, answer yes) y or n\n')
    f.write('Does this run have a pinning (if this run was run as a vortex and is not being passed into ground state again, answer yes) y or n\n')
    f.write(vortex+'\n')
    f.write('\n')
    if (vortex == 'y'):
        for i in range(len(V)):
            for j in range(len(V)):
                if (np.sqrt((xx[i,j])**2 + (yy[i,j])**2) < 1):
                    V[i,j] += 500
    phiOne,phiZero,phiMinusOne,phiOneList,phiZeroList,phiMinusOneList = gst.loadFullResults(mainIterLoad,directory=directory)
else:
    print('Please choose a valid option')
    sys.exit()
    
    
run = input('Do you want to run this order parameter through the ground state code? (y or n)\n')
f.write('Do you want to run this order parameter through the ground state code? (y or n)\n')
f.write(run+'\n')
f.write('\n')
if (run == 'y'):
    mainIter = input('Please select what run you are performing to append onto the end of your saved files\n')
    f.write('Please select what run you are performing to append onto the end of your saved files\n')
    f.write(mainIter+'\n')
    f.write('\n')
    f.close()
    if (initType != 'contd'):
        os.makedirs('run'+str(mainIter))
        time.sleep(1)
        shutil.copy(inputFile,'run'+str(mainIter))
        shutil.move('userInputs.txt','run'+str(mainIter)+'/userInputs'+str(mainIter)+'.txt')
        mgs.groundState(phiOneList, phiZeroList, phiMinusOneList, V, inputFile,directory='run'+str(mainIter)+'/',mainIter=mainIter)
    else:
        shutil.move('userInputs.txt','run'+str(mainIterLoad)+'/userInputs'+str(mainIter)+'.txt')
        mgs.groundState(phiOneList, phiZeroList, phiMinusOneList, V, inputFile,directory='run'+str(mainIterLoad)+'/',mainIter=mainIter)
        
    
else:
    f.close()
    time.sleep(1)
    os.remove('userInputs.txt')



    

"""If not running, all relevant parameters will be now loaded into the interactive console where
the user can call functions from groundStateTools in order to view things graphically or manipulate 
the data in other ways."""
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
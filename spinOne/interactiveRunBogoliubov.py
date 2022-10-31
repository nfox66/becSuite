# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 10:29:18 2022

@author: 16383526
"""

import groundStateTools as gst
from configparser import ConfigParser
import numpy as np
import sys
import bogoliubov as bog
import os
import shutil
import time


f = open('userInputsBog.txt','w')

task = input('Do you wish to load eigenvectors or run Bogoliubov? (load or run)\n')

f.write('Do you wish to load eigenvectors or run Bogoliubov? (load or run)\n')
f.write(task+'\n')
f.write('\n')

if (task == 'load'):
    loadDirectory = input('Please choose a directory to load from (please append / at end)\n')
    mainIterLoad = input('Please provide the number of run that you want to load in\n')
    eigVecs = gst.loadResults('eigVecs',tag = mainIterLoad,directory=loadDirectory)
    eigVals = gst.loadResults('eigVals',tag = mainIterLoad,directory=loadDirectory)

elif(task != 'run'):
    print('Please choose a valid input\n')
    f.close()
    time.sleep(1)
    os.remove('userInputsBog.txt')
    sys.exit()
    
inputType = input('Please choose an input file (options: ferromagnetic, polar, custom)\n')
f.write('Please choose an input file (options: ferromagnetic, polar, custom)\n')
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

potential = input('Do you want a potential? (y or n)\n')
f.write('Do you want a potential? (y or n)\n')
f.write(potential+'\n')
f.write('\n')
if (potential == 'y'):
    V = (1/2)*(xx)**2 + (1/2)*(yy)**2
else:
    V = np.zeros((L-1,L-1))

pinningPotential = input('Do you want a pinning potential? (y or n)\n')
f.write('Do you want a pinning potential? (y or n)\n')
f.write(pinningPotential+'\n')
f.write('\n')
if (pinningPotential == 'y'):
    pinningValue = float(input('What value of pinning would you like? (float value)\n'))
    f.write('What value of pinning would you like? (float value)\n')
    f.write(str(pinningValue)+'\n')
    f.write('\n')
    for i in range(len(V)):
            for j in range(len(V)):
                if (np.sqrt((xx[i,j])**2 + (yy[i,j])**2) < 1):
                    V[i,j] += pinningValue
    
if (task == 'run'):
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
        runNumber = input('Please choose a number for this run\n')
        f.write('Please choose a number for this run\n')
        f.write(runNumber+'\n')
        f.write('\n')
        directory = 'bogRun'+str(runNumber)+'/'
        os.makedirs(directory)
    elif(initType == 'contd'):
        runNumber = input('Please provide the number of run that you want to load in\n')
        f.write('Please provide the number of run that you want to load in\n')
        f.write(runNumber+'\n')
        f.write('\n')
        phiOne,phiZero,phiMinusOne,phiOneList,phiZeroList,phiMinusOneList = gst.loadFullResults(runNumber,directory=directory)
    else:
        print('Please choose a valid option')
        f.close()
        time.sleep(1)
        os.remove('userInputsBog.txt')
        sys.exit()

if (task == 'run'):
    numVecs = input('How many eigenvectors do you want to generate? (int)\n')
    f.write('How many eigenvectors do you want to generate? (int)\n')
    f.write(str(numVecs)+'\n')
    f.write('\n')
f.close()
if (task == 'run'):
    if (directory!=''):
        shutil.copy('userInputsBog.txt',directory+'userInputsBog'+str(runNumber)+'.txt')
        if (initType == 'generic'):
            shutil.copy(inputFile,directory+inputFile)
    bog.bogoliubov(phiOneList, phiZeroList, phiMinusOneList, L, a, b, x, y, V, b0, b1, p, q, mainIter=runNumber, directory=directory)


    




























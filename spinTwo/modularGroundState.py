# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 15:04:16 2022

@author: Nathan

modularGroundState.py

"""

"""Imports"""
import groundStateOneStep as gs
import groundStateTools as gst
import numpy as np
from configparser import ConfigParser
import matplotlib.pyplot as plt







def groundState(phiTwoList,phiOneList,phiZeroList,phiMinusOneList,phiMinusTwoList,V,b0,b1,b2,inputFile,directory='',mainIter=90,phiBTwo=0,phiBOne=0,phiBZero=0,phiBMinusOne=0,phiBMinusTwo=0):
    """Takes in initial guesses for the components of the ground state along with 
    other relevant parameters of the system and runs groundStateOneStep multiple 
    times to perform gradient descent of the energy functional towards the ground 
    state of the system.
    
    tau is varied based on an approximation of what the gradient is at each step and 
    an attempt to ensure convergence is implemented.
    
    Inputs:
        phiTwoList: Initial guess for the m = 2 component of the ground state,
        phiOneList: Initial guess for the m = 1 component of the ground state,
        phiZeroList: Initial guess for the m = 0 component of the ground state,
        phiMinusOneList: Initial guess for the m = -1 component of the ground state,
        phiMinusTwoList: Initial guess for the m = -2 component of the ground state,
        V: A 2D array storing the potential the system is in for each point in our grid,
        inputFile: String specifying the filename of an ini input file containing the following:
            L: The number of lattice spacings in both the x and y direction,
            a: The lower bound of our grid (in both the x and y direction),
            b: The upper bound of our grid (in both the x and y direction),
            x: The list of xpts along the side length x direction of our grid,
            y: The list of ypts along the side length y direction of our grid,
            b0: The dimensionless density interaction term,
            b1: The dimensionless spin density interaction term,
            b2: The dimensionless spin singlet interaction term,
            p: The linear Zeeman term,
            q: The quadratic Zeeman term,
            tau: The initial gradient descent step to take,
            t: the initial energy step we would like ot take between one gradient 
                descent step and the next,
            maxSteps: The maximum number of gradient descent steps to take.
        directory: The directory to store the results of this file. For example
            if the user wants to save the results in a folder called results in the 
            folder where this code is running then directory='results/' should be set.
        mainIter: A number to give the current run. This allows us to identify 
                    the saved files for this run of the code.
        phiBTwo: Boundary list of phiTwo. Default is zero boundary conditions.
        phiBOne: Boundary list of phiOne. Default is zero boundary conditions.
        phiBZero: Boundary list of phiZero. Default is zero boundary conditions.
        phiBMinusOne: Boundary list of phiMinusOne. Default is zero boundary conditions.
        phiBMinusTwo: Boundary list of phiMinusTwo. Default is zero boundary conditions.
    
    This code does not return anything. Instead it saves lists of the ground state 
    components at each step in the gradient descent algorithm."""
    
    """Read in inputs from the ini input file"""
    cp = ConfigParser()
    cp.read(inputFile)
    
    
    L = int(cp.get('latticePts','L'))
    a = float(cp.get('latticePts','a'))
    b = float(cp.get('latticePts','b'))
    
    
    
    p = float(cp.get('zeeman','p'))
    q = float(cp.get('zeeman','q'))
    
    
    
    """tau: The initial gradient descent step.
    maxSteps: The maximum number of gradient descent steps to take before 
                giving up on trying to converge to the ground state.
    t: Benchmark that we are trying to calibrate tau with. We set tau by trying 
    to make the energy difference between the system at one gradient descent step 
    and the next equal to t within some tolerance."""
    tau = float(cp.get('time','tau'))
    maxSteps = int(cp.get('time','maxSteps'))
    t = float(cp.get('time','t'))
    
    x = np.linspace(a,b,L+1)
    y = np.linspace(a,b,L+1)
    
    x = np.delete(x,0)
    x = np.delete(x,-1)
    y = np.delete(y,0)
    y = np.delete(y,-1)
    
    """Lattice spacing."""
    h = (b-a)/L
    
    xx,yy = np.meshgrid(x,y)
    yy*= -1
    
    """Boundary values of phi are set to zero by default."""
    if (phiBTwo == 0 and phiBOne == 0 and phiBZero == 0 and phiBMinusOne == 0 and phiBMinusTwo == 0):
        phiBTwo = np.zeros(4*L)
        phiBOne = np.zeros(4*L)
        phiBZero = np.zeros(4*L)
        phiBMinusOne = np.zeros(4*L)
        phiBMinusTwo = np.zeros(4*L)
    
    
    """Make sure phiTwo,phiOne,phithe wavefunction passed in as the initial condition is normalized."""
    norm = gst.getPhiNorm(phiTwoList,phiOneList,phiZeroList,phiMinusOneList,phiMinusTwoList,h)
    
    phiTwoList /= norm
    phiOneList /= norm
    phiZeroList /= norm
    phiMinusOneList /= norm
    phiMinusTwoList /= norm
    
    #phiTwoList,phiOneList,phiZeroList,phiMinusOneList,phiMinusTwoList = gst.roundPhi(phiTwoList,phiOneList,phiZeroList,phiMinusOneList,phiMinusTwoList)
    
    """Lists to store the accepted components of the wavefunction at each gradient 
    descent step."""
    phiTwo = []
    phiOne = []
    phiZero = []
    phiMinusOne = []
    phiMinusTwo = []
    
    """Store the initial condition as the first state in our list of accepted steps 
    above, we do this to ensure that if no gradient descent steps are accepted below, 
    then we still return the initial condition (ie: The initial condition must have 
    been an exact solution to the equations already, or is the min of the system 
    already)."""
    phiTwo.append(phiTwoList)
    phiOne.append(phiOneList)
    phiZero.append(phiZeroList)
    phiMinusOne.append(phiMinusOneList)
    phiMinusTwo.append(phiMinusTwoList)
    
    """Remains false until the energy difference between the current gradient 
    descent solution for phi and the previous one is below the value of t. This 
    means we can estimate the derivative of the slope we are moving down accurately."""
    accurate = False
    i = 0
    
    while(accurate==False):
        print('i = ' + str(i))
        print('mainIter = ' + str(mainIter))
        
        """Take one gradient descent step."""
        phiTwoCurrent,phiOneCurrent,phiZeroCurrent,phiMinusOneCurrent,phiMinusTwoCurrent = gs.groundState(phiTwoList,phiOneList,phiZeroList,phiMinusOneList,phiMinusTwoList,L,V,h,tau,p,q,b0,b1,b2)
        
        
        
        """Now want to try and keep running groundState function until we get to a 
        local minimum. To do this we need to consider the stepsize, tau, we take in 
        phi space and given a target stepsize, t, in energy space, we would like to 
        be able to vary tau to ensure that the step we make in energy from one run of 
        ground state to the next is of the order of t.
        
        To do this we need to check that we can approximate the derivative of the energy 
        functional properly which is what I'm doing below."""
        
        """Calculate the energies of the wavefunction before and after this gradient descent step."""
        energyPrevious = gst.computeEnergy(phiTwoList,phiOneList,phiZeroList,phiMinusOneList,phiMinusTwoList,L,h,V,b0,b1,b2,p,q)
        energyCurrent = gst.computeEnergy(phiTwoCurrent,phiOneCurrent,phiZeroCurrent,phiMinusOneCurrent,phiMinusTwoCurrent,L,h,V,b0,b1,b2,p,q)
        
        """Calculate the difference in energy. This divided by tau is meant to 
        be an approximation of the derivative of the energy functional at the 
        current phi values. We therefore want this energy difference to be smaller 
        that some initially set t so that we have a good approximation of the derivative 
        and hence a good procedure for varying the value of tau."""
        energyDiff = abs(energyPrevious-energyCurrent)
        
        print('energyDiff = ' + str(energyDiff))
        
        if (energyDiff < t):
            """If the value of energyDiff is less than t, update the stepsize 
            tau to a suitable size based on the derivative of the energy at this 
            point and move to the next step of the code."""
            tau = (t/energyDiff)*tau
            accurate = True
        else:
            """Otherwise, reduce our stepsize to try and get a more accurate 
            approximation for the derivative of the energy functional."""
            tau /= 2
            print('tau = ' + str(tau))
        
        i+=1
    
    """Once we have an accurate approximation of the derivative and hence a 
    reasonable starting value for our gradient descent step size, tau, we can start 
    repeatedly calling the one step gradient descent code to hopefully converge to 
    the ground state of the system."""
    n = 0
    steps = 2
    
    """How accurate we would like to get the ground state."""
    tol = 1e-08
    converged = False
    
    
    
    while(converged == False and n < maxSteps):
        
        print('n = ' + str(n))
        print('mainIter = ' + str(mainIter))
        
        """Perform another gradient descent step with the above phi as our starting point."""
        phiTwoCurrent,phiOneCurrent,phiZeroCurrent,phiMinusOneCurrent,phiMinusTwoCurrent = gs.groundState(phiTwoList,phiOneList,phiZeroList,phiMinusOneList,phiMinusTwoList,L,V,h,tau,p,q,b0,b1,b2)
        
        
        """Calculate the energy difference before and after this step."""
        energyPrevious = gst.computeEnergy(phiTwoList,phiOneList,phiZeroList,phiMinusOneList,phiMinusTwoList,L,h,V,b0,b1,b2,p,q)
        energyCurrent = gst.computeEnergy(phiTwoCurrent,phiOneCurrent,phiZeroCurrent,phiMinusOneCurrent,phiMinusTwoCurrent,L,h,V,b0,b1,b2,p,q)
        
        energyDiff = abs(energyPrevious-energyCurrent)
        
        if ((energyDiff > 0.9*t and energyDiff < 1.1*t) or ((n+1)%steps != 0)):
            
            if (energyDiff > 0.9*t and energyDiff < 1.1*t):
                """If the energy difference is equal to t within some tolerance, 
                accept the phi components after the trial gradient descent step 
                and add them to our list storing the components of phi at each 
                step."""
                print('yay')
                phiTwoList = np.array(phiTwoCurrent)
                phiOneList = np.array(phiOneCurrent)
                phiZeroList = np.array(phiZeroCurrent)
                phiMinusOneList = np.array(phiMinusOneCurrent)
                phiMinusTwoList = np.array(phiMinusTwoCurrent)
                
                phiTwo.append(phiTwoList)
                phiOne.append(phiOneList)
                phiZero.append(phiZeroList)
                phiMinusOne.append(phiMinusOneList)
                phiMinusTwo.append(phiMinusTwoList)
            
            """If we have accepted the gradient descent step or we are not on a multiple 
            of 2 in the number of gradient descent steps we have taken, update tau based 
            off the gradient."""
            tau = (t/energyDiff)*tau
            
            """Increase the step size, tau and target energy difference, t 
            a little bit just to make sure that the gradient has not increased 
            again and we could be taking larger steps towards the ground state 
            again."""
            if (n > 0 and tau < 0.5):
                tau *= 1.1
                t *= 1.1
        
        else:
            """The initial energy difference is never in the above range so it 
            keeps halving t too fast and as such is not working. So I've set it 
            to let it have a go at bringing energyDiff within the range for a few 
            steps by saying it can also go into the statement above if n is not a 
            multiple of steps"""
            
            """Otherwise, halve t and tau to try and take a step that is in line 
            with what the gradient is at the moment."""
            
            t /= 2
            tau /= 2
        
        print('t = ' + str(t))
        print('energyDiff = ' + str(energyDiff))
        print('tau = ' + str(tau))
        
        if (t<tol):
            """We measure if we have converged by looking at what t is in relation 
            to the tolerance we set at the start of the code. If t is small, then the 
            energy difference between steps should be small (ie: We should be near the 
            ground state of the system). We do this with t as we could get an iteration 
            where that gradient happens to be small and so the energy difference is small, 
            however this may not be the min. This will not happen to t and so it is a better 
            measure of convergence."""
            converged = True
            
        n += 1
    
    """Save the lists containing the components of the wavefunction at each gradient 
    descent step that we accepted."""
    gst.saveResults(str(mainIter),directory=directory,phiTwo=phiTwo,phiOne=phiOne,phiZero=phiZero,phiMinusOne=phiMinusOne,phiMinusTwo=phiMinusTwo)
    
    

    
    
    
      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

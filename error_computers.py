#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 18:12:16 2022

@author: v
"""
import os
import copy
import cvxpy as cp
import numpy as np


class ErrorComputerKnown:
    
    def __init__(self, method_params=None):
        self.require_Dmatrix = False
        self.require_landmarks = False
        self.direction = 'RG'
    
    def compute_error(self, R, G, Dx=None, gt_landmarks=None, li=None, iod=None, cpts=None, return_corr=False, li0=None, G0=None):
        N = R.shape[0]
        err = np.sqrt((((R-G[:N,:])**2).sum(axis=1)))
        
        return err
    
    def get_key(self, landmark_computer):
        return 'known'

    def get_label(self):
        return 'Known'


class ErrorComputerChamfer:
    
    def __init__(self, method_params=None, direction='RG', correction='no'):
        self.require_Dmatrix = False
        self.require_landmarks = False
        self.tail = ''
        self.direction = direction
        self.correction = correction
        if self.direction == 'GR':
            self.tail += 'GR'
        if self.correction != 'no':
            self.tail += self.correction
    
    def compute_error(self, R, G, Dx=None, gt_landmarks=None, li=None, return_err=True, iod=None,cpts=None, return_corr=False, li0=None, G0=None):
        N = R.shape[0]
        
        if not return_err:
            return R
        
        if self.direction == 'RG':
            pidx = np.zeros(N)
            for i in range(N):
                pidx[i] = np.argmin(np.sqrt((((R[i,:]-G)**2).sum(axis=1))))
            pidx = pidx.astype(int)
            err_ptwise = np.sqrt((((R-G[pidx])**2).sum(axis=1)))
        elif self.direction == 'GR':
            M = G.shape[0]            
            pidx = np.zeros(M)
            for i in range(M):
                pidx[i] = np.argmin(np.sqrt((((R-G[i,:])**2).sum(axis=1))))
            pidx = pidx.astype(int)
            print(pidx)
            err_ptwise = np.sqrt((((R[pidx]-G)**2).sum(axis=1)))
        
        """
        import matplotlib.pyplot as plt
        plt.subplot(1,2,1)
        plt.plot(R[cpts, 0], R[cpts, 1], '.', markersize=1.5)
        plt.subplot(1,2,2)
        plt.plot(G[pidx, 0], G[pidx, 1], '.', markersize=1.5)
        plt.show()
        """
        
        return err_ptwise
            
    
    def get_key(self, landmark_computer):
        return 'chamfer%s' % self.tail

    def get_label(self):
        if self.correction != 'no':
            return 'ChamfC'
        else:
            return 'Chamfer'

    def __str__(self):
        return self.get_label()






    
    
class ErrorComputerChamferFLR_NIC:
    
    def __init__(self, gamma=1, correction='no', direction='RG'):
        self.direction = direction
        self.require_Dmatrix = True
        self.require_landmarks = True
        self.gamma = gamma
        self.correction = correction
        
        tail = ''
        if self.correction != 'no':
            tail = self.correction

        if self.direction == 'GR':
            tail += 'GR'
        
        self.method_key = 'FLR_NIC%.2f%s' % (gamma, tail)

    
    
    def compute_error(self, R, G, Dx=None, gt_landmarks=None, li=None, iod=None, cpts=None, return_err = True, return_corr=False, li0=None, G0=None):
        Dx = Dx**self.gamma
        Dy = Dx
        Dz = Dx
        
        Dxl = copy.deepcopy(Dx)[li,:]
        Dyl = copy.deepcopy(Dy)[li,:]
        Dzl = copy.deepcopy(Dz)[li,:]
        
        bxl = gt_landmarks[:,0]-R[li,0]
        byl = gt_landmarks[:,1]-R[li,1]
        bzl = gt_landmarks[:,2]-R[li,2]
                
        dxu = self.solve_optimization_problem(Dx, Dxl, bxl)
        dyu = self.solve_optimization_problem(Dy, Dyl, byl)
        dzu = self.solve_optimization_problem(Dz, Dzl, bzl)
        
        N = R.shape[0]
        R2 = np.zeros((N,3))
        
        R2[:,0] = R[:,0] + Dx@dxu
        R2[:,1] = R[:,1] + Dy@dyu
        R2[:,2] = R[:,2] + Dz@dzu
        

        if self.direction == 'RG':
            pidx = np.zeros(N, dtype=int)
            for i in range(N):
                d = np.sqrt(np.sum((R2[i,:]-G)**2, axis=1))
                pidx[i] = np.argmin(d)
        elif self.direction == 'GR':
            M = G.shape[0]
            pidx = np.zeros(M, dtype=int)
            for i in range(M):
                d = np.sqrt(np.sum((R2-G[i,:])**2, axis=1))
                pidx[i] = np.argmin(d)
        
        if self.direction == 'RG':
            return np.sqrt((((R-G[pidx])**2).sum(axis=1)))
        elif self.direction == 'GR':
            return np.sqrt((((R[pidx]-G)**2).sum(axis=1)))
    
    
    def solve_optimization_problem(self, D, Dl, b):
        np.random.seed(1907)
        m = Dl.shape[0]
        n = Dl.shape[1]
        bmax = np.max(np.abs(b))
        #e2max = 0.2*emax;       
        
        Dsub = D[::10,:]
                
        x = cp.Variable(n)
        objective = cp.Minimize(cp.sum_squares(Dl@x-b))
        # constraints = [Dl@x-b <= e, E@x-f <= e2]
        # constraints = [ Dl@x-b <= emax, emin <= Dl@x-b]
        constraints = []
        
        
        constraints.append(cp.norm_inf(Dsub@x)  <= bmax)
        prob = cp.Problem(objective, constraints)
        
        # The optimal objective value is returned by `prob.solve()`.
        result = prob.solve(solver='SCS')
        # The optimal value for x is stored in `x.value`.
        # print(x.value)
        # The optimal Lagrange multiplier for a constraint is stored in
        # `constraint.dual_value`.
        # print(constraints[0].dual_value)
        
        return x.value
    
    
    def get_key(self, landmark_computer):
        return self.method_key + landmark_computer.get_key()

    def get_label(self):
        if self.correction == 'no':
            return 'EFR'
        else:
            return 'EFRC'
            # return 'FLR(c)'
    
    def __str__(self):
        return self.get_label()






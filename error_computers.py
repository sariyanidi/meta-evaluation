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

from scipy import interpolate
from scipy.sparse.linalg import spsolve 
from scipy.sparse import linalg as splinalg
from scipy import sparse

from scipy.spatial.distance import pdist, squareform



def sparse_cholesky(A): # The input matrix A must be a sparse symmetric positive-definite.
    n = A.shape[0]
    LU = splinalg.splu(A,diag_pivot_thresh=0) # sparse LU decomposition
    # if ( LU.perm_r == np.arange(n) ).all() and ( LU.U.diagonal() > 0 ).all(): # check the matrix A is positive definite.
    return LU.L.dot( sparse.diags(LU.U.diagonal()**0.5) )



class LandmarkComputerExact:
    
    def __init__(self):
        pass
    
    def get_key(self):
        return 'exact'
    
    def compute_landmarks(self, Gs_path, Gt_path=None):
        # G = np.loadtxt(Gs_path)
        lpath = Gs_path.replace('txt', '3Dlmks_exact')
        L = np.loadtxt(lpath)
        if L.shape[0] == 68:
            L = L[17:,:]
        return L
    
    def landmarks_exist(self, Gs_path):
        lpath = Gs_path.replace('txt', '3Dlmks_exact')
        return os.path.exists(lpath)
    
        
    
class LandmarkComputerAuto:
    
    def __init__(self):
        pass
    
    def get_key(self):
        return 'auto'
    
    def landmarks_exist(self, Gs_path):
        lpath = Gs_path.replace('txt', '3Dlmks_auto')
        return os.path.exists(lpath)
    
    def compute_landmarks(self, Gs_path, Gt_path=None):
        # G = np.loadtxt(Gs_path)
        lpath = Gs_path.replace('txt', '3Dlmks_auto')
        return np.loadtxt(lpath)
        

class LandmarkComputerNoisy:
    
    def __init__(self, noisesigma):
        self.noisesigma = noisesigma
        
    def get_key(self):
        if self.noisesigma == 'exact':
            return 'exact'
        else:
            return 'n%.5f' % (self.noisesigma)
    
    def compute_landmarks(self, Gs_path, Gt_path=None):
        lpath = Gs_path.replace('txt', '3Dlmks_exact')
        lG = np.loadtxt(lpath)
        iod = np.linalg.norm(lG[28,:]-lG[19,:])
        tmp = int(np.sum(np.abs(1000*copy.deepcopy(lG)/iod)))
        np.random.seed(tmp % 19071)  # standardize noise across subjects        
        
        nlG = lG+np.random.randn(lG.shape[0],3)*self.noisesigma*iod
        # import matplotlib.pyplot as plt
        # plt.plot(lG[:,0], lG[:,1])
        # plt.plot(nlG[:,0], nlG[:,1])
        return nlG

    def landmarks_exist(self, Gs_path):
        lpath = Gs_path.replace('txt', '3Dlmks_exact')
        return os.path.exists(lpath)
     

class LandmarkComputerScaled:
    
    def __init__(self, deltas):
        self.deltas = deltas 
        
    def get_key(self):
        if self.deltas == 0:
            return 'exact'
        else:
            return 'sc%.5f' % (self.deltas)
    
    def landmarks_exist(self, Gs_path):
        lpath = Gs_path.replace('txt', '3Dlmks_exact')
        return os.path.exists(lpath)
  
    def compute_landmarks(self, Gs_path, Gt_path=None):
        lpath = Gs_path.replace('txt', '3Dlmks_exact')
        lG = np.loadtxt(lpath)
        iod = np.linalg.norm(lG[28,:]-lG[19,:])
        tmp = int(np.sum(np.abs(1000*copy.deepcopy(lG)/iod)))
        np.random.seed(tmp % 19071)  # standardize noise across subjects        
        
        nlG = lG*(1+self.deltas*iod)
        # import matplotlib.pyplot as plt
        # plt.plot(lG[:,0], lG[:,1])
        # plt.plot(nlG[:,0], nlG[:,1])
        return nlG
    

class LandmarkComputerTranslated:
    
    def __init__(self, deltat):
        self.deltat = deltat 
        
    def get_key(self):
        if self.deltat == 0:
            return 'exact'
        else:
            return 'tr%.3f' % (self.deltat)
    def landmarks_exist(self, Gs_path):
        lpath = Gs_path.replace('txt', '3Dlmks_exact')
        return os.path.exists(lpath)
 
    def compute_landmarks(self, Gs_path, Gt_path=None):
        lpath = Gs_path.replace('txt', '3Dlmks_exact')
        lG = np.loadtxt(lpath)
        iod = np.linalg.norm(lG[28,:]-lG[19,:])
        tmp = int(np.sum(np.abs(1000*copy.deepcopy(lG)/iod)))
        np.random.seed(tmp % 19071)  # standardize noise across subjects        
        
        nlG = lG+self.deltat*iod
        # import matplotlib.pyplot as plt
        # plt.plot(lG[:,0], lG[:,1])
        # plt.plot(nlG[:,0], nlG[:,1])
        return nlG




def build_cholseky_factor(ads, b, N):
    diags = []
    off_diags = []
    
    sum2_off_diags = 0
    for i in range(N):
        diags.append(np.sqrt(ads[i]-sum2_off_diags))
        if i == N-1:
            break
        off_diags.append((b-sum2_off_diags)/diags[-1])
        sum2_off_diags += off_diags[-1]*off_diags[-1]
    return (diags, off_diags)


def solve_lower_tri(alphas, betas, b):
    N = len(alphas)
    y = np.zeros(N)
    
    prevsum = 0
    for i in range(0,N):
        y[i] = (b[i]-prevsum)/alphas[i]
        if i != N-1:
            prevsum += betas[i]*y[i]
            
    return y


def solve_upper_tri(alphas, betas, y):
    N = len(alphas)
    x = np.zeros(N)
    
    xsum = 0
    betas.append(0)
    for i in reversed(range(0,N)):
        x[i] = (y[i]-xsum*betas[i])/alphas[i]
        xsum += x[i]
            
    return x


def crop_GT(R, G, iod, crop_rate=0.08):
    N = R.shape[0]
    M = G.shape[0]

    idx = []
    for i in range(M):
        mdist = np.min(np.sqrt((((R-G[i,:])**2).sum(axis=1))))
        if mdist < crop_rate*iod:
            idx.append(i)
    
    return G[idx,:], idx




def apply_correction2(X, Y, Xlmks, li, pidx, iod, correction_param):
    
    Nl = Xlmks.shape[0]
    
    X = copy.deepcopy(X)[pidx,:]
    correction_strategy = correction_param.split('-')[0]
    weight_strategy = correction_param.split('-')[1]
    
    weight_power = 'linear'
    if len(correction_param.split('-')) >= 4:
        weight_power = correction_param.split('-')[3]
        
    adists = []
    for i in range(Nl):
        dists = np.sqrt(np.sum(((Xlmks[i]-X)**2), axis=1))
        adists.append(dists)
        dth = 0.01
        dists[np.where(dists < dth)] = dth# max(dists)
    
    madists = np.array(adists).min(axis=0)#/np.median(np.array(adists),axis=0)
    weights_min = (1./madists)
    
    ref_lis = [0, 2, 4, 5, 7, 9, 20, 21, 23, 24, 26, 27, 29, 30,   19, 22, 25, 28,    13, 14, 18, 31, 33, 34, 35, 37, 44, 45, 46, 39, 40, 41, 49, 48, 50]
    adists = []
    for i in range(Nl):
        dists = np.sqrt(np.sum(((Xlmks[i]-X)**2), axis=1))
        adists.append(dists)
    
    mean_dists = np.mean(np.array(adists),axis=0)#/np.median(np.array(adists),axis=0)
    
    weights_mean = (1./np.abs(mean_dists-0.48))
    
    if weight_strategy == 'mixed':
        weights = (weights_mean+weights_min)/2
    elif weight_strategy == 'min':
        weights = weights_min
    elif weight_strategy == 'mean':
        weights = weights_mean
    
    weights[np.where(weights<1)] = 1

    if weight_power == 'square':
        # weights = (weights**2)/(np.max(weights**2)/np.max(weights))
        weights = (weights**2)/(np.mean(weights**2)/np.mean(weights))
    elif weight_power == 'sqrt':
        weights = np.sqrt(weights)
    
    Y = copy.deepcopy(Y)
    
    N = X.shape[0]
    updates = []
    
    del adists
    
    for dim in range(3):
        if correction_strategy == 'trace':
            r = Y[:,dim].reshape(-1,1)
                
            g = X[:,dim].reshape(-1,1)
            e = np.ones(r.shape)
            
            gr = g-r
            r = e*np.sum(gr)-N*gr
            rT = r.T
            
            a = 2*(N-1)
            b = -2
            
            ads = weights*N*2+a 
            
            (diags, off_diags) = build_cholseky_factor(ads, b, N)
            c = -2*rT
            
            y = solve_lower_tri(diags, off_diags, -c.T)
            dx_star = solve_upper_tri(diags, off_diags, y)
            updates.append(dx_star)

        elif correction_strategy == 'pair':
            
            r0 = Y[:,dim]#.reshape(-1,1)
            
            g0 = X[:,dim]

            rix = np.argsort(r0)
            r = copy.deepcopy(r0)[rix]
            g = copy.deepcopy(g0)[rix]
            
            c = np.zeros(N)
            c[0] = -(g[0]-g[1]-(r[0]-r[1]))
            c[-1] = -(g[-1]-g[-2]-(r[-1]-r[-2]))
            
            for i in range(1,N-1):
                c[i] = -(2*g[i]-g[i-1]-g[i+1] - 2*r[i]+r[i-1]+r[i+1])

            d1 = -np.ones(N)#+weights[:N]
            d0 = 2 * np.ones(N)+2*(weights[rix])#/2
            A = (1./2)*sparse.spdiags([d1,d0,d1], [-1,0,1],N,N,format='csc')
            L = sparse_cholesky(A).T
            b = spsolve(L.T, c)
            y = spsolve(L, b)
            irix = np.empty_like(rix)
            irix[rix] = np.arange(rix.size)
            x = y[irix]
            updates.append(x)
    
    dX = np.array(updates).T
    return dX







def apply_correction(G_, R2_, Glmks, li, pidx, iod, correction_param, G0=None, li0=None):
    
    if G0 is None:
        li0 = [19106,19413,19656,19814,19981,20671,20837,20995,21256,21516,8161,8175,8184,8190,6758,7602,8201,8802,9641,1831,3759,5049,6086,4545,3515,10455,11482,12643,14583,12915,11881,5522,6154,7375,8215,9295,10523,10923,9917,9075,8235,7395,6548,5908,7264,8224,9184,10665,8948,8228,7508] 
        G0 = np.loadtxt('data/meanface.txt')    
    iod0 = np.linalg.norm(G0[li0[28],:]-G0[li0[19],:])
    G0 /= iod0
    
    G0l = copy.deepcopy(G0)[li,:]
    Nl = len(li)
        
    correction_strategy = correction_param.split('-')[0]
    weight_strategy = correction_param.split('-')[1]
    
    weight_power = 'linear'
    if len(correction_param.split('-')) >= 4:
        weight_power = correction_param.split('-')[3]
        
    adists = []
    for i in range(Nl):
        dists = np.sqrt(np.sum(((G0l[i,:]-copy.deepcopy(G0))**2), axis=1))
        adists.append(dists)
        dth = 0.01
        dists[np.where(dists < dth)] = dth# max(dists)
    
    madists = np.array(adists).min(axis=0)#/np.median(np.array(adists),axis=0)
    weights_min = (1./madists)
    
    ref_lis = [0, 2, 4, 5, 7, 9, 20, 21, 23, 24, 26, 27, 29, 30,   19, 22, 25, 28,    13, 14, 18, 31, 33, 34, 35, 37, 44, 45, 46, 39, 40, 41, 49, 48, 50]
    adists = []
    if len(li) == 51:
        for i in range(len(ref_lis)):
            dists = np.sqrt(np.sum(((G0[li[ref_lis[i]],:]-G0)**2), axis=1))
            adists.append(dists)
    else:
        for i in range(len(li)):
            dists = np.sqrt(np.sum(((G0[li[i],:]-G0)**2), axis=1))
            adists.append(dists)
    
    mean_dists = np.mean(np.array(adists),axis=0)#/np.median(np.array(adists),axis=0)
    
    weights_mean = (1./np.abs(mean_dists-0.48))
    
    if weight_strategy == 'mixed':
        weights = (weights_mean+weights_min)/2
    elif weight_strategy == 'min':
        weights = weights_min
    elif weight_strategy == 'mean':
        weights = weights_mean
    
    weights[np.where(weights<1)] = 1

    if weight_power == 'square':
        # weights = (weights**2)/(np.max(weights**2)/np.max(weights))
        weights = (weights**2)/(np.mean(weights**2)/np.mean(weights))
    elif weight_power == 'sqrt':
        weights = np.sqrt(weights)
    
    G = copy.deepcopy(G_)[pidx]
    R2 = copy.deepcopy(R2_)
    
    N = G.shape[0]
    updates = []
    
    del adists
    
    for dim in range(3):
        if correction_strategy == 'trace':
            r = R2[:,dim].reshape(-1,1)
                
            g = G[:,dim].reshape(-1,1)
            e = np.ones(r.shape)
            
            gr = g-r
            r = e*np.sum(gr)-N*gr
            rT = r.T
            
            a = 2*(N-1)
            b = -2
            
            ads = weights*N*2+a 
            
            (diags, off_diags) = build_cholseky_factor(ads, b, N)
            c = -2*rT
            
            y = solve_lower_tri(diags, off_diags, -c.T)
            dx_star = solve_upper_tri(diags, off_diags, y)
            updates.append(dx_star)

        elif correction_strategy == 'pair':
            
            r0 = R2[:,dim]#.reshape(-1,1)
            
            g0 = G[:,dim]

            rix = np.argsort(r0)
            r = copy.deepcopy(r0)[rix]
            g = copy.deepcopy(g0)[rix]
            
            c = np.zeros(N)
            c[0] = -(g[0]-g[1]-(r[0]-r[1]))
            c[-1] = -(g[-1]-g[-2]-(r[-1]-r[-2]))
            
            for i in range(1,N-1):
                c[i] = -(2*g[i]-g[i-1]-g[i+1] - 2*r[i]+r[i-1]+r[i+1])

            d1 = -np.ones(N)#+weights[:N]
            d0 = 2 * np.ones(N)+2*(weights[rix])#/2
            A = (1./2)*sparse.spdiags([d1,d0,d1], [-1,0,1],N,N,format='csc')
            L = sparse_cholesky(A).T
            b = spsolve(L.T, c)
            y = spsolve(L, b)
            irix = np.empty_like(rix)
            irix[rix] = np.arange(rix.size)
            x = y[irix]
            updates.append(x)
    
    dG = np.array(updates).T
    
    return dG







class ErrorComputerKnown:
    
    def __init__(self, method_params=None):
        self.require_Dmatrix = False
        self.require_landmarks = False
        self.direction = 'RG'
    
    def compute_error(self, R, G, Dx=None, gt_landmarks=None, li=None, iod=None, cpts=None, return_corr=False, li0=None, G0=None):
        N = R.shape[0]
        err = np.sqrt((((R-G[:N,:])**2).sum(axis=1)))
        
        if return_corr:
            pidx = np.arange(len(err)).astype(int)#.reshape(-1,1)
            
            return (err, pidx)
        else:
            return err
    
    def get_key(self, landmark_computer):
        return 'known'

    def get_label(self):
        return 'Known'



class ErrorComputerLandmarks:
    
    def __init__(self, method_params=None):
        self.require_Dmatrix = False
        self.require_landmarks = True
        self.direction = 'RG'
    
    def compute_error(self, R, G, Dx=None, gt_landmarks=None, li=None, iod=None, cpts=None, return_corr=False, li0=None, G0=None):
        #print(gt_landmarks)
        #print(R[li,:])
        return np.sqrt((((R[li,:]-gt_landmarks)**2).sum(axis=1)))
    
    def get_key(self, landmark_computer):
        return 'lmks'+landmark_computer.get_key()

    def get_label(self):
        return 'Landm'

    def __str__(self):
        return self.get_label()

class ErrorComputerDensityAwareChamfer:
    
    def __init__(self, method_params=None):
        self.require_Dmatrix = False
        self.require_landmarks = False

    
    def compute_error(self, R, G, Dx=None, gt_landmarks=None, li=None, return_err=True, iod=None,cpts=None, return_corr=False, li0=None, G0=None):
        import sys
        sys.path.append('/home/sariyanide/software/other_code/Density_aware_Chamfer_Distance/utils_v2')
        from model_utils import calc_dcd
        import torch
        Rt = torch.from_numpy(R.reshape(1, -1, 3)).cuda().float()
        Gt = torch.from_numpy(G.reshape(1, -1, 3)).cuda().float()
        return calc_dcd(Rt,Gt)
        # print(calc_dcd(Rt,Gt))
        err_ptwise = np.array(calc_dcd(Rt,Gt)[0].cpu())*np.ones(R.shape[0])
        return err_ptwise
        N = R.shape[0]
        
        if not return_err:
            return R
        
        pidx = np.zeros(N)
        for i in range(N):
            pidx[i] = np.argmin(np.sqrt((((R[i,:]-G)**2).sum(axis=1))))
                
        
        pidx = pidx.astype(int)
        """
        import matplotlib.pyplot as plt
        plt.subplot(1,2,1)
        plt.plot(R[:, 0], R[:, 1], '.', markersize=1.5)
        plt.subplot(1,2,2)
        plt.plot(G[pidx, 0], G[pidx, 1], '.', markersize=1.5)
        print('done')
        """
        if return_corr:
            return (np.sqrt((((R-G[pidx])**2).sum(axis=1))), pidx)
        else:
            return np.sqrt((((R-G[pidx])**2).sum(axis=1)))
    
    def get_key(self, landmark_computer):
        return 'dac'

    def get_label(self):
        return 'DAC'





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
        print('done')
        plt.show()
        """
        
        if self.correction == 'no':
            if return_corr:
                return (err_ptwise, pidx)
            else:
                return err_ptwise

        
        if self.direction == 'RG':

            dG = apply_correction(G, R, gt_landmarks, li, pidx, iod, self.correction)
            return np.sqrt((((R-(G[pidx]-dG))**2).sum(axis=1)))
        elif self.direction == 'GR':
            dR = apply_correction2(R, G, R[li,:], li, pidx, iod, self.correction)
            Rupd = copy.deepcopy(R)[pidx]-dR
            return np.sqrt((((Rupd-G)**2).sum(axis=1)))

            
    
    def get_key(self, landmark_computer):
        return 'chamfer%s' % self.tail

    def get_label(self):
        if self.correction != 'no':
            return 'CCE'
        else:
            return 'Chamfer'

    def __str__(self):
        return self.get_label()





class ErrorComputerCPD_FLR_UC:
    
    def __init__(self, gamma=1):
        self.flr =  ErrorComputerChamferFLR_UC(gamma)
        self.method_key = 'CPD' + self.flr.method_key
        self.require_Dmatrix = True 
        self.require_landmarks = True 
    
    def compute_error(self, R, G, Dx=None, gt_landmarks=None, li=None, return_err=True, iod=None, return_corr=False, li0=None, G0=None, cpts=None):
        import time
        import open3d as o3
        from probreg import cpd

        import cupy as cp
        to_cpu = cp.asnumpy
        cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)


        t1 = time.time()
        
        R_ = self.flr.compute_error(R, G, Dx=Dx, gt_landmarks=gt_landmarks, li=li)

        source = cp.asarray(copy.deepcopy(R_))
        target = cp.asarray(copy.deepcopy(G))
        
        tf_param, _, __ = cpd.registration_cpd(source, target, tf_type_name="nonrigid", use_cuda=True)
        result = copy.deepcopy(source)
        result = tf_param.transform(result)
        
        elapsed = time.time() - t1
        print(elapsed)

        R2 = to_cpu(result)
        
        N = R.shape[0]
        
        pidx = np.zeros(N, dtype=int)
        for i in range(N):
            d = np.sqrt(np.sum((R2[i,:]-G)**2, axis=1))
            pidx[i] = np.argmin(d)
        
        if return_corr:
            return (np.sqrt((((R-G[pidx])**2).sum(axis=1))), pidx)
        else:
            return np.sqrt((((R-G[pidx])**2).sum(axis=1)))

    def get_key(self, landmark_computer):
        return self.method_key + landmark_computer.get_key()


    def __str__(self):
        return self.get_label()

    def get_label(self):
        return 'FLR(u)+CPD'







class ErrorComputerChamferFLR_UC:
    
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
            tail += self.direction
        
        self.method_key = 'FLR_UC%.2f%s' % (gamma, tail)
        
        
    def compute_error(self, R, G, Dx=None, gt_landmarks=None, li=None, return_err=True, iod=None, return_corr=False, li0=None, G0=None, cpts=None):
        # import matplotlib.pyplot as plt
        Dx = Dx**self.gamma
        Dy = Dx
        Dz = Dx
        
        Dxl = copy.deepcopy(Dx)[li,:]
        Dyl = copy.deepcopy(Dy)[li,:]
        Dzl = copy.deepcopy(Dz)[li,:]
        
        bxl = gt_landmarks[:,0]-R[li,0]
        byl = gt_landmarks[:,1]-R[li,1]
        bzl = gt_landmarks[:,2]-R[li,2]
        
        dxu = np.linalg.solve(Dxl, bxl)
        dyu = np.linalg.solve(Dyl, byl)
        dzu = np.linalg.solve(Dzl, bzl)
        
        N = R.shape[0]
        R2 = np.zeros((N,3))
        
        R2[:,0] = R[:,0] + Dx@dxu
        R2[:,1] = R[:,1] + Dy@dyu
        R2[:,2] = R[:,2] + Dz@dzu

        if not return_err:
            return R2
        
        pidx = np.zeros(N, dtype=int)
        for i in range(N):
            d = np.sqrt(np.sum((R2[i,:]-G)**2, axis=1))
            pidx[i] = np.argmin(d)
        
        
        #iod = np.linalg.norm(gt_landmarks[28,:]-gt_landmarks[19,:])
        
        
        if self.correction != 'no':
            
            if self.correction[-1] == '1' or self.correction[-1] == '2':
                dG = apply_correction(G, R2, gt_landmarks, li, pidx, iod, self.correction)
                
                updates_full = []
                for dim in range(3): #dim = 0    
                    f = interpolate.interp1d(G[pidx,dim], dG[:,dim], fill_value="extrapolate")
                    dGorig_dim = f(G[:,dim]) 
                    dGorig_dim[np.where(np.isnan(dGorig_dim))] = 0
                    updates_full.append(dGorig_dim)
        
                dGorig = np.array(updates_full).T
        
                Gorig_aug = copy.deepcopy(G)
                Gorig_aug -= dGorig
                pidx2 = np.zeros(N)
                for i in range(N):
                    d = np.sqrt(np.sum((R2[i,:]-(Gorig_aug))**2, axis=1))
                    pidx2[i] = np.argmin(d)
                
                err_corr = np.sqrt((((R-(Gorig_aug[pidx2.astype(int)]))**2).sum(axis=1)))
                
                if self.correction[-1] == '1':
                    return err_corr
                else:
                    return (np.sqrt((((R-(G[pidx]-dG))**2).sum(axis=1)))+err_corr)/2.0
            else:
                dG = apply_correction(G, R2, gt_landmarks, li, pidx, iod, self.correction)
                return np.sqrt((((R-(G[pidx]-dG))**2).sum(axis=1)))
            
        else:
       
            if return_corr:
                return (np.sqrt((((R-G[pidx])**2).sum(axis=1))), pidx)
            else:
                return np.sqrt((((R-G[pidx])**2).sum(axis=1)))
        
    
    def get_key(self, landmark_computer):
        return self.method_key + landmark_computer.get_key()
    def __str__(self):
        return self.get_label()
    
    def get_label(self):
        if self.correction == 'no':
            return 'FLR'
        else:
            return 'FLR (unb+corr.)'



    
    
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
        """
        import matplotlib.pyplot as plt
        plt.subplot(1,2,1)
        plt.plot(R2[cpts,0], R2[cpts,1], '.')
        plt.plot(G[::4,0], G[::4,1], '.')
        plt.subplot(1,2,2)
        plt.plot(R2[cpts,2], R2[cpts,1], '.')
        plt.plot(G[::4,2], G[::4,1], '.')
        plt.show()
        """
        if not return_err:
            return R2

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
        
        if self.correction != 'no':
            if self.correction[-1] == '1' or self.correction[-1] == '2':
                dG = apply_correction(G, R2, gt_landmarks, li, pidx, iod, self.correction)
                
                updates_full = []
                for dim in range(3): #dim = 0    
                    f = interpolate.interp1d(G[pidx,dim], dG[:,dim], fill_value="extrapolate")
                    dGorig_dim = f(G[:,dim]) 
                    dGorig_dim[np.where(np.isnan(dGorig_dim))] = 0
                    updates_full.append(dGorig_dim)
        
                dGorig = np.array(updates_full).T
                
                Gorig_aug = copy.deepcopy(G)
                Gorig_aug -= dGorig
                pidx2 = np.zeros(N)
                for i in range(N):
                    d = np.sqrt(np.sum((R2[i,:]-(Gorig_aug))**2, axis=1))
                    pidx2[i] = np.argmin(d)
                
                err_corr = np.sqrt((((R-(Gorig_aug[pidx2.astype(int)]))**2).sum(axis=1)))
                
                if self.correction[-1] == '1':
                    return err_corr
                else:
                    return (np.sqrt((((R-(G[pidx]-dG))**2).sum(axis=1)))+err_corr)/2.0
            else:
                if self.direction == 'GR':
                    # import matplotlib.pyplot as plt
                    # plt.figure(1)
                    # ss = 1
                    # plt.plot(G[::ss,0], -G[::ss, 1], '.', markersize=3)
                    # plt.plot(R[pidx[::ss],0], -R[pidx[::ss], 1], '.', markersize=2)

                    dR = apply_correction2(R2, G, R[li,:], li, pidx, iod, self.correction)
                    Rupd = copy.deepcopy(R)[pidx]-dR


                    # plt.figure(2)
                    # ss = 1
                    # plt.plot(G[::ss,0], -G[::ss, 1], '.')
                    # plt.plot(Rupd[::ss,0], -Rupd[::ss, 1], '.', markersize=2)
                    return np.sqrt((((Rupd-G)**2).sum(axis=1)))

                elif self.direction == 'RG':
                    dG = apply_correction(G, R2, gt_landmarks, li, pidx, iod, self.correction, li0=li0, G0=G0)
                    return np.sqrt((((R-(G[pidx]-dG))**2).sum(axis=1)))
            
        else:
            if return_corr:
                if self.direction == 'RG':
                    return (np.sqrt((((R-G[pidx])**2).sum(axis=1))), pidx)
                elif self.direction == 'GR':
                    return (np.sqrt((((R[pidx]-G)**2).sum(axis=1))), pidx)
            else:
                if self.direction == 'RG':
                    return np.sqrt((((R-G[pidx])**2).sum(axis=1)))
                elif self.direction == 'GR':
                    return np.sqrt((((R[pidx]-G)**2).sum(axis=1)))
                # return np.sqrt((((R-G[pidx])**2).sum(axis=1)))
    
    
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
            return 'LP+Chamfer'
        else:
            return 'LP+CCE'
            # return 'FLR(c)'
    
    def __str__(self):
        return self.get_label()





    
class ErrorComputerChamferFLR_NIC2:
    
    def __init__(self, gamma=1):
        self.require_Dmatrix = True
        self.require_landmarks = True
        self.gamma = gamma
        self.method_key = 'FLR_NIC2%.2f' % (self.gamma)
        
        NLANDMARKS = 51
        
        self.li_lb = [0,1,2,3,4]
        self.li_rb = [5,6,7,8,9]
        self.li_le = [19,20,21,22,23,24]
        self.li_re = [25,26,27,28,29,30]
        self.li_no = [10,11,12,13,14,15,16,17,18]
        self.li_ul = [31,32,33,34,35,36,37,43,44,45,46,47]
        self.li_ll = [31,37,38,39,40,41,42,47,48,49,50, 43]
    
    
    def compute_error(self, R, G, Dx=None, gt_landmarks=None, li=None, iod=None, li0=None, G0=None):
        Dx = Dx**self.gamma
        Dy = Dx
        Dz = Dx
        
        Dxl = copy.deepcopy(Dx)[li,:]
        Dyl = copy.deepcopy(Dy)[li,:]
        Dzl = copy.deepcopy(Dz)[li,:]
        
        bxl = gt_landmarks[:,0]-R[li,0]
        byl = gt_landmarks[:,1]-R[li,1]
        bzl = gt_landmarks[:,2]-R[li,2]
                
        dxu = self.solve_optimization_problem(Dx, Dxl,  bxl)
        dyu = self.solve_optimization_problem(Dy, Dyl, byl)
        dzu = self.solve_optimization_problem(Dz, Dzl, bzl)
        
        
        N = R.shape[0]
        R2 = np.zeros((N,3))
        
        R2[:,0] = R[:,0] + Dx@dxu
        R2[:,1] = R[:,1] + Dy@dyu
        R2[:,2] = R[:,2] + Dz@dzu       
        
        pidx = np.zeros(N, dtype=int)
        for i in range(N):
            d = np.sqrt(np.sum((R2[i,:]-G)**2, axis=1))
            pidx[i] = np.argmin(d)
        
        return np.sqrt((((R-G[pidx])**2).sum(axis=1)))
        
    
    def solve_optimization_problem(self, D, Dl, b):
        np.random.seed(1907)
        m = Dl.shape[0]
        n = Dl.shape[1]
        #e2max = 0.2*emax;       
        
        Dsub = D[::20,:]
        bmax = np.max(np.abs(b))
        
        
        Dalt = np.zeros(Dsub.shape)
        balt = np.zeros(Dsub.shape[0])
        
        for i in range(Dalt.shape[0]):
            ix = np.argmax(Dsub[i,:])
            Dalt[i,ix] = 1
            balt[i] = np.abs(b[ix])
        
        x = cp.Variable(n)
        objective = cp.Minimize(cp.norm(Dl@x-b, p=1))
        constraints = []
        
        
        constraints.append(cp.abs(Dalt@x)  <= balt)
        prob = cp.Problem(objective, constraints)
        
        result = prob.solve(solver='SCS')
        
        return x.value
    
    
    def get_key(self, landmark_computer):
        return self.method_key + landmark_computer.get_key()
    





    
    
   
class ErrorComputerChamferFLR_IC:
    
    def __init__(self, gamma=1, err_bounds_param='bounds/ebounds_90.txt', unbiased_features=True):
        self.require_Dmatrix = True
        self.require_landmarks = True
        self.unbiased_features = unbiased_features
        self.gamma = gamma
        self.method_key = 'FLR_IC%.2f%d' % (gamma, unbiased_features)
        
        NLANDMARKS = 51
        if isinstance(err_bounds_param, float):
            self.err_bound_max = err_bounds_param*np.ones((NLANDMARKS*3,1))
            self.err_bound_min = -self.err_bound_max 
            self.method_key += 'err-%f' % err_bounds_param
        elif isinstance(err_bounds_param, str):
            err_bound = np.loadtxt(err_bounds_param)
            self.err_bound_min = err_bound[:,0:1]
            self.err_bound_max = err_bound[:,1:2]
            self.method_key += os.path.basename(err_bounds_param)
        
        self.li_lb = [0,1,2,3,4]
        self.li_rb = [5,6,7,8,9]
        self.li_le = [19,20,21,22,23,24]
        self.li_re = [25,26,27,28,29,30]
        self.li_no = [10,11,12,13,14,15,16,17,18]
        self.li_ul = [31,32,33,34,35,36,37,43,44,45,46,47]
        self.li_ll = [31,37,38,39,40,41,42,47,48,49,50, 43]
    
    
    def compute_error(self, R, G, Dx=None, gt_landmarks=None, li=None, iod=None, li0=None, G0=None):
        Dx = Dx**self.gamma
        Dy = Dx
        Dz = Dx
        
        Dxl = copy.deepcopy(Dx)[li,:]
        Dyl = copy.deepcopy(Dy)[li,:]
        Dzl = copy.deepcopy(Dz)[li,:]
        
        bxl = gt_landmarks[:,0]-R[li,0]
        byl = gt_landmarks[:,1]-R[li,1]
        bzl = gt_landmarks[:,2]-R[li,2]
        
        (Ex, Ey, Ez, fx, fy, fz) = self.create_equality_constaints(Dx, Dy, Dz, bxl, byl, bzl, li)

        iod = np.linalg.norm(gt_landmarks[28,:]-gt_landmarks[19,:])
        
        dxu = self.solve_optimization_problem(Dxl, self.err_bound_min[::3,0], self.err_bound_max[::3,0],   Ex, bxl, fx, iod)
        dyu = self.solve_optimization_problem(Dyl, self.err_bound_min[1::3,0], self.err_bound_max[1::3,0], Ey, byl, fy, iod)
        dzu = self.solve_optimization_problem(Dzl, self.err_bound_min[2::3,0], self.err_bound_max[2::3,0], Ez, bzl, fz, iod)
        
        
        N = R.shape[0]
        R2 = np.zeros((N,3))
        
        R2[:,0] = R[:,0] + Dx@dxu
        R2[:,1] = R[:,1] + Dy@dyu
        R2[:,2] = R[:,2] + Dz@dzu       
        
        pidx = np.zeros(N, dtype=int)
        for i in range(N):
            d = np.sqrt(np.sum((R2[i,:]-G)**2, axis=1))
            pidx[i] = np.argmin(d)
        
        return np.sqrt((((R-G[pidx])**2).sum(axis=1)))
    
    
    def create_equality_constaints(self, Dx, Dy, Dz, diffx, diffy, diffz, li):
        nli = np.array(li)
        Cx_lb = np.ones((1, len(self.li_lb))) @ Dx[nli[self.li_lb], :]
        Cy_lb = np.ones((1, len(self.li_lb))) @ Dy[nli[self.li_lb], :]
        Cz_lb = np.ones((1, len(self.li_lb))) @ Dz[nli[self.li_lb], :]
        dx_lb = np.ones((1, len(self.li_lb))) @ diffx[self.li_lb]
        dy_lb = np.ones((1, len(self.li_lb))) @ diffy[self.li_lb]
        dz_lb = np.ones((1, len(self.li_lb))) @ diffz[self.li_lb]
        
        Cx_rb = np.ones((1, len(self.li_rb))) @ Dx[nli[self.li_rb], :]
        Cy_rb = np.ones((1, len(self.li_rb))) @ Dy[nli[self.li_rb], :]
        Cz_rb = np.ones((1, len(self.li_rb))) @ Dz[nli[self.li_rb], :]
        dx_rb = np.ones((1, len(self.li_rb))) @ diffx[self.li_rb]
        dy_rb = np.ones((1, len(self.li_rb))) @ diffy[self.li_rb]
        dz_rb = np.ones((1, len(self.li_rb))) @ diffz[self.li_rb]
        
        Cx_le = np.ones((1, len(self.li_le))) @ Dx[nli[self.li_le], :]
        Cy_le = np.ones((1, len(self.li_le))) @ Dy[nli[self.li_le], :]
        Cz_le = np.ones((1, len(self.li_le))) @ Dz[nli[self.li_le], :]
        dx_le = np.ones((1, len(self.li_le))) @ diffx[self.li_le]
        dy_le = np.ones((1, len(self.li_le))) @ diffy[self.li_le]
        dz_le = np.ones((1, len(self.li_le))) @ diffz[self.li_le]
        
        Cx_re = np.ones((1, len(self.li_re))) @ Dx[nli[self.li_re], :]
        Cy_re = np.ones((1, len(self.li_re))) @ Dy[nli[self.li_re], :]
        Cz_re = np.ones((1, len(self.li_re))) @ Dz[nli[self.li_re], :]
        dx_re = np.ones((1, len(self.li_re))) @ diffx[self.li_re]
        dy_re = np.ones((1, len(self.li_re))) @ diffy[self.li_re]
        dz_re = np.ones((1, len(self.li_re))) @ diffz[self.li_re]
        
        Cx_no = np.ones((1, len(self.li_no))) @ Dx[nli[self.li_no], :]
        Cy_no = np.ones((1, len(self.li_no))) @ Dy[nli[self.li_no], :]
        Cz_no = np.ones((1, len(self.li_no))) @ Dz[nli[self.li_no], :]
        dx_no = np.ones((1, len(self.li_no))) @ diffx[self.li_no]
        dy_no = np.ones((1, len(self.li_no))) @ diffy[self.li_no]
        dz_no = np.ones((1, len(self.li_no))) @ diffz[self.li_no]
        
        Cx_ul = np.ones((1, len(self.li_ul))) @ Dx[nli[self.li_ul], :]
        Cy_ul = np.ones((1, len(self.li_ul))) @ Dy[nli[self.li_ul], :]
        Cz_ul = np.ones((1, len(self.li_ul))) @ Dz[nli[self.li_ul], :]
        dx_ul = np.ones((1, len(self.li_ul))) @ diffx[self.li_ul]
        dy_ul = np.ones((1, len(self.li_ul))) @ diffy[self.li_ul]
        dz_ul = np.ones((1, len(self.li_ul))) @ diffz[self.li_ul]
        
        Cx_ll = np.ones((1, len(self.li_ll))) @ Dx[nli[self.li_ll], :]
        Cy_ll = np.ones((1, len(self.li_ll))) @ Dy[nli[self.li_ll], :]
        Cz_ll = np.ones((1, len(self.li_ll))) @ Dz[nli[self.li_ll], :]
        dx_ll = np.ones((1, len(self.li_ll))) @ diffx[self.li_ll]
        dy_ll = np.ones((1, len(self.li_ll))) @ diffy[self.li_ll]
        dz_ll = np.ones((1, len(self.li_ll))) @ diffz[self.li_ll]
        
        Ex = np.concatenate((Cx_lb, Cx_rb, Cx_le, Cx_re, Cx_no, Cx_ul, Cx_ll), axis=0)
        Ey = np.concatenate((Cy_lb, Cy_rb, Cy_le, Cy_re, Cy_no, Cy_ul, Cy_ll), axis=0)
        Ez = np.concatenate((Cz_lb, Cz_rb, Cz_le, Cz_re, Cz_no, Cz_ul, Cz_ll), axis=0)
        
        fx = np.concatenate((dx_lb, dx_rb, dx_le, dx_re, dx_no, dx_ul, dx_ll), axis=0)
        fy = np.concatenate((dy_lb, dy_rb, dy_le, dy_re, dy_no, dy_ul, dy_ll), axis=0)
        fz = np.concatenate((dz_lb, dz_rb, dz_le, dz_re, dz_no, dz_ul, dz_ll), axis=0)
        
        return (Ex, Ey, Ez, fx, fy, fz)


    def solve_optimization_problem(self, Dl, emin, emax, E, b, f, iod):
        np.random.seed(1907)
        m = Dl.shape[0]
        n = Dl.shape[1]
        e2min = iod*0.2*np.mean(emin)*np.ones(len(f));       
        e2max = iod*0.2*np.mean(emax)*np.ones(len(f));       

        
        bmax = np.max(np.abs(b))
        #e2max = 0.2*emax;       
        
        x = cp.Variable(n)
        objective = cp.Minimize(cp.sum_squares(x))
        # constraints = [Dl@x-b <= e, E@x-f <= e2]
        constraints = [ Dl@x-b <= emax*iod, emin*iod <= Dl@x-b]

        if self.unbiased_features:
            constraints.append(E@x-f <= e2max)
            constraints.append(e2min <= E@x-f)
        
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
        return "IC"

    def __str__(self):
        return self.get_label()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 09:08:08 2022

@author: v
"""
import os
import sys
import itertools
import numpy as np
from error_computers import ErrorComputerKnown, ErrorComputerChamfer, ErrorComputerLandmarks, ErrorComputerChamferFLR_UC, ErrorComputerChamferFLR_IC, ErrorComputerChamferFLR_NIC2, ErrorComputerChamferFLR_NIC, ErrorComputerDensityAwareChamfer
from error_computers import LandmarkComputerExact, LandmarkComputerAuto
from performance_reporter import PerformanceReporter
iod_ptg = None # 0.2
#dbname = 'db1'
dbname = 'synth_100'
dbname = 'synth_020'
dbname = sys.argv[1]
dbroot  = '/media/v/SSD1TB/data/processed/meshes'
cacheroot = '/media/v/SSD1TB/data/cache/3Drec'

if not os.path.exists(dbroot):
    dbroot = '/offline_data/face/meshes'
    cacheroot = '/offline_data/cache/geometric_error'
"""
which_pts = sys.argv[2]
if which_pts == 'all':
    imin = 10.15
    imean = 10.60
elif which_pts == 'h':
    imin = 0.15
    imean = 0.60
elif which_pts == 'lth':
    imin = 0.12
    imean = 0.50
"""

imean = float(sys.argv[2])

num_lmks = 28 # 51
#num_lmks = 51
num_lmks = 11 # 51
num_lmks = 5# 51
num_lmks =  51
auto_lmks = int(sys.argv[3])
num_lmks = int(sys.argv[4])

# ecs = [ErrorComputerLandmarks()]
ecs = [ErrorComputerChamfer()]
ecs += [ErrorComputerChamferFLR_NIC()]
ecs += [ErrorComputerChamferFLR_NIC(correction='pair-mixed-r2-sqrt')]
ecs += ['NICPRauto']
ecs += ['NICPR2auto']
# ecs += [ErrorComputerChamferFLR_NIC(direction='GR', correction='pair-mixed-r2-sqrt')]
#ecs += [[ErrorComputerChamferFLR_NIC(direction='GR', correction='pair-mixed-r2-sqrt'), ErrorComputerChamferFLR_NIC(correction='pair-mixed-r2-sqrt')]]
#ecs += [ErrorComputerChamferFLR_NIC(correction='pair-mixed-r2-linear')]
#ecs += [ErrorComputerChamferFLR_NIC(correction='trace-mixed-r2-sqrt')]

#ecs += [ErrorComputerChamferFLR_UC()]
#ecs += [ErrorComputerChamferFLR_NIC(correction='pair-mean')]
###ecs += [ErrorComputerChamferFLR_NIC(correction='pair-mean-r2-tmp')]
median = int(sys.argv[5])

#methods = ['Deep3DFace-M09', 'Deep3DFace-M63', '3DI-M09', '3DI-M63', '3DDFAv2-M09', '3DDFAv2-M63', 'INORig-M09', 'INORig-M63', 'meanface' ]
methods = ['3DDFAv2-F01', '3DDFAv2-F50', '3DI-M05-F50', 'Deep3DFace-F10', 'INORig-M05-F50', '3DDFAv2-F10', '3DI-M05-F10', 'Deep3DFace-F01', 'Deep3DFace-F50', 'INORig-M05-F10', 'meanface-F01']
methods = ['3DDFAv2-F01', '3DDFAv2-F50', '3DI-M05-F50', 'Deep3DFace-F10', 'INORig-M05-F50', '3DI-M05-F10', 'Deep3DFace-F01', 'Deep3DFace-F50', 'meanface-F01']
methods = ['3DDFAv2-F01', '3DDFAv2-F50', 'INORig-M05-F10', 'INORig-M05-F50','Deep3DFace-F10', '3DI-M05-F10', '3DI-M05-F50', 'Deep3DFace-F01', 'Deep3DFace-F50', 'meanface-F01']
#methods = ['3DI-M05-F50', '3DI-M05-F10', 'Deep3DFace-F50', 'Deep3DFace-F10', 'Deep3DFace-F01', 'INORig-M05-F50', 'INORig-M05-F10', '3DDFAv2-F50', '3DDFAv2-F01']
methods = ['3DI-M05-F50', '3DI-M05-F10', 'Deep3DFace-F50', 'Deep3DFace-F10', 'Deep3DFace-F01', 'INORig-M05-F50', 'INORig-M05-F10', '3DDFAv2-F50', '3DDFAv2-F01', 'meanface-F01']
methods = ['3DI-M05-F50', '3DI-M05-F10','INORig-M05-F10', 'INORig-M05-F50','Deep3DFace-F50', 'Deep3DFace-F10', 'Deep3DFace-F01', '3DDFAv2-F50', '3DDFAv2-F10','3DDFAv2-F01', 'meanface-F01']
#methods = ['3DI-M05-F50', '3DI-M05-F10', 'Deep3DFace-F50', 'Deep3DFace-F10', 'Deep3DFace-F01', 'INORig-M05-F50', 'INORig-M05-F10', '3DDFAv2-F50', '3DDFAv2-F01', 'meanface-F01']
##methods = ['INORig-M05-F50', 'Deep3DFace-F01']
#methods = ['3DDFAv2-F01', '3DDFAv2-F50', '3DI-M05-F50', 'Deep3DFace-F10', 'INORig-M05-F50', '3DI-M05-F10', 'Deep3DFace-F01', 'Deep3DFace-F50']
methods = ['3DI001-M05-F50',  '3DDFAv2-F50', 'Deep3DFace-F50', 'INORig-M05-F50', 'meanface-F01']
 
#methods = ['3DDFAv2-F01', '3DDFAv2-F50', '3DI-M05-F50',  'INORig-M05-F50',   'Deep3DFace-F01', 'Deep3DFace-F50', 'meanface-F01']
class PoolParams:
    def __init__(self, ec, subj_id, method, lc):
        self.ec = ec
        self.lc = lc 
        self.method = method
        self.subj_id = subj_id

Nsubj = 100 

param_sets = []
for ec in ecs:
    if type(ec) is list:
        continue
    for subj_id in range(Nsubj):
        for method in methods:
            if not auto_lmks:
                param_sets.append(PoolParams(ec, subj_id, method, LandmarkComputerExact()))
            else:
                param_sets.append(PoolParams(ec, subj_id, method, LandmarkComputerAuto()))

#print(len(param_sets))

if auto_lmks:
    pr = PerformanceReporter(dbname, dbroot, cacheroot, landmark_computer=LandmarkComputerAuto(), aligner='auto', num_lmks=num_lmks, iod_ptg=imean, figdir_root='figures') 
    pr0 = PerformanceReporter(dbname, dbroot, cacheroot, landmark_computer=LandmarkComputerAuto(), aligner='auto', iod_ptg=imean, figdir_root='figures') 
else:
    pr = PerformanceReporter(dbname, dbroot, cacheroot, landmark_computer=LandmarkComputerExact(), aligner='exact', num_lmks=num_lmks, iod_ptg=imean, figdir_root='figures') 
    pr0 = PerformanceReporter(dbname, dbroot, cacheroot, landmark_computer=LandmarkComputerExact(), aligner='exact', iod_ptg=imean, figdir_root='figures') 

        
def func(pp):
    subj_fname = 'subj%03d' % pp.subj_id
    if pr.subject_exists(pp.subj_id):
        try:
            if type(pp.ec) is ErrorComputerLandmarks:
                pr0.compute_err_ptwise(subj_fname, pp.ec, pp.method)
            else:
                pr.compute_err_ptwise(subj_fname, pp.ec, pp.method)
        except Exception as e:
            print("problem with %s" % pp.subj_id)
            print(pp.ec)
            print(pp.method)
            print(e)
        # print(subj_fname + ec.method_key)
    else:
        print('no subj')


import multiprocessing as mp
pool = mp.Pool(processes=18)
pool.map(func, param_sets)

#%%

print("=============================================================")
import matplotlib.pyplot as plt




if dbname[:5] == 'synth':
    ecss =  [ErrorComputerKnown()]+ecs
else:
    ecss = ecs
ecs0 = ['comb' if type(ec) is list else ec if type(ec) is str else ec.get_label() for ec in ecss] 
ecs0 = [''] + ecs0
print('\t'+'\t'.join(['%-7s' %  x for x in ecs0]))

if auto_lmks:
    pr = PerformanceReporter(dbname, dbroot, cacheroot, landmark_computer=LandmarkComputerAuto(), aligner='auto', num_lmks=num_lmks, iod_ptg=imean, figdir_root='figures')
else:
    pr = PerformanceReporter(dbname, dbroot, cacheroot, landmark_computer=LandmarkComputerExact(), aligner='exact', num_lmks=num_lmks, iod_ptg=imean, figdir_root='figures')

for err_computer in ecs:
    pr.compute_ptwise_scatterplots_cumulative(Nsubj, err_computer, methods, Npts=25000)

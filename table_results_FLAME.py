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

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dbname', type=str, default='FLAMEdataUC_020')
parser.add_argument('--imean', type=float, default=2.0)
parser.add_argument('--auto_lmks', type=int, default=0)
parser.add_argument('--num_lmks', type=float, default=18)
parser.add_argument('--Nsubjs', type=int, default=100)
parser.add_argument('--dbroot', type=str, default='data/meshes')
parser.add_argument('--cacheroot', type=str, default='data/cache')
parser.add_argument('--num_processes', type=int, default=2)

args = parser.parse_args()
iod_ptg = None 
dbname = args.dbname

imean = args.imean

auto_lmks = args.auto_lmks
num_lmks =  args.num_lmks

ecs = [ErrorComputerKnown(), ErrorComputerLandmarks(), ErrorComputerChamfer()]
median = 0 
methods = ['RingNet-F01', 'DECA-F01', 'meanface_FLAME-F01']

class PoolParams:
    def __init__(self, ec, subj_id, method, lc):
        self.ec = ec
        self.lc = lc 
        self.method = method
        self.subj_id = subj_id
        
Nsubj = args.Nsubjs

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

mt = 'FLAME'

pr = PerformanceReporter(dbname, args.dbroot, args.cacheroot, landmark_computer=LandmarkComputerExact(), aligner='exact', num_lmks=num_lmks, iod_ptg=imean, figdir_root='figures', meshtype=mt, meshtopology='FLAMEcropped') 
pr0 = PerformanceReporter(dbname, args.dbroot, args.cacheroot, landmark_computer=LandmarkComputerExact(), aligner='exact', iod_ptg=imean, figdir_root='figures', meshtype=mt, meshtopology='FLAMEcropped') 

        
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


import multiprocessing as mp
pool = mp.Pool(processes=args.num_processes)
pool.map(func, param_sets)

print("=============================================================")
import matplotlib.pyplot as plt

N0 = 0

if dbname.find('Floren') >= 0:
    N0=1

ecss = ecs
ecs0 = ['comb' if type(ec) is list else ec if type(ec) is str else ec.get_label() for ec in ecss] 
ecs0 = [''] + ecs0
print('\t'+'\t'.join(['%-7s' %  x for x in ecs0]))


for method in methods:
    errs = []
    for err_computer in ecss: 

        if type(err_computer) is ErrorComputerLandmarks:
            errs.append(89*np.mean(np.array(pr0.compute_avg_err_on_dataset(err_computer, method, Nsubj, median=median, N0=N0))))
        else:
            errs.append(89*np.mean(np.array(pr.compute_avg_err_on_dataset(err_computer, method, Nsubj, median=median, N0=N0))))

    print('\t'.join(['%-7s' % method] + ['%-7s' %  x for x in ['%.2f' % (x) for x in errs]]))

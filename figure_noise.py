#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 09:08:08 2022

@author: v
"""
import os
import sys
import numpy as np
from error_computers import ErrorComputerLandmarks, ErrorComputerChamferFLR_UC, ErrorComputerChamferFLR_IC, ErrorComputerChamferFLR_NIC
from error_computers import LandmarkComputerScaled, LandmarkComputerNoisy, LandmarkComputerTranslated
from performance_reporter import PerformanceReporter

dbname = sys.argv[1]
dbroot  = '/media/v/SSD1TB/data/processed/meshes'
cacheroot = '/media/v/SSD1TB/data/cache/3Drec'

if not os.path.exists(dbroot):
    dbroot = '/offline_data/face/meshes'
    cacheroot = '/offline_data/cache/geometric_error'


for noisetype in ['translation', 'scaling', 'gaussian']:
    Nsubj = 50

    if len(sys.argv) >= 6:
        Nsubj = int(sys.argv[5])

    ecs = [ErrorComputerLandmarks()]
    ecs += [ErrorComputerChamferFLR_NIC(gamma=1)]
    ecs += [ErrorComputerChamferFLR_NIC(gamma=1, correction='pair-mixed-r2-sqrt')]

    if noisetype == 'translation':
        landmark_computer_template = LandmarkComputerTranslated
        lnss = np.arange(0, 0.0161, 0.0020)
    elif noisetype == 'scaling':
        landmark_computer_template = LandmarkComputerScaled
        lnss = np.arange(0, 0.0201, 0.0020)
    elif noisetype == 'gaussian':
        landmark_computer_template = LandmarkComputerNoisy
        lnss = np.arange(0, 0.0161, 0.0020)

    #methods = ['Deep3DFace-M09', 'Deep3DFace-M63', '3DI-M09', '3DI-M63', '3DDFAv2-M09', '3DDFAv2-M63', 'INORig-M09', 'INORig-M63', 'meanface' ]
    methods = ['3DDFAv2-F01', '3DDFAv2-F50', '3DI-M05-F50', 'Deep3DFace-F10', 'INORig-M05-F50', '3DI-M05-F10', 'Deep3DFace-F01', 'Deep3DFace-F50', 'meanface-F01']


    class PoolParams:
        def __init__(self, ec, subj_id, method, ns):
            self.ec = ec
            self.ns = ns
            self.method = method
            self.subj_id = subj_id

    param_sets = []
    for ec in ecs:
        for subj_id in range(Nsubj):
            for method in methods:
                for ns in lnss:
                    param_sets.append(PoolParams(ec, subj_id, method, ns))

    print(len(param_sets))

    def func(pp):
        pr = PerformanceReporter(dbname, dbroot, cacheroot, landmark_computer=landmark_computer_template(pp.ns), aligner='exact') 

        subj_fname = 'subj%03d' % pp.subj_id
        pr.compute_err_ptwise(subj_fname, pp.ec, pp.method)

    import multiprocessing as mp
    pool = mp.Pool(processes=26)
    pool.map(func, param_sets)

    import matplotlib.pyplot as plt


    from matplotlib import rc

    rc('text', usetex=True)
    rc('font', family='serif')

    plt.figure(figsize=(4.5,3.0))

    ays = {}


    for err_computer in ecs:
        ays = []
        for method in methods:
            ys = []
            prs = [PerformanceReporter(dbname, dbroot, cacheroot, landmark_computer=landmark_computer_template(ns), aligner='exact') for ns in lnss]
            
            for pr in prs:
                ys.append(np.mean(pr.compute_avg_err_on_dataset(err_computer, method, Nsubj)))

            ys = np.array(ys)
            ys /= ys[0]
            ays.append(ys)
        
        y = np.array(ays)
        plt.errorbar(89*lnss, y.mean(axis=0), np.std(y, axis=0)/np.sqrt(y.shape[0]), label=err_computer.get_label())

        plt.legend()

    plt.title('Performance variation vs. noise (%s)' % noisetype)
    plt.ylim((1, 1.6))
    plt.ylabel('$\epsilon({\sigma})/\epsilon(0)$')
    plt.xlabel('$\sigma$')
    plt.tight_layout()

    import os
    if not os.path.exists('figures'):
        os.mkdir('figures')

    plt.savefig('figures/fig_noise-%s.pdf' % noisetype)
    plt.savefig('figures/fig_noise-%s.png' % noisetype)

    #plt.show()





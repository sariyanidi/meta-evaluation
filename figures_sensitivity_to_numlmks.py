"""
Created on Mon Dec  5 09:08:08 2022
@author: v
"""
import sys
import os
import numpy as np
import itertools
from error_computers import ErrorComputerKnown, ErrorComputerChamfer, ErrorComputerChamferFLR_UC, ErrorComputerChamferFLR_NIC
from error_computers import LandmarkComputerAuto, LandmarkComputerExact
from performance_reporter import PerformanceReporter

import matplotlib.pyplot as plt

dbname = sys.argv[1]
dbroot  = '/media/v/SSD1TB/data/processed/meshes'
cacheroot = '/media/v/SSD1TB/data/cache/3Drec'

if not os.path.exists(dbroot):
    dbroot = '/offline_data/face/meshes'
    cacheroot = '/offline_data/cache/geometric_error'

methods = ['3DDFAv2-F01',  '3DI-M05-F50', 'Deep3DFace-F10', 'INORig-M05-F50', '3DI-M05-F10', 'Deep3DFace-F01', 'Deep3DFace-F50']


ecs = [ErrorComputerKnown()]

ecs += [ErrorComputerChamferFLR_NIC(gamma=1)]
ecs += [ErrorComputerChamferFLR_NIC(gamma=1, correction='pair-mixed-r2-sqrt')]

aligner = 'auto' # this is for the rigid aligner
auto_lmks = True

#%%


class PoolParams:
    def __init__(self, ec, subj_id, method, lc, num_lmks):
        self.ec = ec
        self.lc = lc 
        self.method = method
        self.subj_id = subj_id
        self.num_lmks = num_lmks

Nsubj = 10

all_num_lmks = [5,   18,  32,   51]
param_sets = []
for subj_id in range(Nsubj):
    for method in methods:
        for ec in ecs:
            #for num_lmks in [5, 7, 11, 14, 16, 18, 20, 28, 32, 33, 37, 39, 47, 49, 51]:
            for num_lmks in all_num_lmks:
                if not auto_lmks:
                    param_sets.append(PoolParams(ec, subj_id, method, LandmarkComputerExact(), num_lmks))
                else:
                    param_sets.append(PoolParams(ec, subj_id, method, LandmarkComputerAuto(), num_lmks))

print(len(param_sets))

        
def func(pp):
    if auto_lmks:
        pr = PerformanceReporter(dbname, dbroot, cacheroot, landmark_computer=pp.lc, aligner='auto', num_lmks=pp.num_lmks) 
    else:
        pr = PerformanceReporter(dbname, dbroot, cacheroot, landmark_computer=pp.lc, aligner='exact', num_lmks=pp.num_lmks) 

    subj_fname = 'subj%03d' % pp.subj_id
    pr.compute_err_ptwise(subj_fname, pp.ec, pp.method)
        # print(subj_fname + ec.method_key)


import multiprocessing as mp
pool = mp.Pool(processes=10)
pool.map(func, param_sets)
print('done')




# plt.figure(0, figsize=(4,4))
plt.figure()

avgr2s = []
stdr2s = []

#all_num_lmks =  [5, 7, 11, 14, 16, 18, 20, 28, 32, 33, 37, 39, 47, 49, 51]
for num_lmks in all_num_lmks:
    print(num_lmks)
    r2s = []
    
    for method in methods:
        landmark_computer = LandmarkComputerAuto()
        if aligner == 'exact':
            landmark_computer = LandmarkComputerExact()
        
        pr = PerformanceReporter(dbname, dbroot, cacheroot, landmark_computer=landmark_computer, aligner=aligner, figdir_root='figures', num_lmks=num_lmks)
        r2s.append(pr.compute_intersubj_r2_on_dataset(ec, method, Nsubj=Nsubj, save_figure=False))

    avgr2s.append(np.mean(r2s))
    stdr2s.append(np.std(r2s)/len(r2s))

# plt.plot(avgr2s)


plt.errorbar(all_num_lmks, avgr2s, stdr2s)
plt.xlabel('number of landmarks')
plt.ylabel('R2 score (mean +/ std err)')
plt.savefig('figures/sensitivity_to_numlmks.pdf')
plt.show()



"""
Created on Mon Dec  5 09:08:08 2022
@author: v
"""
import os
import itertools
from error_computers import ErrorComputerChamfer, ErrorComputerChamferFLR_UC, ErrorComputerChamferFLR_NIC
from error_computers import LandmarkComputerAuto, LandmarkComputerExact, LandmarkComputerNoisy
from performance_reporter import PerformanceReporter

import matplotlib.pyplot as plt

dbname = 'db1'
dbroot  = '/media/v/SSD1TB/data/processed/meshes'
cacheroot = '/media/v/SSD1TB/data/cache/3Drec'

if not os.path.exists(dbroot):
    dbroot = '/offline_data/face/synth/meshes'
    cacheroot = '/offline_data/cache/geometric_error'

methods = ['Deep3DFace-M09', 'Deep3DFace-M63', '3DI-M09', '3DI-M63', '3DDFAv2-M09', '3DDFAv2-M63', 'INORig-M09', 'INORig-M63', 'meanface']
ecs = [ErrorComputerChamfer(), ErrorComputerChamferFLR_UC(), ErrorComputerChamferFLR_NIC()]
aligners = ['exact', 'auto']

# plt.figure(0, figsize=(4,4))
plt.figure()

for method, ec, aligner in itertools.product(methods, ecs, aligners):
    landmark_computer = LandmarkComputerAuto()
    if aligner == 'exact':
        landmark_computer = LandmarkComputerExact()
    
    pr = PerformanceReporter(dbname, dbroot, cacheroot, landmark_computer=landmark_computer, aligner=aligner, figdir_root='figures')
    pr.compute_r2_subjectwise_on_dataset(ec, method, Nsubj=100, save_figure=True)


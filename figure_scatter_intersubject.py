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
from performance_reporter import PerformanceReporter, method_translator
dbtype = sys.argv[1]


if dbtype == 'synth':
    dbnames = ['synth_020'] #, 'synth_060', 'synth_100'] 
elif dbtype == 'synthb':
    dbnames = ['synthb_020'] 
elif dbtype == 'synthbb':
    dbnames = ['synthbb_020'] 
elif dbtype == 'ssynth':
    dbnames = ['synth_020', 'synthbbb_020'] 
elif dbtype == 'synthbbb':
    dbnames = ['synthbbb_020'] 
elif dbtype == 'Florence':
    dbnames = ['Florence_020', 'Florence_060', 'Florence_100'] 
elif dbtype == 'BU4DFE':
    dbnames = ['BU4DFE_020', 'BU4DFE_060', 'BU4DFE_100'] 
dbroot  = '/media/v/SSD1TB/data/processed/meshes'
cacheroot = '/media/v/SSD1TB/data/cache/3Drec'

if not os.path.exists(dbroot):
    dbroot = '/offline_data/face/meshes'
    cacheroot = '/offline_data/cache/geometric_error'

imean = float(sys.argv[2])
auto_lmks = int(sys.argv[3])
num_lmks = int(sys.argv[4])
median = int(sys.argv[5])
mean_or_median = 'mean'
if median:
    mean_or_median = 'median'
#median = False 

odir = 'figures/TMP%s-%s/%.4f/' % (dbtype, mean_or_median, imean)
if not os.path.exists(odir):
    os.makedirs(odir)



# ecs = [ErrorComputerLandmarks()]
ecs = [ErrorComputerChamfer()]
ecs += [ErrorComputerLandmarks()]
ecs += [[ErrorComputerChamfer(), ErrorComputerChamfer(direction='GR')]]
ecs += [[ErrorComputerChamferFLR_NIC(), ErrorComputerChamferFLR_NIC(direction='GR')]]
#ecs += [ErrorComputerChamferFLR_NIC()]
ecs += [ErrorComputerChamferFLR_NIC(correction='pair-mixed-r2-sqrt')]

ecs = [ErrorComputerChamfer()]
methods = [ '3DDFAv2-F50', '3DI001-M05-F50', 'INORig-M05-F50', 'Deep3DFace-F50', 'meanface-F01']
methods = [ '3DDFAv2-F50', '3DI-M05-F50', 'INORig-M05-F50', 'Deep3DFace-F50', 'meanface-F01']
do_exit = True 
do_exit = False 
for m in range(len(methods)):
    figpath = '%s/subjectwise_err_%d.pdf' % (odir, m+1)
    if not os.path.exists(figpath):
        do_exit = False
        break
if do_exit:
    exit()

Nsubj = 50 # 100
Nsubj = 100


import matplotlib.pyplot as plt




if dbtype[:5] == 'synth' or dbtype[:5] == 'ssynt':
    ecss =  [ErrorComputerKnown()]+ecs
else:
    ecss = ecs



errs_per_ec = {x: [] for x in range(len(ecss))}
errs_per_ecm = {x: [] for x in range(len(ecss))}

for ei, err_computer in enumerate(ecss): 
    errs_per_ecm[ei] = []
    for method in methods:
        ecm = []
        for dbname in dbnames:
            
            if auto_lmks:
                pr = PerformanceReporter(dbname, dbroot, cacheroot, landmark_computer=LandmarkComputerAuto(), aligner='auto', num_lmks=num_lmks, iod_ptg=imean, figdir_root='figures')
                pr0 = PerformanceReporter(dbname, dbroot, cacheroot, landmark_computer=LandmarkComputerAuto(), aligner='auto', num_lmks=18,  iod_ptg=imean, figdir_root='figures')
            else:
                pr= PerformanceReporter(dbname, dbroot, cacheroot, landmark_computer=LandmarkComputerExact(), aligner='exact', num_lmks=num_lmks, iod_ptg=imean, figdir_root='figures')
                pr0 = PerformanceReporter(dbname, dbroot, cacheroot, landmark_computer=LandmarkComputerExact(), aligner='exact',  num_lmks=18, iod_ptg=imean, figdir_root='figures')

            
            if type(err_computer) is ErrorComputerLandmarks:
                tmp = pr0.compute_avg_err_on_dataset(err_computer, method, Nsubj, median=median)
            else:
                tmp = pr.compute_avg_err_on_dataset(err_computer, method, Nsubj, median=median)
            errs_per_ec[ei] += tmp
            ecm += tmp
        errs_per_ecm[ei].append(ecm)
    

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures



Ke = len(errs_per_ec)
ix = 0
rng = (89*min([min(errs_per_ec[x]) for x in errs_per_ec]), 
        89*max([max(errs_per_ec[x]) for x in errs_per_ec]))
rng = (1,  89*max([max(errs_per_ec[x]) for x in errs_per_ec]))


if dbtype[:5] == 'synth' or dbtype[:5] == 'ssynt':
    pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)]#, (0,4), (2, 1), (1,2)]
    pairs = [(0, 1), (0, 2), (0, 3), (0, 4)]#, (0,4), (2, 1), (1,2)]
    pairs = [(0, 1), (0, 2)]#, (0,4), (2, 1), (1,2)]
    pairs = [(0, 1), (0, 2), (0, 3)]#, (0,4), (2, 1), (1,2)]
    pairs = [(0, 1), (0, 2), (0, 3), (0, 4)]#, (0,4), (2, 1), (1,2)]
    pairs = []
    for ei in range(len(ecs)):
        pairs.append((0, ei+1))
    # pairs = [(0, 1)]#, (0,4), (2, 1), (1,2)]
else:
    pairs = [(2, 0), (3, 0)]


import random
random.seed(1907)
fix=0

colors = list(iter([plt.cm.tab10(i) for i in range(10)]))


ix = list(range(len(errs_per_ecm[0][0])))
random.shuffle(ix)
# ix = ix[:144]

from sklearn.linear_model import LinearRegression

plt_titles = {1: '(a)', 2: '(b)', 3: '(c)', 4: '(d)', 5: '(e)', 6: '(f)'}
for ei, ej in pairs:
    plt.figure(fix, figsize=(3,3))
    forxs = 89*np.array([errs_per_ecm[ei][m] for m in range(len(errs_per_ecm[ei]))]).reshape(-1,)
    forys = 89*np.array([errs_per_ecm[ej][m] for m in range(len(errs_per_ecm[ej]))]).reshape(-1,)

    xmin = 0.5 #np.min(forxs);
    xmax = 5.5 #np.max(forxs);
    ymin = np.min(forys);
    ymax = np.max(forys);

    fix += 1
    if ei == ej:
        continue 
    xs = []
    ys = []
    print('=============================================')
    slopes = []
    for m in range(len(errs_per_ecm[ei])):
        xs += errs_per_ecm[ei][m]
        ys += errs_per_ecm[ej][m]
        x = 89*np.array(errs_per_ecm[ei][m]).reshape(-1,1)#[ix,:]
        y = 89*np.array(errs_per_ecm[ej][m]).reshape(-1,1)#[ix,:]
        plt.scatter(x[ix], y[ix], alpha=0.35, color=colors[m], s=24)
        
        rngx = (xmin, xmax)
        rngy = (ymin, ymax)

        if type(ecss[ei]) is list:
            xlbl = 'comb'
        elif type(ecss[ei]) is str:
            xlbl = method_translator[ecss[ei]]
        else:
            xlbl = ecss[ei].get_label()

        if type(ecss[ej]) is list:
            ttl = 'comb'
        elif type(ecss[ej]) is str:
            ttl = method_translator[ecss[ej]]
        else:
            ttl = ecss[ej].get_label()

        plt.xlabel('True per-subject error')
        plt.ylabel('Est. per-subject error')
        plt.title('%s %s' % (plt_titles[fix], ttl))
        #rng = (min(errs_per_ec[0]), max(errs_per_ec[0]))

        mdl = LinearRegression(fit_intercept=False)
        mdl.fit(x, y)
        slope = mdl.coef_[0]

        print('%s: %.2f' % (methods[m], slope))
        plt.plot([rngx[0], rngx[1]], [slope*(rngx[0]), slope*(rngx[1])], color=colors[m], alpha=1)
        slopes.append(slope)

    mslope = np.max(slope)
    plt.xlim(rngx)
    plt.ylim((mslope*rngx[0], mslope*rngx[1]))
    rate_of_inconsistency = np.std(slopes)/np.mean(slopes)



    xs = 89*np.array(xs).reshape(-1,1)
    ys = 89*np.array(ys).reshape(-1,1)
    model = LinearRegression(fit_intercept=False)
    model.fit(xs, ys)
    score = model.score(xs, ys)

    xmin = rngx[0]
    xmax = rngx[1]
    ymin = mslope*rngx[0]
    ymax = mslope*rngx[1]
    
    dx = xmax-xmin
    dy = ymax-ymin
    scorename = 'R^2'
    plt.text(xmin+dx*0.075, ymax-dy*0.125, r"$%s=%.2f$" % (scorename, score), fontsize=12)
    plt.text(xmax-dx*0.33, ymin+dy*0.1, r"$\eta=%.1f%%$" % (100*rate_of_inconsistency), fontsize=12)


    #plt.legend(methods)
    """
    x = 89*np.array(errs_per_ec[ei]).reshape(-1,1)[ix,:]
    y = 89*np.array(errs_per_ec[ej]).reshape(-1,1)[ix,:]
    plt.scatter(x, y, alpha=0.1)
    plt.xlabel(ecss[ei].get_label())
    plt.ylabel(ecss[ej].get_label())
    #rng = (min(errs_per_ec[0]), max(errs_per_ec[0]))
    plt.xlim(rng)
    plt.ylim(rng)
    poly_reg = PolynomialFeatures(degree = 2)
    x_poly = poly_reg.fit_transform(x)
    poly_reg.fit(x_poly, y)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(x_poly, y)
    """


    ## Visualising the Polynomial Regression results
    #plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color = 'blue')

    plt.tight_layout()
    figpath = '%s/subjectwise_err_%d.pdf' % (odir, fix)
    plt.savefig(figpath)
# plt.show()
#    break





"""
    #plt.subplot(1, len(ecs), idx)
    #plt.hist(danz, label=err_computer.get_label())
    #plt.title(err_computer.get_label())
    #plt.xlim((0, 0.3))
    data.append(danz)
# plt.boxplot(data, whis=(0,100))
fig, ax1 = plt.subplots(1,1)
ax1.violinplot(data, showmeans=True)#, whis=(0,100))

ax1.set_xticks([x+1 for x in list(range(len(ecs0[2:])))])
ax1.set_xticklabels(ecs0[2:], rotation=45, fontsize=8)
#plt.show()

plt.ylim((0,  0.3))
plt.savefig('figures/violin/%s-%.2f-%d.png' % (dbname, imean, auto_lmks))
"""




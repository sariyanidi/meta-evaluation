import matplotlib.pyplot as plt
import cvxpy as cp
import numpy as np
import os

import itertools
import matplotlib as mpl
import matplotlib.cm as cm

from error_computers import LandmarkComputerAuto, LandmarkComputerExact, ErrorComputerKnown, ErrorComputerLandmarks
from scipy.spatial.distance import pdist, squareform

from sklearn.linear_model import LinearRegression, Lasso
from scipy import stats

method_translator = {'NICPRauto': 'NICP',
        'NICPR2auto': 'LP+NICP'}

def procrustes(X, Y, scaling=True, reflection='best'):
    """
    Taken from https://github.com/patrikhuber/fg2018-competition
    A port of MATLAB's `procrustes` function to Numpy.
    Code from: https://stackoverflow.com/a/18927641.
    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.
        d, Z, [tform] = procrustes(X, Y)
    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.
    scaling
        if False, the scaling component of the transformation is forced
        to 1
    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.
    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()
    Z
        the matrix of transformed Y-values
    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y
    """

    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA ** 2

        # transformed coords
        Z = normX * traceTA * np.dot(Y0, T) + muX
        # print(Y)

    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)

    # transformation values
    tform = {'rotation': T, 'scale': b, 'translation': c}

    return d, Z, tform



def interpolate_missing_pixels(
        image: np.ndarray,
        mask: np.ndarray,
        method: str = 'cubic',
        fill_value: int = 0
):
    """
    :param image: a 2D image
    :param mask: a 2D boolean image, True indicates missing values
    :param method: interpolation method, one of
        'nearest', 'linear', 'cubic'.
    :param fill_value: which value to use for filling up data outside the
        convex hull of known pixel values.
        Default is 0, Has no effect for 'nearest'.
    :return: the image with missing values interpolated
    """
    from scipy import interpolate

    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = image[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]

    interp_values = interpolate.griddata(
        (known_x, known_y), known_v, (missing_x, missing_y),
        method=method, fill_value=fill_value
    )

    interp_image = image.copy()
    interp_image[missing_y, missing_x] = interp_values

    return interp_image



class PerformanceReporter:
    
    def __init__(self, dbname, dbdir, cacheroot, landmark_computer, aligner='exact', figdir_root=None, num_lmks=51, lrigid=[13, 19, 28, 31, 37],
            iod_ptg=None, rigid_alignment=True, enforce_MeqN=False, meshtype='BFM', meshtopology='BFMcropped'):
        
        self.dbname = dbname
        self.dbdir = dbdir
        self.aligner = aligner
        self.cacheroot = cacheroot + '/' + self.dbname
        self.figdir_root = figdir_root
        self.num_lmks = num_lmks
        self.rigid_alignment = rigid_alignment
        self.enforce_MeqN = enforce_MeqN
        self.lrigid = lrigid
        self.meshtopology = meshtopology
        
        
        self.iod_ptg = iod_ptg
        
        self.cpts = None
        
        if self.figdir_root is not None:
            if not os.path.exists(self.figdir_root):
                os.mkdir(self.figdir_root)
            
            self.figdir_root += '/' + dbname
        
            if not os.path.exists(self.figdir_root):
                os.mkdir(self.figdir_root)
                        
        # Hierarchy of landmarks
        all_lis = [[13, 19, 28, 31, 37],
                    [19, 22, 25, 28, 31, 37],
                    [13, 19, 22, 25, 28, 31, 37],
                    [0, 4, 5, 9, 13, 19, 22, 25, 28, 31, 37],
                    [0, 4, 5, 9, 10, 13, 14, 18, 19, 22, 25, 28, 31, 37],
                    [0, 2, 4, 5, 7, 9, 10, 13, 14, 18, 19, 22, 25, 28, 31, 37],
                    [0, 2, 4, 5, 7, 9, 10, 13, 14, 18, 19, 22, 25, 28, 31, 34, 37, 40],
                    [0, 2, 4, 5, 7, 9, 10, 13, 14, 18, 19, 22, 25, 28, 31, 34, 37, 40, 45, 49],
                    [0, 2, 4, 5, 7, 9, 10, 13, 14, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 37, 40, 45, 49],
                    [0, 2, 4, 5, 7, 9, 10, 13, 14, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 37, 39, 40, 41, 45, 49],
                    [0, 2, 4, 5, 7, 9, 10, 13, 14, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 37, 39, 40, 41, 45, 49],
                    [0, 2, 4, 5, 7, 9, 10, 13, 14, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 37, 39, 40, 41, 44, 45, 46, 48, 49, 50],
                    [0, 2, 4, 5, 7, 9, 10, 11, 12, 13, 14, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 37, 39, 40, 41, 44, 45, 46, 48, 49, 50],
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 48, 49, 50],
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]]
        
        self.lis_map = {len(li): li for li in all_lis}
        # the keys of the map above are = [5, 7, 11, 14, 16, 18, 20, 28, 32, 33, 37, 39, 47, 49, 51]
        
        liBFM = [19106,19413,19656,19814,19981,20671,20837,20995,21256,21516,8161,8175,8184,8190,6758,7602,8201,8802,9641,1831,3759,5049,6086,4545,3515,10455,11482,12643,14583,12915,11881,5522,6154,7375,8215,9295,10523,10923,9917,9075,8235,7395,6548,5908,7264,8224,9184,10665,8948,8228,7508] 
        # liFLAME = [729,1560,4,1832,855,71,1973,0,833,44,1707,1712,1670,1714,1281,1323,1710,570,509,1068,1023,1124,1770,936,976,1943,340,309,296,154,152,1248,1342,1303,1699,550,589,476,676,725,1672,1474,1425,1374,1314,1691,561,475,714,1678,1463] # We don't use this anymore
        liFLAME = [748,595,1250,1350,1352,1356,1359,1382,1537,1541,1549,1552,1556,1558,602,23,78,597,1339,4,1594,720,50,1732,0,705,25,1474,1478,1439,1480,1075,1112,1476,450,407,884,863,926,1538,785,825,1705,256,239,226,117,115,1047,1131,1097,1467,435,469,379,548,587,1441,1245,1206,1163,1103,1459,441,378,582,1447,1240]
        liPRNet = [22396,27792,33074,38077,41382,42506,42234,42101,42986,42132,42307,42627,41547,38268,33277,28003,22621,10877,9168,8444,8702,9203,9246,8763,8529,9281,11016,13407,16851,20049,22509,26188,26682,27175,26693,26209,15325,13611,13866,14857,15589,15826,14908,13931,13694,15424,15907,15652,33375,30926,30014,30250,30021,30949,33413,34699,35119,35115,35110,34680,33158,31838,31618,31849,33412,33617,33395,33608]
        liExpressionNet = [21490,21741,21303,20786,40505,41616,42365,42991,43507,44024,44659,45422,46524,32985,32417,31943,32156,37594,38040,38265,38390,38510,38986,39105,39227,39472,39845,8161,8175,8184,8190,6758,7602,8201,8802,9641,1830,3759,5049,6086,4545,3515,10453,11480,12641,14581,12913,11879,5522,6154,7375,8215,9295,10521,10921,9917,9075,8235,7395,6548,5908,7264,8224,9184,10663,8948,8228,7388] 

        
        self.mesh2li = {'BFMcropped': liBFM, 'FLAMEcropped': liFLAME[17:], 'PRNet': liPRNet[17:], 'ExpressionNet': liExpressionNet[17:]}
        
        G0 = np.loadtxt('data/meanface.txt')
        meshname = 'BFMcropped'

        if self.dbname.find('FLAME') >= 0 or meshtype == 'FLAME':
            G0 = np.loadtxt('data/meanface_FLAME.txt')
            cidx = np.loadtxt('idxs/FLAME_face.txt').astype(int)
            G0 = G0[cidx,:]
            meshname = 'FLAMEcropped'

        elif meshtype == 'PRNet':
            G0 = np.loadtxt('data/meanface_PRNet.txt')
            meshname = 'PRNet'

        elif meshtype == 'ExpressionNet':
            G0 = np.loadtxt('data/meanface_ExpressionNet.txt')
            meshname = 'ExpressionNet'

        self.G0 = G0
        self.li0 = self.mesh2li[meshname]
        
        curli = self.mesh2li[meshname]
        eoc1 = curli[28]
        eoc2 = curli[19]       
        iod = np.linalg.norm(G0[eoc1,:]-G0[eoc2,:])
        
        if self.iod_ptg is not None:
            # ref_lis = all_lis[5]
            ref_lis = [0, 2, 4, 5, 7, 9, 20, 21, 23, 24, 26, 27, 29, 30, 19, 22, 25, 28, 13, 14, 18, 31, 33, 34, 35, 37, 44, 45, 46, 39, 40, 41, 49, 48, 50]
            adists = []
            for i in range(len(ref_lis)):
                dists = np.sqrt(np.sum(((G0[curli[ref_lis[i]],:]-G0)**2), axis=1))
                adists.append(dists)
            
            mean_dists = np.mean(np.array(adists),axis=0)#/np.median(np.array(adists),axis=0)
            mean_dists -= mean_dists.min()

            adists = []
            for i in range(len(curli)):
                dists = np.sqrt(np.sum(((G0[curli[i],:]-G0)**2), axis=1))
                adists.append(dists)

            min_dists = np.min(np.array(adists),axis=0)#/np.median(np.array(adists),axis=0)
            
            self.cpts = []
            
            if self.iod_ptg is not None:
                self.cpts += np.where(mean_dists < iod*(1.0*self.iod_ptg))[0].tolist() # madists[np.where(madists < 0.2)]
                self.cpts += np.where(min_dists < iod*1.0*self.iod_ptg)[0].tolist() # madists[np.where(madists < 0.2)]
                self.cpts = np.unique(self.cpts)

            """
            plt.figure(figsize=(5, 6.25))
            plt.plot(G0[self.cpts,0], G0[self.cpts,1], '.', markersize=1)
            plt.axis('off')
            np.savetxt(f'figures/pt_idx-{len(self.cpts)}.txt', self.cpts)
            plt.savefig('figures/mesh%05d.pdf' % len(self.cpts))
            plt.show()
            dist_weights = mean_dists+min_dists
            
            self.cpts = list(set(self.cpts))
            print(len(self.cpts))
            ss = 2
            plt.figure(figsize=(10, 7))
            plt.subplot(1,2,1)
            plt.plot(G0[self.cpts[::ss], 0], G0[self.cpts[::ss], 1], '.')
            plt.ylim((-1, 1))
            plt.xlim((-.7, .7))

            plt.subplot(1,2,2)
            
            # plt.plot(G0[self.cpts[::ss], 2], G0[self.cpts[::ss], 1], '.')
            # plt.figure(figsize=(5, 7))
            
            plt.scatter(G0[::ss,0], G0[::ss,1], c=dist_weights[::ss], cmap='jet')
            plt.ylim((-1, 1))
            plt.xlim((-.7, .7))
            """
                
        self.tail = ''
        if self.num_lmks != 51:
            self.tail = 'L%d' % self.num_lmks
            
        if self.enforce_MeqN:
            self.tail += 'MN'
            
        if not os.path.exists(self.cacheroot):
            os.makedirs(self.cacheroot, exist_ok=True)
        
        self.landmark_computer = landmark_computer
        
    
    def compute_r2_ptwise2(self, subj_fname, err_computer1, err_computer2, rec_method):
        ref_err_computer = ErrorComputerKnown()
        ref_err_ptwise = self.compute_err_ptwise(subj_fname, ref_err_computer, rec_method)
        err_ptwise1 = self.compute_err_ptwise(subj_fname, err_computer1, rec_method)
        err_ptwise2 = self.compute_err_ptwise(subj_fname, err_computer2, rec_method)
        err_ptwise = (err_ptwise1+err_ptwise2)/2
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(ref_err_ptwise, err_ptwise)

        #initiate linear regression model
        model = LinearRegression()
        model.fit(ref_err_ptwise.reshape(-1,1), err_ptwise.reshape(-1,1))
        r_squared = model.score(ref_err_ptwise.reshape(-1, 1), err_ptwise.reshape(-1,1))
            
        # plt.scatter(ref_err_ptwise, err_ptwise, alpha=0.05)
        # import scipy.stats
        # print(scipy.stats.pearsonr(ref_err_ptwise, err_ptwise))
        # np.savetxt('x.txt', ref_err_ptwise)
        # np.savetxt('y.txt', err_ptwise)
        # plt.show()
        return r_squared#**2
    
    
    def compute_ptwise_scatterplots_cumulative(self, Nsubj, err_computer, rec_methods, Npts=10000):
        ref_err_computer = ErrorComputerKnown()
        ref_errs = []
        errs = []
        import random
        
        for rec_method in rec_methods:
            for subj_id in range(Nsubj):
                if not self.subject_exists(subj_id):
                    continue
                subj_fname = 'subj%03d' % subj_id
                mm = False
                if type(ref_err_computer) is str:
                    mm = True
                ref_err_ptwise = self.compute_err_ptwise(subj_fname, ref_err_computer, rec_method, mm=False)
                errs_ptwise = self.compute_err_ptwise(subj_fname, err_computer, rec_method, mm=mm)
                   
                if self.cpts is not None:
                    ix = list(range(len(self.cpts)))
                else:
                    ix = list(range(len(errs_ptwise)))
                random.shuffle(ix)
                ix = ix[:2000]
                    
                ref_errs += ref_err_ptwise[ix].tolist()
                errs += errs_ptwise[ix].tolist()
        
        
        ix = list(range(len(ref_errs)))
        random.shuffle(ix)
        ix = np.array(ix, dtype=int)
        
        errs = 89*np.array(errs)[ix[:Npts]]
        ref_errs = 89*np.array(ref_errs)[ix[:Npts]]
        
        plt.figure(figsize=(3,3))
        plt.scatter(ref_errs, errs, s=16, alpha=0.0075)
        mn = 0  #np.percentile(ref_errs, 1)
        mx = 6 # np.percentile(ref_errs, 99)
        plt.ylim((mn, mx))
        plt.xlim((mn, mx))
        # plt.xlabel('Error (known top.)')
        plt.xlabel('True error (mm)')
        plt.ylabel('Estimated error')
        if type(err_computer) is str:
            plt.title('%s' % method_translator[err_computer])
        else:
            plt.title('%s' % err_computer.get_label())
        # plt.title('Error vs. estimated error (%d pts)' % len(errs_ptwise))
        
        
        model = LinearRegression(fit_intercept=False)
        model.fit(np.array(ref_errs).reshape(-1,1), np.array(errs).reshape(-1,1))
        r_squared = model.score(np.array(ref_errs).reshape(-1,1), np.array(errs).reshape(-1,1))
        slope = model.coef_[0]
        
        figdir = self.figdir_root + '/cumulative_scatterplots/'
        
        if not os.path.exists(figdir):
            os.mkdir(figdir)
            
        
        #plt.savefig('%s/%s_%s-%d.png' % (figdir, err_computer.get_key(self.landmark_computer), str(self.iod_ptg), self.enforce_MeqN))

        plt.plot([mn, mx],[mn*slope, mx*slope], 'r:', linewidth=1.75, alpha=0.45)
        
        d = mx-mn
        
        plt.xticks([0,2,4,6])
        plt.yticks([0,2,4,6])
        plt.text(mn+d*0.07, mx-d*0.1, r"slope=%.2f" % slope,
                color="black", fontsize=12,
                horizontalalignment="left", verticalalignment="center",
                bbox=dict(boxstyle="round", fc="white", ec="white", pad=0.1))
        
        if type(err_computer) is str:
            ecstr = err_computer
        else:
            ecstr = err_computer.get_key(self.landmark_computer)
        plt.text(mn+d*0.07, mx-d*0.20, r"$R^2$=%.2f" % r_squared,
                color="black", fontsize=12,
                horizontalalignment="left", verticalalignment="center",
                bbox=dict(boxstyle="round", fc="white", ec="white", pad=0.1))

        plt.savefig('%s/%s_%s_s-%d.png' % (figdir, ecstr, str(self.iod_ptg), self.enforce_MeqN), dpi=500, bbox_inches='tight')

    


        
    
    def get_li(self, rec_method):
        rec_method_base = rec_method.split('-')[0]
        # meshname = self.recmethod2mesh[rec_method_base]
        return self.mesh2li[self.meshtopology]
    
    
    
    def compute_r2_ptwise(self, subj_fname, err_computer, rec_method, save_figure=False):
        ref_err_computer = ErrorComputerKnown()
        ref_err_ptwise = self.compute_err_ptwise(subj_fname, ref_err_computer, rec_method)
        err_ptwise = self.compute_err_ptwise(subj_fname, err_computer, rec_method)
        
        # slope, intercept, r_value, p_value, std_err = stats.linregress(ref_err_ptwise, err_ptwise)
        
        #initiate linear regression model
        model = LinearRegression()
        model.fit(ref_err_ptwise.reshape(-1,1), err_ptwise.reshape(-1,1))
        r_squared = model.score(ref_err_ptwise.reshape(-1, 1), err_ptwise.reshape(-1,1))
        
        if save_figure:
            mx = max(np.percentile(err_ptwise, 99), np.percentile(ref_err_ptwise, 99))
            mn = max(err_ptwise.min(), ref_err_ptwise.min())
            
            # maxerr = 6.25

            ticks = np.linspace(0, 1, 4).round(1)
            
            assert self.figdir_root is not None
            figdir = self.figdir_root + "/r2_intrasubj/" + err_computer.get_key(self.landmark_computer) + "_" + self.aligner + self.tail +"/" 
            
            if not os.path.exists(figdir):
                os.makedirs(figdir)
            
            # figpath_pdf = '%s/%s-%s.pdf' % (figdir, rec_method, subj_fname)    
            figpath_png = '%s/%s-%s.png' % (figdir, rec_method, subj_fname)   
            
            # plt.figure(figsize=(4,4))
            plt.clf()
            plt.axis('square')
            plt.scatter((ref_err_ptwise-mn)/(mx-mn), (err_ptwise-mn)/(mx-mn), alpha=0.007)
            plt.xlim((-0.01, 1.01))
            plt.ylim((-0.01, 1.01))
            plt.ylabel('Error (known topology)' )
            plt.xlabel('Error (%s)' % err_computer.get_label())
            plt.yticks([0, 1])
            plt.xticks([0, 1])
            
            plt.tight_layout()
            # plt.savefig(figpath_pdf)
            plt.savefig(figpath_png, dpi=100, bbox_inches='tight', pad_inches=0)
            
            


        import scipy.stats    
        return scipy.stats.pearsonr(ref_err_ptwise, err_ptwise)[0]
        # plt.scatter(ref_err_ptwise, err_ptwise, alpha=0.05)
        # import scipy.stats
        # print(scipy.stats.pearsonr(ref_err_ptwise, err_ptwise))
        # np.savetxt('x.txt', ref_err_ptwise)
        # np.savetxt('y.txt', err_ptwise)
        # plt.show()
        return r_squared#**2
    
    
    def compute_all_heatmaps(self, err_computers, rec_method, Nsubj=100):
        for subj_id in range(Nsubj):
            if not self.subject_exists(subj_id):
                continue
            self.compute_heatmaps(subj_id, err_computers, rec_method, save_figure=True)

    def compute_heatmaps(self, subj_id, err_computers, rec_method, save_figure=False):
        if not self.subject_exists(subj_id):
            return False
        
        subj_fname = 'subj%03d' % subj_id
        figdir = self.figdir_root + "/heatmaps/" + err_computers[0].get_key(self.landmark_computer) + "_" + self.aligner + self.tail +"/" 
        figpath_png = '%s/%s-%s.png' % (figdir, rec_method, subj_fname)   

        #if os.path.exists(figpath_png):
        #    print(figpath_png)
        #    return True

        mms = [True if type(err_computer) is str else False for err_computer in err_computers]
        errs_ptwise = [self.compute_err_ptwise(subj_fname, err_computers[i], rec_method, return_full=False, mm=mms[i]) for i in range(len(err_computers))]
        err_maxs = [np.percentile(err_ptwise, 95) for err_ptwise in errs_ptwise]
        err_max = max(err_maxs)

        Ds = [self.compute_heatmap(subj_fname, err_computers[i], rec_method, err_max, mm=mms[i]) for i in range(len(err_computers))]
        
        plt.clf()
        for i in range(len(Ds)):
            plt.subplot(1, len(Ds), i+1)
            plt.imshow(Ds[i])
            plt.axis('off')
            
        if save_figure:
            assert self.figdir_root is not None
            
            if not os.path.exists(figdir):
                os.makedirs(figdir)
            
            plt.savefig(figpath_png, dpi=100, bbox_inches='tight', pad_inches=0)

    def fetch_precomputed_ptwise_err(self, subj_fname, err_computer_string, rec_method):
        Gpath = '%s/%s/ground_truth/%s.txt' % (self.dbdir, self.dbname, subj_fname)
        landmarks = self.landmark_computer.compute_landmarks(Gpath)
        iod = np.linalg.norm(landmarks[28,:]-landmarks[19,:])
        err_path = '%s/%s/%s/%s.%s.err' % (self.dbdir, self.dbname, rec_method, subj_fname, err_computer_string)
        err_ptwise = np.loadtxt(err_path)/iod
        if self.cpts is not None:
            err_ptwise = err_ptwise[self.cpts]
        return err_ptwise

    def plot_interactive_3d(self, subj_id,  err_computer, rec_method):
        import open3d as o3d
        subj_fname = 'subj%03d' % subj_id
        
        Rpath = '%s/%s/%s/%s.txt' % (self.dbdir, self.dbname, rec_method, subj_fname)
        Gpath = '%s/%s/ground_truth/%s.txt' % (self.dbdir, self.dbname, subj_fname)
        
        (_, pidx) = self.compute_err_ptwise(subj_fname, err_computer, rec_method, return_corr=True)
        pidx = pidx.astype(int)
        (R, G) = self.get_aligned_data(Gpath, Rpath, rec_method)
        
        G[:,0] -= 3
        
        Rp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(R))
        Gp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(G[pidx]))
        
        # ix3di = np.loadtxt('/home/v/car-vision/python/geometric_error/ridxs/ix_3di.txt').astype(int)
        # ix_common = np.loadtxt('/home/v/car-vision/python/geometric_error/ridxs/ix_common.txt').astype(int)
        
        # shpG = np.zeros((53490,3))
        # shpG[ix_common] = G
        # shpG = shpG[ix3di]

                
        # shpG2 = np.zeros((53490,3))
        # shpG2[ix_common] = G[pidx]
        # shpG2 = shpG2[ix3di]
        
                
        # shpR = np.zeros((53490,3))
        # shpR[ix_common] = R
        # shpR = shpR[ix3di]
        
        # figdir = self.figdir_root + "/visual_lmks/" + err_computer.get_key(self.landmark_computer) + "_" + self.aligner + self.tail +"/" + rec_method
        
        # if not os.path.exists(figdir):
        #     os.makedirs(figdir)
        


        # tl = np.loadtxt('/home/v/car-vision/cuda/build-3DI/models/dat_files/tl.dat').astype(int)
        
        # meshG = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(shpG), o3d.utility.Vector3iVector(tl-1)) 
        # meshG.compute_vertex_normals()
        
        Rp.paint_uniform_color([0.9, 0.1, 0.1])
        Gp.paint_uniform_color([0.1, 0.9, 0.1])
        
        vcd = o3d.visualization.ViewControl()
        
        vcd.change_field_of_view(220.1)
        # vcd.draw_geometries([Rp, Gp])
        o3d.visualization.draw_geometries([Rp, Gp])
        
    
    def visualize_subj(self, subj_id,  rec_method, err_computer, save_figure=False, 
                       zoomed=False, do_landmarks=True, do_R=True, do_G=True, do_mesh=True,
                       do_correction=False, warped_R=False):
        
        import open3d as o3d
        cur_li = self.get_li(rec_method)
        
        subj_fname = 'subj%03d' % subj_id
        
        Rpath = '%s/%s/%s/%s.txt' % (self.dbdir, self.dbname, rec_method, subj_fname)
        Gpath = '%s/%s/ground_truth/%s.txt' % (self.dbdir, self.dbname, subj_fname)
        
        (_, pidx) = self.compute_err_ptwise(subj_fname, err_computer, rec_method, return_corr=True)
        pidx = pidx.astype(int)
        (R, G) = self.get_aligned_data(Gpath, Rpath, rec_method)

        Dx = None
        landmarks = None
        
        if err_computer.require_Dmatrix:
            cache_fpathD = '%s/D%s-%s.npy' % (self.cacheroot, rec_method, subj_fname)
            
            if os.path.exists(cache_fpathD):
                Dx = np.load(cache_fpathD)
            else:
                Dx = squareform(pdist(R)).T
                Dx = Dx[:,cur_li]
                
                for j in range(Dx.shape[1]):
                    Dx[:,j] = 1-Dx[:,j]/Dx[:,j].max()
                
                np.save(cache_fpathD, Dx)
                
            Dx = Dx[:,self.lis_map[self.num_lmks]]
        
        # Here we assume that we are reading 51 landmarks
        landmarks = self.landmark_computer.compute_landmarks(Gpath)
        iod = np.linalg.norm(landmarks[28,:]-landmarks[19,:])
        

        landmarks = landmarks[self.lis_map[self.num_lmks],:]

        
        if warped_R:
            R = err_computer.compute_error(R, G, Dx=Dx, gt_landmarks=landmarks, li=cur_li, return_err=False, iod=iod, return_corr=False)

        
        ix3di = np.loadtxt('/home/v/car-vision/python/geometric_error/ridxs/ix_3di.txt').astype(int)
        ix_common = np.loadtxt('/home/v/car-vision/python/geometric_error/ridxs/ix_common.txt').astype(int)
        
        shpG = np.zeros((53490,3))
        shpG[ix_common] = G
        shpG = shpG[ix3di]

                
        shpG2 = np.zeros((53490,3))
        shpG2[ix_common] = G[pidx]
        shpG2 = shpG2[ix3di]
        
                
        shpR = np.zeros((53490,3))
        shpR[ix_common] = R
        shpR = shpR[ix3di]
        
        if do_landmarks:
            figdir = self.figdir_root + "/visual_lmks/" + err_computer.get_key(self.landmark_computer) + "_" + self.aligner + self.tail +"/" + rec_method
        else:
            figdir = self.figdir_root + "/visual_mesh/" + err_computer.get_key(self.landmark_computer) + "_" + self.aligner + self.tail +"/" + rec_method
            
        if not os.path.exists(figdir):
            os.makedirs(figdir)
        
        tl = np.loadtxt('/home/v/car-vision/cuda/build-3DI/models/dat_files/tl.dat').astype(int)
        
        meshG = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(shpG), o3d.utility.Vector3iVector(tl-1)) 
        meshG.compute_vertex_normals()
        
        meshR = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(shpR), o3d.utility.Vector3iVector(tl-1)) 
        meshR.compute_vertex_normals()
        
        if not zoomed:
            render = o3d.visualization.rendering.OffscreenRenderer(800, 800)
        else:
            render = o3d.visualization.rendering.OffscreenRenderer(2000, 2000)
        
        # Define a simple unlit Material.
        # (The base color does not replace the arrows' own colors.)
        mat = o3d.visualization.rendering.MaterialRecord()
        # mat.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
        mat.shader = "defaultLit"
        
        yellow = o3d.visualization.rendering.MaterialRecord()
        yellow.base_color = [1.0, 0.75, 0.0, 1.0]
        yellow.shader = "defaultLit"
        
        if do_landmarks:
            PG = G[pidx][cur_li,:]
            PG[:,2] -= 0.2
            # P2 = G[self.li,:]
            # P2[:,2] -= 0.2
            PR = shpR[cur_li,:]
            PR[:,2] -= 0.4
        else:
            ss =  2 #3
            PG = G[pidx][::ss]
            
            if do_correction:
                from error_computers import apply_correction
                R2 = err_computer.compute_error(R, G, Dx=Dx, gt_landmarks=landmarks, li=cur_li, return_err=False, iod=iod, return_corr=False)

                dG = apply_correction(G, R2, Glmks=landmarks, li=cur_li, pidx=pidx, iod=iod, correction_param='pair-mixed-r2-sqrt')
                PG += dG[::ss]
                
            
            PG[:,2] -= 0.2
            # P2 = G[::ss]
            # P2[:,2] -= 0.2
            PR = shpR[::ss]
            PR[:,2] -= 0.4
            
        ptsG = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(PG))
        ptsR = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(PR))
        # pts3 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P3))
        
        # source = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(G), o3d.utility.Vector3iVector(tl-1)) #(G, tl-1)
        # source.compute_vertex_normals()
        
        
        cam = o3d.camera.PinholeCameraParameters()
        cam.extrinsic = np.asarray([[15,0,0,0],[0,15,0,0],[0,0,1,89],[0,0,0,3]]) # Translate infront of camera by 5 unit
        cam.intrinsic = o3d.camera.PinholeCameraIntrinsic(640, 640, 640, 640, 320, 320)
        
        render.setup_camera(cam.intrinsic, cam.extrinsic)   
        grey = o3d.visualization.rendering.MaterialRecord()
        gc = 0.80
        grey.base_color = [gc, gc, gc, 1.0]
        
        grey.base_metallic = 0.7
        grey.base_reflectance =  1.0
        grey.base_clearcoat = 0.8
        grey.base_clearcoat_roughness = 0.6
        grey.base_anisotropy=0.55
        
        ptsize = 7
        
        if not do_landmarks:
            ptsize = 10 # 8.5
        
        if do_landmarks and zoomed:
            ptsize = 10
        
        alpha = 1
        
        mat_red = o3d.visualization.rendering.MaterialRecord()
        mat_red.shader = 'defaultUnlit'
        mat_red.point_size = ptsize
        
        if zoomed:
            if do_landmarks:
                mat_red.point_size += 3
        mat_red.base_color = [48/255.0, 123/255.0, 166/255.0, alpha]
        
        if do_landmarks:
            mat_red.base_color = [255.0/255.0, 127.0/255.0, 0.0, alpha]
            
        
        mat_blue = o3d.visualization.rendering.MaterialRecord()
        mat_blue.shader = 'defaultUnlit'
        mat_blue.point_size = ptsize
        mat_blue.base_color = [188/255., 66/255., 131/255.0, alpha]
        if do_landmarks:
            mat_blue.base_color = [55./255., 126./255., 184.0/255., alpha]
            
        
        mat_green = o3d.visualization.rendering.MaterialRecord()
        mat_green.shader = 'defaultUnlit'
        mat_green.point_size = ptsize
        mat_green.base_color = [0/255., 66/255., 131/255.0, alpha]
        



        if do_landmarks:
            mat_green.base_color = [0.1, 1.0, 0.1, alpha]
        
        grey.shader = "defaultLit"
        
        a = np.array([7,10,20]); 
        a = a/np.linalg.norm(a)
        
        clr = 1.0
        render.scene.scene.set_sun_light(a, [clr,clr,clr],
                                         280000)
        
        render.scene.scene.enable_sun_light(True)
        bg = 1.25
        render.scene.set_background(np.array([bg,bg,bg,bg]))
        
        tail = ''
        if do_G:
            tail += 'G'
            if do_mesh:
                render.scene.add_geometry("pcd3", meshG, grey)
            render.scene.add_geometry("pts1", ptsG, mat_red)
            
        if do_R:
            tail += 'R'
            if do_mesh:
                render.scene.add_geometry("pcd3", meshR, grey)
                
            render.scene.add_geometry("pts2", ptsR, mat_blue)        
            if warped_R:
                tail += 'W'
                
        if do_correction:
            tail += 'C'
        # render.scene.add_geometry("pts3", pts3, mat_green)
        # render.setup_camera(80.0, [0.5, 0.5, -10000.5], [0.5, 2, 0.5], [0, 0, 1])
        img = render.render_to_image()
        o3d.io.write_image(figdir + "/%s_%d_%s.png" % (subj_fname, zoomed, tail), img)
        del render
        



    def visualize_subj_lmks_overlaid(self, subj_id,  rec_method, err_computer, save_figure=False, 
                       zoomed=False, do_landmarks=True, do_R=True, do_G=True, do_mesh=True,
                       do_correction=False):
        
        import open3d as o3d
        subj_fname = 'subj%03d' % subj_id
        cur_li = self.get_li(rec_method)

        
        Rpath = '%s/%s/%s/%s.txt' % (self.dbdir, self.dbname, rec_method, subj_fname)
        Gpath = '%s/%s/ground_truth/%s.txt' % (self.dbdir, self.dbname, subj_fname)
        
        (_, pidx) = self.compute_err_ptwise(subj_fname, err_computer, rec_method, return_corr=True)
        pidx = pidx.astype(int)
        (R, G) = self.get_aligned_data(Gpath, Rpath, rec_method)
        
        ix3di = np.loadtxt('/home/v/car-vision/python/geometric_error/ridxs/ix_3di.txt').astype(int)
        ix_common = np.loadtxt('/home/v/car-vision/python/geometric_error/ridxs/ix_common.txt').astype(int)
        
        shpG = np.zeros((53490,3))
        shpG[ix_common] = G
        shpG = shpG[ix3di]

                
        shpG2 = np.zeros((53490,3))
        shpG2[ix_common] = G[pidx]
        shpG2 = shpG2[ix3di]
        
                
        shpR = np.zeros((53490,3))
        shpR[ix_common] = R
        shpR = shpR[ix3di]
        
        if do_landmarks:
            figdir = self.figdir_root + "/visual_lmks/" + err_computer.get_key(self.landmark_computer) + "_" + self.aligner + self.tail +"/" + rec_method
        else:
            figdir = self.figdir_root + "/visual_mesh/" + err_computer.get_key(self.landmark_computer) + "_" + self.aligner + self.tail +"/" + rec_method
            
        if not os.path.exists(figdir):
            os.makedirs(figdir)
        
        tl = np.loadtxt('/home/v/car-vision/cuda/build-3DI/models/dat_files/tl.dat').astype(int)
        
        meshG = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(shpG), o3d.utility.Vector3iVector(tl-1)) 
        meshG.compute_vertex_normals()
        
        meshR = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(shpR), o3d.utility.Vector3iVector(tl-1)) 
        meshR.compute_vertex_normals()
        
        if not zoomed:
            render = o3d.visualization.rendering.OffscreenRenderer(800, 800)
        else:
            render = o3d.visualization.rendering.OffscreenRenderer(2000, 2000)
        
        # Define a simple unlit Material.
        # (The base color does not replace the arrows' own colors.)
        mat = o3d.visualization.rendering.MaterialRecord()
        # mat.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
        mat.shader = "defaultLit"
        
        yellow = o3d.visualization.rendering.MaterialRecord()
        yellow.base_color = [1.0, 0.75, 0.0, 1.0]
        yellow.shader = "defaultLit"
        
        # L = G[[self.li,:]
        
        if do_landmarks:
            PG = G[pidx][cur_li,:]
            PG[:,2] -= 0.2
            # P2 = G[self.li,:]
            # P2[:,2] -= 0.2
            PR = shpR[cur_li,:]
            PR[:,2] -= 0.4
        else:
            ss =  2 #3
            PG = G[pidx][::ss]
            
            if do_correction:
                from error_computers import apply_correction
                Dx = None
                landmarks = None
                
                if err_computer.require_Dmatrix:
                    cache_fpathD = '%s/D%s-%s.npy' % (self.cacheroot, rec_method, subj_fname)
                    
                    if os.path.exists(cache_fpathD):
                        Dx = np.load(cache_fpathD)
                    else:
                        Dx = squareform(pdist(R)).T
                        Dx = Dx[:,cur_li]
                        
                        for j in range(Dx.shape[1]):
                            Dx[:,j] = 1-Dx[:,j]/Dx[:,j].max()
                        
                        np.save(cache_fpathD, Dx)
                        
                    Dx = Dx[:,self.lis_map[self.num_lmks]]
                
                # Here we assume that we are reading 51 landmarks
                landmarks = self.landmark_computer.compute_landmarks(Gpath)
                iod = np.linalg.norm(landmarks[28,:]-landmarks[19,:])
                
                landmarks = landmarks[self.lis_map[self.num_lmks],:]
                R2 = err_computer.compute_error(R, G, Dx=Dx, gt_landmarks=landmarks, li=cur_li, return_err=False, iod=iod, return_corr=False)

                dG = apply_correction(G, R2, Glmks=landmarks, li=cur_li, pidx=pidx, iod=iod, correction_param='pair-mixed-r2-sqrt')
                PG += dG[::ss]


            
            
            PG[:,2] -= 0.2
            # P2 = G[::ss]
            # P2[:,2] -= 0.2
            PR = shpR[::ss]
            PR[:,2] -= 0.4
            
        ptsG = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(PG))
        ptsR = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(PR))
        
        
        cam = o3d.camera.PinholeCameraParameters()
        cam.extrinsic = np.asarray([[15,0,0,0],[0,15,0,0],[0,0,1,89],[0,0,0,3]]) # Translate infront of camera by 5 unit
        cam.intrinsic = o3d.camera.PinholeCameraIntrinsic(640, 640, 640, 640, 320, 320)
        
        render.setup_camera(cam.intrinsic, cam.extrinsic)   
        grey = o3d.visualization.rendering.MaterialRecord()
        gc = 0.80
        grey.base_color = [gc, gc, gc, 1.0]
        
        grey.base_metallic = 0.7
        grey.base_reflectance =  1.0
        grey.base_clearcoat = 0.8
        grey.base_clearcoat_roughness = 0.6
        grey.base_anisotropy=0.55
        
        ptsize = 7
        
        if not do_landmarks:
            ptsize = 10 # 8.5
        
        if do_landmarks and zoomed:
            ptsize = 10
        
        alpha = 1
        
        mat_red = o3d.visualization.rendering.MaterialRecord()
        mat_red.shader = 'defaultUnlit'
        mat_red.point_size = ptsize
        
        if zoomed:
            if do_landmarks:
                mat_red.point_size += 3
                
        mat_red.base_color = [255.0/255.0, 127.0/255.0, 0.0, alpha]
        
        mat_blue = o3d.visualization.rendering.MaterialRecord()
        mat_blue.shader = 'defaultUnlit'
        mat_blue.point_size = ptsize
        mat_blue.base_color = [55./255., 126./255., 184.0/255., alpha]
        # mat_blue.base_color = [ 77.0/255.0, 175.0/255.0, 74.0/255.0, alpha]
        
        mat_green = o3d.visualization.rendering.MaterialRecord()
        mat_green.shader = 'defaultUnlit'
        mat_green.point_size = ptsize
        mat_green.base_color = [0.1, 1.0, 0.1, alpha]
        
        if not do_mesh:
            mat_red.point_size += 4
            mat_blue.point_size += 2


        grey.shader = "defaultLit"
        
        a = np.array([7,10,20]); 
        a = a/np.linalg.norm(a)
        
        clr = 1.0
        render.scene.scene.set_sun_light(a, [clr,clr,clr],
                                         280000)
        
        render.scene.scene.enable_sun_light(True)
        bg = 1.25
        render.scene.set_background(np.array([bg,bg,bg,bg]))
        
        tail = ''
        if do_G:
            tail += 'G'
            if do_mesh:
                render.scene.add_geometry("pcd3", meshG, grey)
            
        if do_R:
            tail += 'R'
            if do_mesh:
                render.scene.add_geometry("pcd3", meshR, grey)
        
        render.scene.add_geometry("pts1", ptsG, mat_red)
        render.scene.add_geometry("pts2", ptsR, mat_blue)
        if do_correction:
            tail += 'C'
            
        if not do_mesh:
            tail += 'NM'
        # render.scene.add_geometry("pts3", pts3, mat_green)
        # render.setup_camera(80.0, [0.5, 0.5, -10000.5], [0.5, 2, 0.5], [0, 0, 1])
        img = render.render_to_image()
        o3d.io.write_image(figdir + "/%s_%d_%s.png" % (subj_fname, zoomed, tail), img)
        del render
        




            

            
    
    
    def compute_heatmap(self, subj_fname, err_computer, rec_method, norm_coef, mm=False):
        
        cur_li = self.get_li(rec_method)
        
        err_ptwise = self.compute_err_ptwise(subj_fname, err_computer, rec_method, return_full=True, mm=mm)
        # print('%f' % (np.mean(err_ptwise.mean())*89))
        # Gpath = '%s/%s/ground_truth/%s.txt' % (self.dbdir, self.dbname, subj_fname)        
        Rpath = '%s/%s/%s/%s.txt' % (self.dbdir, self.dbname, rec_method, subj_fname)

        R = np.loadtxt(Rpath)

        # @@@ This needs to be replaced with each methods landmark ID
        iod = np.linalg.norm(R[cur_li[28],:]-R[cur_li[19],:])
        R /= iod

        # R[:,1] *= -1
        
        coef = 50
        if self.dbname[:5] == 'synth':
            coef = 50
        x = ((R[:,0]-R[:,0].min())*coef).round().reshape(-1,).astype(int)
        y = ((R[:,1]-R[:,1].min())*coef).round().reshape(-1,).astype(int)
        
        H = np.nan*np.ones((y.max(), x.max()))
        
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        m = cm.ScalarMappable(norm=norm,cmap='jet')

        for i in range(len(x)):
            if self.cpts is None:
                H[H.shape[0]-(y[i]+1), x[i]-1] = err_ptwise[i]
            else:
                if i in self.cpts:
                    H[H.shape[0]-(y[i]+1), x[i]-1] = err_ptwise[i]
                else:
                    H[H.shape[0]-(y[i]+1), x[i]-1] = 0
                    
        D = interpolate_missing_pixels(H,np.isnan(H))
        D = D/norm_coef
                        

        if self.cpts is not None:
            Hmask = np.nan*np.ones((y.max(), x.max()))
    
            for i in range(len(x)):
                if self.cpts is None:
                    Hmask[Hmask.shape[0]-(y[i]+1), x[i]-1] = 1
                else:
                    if i in self.cpts:
                        Hmask[Hmask.shape[0]-(y[i]+1), x[i]-1] = 1
                    else:
                        Hmask[Hmask.shape[0]-(y[i]+1), x[i]-1] = 0

            Dmask = interpolate_missing_pixels(Hmask,np.isnan(Hmask))
        
        Dc = m.to_rgba(D)[:,:,:3]
        for i, j in itertools.product(range(Dc.shape[0]), range(Dc.shape[1])):
            if self.cpts is not None:
                if D[i,j] == 0 or Dmask[i,j] < 0.95:
                    Dc[i,j,0] = 1
                    Dc[i,j,1] = 1
                    Dc[i,j,2] = 1
            else:
                if D[i,j] == 0:
                    Dc[i,j,0] = 1
                    Dc[i,j,1] = 1
                    Dc[i,j,2] = 1

        return Dc
    
    
    def get_aligned_data(self, Gpath, Rpath, rec_method):
        lpath = Gpath.replace('txt', '3Dlmks_' + self.aligner)
        if self.aligner == 'exact':
            lc = LandmarkComputerExact()
            lG = lc.compute_landmarks(Gpath)
        elif self.aligner == 'auto':
            lc = LandmarkComputerAuto()
            lG = lc.compute_landmarks(Gpath)
        G = np.loadtxt(Gpath)
        R = np.loadtxt(Rpath)
        """
        plt.subplot(1,3,1)
        plt.plot(R[:,0], R[:,1], '.')
        plt.plot(G[:,0], G[:,1], '.')
        """
        cur_li = self.get_li(rec_method)


        

        if Rpath.find('NICP') >= 0:
            #lR = R[self.li,:]
            #iod = np.linalg.norm(lR[28,:]-lR[19,:])
            #R /= iod
            #lR /= iod
            #_, __, tform = procrustes(lR[lrigid,:], lG[lrigid,:])
            _, __, tform = procrustes(R, G)
            
            G = tform['scale']*(G @ tform['rotation'])+tform['translation']
        else:
            lR = R[cur_li,:]
            _, __, tform = procrustes(lG[self.lrigid,:], lR[self.lrigid,:])
            
            """
            plt.subplot(1,3,2)
            #plt.plot(lR[:,0], lR[:,1])
            plt.plot(lG[lrigid,0], lG[lrigid,1])
            plt.plot(__[:,0], __[:,1])
            """
            R = tform['scale']*(R @ tform['rotation'])+tform['translation']
        # iod = np.linalg.norm(lG[28,:]-lG[19,:])
        # R[:,1] *= -1
        # G[:,1] *= -1
        
        # R /= iod
        # G /= iod


        if self.enforce_MeqN:
            G = G[:R.shape[0], :]
        
        """
        plt.subplot(1,3,3)
        plt.plot(R[:,0], R[:,1], '.')
        plt.plot(G[:,0], G[:,1], '.')
        plt.show()
        """
        return (R, G)
    
    
    def subject_exists(self, subj_id):
        Gpath = '%s/%s/ground_truth/subj%03d.txt' % (self.dbdir, self.dbname, subj_id)
        lexists = self.landmark_computer.landmarks_exist(Gpath)
        gexists = os.path.exists(Gpath)
        return lexists and gexists


    def compute_iod(self, subj_fname):
        Gpath = '%s/%s/ground_truth/%s.txt' % (self.dbdir, self.dbname, subj_fname)
        landmarks = self.landmark_computer.compute_landmarks(Gpath)
        iod = np.linalg.norm(landmarks[28,:]-landmarks[19,:])
        return iod



    def compute_err_ptwise(self, subj_fname, err_computer, rec_method, return_full=False, 
                           return_corr=False, mm=True, bypass_cache=False):

        if type(err_computer) is str:
            return self.fetch_precomputed_ptwise_err(subj_fname, err_computer, rec_method)
        
        cur_li = self.get_li(rec_method)

        
        tail2 = ''
        if return_corr:
            tail2 = 'C'

        if self.rigid_alignment == False:
            tail2 += 'N'
        
        cache_fpath = '%s/%s-%s-%s-%s%s%s.npy' % (self.cacheroot, rec_method, subj_fname, self.aligner,
                                              err_computer.get_key(self.landmark_computer), self.tail, tail2)
        
        if not bypass_cache and os.path.exists(cache_fpath):
            if not return_corr:
                err_ptwise = np.load(cache_fpath)
                iod = 1
                if mm:
                    iod = self.compute_iod(subj_fname)
                err_ptwise = iod*err_ptwise
                if self.cpts is not None and not isinstance(err_computer, ErrorComputerLandmarks) and return_full is False and err_computer.direction == 'RG':
                    err_ptwise = err_ptwise[self.cpts]
                return err_ptwise.reshape(-1,)
            else:
                (err_ptwise, pidx) = np.load(cache_fpath)
                iod = 1
                if mm:
                    iod = self.compute_iod(subj_fname)
                err_ptwise = iod*err_ptwise
                if self.cpts is not None and not isinstance(err_computer, ErrorComputerLandmarks) and return_full is False and err_computer.direction == 'RG':
                    err_ptwise = err_ptwise[self.cpts]
                return (err_ptwise.reshape(-1,), pidx)
                

        Rpath = '%s/%s/%s/%s.txt' % (self.dbdir, self.dbname, rec_method, subj_fname)
        Gpath = '%s/%s/ground_truth/%s.txt' % (self.dbdir, self.dbname, subj_fname)
        
        if self.rigid_alignment:
            (R, G) = self.get_aligned_data(Gpath, Rpath, rec_method)
        else:
            R = np.loadtxt(Rpath)
            G = np.loadtxt(Gpath)
        
        
        if bypass_cache:
            plt.figure(figsize=(36,36))
            ss = 1
            plt.subplot(2,2,1)
            plt.plot(R[self.cpts[::ss],0], -R[self.cpts[::ss],1], '.')
            plt.xlim((-3, 3))
            plt.ylim((-3, 3))
            plt.subplot(2,2,2)
            plt.plot(G[::ss,0], -G[::ss,1], '.')
            plt.xlim((-3, 3))
            plt.ylim((-3, 3))
    
            plt.subplot(2,2,3)
            plt.plot(R[self.cpts[::ss],2], -R[self.cpts[::ss],1], '.')
            plt.xlim((-3, 3))
            plt.ylim((-3, 3))
            plt.subplot(2,2,4)
            plt.plot(G[::ss,2], -G[::ss,1], '.')
            plt.xlim((-3, 3))
            plt.ylim((-3, 3))
                    # print(len(self.cpts))
            print('plotted')
            """ """
        
        # R = np.loadtxt(Rpath)
        # G = np.loadtxt(Gpath)
        # plt.plot(R[:,0], R[:,1])
        # plt.plot(G[:,0], G[:,1])
        
        Dx = None
        landmarks = None
        
        if err_computer.require_Dmatrix:
            cache_fpathD = '%s/D%s-%s.npy' % (self.cacheroot, rec_method, subj_fname)
            
            if os.path.exists(cache_fpathD):
                Dx = np.load(cache_fpathD)
            else:
                Dx = squareform(pdist(R)).T
                Dx = Dx[:,cur_li]
                
                for j in range(Dx.shape[1]):
                    Dx[:,j] = 1-Dx[:,j]/Dx[:,j].max()
                
                np.save(cache_fpathD, Dx)
                
            Dx = Dx[:,self.lis_map[self.num_lmks]]
        
        # Here we assume that we are reading 51 landmarks
        landmarks = self.landmark_computer.compute_landmarks(Gpath)
        iod = np.linalg.norm(landmarks[28,:]-landmarks[19,:])
        if err_computer.direction == 'GR':
            if G.shape[0] != R.shape[0]:
                if self.dbname.find('synth') >= 0:
                    pass
                else:
                    cidx = np.loadtxt('%s/%s/ground_truth/%s.cidx' % (self.dbdir, self.dbname, subj_fname)).astype(int)
                    #print(cidx)
                    #print(G.shape)
                    G = G[cidx,:]
        
        
        landmarks = landmarks[self.lis_map[self.num_lmks],:]
        
        output = err_computer.compute_error(R, G, Dx=Dx, gt_landmarks=landmarks, 
                                                li=np.array(cur_li)[self.lis_map[self.num_lmks]], iod=iod,
                                                cpts=self.cpts, return_corr=return_corr,
                                                G0=self.G0, li0=self.li0)
        if return_corr:
            err_ptwise = output[0]
            pidx = output[1]
        else:
            err_ptwise = output
        
        err_ptwise /= iod
        np.save(cache_fpath, output)
        
        if self.cpts is not None and not isinstance(err_computer, ErrorComputerLandmarks) and return_full is False and err_computer.direction == 'RG':
            err_ptwise = err_ptwise[self.cpts]

        if mm:
            err_ptwise *= iod

        if return_corr:
            return(err_ptwise, pidx)
        else:
            return err_ptwise
        
    

    
    def compute_avg_err_on_dataset(self, err_computer, rec_method, Nsubj=None, median=False, N0=0, mm=False):
        
        if Nsubj is None:
            if self.dbname[:5] == 'synth':
                Nsubj = 100
            elif self.dbname[:5] == 'Flore':
                Nsubj = 54
        
        errs = []
        for subj_id in range(N0, Nsubj):
            if not self.subject_exists(subj_id):
                continue
            if self.dbname[:9] == 'BU4DFEexp':
                if subj_id in [19, 28, 78, 84]:
                    continue
            if self.dbname[:5] == 'Flore':
                if subj_id in [3, 28, 41,44]:
                    continue
            #try:
            if True:
                if type(err_computer) is list:
                    if median:
                        ce = np.mean([np.median(self.compute_err_ptwise('subj%03d' % subj_id, ec, rec_method,mm=mm)) for ec in err_computer])
                    else: 
                        ce = np.mean([np.mean(self.compute_err_ptwise('subj%03d' % subj_id, ec, rec_method,mm=mm)) for ec in err_computer])
                elif type(err_computer) is str:
                    err_ptwise = self.fetch_precomputed_ptwise_err('subj%03d' % subj_id, err_computer, rec_method)
                    if median:
                        ce = np.median(err_ptwise)
                    else:
                        ce = np.mean(err_ptwise)
                else:
                    err_ptwise = self.compute_err_ptwise('subj%03d' % subj_id, err_computer, rec_method,mm=mm)
                    if median:
                        ce = np.median(err_ptwise)
                    else:
                        ce = np.mean(err_ptwise)
                if ce>15:
                    continue

                errs.append(ce)
            try:
                esdasd=1
            except:
                print('Problem with %d' % subj_id)
        
        return errs
        
     
    def compute_avg_intrasubj_r2_on_dataset(self, err_computer, rec_method, Nsubj=100, save_figure=True):
        
        r2s = []
        for subj_id in range(Nsubj):
            r2 = self.compute_r2_ptwise('subj%03d' % subj_id, err_computer, rec_method, save_figure)
            r2s.append(r2)
        
        return np.mean(r2s)
        
    
    def compute_intersubj_r2_on_dataset(self, err_computer, rec_method, Nsubj=100, save_figure=False):
        ref_err_computer = ErrorComputerKnown()

        err_refs = []
        errs = []
        err_p_s = {}
        for subj_id in range(Nsubj):
            try:
                ref_err_ptwise = self.compute_err_ptwise('subj%03d' % subj_id, ref_err_computer, rec_method)
                err_ptwise = self.compute_err_ptwise('subj%03d' % subj_id, err_computer, rec_method)
            except:
                continue
            errs.append(np.mean(err_ptwise))
            err_refs.append(np.mean(ref_err_ptwise))
        
        err_refs = np.array(err_refs)
        errs = np.array(errs)
        model = LinearRegression()
        
        model.fit(err_refs.reshape(-1,1), errs.reshape(-1,1))
        r_squared = model.score(err_refs.reshape(-1, 1), errs.reshape(-1,1))
        
        if save_figure:
            assert self.figdir_root is not None
            figdir = self.figdir_root + "/r2_intersubj/" + err_computer.get_key(self.landmark_computer) + "_" + self.aligner + self.tail +"/" 
            
            if not os.path.exists(figdir):
                os.makedirs(figdir)
            
            mx = max(100*err_refs)
            mn = min(min(100*err_refs), min(100*errs))
            # mn = 1
            # mn = 0
            ticks = np.linspace(mn, mx, 4).round(2)
            
            d = mx-mn
            
            
            plt.clf()
            plt.axis('square')
            plt.scatter(100*err_refs, 100*errs)
            
            plt.text(mn+d*0.23, mx-d*0.11, r"$R^2=%.2f$" % r_squared,
                    color="black", fontsize=16,
                    horizontalalignment="center", verticalalignment="center",
                    bbox=dict(boxstyle="round", fc="white", ec="white", pad=0.1))


            plt.ylim((mn-d*0.025, mx+d*0.025))
            plt.xlim((mn-d*0.025, mx+d*0.025))
            plt.xticks(ticks)
            plt.yticks(ticks)
            plt.xlabel('Error (known topology)')
            plt.ylabel('Error (%s)' % err_computer.get_label())
            plt.title(rec_method)
            
            figpath_png = '%s/%s.png' % (figdir, rec_method)   
            plt.savefig(figpath_png, dpi=100, bbox_inches='tight', pad_inches=0)
            

        return r_squared
    
    
    

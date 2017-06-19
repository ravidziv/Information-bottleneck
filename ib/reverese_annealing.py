import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import scipy.io as sio
import numpy as np
import cPickle
from sklearn import neighbors
from idnns import information_network as inet

from idnns.information import information_utilities as iu
from joblib import Parallel, delayed
import multiprocessing

NUM_CORES = multiprocessing.cpu_count()


def calc_probs_cond(fill_p_y_given_x, x_index, p_x, new_p_y_given_t, beta):
    current_p_y_given_x = fill_p_y_given_x[:, x_index]
    D_KL = np.array([iu.KL(current_p_y_given_x, c_p_y_given_t) for c_p_y_given_t in new_p_y_given_t.T])
    new_p_x_given_t = (np.tile(p_x[x_index], (new_p_y_given_t.shape[1], 1)) * np.exp(-beta * np.vstack(D_KL))).T
    return new_p_x_given_t


def calc_IB_combained(p_t_given_x, pxs, pys, p_y_given_x, beta, iter, p_x_emp, p_y_given_x_emp, sampled_indexes, type):
    mask = np.zeros((pxs.shape[0], 1), dtype=bool)  # np.ones_like(a,dtype=bool)
    mask[sampled_indexes] = True
    #mask[:,:]=True
    #p_y_given_x_emp = p_y_given_x
    mask = mask[:, 0]
    probTgivenXs = p_t_given_x
    #probTgivenXs = np.eye(pxs.shape[0])
    #probTgivenXs = np.array([np.random.permutation(probTgivenXs[:,i]) for i in range(probTgivenXs.shape[1])]).T
    pts_emp = np.dot(probTgivenXs, pxs)

    for i in range(0, iter):
        p_t_given_x_emp = probTgivenXs[:, :]
        if type==0:
            pass
            #pts_emp = np.dot(p_t_given_x_emp, p_x_emp)
        else:
            pts_emp = np.dot(probTgivenXs, pxs)

        p_yx = np.multiply(p_y_given_x, np.tile(pxs, (p_y_given_x.shape[0], 1)))
        p_t_given_x_divide_p_t= np.multiply(probTgivenXs, np.tile(1. / (pts_emp+np.spacing(1)), (probTgivenXs.shape[1], 1)).T)
        PYgivenTs_update = np.dot(p_yx[:,:],p_t_given_x_divide_p_t[:,:].T)

        PYgivenTs_update = PYgivenTs_update / np.tile(np.nansum(PYgivenTs_update, axis=0), (PYgivenTs_update.shape[0], 1))
        PYgivenTs_update[np.isnan(PYgivenTs_update)] = 0
        d1 = np.tile(np.nansum(np.multiply(p_y_given_x_emp, np.log(p_y_given_x_emp)), axis=0), (p_t_given_x_emp.shape[0], 1))
        d2 = np.dot(-np.log(PYgivenTs_update.T + np.spacing(1)), p_y_given_x_emp)
        DKL =np.tile(pts_emp, (probTgivenXs.shape[1],1))
        DKL[:, mask] = d1 + d2
        probTgivenXs = np.exp(-beta * (DKL)) * pts_emp[:, np.newaxis]
        probTgivenXs = probTgivenXs / np.tile(np.nansum(probTgivenXs, axis=0), (probTgivenXs.shape[0], 1))
        probTgivenXs[:, np.isnan(probTgivenXs)[0, :]] = np.tile(pts_emp, (np.sum(np.isnan(probTgivenXs), axis=1)[0], 1)).T

        #probTgivenXs[np.isnan(probTgivenXs)] = 0
    return probTgivenXs, PYgivenTs_update, pts_emp

def calc_IB_combained_second(p_t_given_x, pxs, pys, p_y_given_x, beta, iter, p_x_emp, p_y_given_x_emp, sampled_indexes, type):
    pys = np.mean(p_y_given_x[:,sampled_indexes], axis=1)
    mask = np.zeros((pxs.shape[0], 1), dtype=bool)  # np.ones_like(a,dtype=bool)
    mask[sampled_indexes] = True
    #p_y_given_x_emp = p_y_given_x
    mask = mask[:, 0]
    probTgivenXs = p_t_given_x
    #probTgivenXs = np.eye(pxs.shape[0])
    #probTgivenXs = np.array([np.random.permutation(probTgivenXs[:,i]) for i in range(probTgivenXs.shape[1])]).T
    pts_emp = np.dot(probTgivenXs, pxs)

    for i in range(0, iter):
        p_t_given_x_emp = probTgivenXs[:,  :]
        if type==0:
            pass
            #pts_emp = np.dot(p_t_given_x_emp, p_x_emp)
        else:
            pts_emp = np.dot(probTgivenXs, pxs)
            #pts_emp = np.dot(p_t_given_x_emp, p_x_emp)

        p_yx = np.multiply(p_y_given_x, np.tile(pxs, (p_y_given_x.shape[0], 1)))
        p_t_given_x_divide_p_t= np.multiply(probTgivenXs, np.tile(1. / (pts_emp+np.spacing(1)), (probTgivenXs.shape[1], 1)).T)
        PYgivenTs_update_s = np.dot(p_yx[:,sampled_indexes],p_t_given_x_divide_p_t[:,sampled_indexes].T)
        PYgivenTs_update_s[PYgivenTs_update_s<np.exp(-40)]=0
        PYgivenTs_update_before = PYgivenTs_update_s / np.tile(np.nansum(PYgivenTs_update_s, axis=0), (PYgivenTs_update_s.shape[0], 1))
        PYgivenTs_update = PYgivenTs_update_before
        PYgivenTs_update[:, np.isnan(PYgivenTs_update)[0,:]] = np.tile(pys,(np.sum(np.isnan(PYgivenTs_update), axis=1)[0],1)).T
        d1 = np.tile(np.nansum(np.multiply(p_y_given_x_emp, np.log(p_y_given_x_emp)), axis=0), (p_t_given_x_emp.shape[0], 1))
        d2 = np.dot(-np.log(PYgivenTs_update.T + np.spacing(1)), p_y_given_x_emp)
        DKL =np.tile(pts_emp, (probTgivenXs.shape[1],1))
        DKL[:, mask] = d1 + d2
        probTgivenXs = np.exp(-beta * (DKL)) * pts_emp[:, np.newaxis]
        probTgivenXs = probTgivenXs / np.tile(np.nansum(probTgivenXs, axis=0), (probTgivenXs.shape[0], 1))
        #probTgivenXs[np.isnan(probTgivenXs)] = 0
        probTgivenXs[:, np.isnan(probTgivenXs)[0, :]] = np.tile(pts_emp, (np.sum(np.isnan(probTgivenXs), axis=1)[0], 1)).T

    return probTgivenXs, PYgivenTs_update, pts_emp


def calc_IB_combained_third(p_t_given_x, pxs, pys, p_y_given_x, beta, iter, p_x_emp, p_y_given_x_emp, sampled_indexes, type,choosen_indeces, mask):
    p_yx = (p_y_given_x.T * pxs[:, np.newaxis]).T
    probTgivenXs = p_t_given_x
    #probTgivenXs[:, ~mask] = p_t_given_x[:, choosen_indeces[~mask]]
    not_mask_indexes = [si[0] for si in np.argwhere(~mask)]
    for i in range(0, iter):
        #choosen_indeces =np.random.choice(p_t_given_x.shape[1], np.sum(~mask), replace=False)
        pts_emp = np.dot(probTgivenXs, pxs)
        one_over_pt = 1. / (pts_emp + np.spacing(1))
        p_t_given_x_divide_p_t = np.multiply(probTgivenXs, one_over_pt[:, np.newaxis])
        PYgivenTs_update_s = np.dot(p_yx[:, :], p_t_given_x_divide_p_t[ :, :].T)
        PYgivenTs_update_before = PYgivenTs_update_s.T / (np.nansum(PYgivenTs_update_s, axis=0))[:, np.newaxis]
        PYgivenTs_update = PYgivenTs_update_before.T
        nan_indexes_y = np.unique([t[1] for t in np.argwhere(np.isnan(PYgivenTs_update)[:, :])])
        if nan_indexes_y.shape[0] > 0:
            injected_x = p_y_given_x[:, nan_indexes_y]
            PYgivenTs_update[:, nan_indexes_y] = injected_x
        d1 = np.tile(np.nansum(np.multiply(p_y_given_x, np.log(p_y_given_x+np.spacing(1))), axis=0), (probTgivenXs.shape[0], 1))
        d2 = np.dot(-np.log(PYgivenTs_update.T+np.spacing(1)), p_y_given_x)
        DKL = d1 + d2
        DKL[DKL<np.exp(-20)]=0
        probTgivenXs = np.exp(-beta * (DKL)) * pts_emp[:, np.newaxis]
        probTgivenXs = probTgivenXs / (np.nansum(probTgivenXs, axis=0)[np.newaxis,:])
        if True and len(choosen_indeces)>0:
            probTgivenXs[:, ~mask] = p_t_given_x[:,choosen_indeces ]

            #probTgivenXs[:, ~mask] = p_t_given_x[:,choosen_indeces[~mask] ]
        nan_indexes =np.unique([t[1] for t in np.argwhere(np.isnan(probTgivenXs)[:, :])])
        if nan_indexes.shape[0]>0:
            choosen_indeces_x = np.random.choice(p_t_given_x.shape[1], nan_indexes.shape[0], replace=False)
            injected_p_t_give_x = p_t_given_x[:, choosen_indeces_x]
            probTgivenXs[:, nan_indexes] = injected_p_t_give_x
    return probTgivenXs, PYgivenTs_update, pts_emp, pxs

def calc_IB(p_t_x, PXs, PYgivenXs, beta, iter):
    PYgivenXs = PYgivenXs.astype(np.longdouble)
    PXs = PXs.astype(np.longdouble)
    p_t_x = p_t_x.astype(np.longdouble)
    probTgivenXs = p_t_x
    for i in range(0, iter):
        pts = np.dot(probTgivenXs, PXs)
        ProbYGivenT_b = np.multiply(PYgivenXs, np.tile(PXs, (PYgivenXs.shape[0], 1)))
        PYgivenTs_update = np.dot(ProbYGivenT_b,
                                  np.multiply(probTgivenXs, np.tile(1. / (pts), (probTgivenXs.shape[1], 1)).T).T)
        d1 = np.tile(np.nansum(np.multiply(PYgivenXs, np.log(PYgivenXs)), axis=0), (probTgivenXs.shape[0], 1))
        d2 = np.dot(-np.log(PYgivenTs_update.T + np.spacing(1)), PYgivenXs)
        DKL = d1 + d2
        probTgivenXs = np.exp(-beta * (DKL)) * pts[:, np.newaxis]
        probTgivenXs = probTgivenXs / np.tile(np.nansum(probTgivenXs, axis=0), (probTgivenXs.shape[0], 1))
    pts = np.dot(probTgivenXs, PXs)
    return probTgivenXs, PYgivenTs_update, pts


"""Deprecated"""


def calcXI(probTgivenXs, PYgivenTs, PXs, PYs):
    probTgivenXs = probTgivenXs.astype(np.longdouble)
    PYgivenTs = PYgivenTs.astype(np.longdouble)
    # PYgivenXs = PYgivenXs.astype(np.longdouble)
    PXs = PXs.astype(np.longdouble)
    PYs = PYs.astype(np.longdouble)
    PTs = np.nansum(probTgivenXs * PXs, axis=1)

    Ht = np.nansum(-np.dot(PTs, np.log2(PTs)))
    # Ht_n = np.nansum(-np.dot(PTS_new, np.log2(PTS_new)))
    Htx = - np.nansum((np.dot(np.multiply(probTgivenXs, np.log2(probTgivenXs)), PXs)))
    # Htx_n = - np.nansum((np.dot(np.multiply(probTgivenXs_n, np.log2(probTgivenXs_n)), PXs_n)))
    # Hyt_n = - np.nansum(np.dot(np.multiply(PYgivenTs_n,  np.log2(PYgivenTs_n)), PTS_new))
    Hyt = - np.nansum(np.dot(np.multiply(PYgivenTs, np.log2(PYgivenTs)), PTs))
    # con_Htx= - np.nansum((np.dot(np.multiply(probTgivenXs_n, np.log2(probTgivenXs_n)), PXs[0:500])))
    # con_Hyt = - np.nansum((np.dot(np.multiply(probTgivenXs_n, np.log2(probTgivenXs_n)), PXs[0:500])))
    Hy = np.nansum(-PYs * np.log2(PYs))
    # IYT1 = np.sum([np.nansum(np.log2(prob_y_given_t / PYs) * (prob_y_given_t * prob_t))
    #               for prob_y_given_t, prob_t in zip(PYgivenTs.T, PTs)], axis=0)
    # Hyx = - np.nansum(np.dot(np.multiply(PYgivenXs,  np.log2(PYgivenXs)), PXs))
    IYT = Hy - Hyt
    ITX = Ht - Htx

    # IXY = Hy - Hyx
    # print (IXY,IYT1,Hyt )
    return ITX, IYT


def load_temp_data(name, initial_beta=0.9, max_beta=100, interval_beta=0.5):
    d = sio.loadmat(name + '.mat')
    y = d['y']
    PYX = np.squeeze(np.concatenate((y[None, :], 1 - y[None, :]), axis=0))
    PXs = np.ones(PYX.shape[1]) / PYX.shape[1]
    PXs = PXs[:]
    PXs = PXs / np.sum(PXs)
    PYX = PYX[:, :PXs.shape[0]]
    mybetaS = 2 ** np.arange(np.log2(initial_beta), np.log2(max_beta), interval_beta)
    mybetaS = mybetaS[::-1]
    to_add =np.random.rand(len(mybetaS))*2 - 1
    #to_add1 =np.random.rand(len(mybetaS))*2 - 1

    #print np.max(to_add), np.min(to_add),np.max(to_add1), np.min(to_add1)

    mybetaS = np.abs(mybetaS+mybetaS*to_add)
    mybetaS = np.sort(mybetaS)[::-1]
    PTX0 = np.eye(PXs.shape[0])
    PYs = np.mean(y)
    PYs = np.vstack(np.array([PYs, 1 - PYs]))
    F = d['F']
    return mybetaS, np.squeeze(PTX0), np.squeeze(PXs), np.squeeze(PYX), np.squeeze(PYs), F


def loadData(name='data/all'):
    d = sio.loadmat(name + '.mat')
    what_to_do = d['what_to_do'][0]
    pertub_probS = d['pertub_probS'][0]
    temperatureS = d['temperatureS'][0]
    temperatureS = [1e-5]
    pertub_probS = [0.2]
    # pertub_probS = [pertub_probS[0], pertub_probS[2]]
    return what_to_do, pertub_probS, temperatureS


def do_IB_iteation(pxs, pys, p_t_given_x, p_y_given_x, beta, iter):
    """Regulear IB iteration"""
    probTgivenXs, PYgivenTs_update, pts = calc_IB(p_t_given_x, pxs, p_y_given_x, beta, iter)
    ITX, IYT = iu.calc_information(probTgivenXs, PYgivenTs_update, pxs, pys)
    L = ITX - beta * IYT
    # print ITX, IYT
    return ITX, IYT, L, probTgivenXs, PYgivenTs_update, pts


def do_IB_iteation_combained(pxs, pys, p_tx, pyx, beta, iter, p_x_emp, p_t_given_x_emp, p_ygiven_x_emp,
                             sampled_indexes, ind,choosen_indeces, mask):
    """Combination of encoder and decoder from diffrnet type"""
    #probTgivenXs_new, p_y_given_t_new, pts = calc_IB(p_tx, pxs, pyx, beta, iter)
    #p_t_given_x_emp_new, p_y_given_t_emp_new, pts_emp = calc_IB(p_t_given_x_emp, p_x_emp, p_ygiven_x_emp, beta, iter)
    #p_t_given_x_comb_new, p_y_given_t_comb, pts_comb = calc_IB_combained(p_tx, pxs, pyx, beta, iter, p_x_emp,
    #                                                                    p_ygiven_x_emp, sampled_indexes, 0)
    if ind==0:
        p_t_given_x_comb_new_0, p_y_given_t_comb_0, pts_comb_0 = calc_IB_combained(p_tx.astype(np.longdouble), pxs.astype(np.longdouble),pys.astype(np.longdouble), pyx.astype(np.longdouble)
                                                                               , beta, iter, p_x_emp.astype(np.longdouble),
                                                                         p_ygiven_x_emp.astype(np.longdouble), sampled_indexes, 1)
    elif ind==1:
        p_t_given_x_comb_new_0, p_y_given_t_comb_0, pts_comb_0 = calc_IB_combained_second(p_tx.astype(np.longdouble),
                                                                                          pxs.astype(np.longdouble),
                                                                                          pys.astype(np.longdouble),
                                                                                          pyx.astype(np.longdouble)
                                                                                          , beta, iter,
                                                                                          p_x_emp.astype(np.longdouble),
                                                                                          p_ygiven_x_emp.astype(
                                                                                              np.longdouble),
                                                                                          sampled_indexes, 1)
    elif ind==2:
        p_t_given_x_comb_new_0, p_y_given_t_comb_0, pts_comb_0, pxs = calc_IB_combained_third(p_tx.astype(np.longdouble),
                                                                                          pxs.astype(np.longdouble),
                                                                                          pys.astype(np.longdouble),
                                                                                          pyx.astype(np.longdouble)
                                                                                          , beta, iter,
                                                                                          p_x_emp.astype(np.longdouble),
                                                                                          p_ygiven_x_emp.astype(
                                                                                              np.longdouble),
                                                                                          sampled_indexes, 1,choosen_indeces, mask)
    elif ind ==3:
        p_t_given_x_comb_new_0, p_y_given_t_comb_0, pts_comb_0 = calc_IB(p_tx, pxs, pyx, beta, iter)
    ITXs, IYYs = [], []
    for type in range(1):
        """
        if type == 2:
            ITX, IYT = iu.calc_information(p_t_given_x_comb_new, p_y_given_t_comb, pxs, pys, pts_comb,
                                           p_t_given_x_emp_new,
                                           p_y_given_t_emp_new, p_x_emp, pts_emp, 0)
        """
        """
        if type == 2:
            ITX, IYT = iu.calc_information(p_t_given_x_comb_new_0, p_y_given_t_comb_0, pxs, pys, pts_comb_0)

            ITX, IYT = ITX, IYT*(4096/float(1500))
        else:
            ITX, IYT = iu.calc_information(probTgivenXs_new, p_y_given_t_new, pxs, pys, pts, p_t_given_x_emp_new,
                                           p_y_given_t_emp_new, p_x_emp, pts_emp, type)
        """
        ITX, IYT = iu.calc_information_1(p_t_given_x_comb_new_0, p_y_given_t_comb_0, pxs, pys, pts_comb_0)
        #print ITX, IYT
        ITXs.append(ITX)
        IYYs.append(IYT)
    # ITX_emp, IYT_emp = iu.calc_information(p_t_given_x_emp_new, p_y_given_t_emp_new, p_x_emp, pys)
    # ITX_combine, IYT_combine = iu.calc_information(probTgivenXs_new, p_y_given_t_emp_new, pxs, pys)
    ind = 0
    L = ITXs[ind] - beta * IYYs[ind]
    #print ITX , IYT
    # print ITX_full, IYT_full, ITX_emp, IYT_emp
    # all_information_x = np.array((ITX_full, ITX_emp))
    # all_information_y = np.array((IYT_full, IYT_emp))
    return ITXs[ind], IYYs[
        ind], L, p_t_given_x_comb_new_0, p_y_given_t_comb_0, pts_comb_0, p_t_given_x_comb_new_0, p_y_given_t_comb_0, pts_comb_0, ITXs, IYYs, pxs


def do_annealing(x, y, PCX0, PYX, ITER, beta, what_to_do, pertub_probS, temperatureS, emp_x_indeces, x_n, PYX_n, PCX0_n,
                 sampled_indexes, ind,choosen_indeces, mask):
    Ls = []  # list of Lagrangian values
    # Keep track of best over all
    bbL = 9999999999
    bbPCX, bbPTY, bbPTC, bbPYC = [], [], [], []
    cnt = 0  # count all iterations...
    PCX = PCX0
    PCX_n = PCX0_n
    ITERsmall = 3
    ITERpertub = 1

    for temperature in temperatureS:
        for pertub_prob in pertub_probS:
            iter = 1
            while (iter <= ITER):
                cnt = cnt + 1
                # Run current solution
                IXT, ITY, L, PCX, PYC, PTs, PCX_n, PYC_n, pts_n, all_information_x, all_information_y, x = do_IB_iteation_combained(
                    x, y, PCX, PYX, beta, ITERsmall, x_n, PCX_n, PYX_n, sampled_indexes, ind,choosen_indeces, mask)
                iter = iter + ITERsmall
                # Pertub solution and run once
                Z = np.random.rand(PCX.shape[0], PCX.shape[1])
                Z = np.divide(Z, np.tile(np.sum(Z, axis=0), (Z.shape[0], 1)))
                PCX1 = (1 - pertub_prob) * PCX + pertub_prob * Z
                PCX1 = np.divide(PCX1, np.tile(np.sum(PCX1,axis=0), (PCX1.shape[0], 1)))
                Z_n = np.random.rand(PCX_n.shape[0], PCX_n.shape[1])
                Z_n = np.divide(Z_n, np.tile(np.sum(Z_n, axis=0), (Z_n.shape[0], 1)))
                PCX1_n = (1 - pertub_prob) * PCX_n + pertub_prob * Z_n
                PCX1_n = np.divide(PCX1_n, np.tile(np.sum(PCX1_n, axis=0), (PCX1_n.shape[0], 1)))
                if (pertub_prob > 0.15) and (pertub_prob <= 0.7):
                    nn = np.random.randint(PCX1.shape[1])
                    PCX1[np.random.permutation(PCX1.shape[0]), nn]
                IXT1, ITY1, L1, PCX1, PYC1, PTs1, PCX1_n, PYC1_n, PTs1_n, all_information_x1, all_information_y1,x1 = do_IB_iteation_combained(
                    x, y, PCX1, PYX, beta, ITERpertub, x_n, PCX1_n, PYX_n, sampled_indexes, ind,choosen_indeces, mask)
                iter = iter + ITERpertub
                # Local search move
                if L1 <= L:
                    # print (IXT1, ITY1)
                    x = x1
                    PCX = PCX1
                    PYC = PYC1
                    PTs = PTs1
                    IXT_t, ITY_t = IXT1, ITY1
                    all_information_x, all_information_y = all_information_x1, all_information_y1
                    PCX_n = PCX1_n
                    PYC_n = PYC1_n
                    PTs_n = PTs1_n
                    IXT_n, ITY_n = IXT1, ITY1
                    L = L1
                if L < bbL:
                    # print (IXT, ITY)
                    # print ("3")
                    bbPCX = PCX
                    bbPYC = PYC
                    bbPTs = PTs
                    bbL = L
                    bbPCX_n = PCX_n
                    all_information_x_t, all_information_y_t = all_information_x, all_information_y
                if L1 > L:
                    if np.random.rand(1) < np.exp((L - L1) / temperature):
                        PCX = PCX1
                        PCX_n = PCX1_n

                        PYC = PYC1
                        PTs = PTs1
                        L = L1
                Ls = [Ls, L]
    return bbPCX, bbPYC, bbPTs, bbL, Ls, bbPCX_n, IXT, ITY, all_information_x_t, all_information_y_t, x

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def choosen_method(percent_index, method_type,not_mask_indexes,PXs,mask,emp_x_indeces, F, k_neighbors, p_y_given_x,p_t_given_x):

    if method_type ==0:
        choosen_indeces = np.random.choice(PXs.shape[0], np.sum(~mask), replace=False)
    elif method_type ==1:
        #emp_x_indeces = np.arange(0, PXs.shape[0])
        coosen_indexs_array = []
        all_norms = np.array([np.linalg.norm(F[emp_x_indeces[index], :]) for index in range(emp_x_indeces.shape[0])])
        for i in range(len(not_mask_indexes)):
            current_norm = np.linalg.norm(F[not_mask_indexes[i], :])
            # idx = (np.abs(all_norms - current_norm)).argmin()
            x = np.abs(all_norms - current_norm)
            ids = np.where(x == x.min())[0]
            idx = np.random.choice(ids, 1)[0]
            coosen_indexs_array.append(emp_x_indeces[idx])
        choosen_indeces = np.array(coosen_indexs_array)
        if len(not_mask_indexes) == 0:
            choosen_indeces = []

    elif method_type ==2:
        choosen_indeces = []
        knn = neighbors.KNeighborsRegressor(k_neighbors, weights='distance')
        for index, point in enumerate(not_mask_indexes):
            #print index
            target_p_y_given_x = p_y_given_x[:,point]
            D_KLs = np.array([iu.KL(c_p_y_given_x, target_p_y_given_x) for  c_p_y_given_x in p_y_given_x[:,: ].T ])
            #y = knn.fit(F[emp_x_indeces,:], D_KLs).predict(F[point, :])
            ids = np.where(D_KLs == D_KLs.min())[0]
            idx = np.random.choice(ids, 1)[0]
            choosen_indeces.append(idx)
        choosen_indeces = np.array(choosen_indeces)

    elif method_type ==3:
        choosen_indeces = []
        rand_int = np.random.randint(0,20)
        net = inet.informationNetwork(num_of_samples = percent_index, rand_int = rand_int)
        net.print_information()
        net.run_network()
        pred = net.inferance(F)
        info = net.calc_information_last()
        ind_worng = []
        for index, point in enumerate(not_mask_indexes):
            vec = np.abs(p_y_given_x[0] - pred[point, 0])
            ids = np.where(vec == vec.min())[0]
            #ids = np.where(np.abs(p_y_given_x[0] - pred[x_j]))
            #if vec.min()>0.006:
            ind_worng.append(point)
            idx = np.random.choice(ids, 1)[0]

            choosen_indeces.append(idx)
        print len(ind_worng)
        p_y_given_x =p_y_given_x.astype(np.float64)
        #p_y_given_x[:, :] = np.vstack(np.squeeze(info)[5,3]).T
        #p_t_given_x  = np.vstack(np.vstack(np.squeeze(info))[0,2])
        choosen_indeces = np.array(choosen_indeces).astype(np.int32)
        ind_worng = np.array(ind_worng)
        #print ind_worng.shape
        if len(not_mask_indexes) == 0:
            choosen_indeces = []
        mask = np.ones((PXs.shape[0], 1), dtype=bool)  # np.ones_like(a,dtype=bool)
        #choosen_indeces = choosen_indeces[]
        if len(not_mask_indexes) > 0:
            mask[ind_worng] = False
            mask = mask[:, 0]

            p_y_given_x[:,~mask] = p_y_given_x[:,choosen_indeces]

            #choosen_indeces = np.array(range(0, p_y_given_x.shape[1]))
    if len(not_mask_indexes) == 0:
        choosen_indeces = []
    return choosen_indeces, p_y_given_x, p_t_given_x, mask


def run_annealing(percent_index, mybetaS, PTX0, PXs, PYX, PYs, what_to_do, pertub_probS, temperatureS, ITER, emp_x_indeces, ind,initial_beta,interval_beta, max_beta, F):
    NB = len(mybetaS)
    PCXs, PYCs, ICXs, IYCs, PTs_all = [], [], [], [], []
    PCX = PTX0
    x_n = PXs[emp_x_indeces]
    x_n = x_n / np.sum(x_n)
    PYX_n = PYX[:, emp_x_indeces]
    PCX_n = np.eye(x_n.shape[0])
    t = 0
    k =0
    mask = np.zeros((PXs.shape[0], 1), dtype=bool)  # np.ones_like(a,dtype=bool)
    mask[emp_x_indeces] = True
    mask = mask[:, 0]
    # p_y_given_x_emp = p_y_given_x
    method_type = 3
    k_neighbors = 5
    s = np.argwhere(~mask)
    not_mask_indexes = [si[0] for si in s]
    choosen_indeces,PYX, PCX, mask = choosen_method(percent_index, method_type, not_mask_indexes, PXs, mask, emp_x_indeces, F, k_neighbors, PYX,PCX)

    while (k<NB):
    #for k in range(0, NB):
        mybeta = mybetaS[k]
        PCX_old = PCX
        print ('Running beta ={0:.2f}, indexs {1} from {2}'.format (mybeta, k, NB))
        [PCX, PYC, PTs, bL, Ls, PCX_n, ICX, IYC, all_ICX, all_IYC, PXs] = do_annealing(PXs, PYs, PCX, PYX, ITER, mybeta,
                                                                                  what_to_do,
                                                                                  pertub_probS, temperatureS,
                                                                                  emp_x_indeces, x_n, PYX_n, PCX_n,
                                                                                  emp_x_indeces, ind,choosen_indeces, mask)
        print all_ICX[0], all_IYC[0]
        if all_ICX[0]<.1 and k == 0 and t<1:
            max_beta *=100
            #interval_beta *=3
            mybetaS = 2 ** np.arange(np.log2(initial_beta), np.log2(max_beta), interval_beta)
            mybetaS = mybetaS[::-1]
            t+=1
            PCX = PCX_old
            NB = len(mybetaS)

            continue
        #PCXs.append(PCX)
        #PYCs.append(PYC)
        #PTs_all.append(PTs)
        print ('Final information - ', all_ICX, all_IYC)
        ICXs.append(all_ICX)
        IYCs.append(all_IYC)
        k+=1
    return ICXs, IYCs


def main_from_source(percent_index, mybetaS, PTX0, PXs, PYX, PYs, emp_x_indeces, ind,initial_beta,interval_beta, max_beta,  F, ITER=7):
    what_to_do, pertub_probS, temperatureS = loadData()
    [ICX, IYC] = run_annealing(percent_index, mybetaS, PTX0, PXs, PYX, PYs, what_to_do, pertub_probS, temperatureS, ITER,
                                         emp_x_indeces, ind,initial_beta,interval_beta, max_beta,F)
    return ICX, IYC


def calc_reverase_annleaing(percent_index, name, num_of_indices, ind,max_beta,initial_beta = 0.9,interval_beta =0.5):
    np.random.seed(None)
    [mybetaS, PTX0, PXs, PYX, PYs, F] = load_temp_data(name, max_beta=max_beta, initial_beta=initial_beta, interval_beta=interval_beta)
    emp_x_indeces = np.sort(np.random.choice(PTX0.shape[0], num_of_indices, replace=False))
    [ICX, IYC] = main_from_source(percent_index, mybetaS, PTX0, PXs, PYX, PYs, emp_x_indeces, ind,initial_beta,interval_beta, max_beta, F)
    return np.array(ICX), np.array(IYC)


def main():
    ITER = 7
    # [PTX0, PXs, PYX, PYs, what_to_do, pertub_probS, temperatureS, mybetaS]= loadData()
    # [ICX, IYC] = run_annealing(mybetaS,PTX0, PTY0, PXs, PYX, PYs, what_to_do, pertub_probS, temperatureS, mybetaS,ITER)
    name = 'var_u'
    [mybetaS, PTX0, PXs, PYX, PYs] = load_temp_data(name)
    # print (PYX.shape)
    num_of_indices = 500
    emp_x_indeces = np.sort(np.random.choice(PTX0.shape[0], num_of_indices, replace=False))
    print iu.calc_information(PTX0, PYX, PXs, PYs)

    [ICX, IYC] = main_from_source(mybetaS, PTX0, PXs, PYX, PYs, emp_x_indeces)
    with open('data.pickle', 'wb') as f:
        cPickle.dump([ICX, IYC], f, protocol=2)
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    ax4.plot(ICX, IYC)
    # fig4.save('plot_fi.png')


if __name__ == "__main__":
    main()

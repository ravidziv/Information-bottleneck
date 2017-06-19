import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from ib import reverese_annealing as ra
from ib import reverese_annealing_new as ra_n

import scipy.io as sio
import numpy as np

from joblib import Parallel, delayed
import multiprocessing

NUM_CORES = multiprocessing.cpu_count()

# Force matplotlib to not use any Xwindows backend.
#matplotlib.use('Agg')
import cPickle

def plot_figure():
    initial_beta = 0.8
    interval_beta = 0.005
    name = 'reg_90_s72_total_.pickle'
    max_beta = 2000

    mybetaS = 2 ** np.arange(np.log2(initial_beta), np.log2(max_beta), interval_beta)

    mybetaS = mybetaS[::-1]
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.logspace(-1, 0, mybetaS[0]+ 1)]
    with open(name, 'rb') as f:
        total_data = cPickle.load(f)[0]
    arrays = total_data[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(len(arrays)):
        x = arrays[0][0]
        y = arrays[0][1]

        x1 = arrays[0][2]*1.25
        y1 = arrays[0][3]
        y1 = y1 / np.max(y)
        y = y / np.max(y)
        ax.plot(x, y, ':', color='r', alpha=1)
        ax.scatter(x1, y1, color='g', alpha=.5)
        #ax.set_xlim([0,12])
    ax.legend(loc='best')
    ax.set_xlabel('I(X;T)')
    ax.set_ylabel('I(T;Y)')

    plt.show()

def main():
    name = 'data/regular_90'
    str_name = 'reg_90_s7'
    save_probs = False
    ICX_array = []
    IYC_array = []
    num_of_indices = 1500
    ind =2
    max_beta = 1000
    #num_of_indices_array_init = [20]
    num_of_indices_array_init = np.linspace(10, 100, num=40, dtype= np.int32)[:]
    num_of_indices_array_init = [4]
    #print NUM_CORES
    num_of_indices_array = np.array([round(i / float(100) * 4096) for i in num_of_indices_array_init]).astype(np.int32)
    total_data = []
    num_of_repeat = 1
    method_type = 1
    k_neighbors = 1
    print NUM_CORES
    #total_data = Parallel(n_jobs=1)(delayed(ra.calc_reverase_annleaing)(name, num_of_indices_array[0], ind, max_beta)
    #                                for i in range(0, num_of_repeat))

    for i in range(0,num_of_repeat):
        print ('Trial number {0}'.format(i))
        #arrays = Parallel(n_jobs=NUM_CORES)(delayed(ra.calc_reverase_annleaing)(name,num_of_indices, ind,method_type, k_neighbors,max_beta)
        #                                for num_of_indices in num_of_indices_array)
        arrays = [ra_n.calc_reverase_annleaing(name, num_of_indices,method_type, k_neighbors,  ind, max_beta)
            for num_of_indices in num_of_indices_array]
        cPickle.dump([arrays], file(str_name+str(ind)+'_'+str(i)+'.pickle', 'wb'))
        total_data.append(arrays)
    cPickle.dump([total_data], file(str_name + str(ind) + '_total_' + '.pickle', 'wb'))
    """
    for num_of_indices in num_of_indices_array:
        print ('Indexs number -', num_of_indices)
        ICXs, IYCs,PCXs, PYCs = ra.calc_reverase_annleaing(name,num_of_indices, ind, max_beta)
        ICX_array.append(ICXs)
        IYC_array.append(IYCs)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    arr1 = arrays[0][0]
    arr2 = arrays[0][1]
    for i in range(len(arrays)):
        x = arrays[0][0]
        y = arrays[0][1]
        x1 = arrays[0][2]
        y1 =arrays[0][3]

        ax.scatter(x,y, color = 'r',alpha=.5)
        ax.scatter(x1,y1, color = 'g', alpha=.5)
    ax.set_xlim([0, 12])
    ax.legend(loc= 'best')
    plt.show()

    cPickle.dump([arrays], file(name+'_informationCurve1.pickle', 'wb'))
    """
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    leg = []
    for i in range(ICXs.shape[1]):
        ax4.plot(ICXs[:,i], IYCs[:,i])
        leg.append(str(i))
    ax4.legend(leg, loc= 'best')
    fig4.savefig(name + 'Fig2.png')

    plt.show()
    """
if __name__ == "__main__":
    main()
    #plot_figure()
    plt.show()
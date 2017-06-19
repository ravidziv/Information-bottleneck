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
#matplotlib.use('Agg')
import cPickle
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1') or v==True:
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def plot_figure(name):
    name = 'reg_90_s92_total_.pickle'
    with open(name, 'rb') as f:
        total_data = cPickle.load(f)[0]
    arrays = total_data[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(arrays)):
        x = arrays[0][0]
        y = arrays[0][1]

        x1 = arrays[0][2]
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

def main(args):
    name = 'data/regular_90'
    str_name = 'reg_90_s9'
    save_probs = False
    ICX_array = []
    IYC_array = []
    num_of_indices = 1500
    ind =2
    #num_of_indices_array_init = [20]
    num_of_indices_array_init = np.linspace(10, 100, num=40, dtype= np.int32)[:]
    num_of_indices_array_init = [4]
    #print NUM_CORES
    num_of_indices_array = np.array([round(i / float(100) * 4096) for i in num_of_indices_array_init]).astype(np.int32)
    total_data = []
    num_of_repeat = 1
    method_type = 1
    k_neighbors = 1
    #total_data = Parallel(n_jobs=1)(delayed(ra.calc_reverase_annleaing)(name, num_of_indices_array[0], ind, max_beta)
    #                                for i in range(0, num_of_repeat))

    for i in range(0,num_of_repeat):
        print ('Trial number {0}'.format(i))
        #arrays = Parallel(n_jobs=NUM_CORES)(delayed(ra.calc_reverase_annleaing)(name,num_of_indices, ind,method_type, k_neighbors,max_beta)
        #                                for num_of_indices in num_of_indices_array)
        arrays = [ra_n.calc_reverase_annleaing(name, num_of_indices,method_type, k_neighbors,  ind, args.max_beta)
            for num_of_indices in num_of_indices_array]
        cPickle.dump([arrays], file(str_name+str(ind)+'_'+str(i)+'.pickle', 'wb'))
        total_data.append(arrays)
    cPickle.dump(total_data, file('data/' +str_name + str(ind) + '_total_' + '.pickle', 'wb'))
    fig = plt.figure()
    ax = fig.add_subplot(111)
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
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-run',
                        '-r', dest="run_IB",type=str2bool, nargs='?', const=True,default=True,
                       help='Run the reverse annleaing of the IB or plot if')
    parser.add_argument('-max_beta',
                        '-mb', dest="max_beta", type=int, default=10000,
                        help='The maximal value of beta')

    args = parser.parse_args()
    if args.run_IB:
        main(args)
    else:
        plot_figure()
    plt.show()
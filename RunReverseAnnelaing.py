import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from ib import reverese_annealing as ra
import scipy.io as sio
import numpy as np

from joblib import Parallel, delayed
import multiprocessing

NUM_CORES = multiprocessing.cpu_count()

# Force matplotlib to not use any Xwindows backend.
#matplotlib.use('Agg')
import cPickle

def main():
    name = 'data/var_u'
    str_name = 'logs/var_u_y_k'
    save_probs = False
    ICX_array = []
    IYC_array = []
    num_of_indices = 1500
    ind =2
    max_beta = 2500
    #num_of_indices_array_init = [100]
    num_of_indices_array_init = np.linspace(10, 100, num=100, dtype= np.int32)[:]
    #num_of_indices_array_init = [100]
    #print NUM_CORES
    num_of_indices_array = np.array([round(i / float(100) * 4096) for i in num_of_indices_array_init]).astype(np.int32)
    total_data = []
    num_of_repeat = 50
    print NUM_CORES
    print num_of_indices_array
    print '++++++++++++++++++', num_of_indices
    """"
    total_data_i = Parallel(n_jobs=NUM_CORES)(
        delayed(ra.calc_reverase_annleaing)(percent_index, name, num_of_indices, ind, max_beta)
        for percent_index, num_of_indices in zip(num_of_indices_array_init, num_of_indices_array))
    """
    for percent_index, num_of_indices in zip(num_of_indices_array_init, num_of_indices_array):
        print '++++++++++++++++++', num_of_indices
        total_data_i = Parallel(n_jobs=NUM_CORES)(delayed(ra.calc_reverase_annleaing)(percent_index, name, num_of_indices, ind, max_beta)
                                    for i in range(0, num_of_repeat))

        total_data.append(total_data_i)
        cPickle.dump(total_data_i, file(str_name + str(ind) + '_total_inner' +str(num_of_indices) + '.pickle', 'wb'))


    cPickle.dump(total_data, file(str_name + str(ind) + '_total_' + '.pickle', 'wb'))

    """"
    for i in range(3,num_of_repeat):
        print ('Trial number {0}'.format(i))
        arrays = Parallel(n_jobs=1)(delayed(ra.calc_reverase_annleaing)(name,num_of_indices, ind, max_beta)
                                        for num_of_indices in num_of_indices_array)
        cPickle.dump([arrays], file(str_name+str(ind)+'_'+str(i)+'.pickle', 'wb'))
        total_data.append(arrays)
    """
    #cPickle.dump([total_data], file(str_name + str(ind) + '_total_' + '.pickle', 'wb'))
    """
    for num_of_indices in num_of_indices_array:
        print ('Indexs number -', num_of_indices)
        ICXs, IYCs,PCXs, PYCs = ra.calc_reverase_annleaing(name,num_of_indices, ind, max_beta)
        ICX_array.append(ICXs)
        IYC_array.append(IYCs)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(arrays)):
        x = arrays[i][0]
        y = arrays[i][1]

        ax.plot(x,y, label = str(num_of_indices_array_init[i])+'%')
    ax.legend(loc= 'best')
    """
    if save_probs:
        cPickle.dump([ICXs, IYCs,PCXs, PYCs], file(name+'_informationCurve1.pickle', 'wb'))
    else:
        cPickle.dump([ICXs, IYCs,PCXs, PYCs], file(name+'_informationCurve1.pickle', 'wb'))
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
    plt.show()

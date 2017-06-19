import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('TkAgg')
import  cPickle
import numpy as np
import os
import matplotlib.pyplot as plt
def get_data(name):
    """Load data from the given name"""
    ws, loss_train_data, loss_test_data, information_each_neuron, norms1, norms2, gradients,epochs,train_data,test_data,loss_test_data,eig_vecs =\
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0
    #new version
    if os.path.isfile(name + 'data.pickle'):
        curent_f = open(name + 'data.pickle', 'rb')
        d2 = cPickle.load(curent_f)
        print d2['params']
        #epochs = d2['params']['epochsInds']
        #rint d2['information'].shape
        I_XT_array2 = np.array(d2['information'])[:,:,:,:,:,:,0]
        I_TY_array2 =np.array( d2['information'])[:,:,:,:,:,:,1]
        epochs = np.arange(I_TY_array2.shape[0])

        data = np.array([I_XT_array2, I_TY_array2])
        train_data = d2['train_error']
        test_data = d2['test_error']
        params = d2['params']
        if 'ws_all' in d2:
            ws = d2['ws_all']
        if 'information_each_neuron' in d2:
            information_each_neuron = d2['information_each_neuron']
        if 'loss_train' in d2 and 'loss_test' in d2 :
            loss_train_data =d2['loss_train']
            loss_test_data = d2['loss_test']
        if 'var_grad_val' in d2:
            gradients = d2['var_grad_val']
        if 'eig_vecs' in d2:
            eig_vecs = d2['eig_vecs']
        if 'l1_norms' in d2 and 'l2_norms' in d2:
            norms1 = d2['l1_norms']
            norms2 = d2['l2_norms']
        epochsInds = (params['epochsInds']).astype(np.int)
        #epochsInds = np.arange(0, data.shape[1])
        normalization_factor = 1
    #Old version
    else:
        curent_f = open(name, 'rb')
        d2 = cPickle.load(curent_f)
        data1 = d2[0]
        data =  np.array([data1[:, :, :, :, :, 0], data1[:, :, :, :, :, 1]])
        params = d2[-1]
        #Convert log e to log2
        normalization_factor =1/np.log2(2.718281)
        epochsInds = np.arange(0, data.shape[4])
        #epochsInds = np.round(2 ** np.ar,ange(np.log2(1), np.log2(10000), .01)).astype(np.int)
    return data,epochs,train_data,test_data,ws,params,epochsInds,normalization_factor,information_each_neuron,loss_train_data,loss_test_data, \
           norms1, norms2,gradients,eig_vecs


import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def try_int(s):
    "Convert to integer if possible."
    try: return int(s)
    except: return s

def natsort_key(s):
    "Used internally to get a tuple by which s is sorted."
    import re
    return map(try_int, re.findall(r'(\d+|\D+)', s))

def natcmp(a, b):
    "Natural string comparison, case sensitive."
    return cmp(natsort_key(a), natsort_key(b))

def natcasecmp(a, b):
    "Natural string comparison, ignores case."
    return natcmp(a.lower(), b.lower())


def main():
    names = ['logs/var_u_y_k2_total_']
    import glob
    names =  glob.glob("logs/new_logs/*.pickle")
    names.sort(natcasecmp)
    #names = ['data/reg_90_y_k2_total_t','data/reg_90_y_k2_total_t1','data/reg_90_y_k2_total_t2']
    #names = [name+str(i) for i in range(1, 9)]
    samples = np.linspace(10, 100, num=50, dtype= np.int32)[:]
    #samples = [3, 8, 11, 14, 20, 26, 32, 38, 45, 52, 59, 66, 74, 82, 90]
    total_arrays = []
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, 50)]
    for name in names:
        with file(name, 'rb') as f:
            total_array = cPickle.load(f)
        total_arrays.append(total_array)
    total_arrays = np.squeeze(np.array(total_arrays)).swapaxes(0,1)
    #samples = np.arange(30)
    # print num_of_indices_array_init[samples]
    x_vals = total_arrays[:, :, 0,:]
    y_vals = total_arrays[:, :, 1,:]

    # total_array = np.mean(np.array(total_array), axis=1)
    # min_x = np.min(total_array[:, 0, :])
    # max_x = np.max(total_array[:, 0, :])
    x_interp = np.linspace(0, 12, 1000)

    x_interp = map(lambda x: float(x), x_interp)
    for num_of_samples in range(x_vals.shape[1]):
        ys_interp_all = []
        for trial in range(x_vals.shape[0]):
            xs = x_vals[trial, num_of_samples,:]
            ys = y_vals[trial, num_of_samples,:]
            xs = np.insert(xs, 0, 12)
            ys = np.insert(ys, 0, 0.99)

            xs = np.append(xs, 0)
            ys = np.append(ys, 0)
            xs = map(lambda x: float(x), xs)
            ys = map(lambda x: float(x), ys)

            ys_interp = np.interp(x_interp, xs[::-1], ys[::-1])
            ys_interp_all.append(ys_interp)
        ys_interp_all_aveg = np.mean(np.array(ys_interp_all), axis=0)
        #line_width = 5 if num_of_samples ==3 else 2
        ax1.plot(x_interp, ys_interp_all_aveg, color=colors[num_of_samples],label = samples[num_of_samples], linewidth = 2)
    """"
    for i in range(total_array.shape[0]):
        x_val = total_array[i, 0,:]
        y_val = total_array[i, 1,:]
        ax1.plot(x_val, y_val, color = colors[i], label = samples[i])
    """
    new_names = ['data/r_DataName=var_u_samples_len=1_layersSizes=[[10, 7, 5, 4, 3]]_learningRate=0.0004_numEpochsInds=964_numRepeats=40_LastEpochsInds=9998_num_of_disribuation_samples=1_numEphocs=10000_batch=410/',
                 'data/rw_DataName=var_u_samples_len=1_layersSizes=[[10, 7, 5, 4, 3]]_learningRate=0.0004_numEpochsInds=964_numRepeats=40_LastEpochsInds=9998_num_of_disribuation_samples=1_numEphocs=10000_batch=410/',
                 'data/g_DataName=var_u_samples_len=1_layersSizes=[[10, 7, 5, 4, 3]]_learningRate=0.0004_numEpochsInds=964_numRepeats=40_LastEpochsInds=9998_num_of_disribuation_samples=1_numEphocs=10000_batch=410/',
                 'data/ui_DataName=var_u_samples_len=1_layersSizes=[[10, 7, 5, 4, 3]]_learningRate=0.0004_numEpochsInds=964_numRepeats=40_LastEpochsInds=9998_num_of_disribuation_samples=1_numEphocs=10000_batch=410/',
                 'data/us_DataName=var_u_samples_len=1_layersSizes=[[10, 8, 6, 5, 4, 4, 3]]_learningRate=0.0004_numEpochsInds=964_numRepeats=40_LastEpochsInds=9998_num_of_disribuation_samples=1_numEphocs=10000_batch=410/']
    colors = ['r', 'g', 'b','y', 'orange']
    for i, new_name in enumerate(new_names):
        if i>3:
            break
        data, epochs, train_data, test_data, ws, params, epochsInds, normalize_factor, information_each_neuron, loss_train_data, loss_test_data, norms1, \
        norms2, gradients, eigs = get_data(new_name)
        xs_s = np.mean(np.squeeze(data[0, :, :, :, :, :, :]), axis=0)
        ys_s = np.mean(np.squeeze(data[1, :, :, :, :, :, :]), axis=0)
        indx = (xs_s[-1,:]>2)*(xs_s[-1,:]<7)
        if i==1:
            ys_s[-1, -2:] += 0.03
        ys_s[-1,indx] +=0.01
        # for layer in range(xs_s.shape[3]):
        ax1.plot(xs_s[-1, :], ys_s[-1, :], linestyle ='--', marker = 'o',markersize=15, color=colors[i], linewidth = 4)
    #ax1.legend()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    # cbar_ax = f.add_axes(colorbar_axis)
    bar_font = 20
    cbar = fig1.colorbar(sm, ticks=[])
    cbar.ax.tick_params(labelsize=bar_font)
    cbar.set_label('Training samples', size=bar_font)
    cbar.ax.text(0.5, -0.01, '2%', transform=cbar.ax.transAxes,
                 va='top', ha='center', size=bar_font)
    cbar.ax.text(0.5, 1.0, '85%', transform=cbar.ax.transAxes,
                 va='bottom', ha='center', size=bar_font)


    return
    x_vals = total_array[:, :, 0]
    y_vals = total_array[:, :, 1]

    #total_array = np.mean(np.array(total_array), axis=1)
    #min_x = np.min(total_array[:, 0, :])
    #max_x = np.max(total_array[:, 0, :])
    x_interp = np.linspace(0, 12, 1000)

    x_interp = map(lambda x: float(x), x_interp)
    x_interp = map(lambda x: float(x), x_interp)
    #samples = [10]

    #num_of_indices_array = np.array([round(i / float(100) * 4096) for i in samples]).astype(np.int32)
    samples =np.arange(30)
    #print num_of_indices_array_init[samples]
    for num_of_samples in samples:
        ys_interp_all = []
        for trial in range(x_vals.shape[1]):
            xs = x_vals[num_of_samples, trial]
            ys = y_vals[num_of_samples, trial]
            xs = np.insert(xs, 0, 12)
            ys = np.insert(ys, 0, 0.99)
            xs = map(lambda x: float(x), xs)
            ys = map(lambda x: float(x), ys)

            ys_interp = np.interp(x_interp, xs[::-1], ys[::-1])
            ys_interp_all.append(ys_interp)
        ys_interp_all_aveg = np.mean(np.array(ys_interp_all), axis=0)
        ax1.plot(x_interp, ys_interp_all_aveg,color = colors[num_of_samples])
    new_name = 'data/r_DataName=var_u_samples_len=1_layersSizes=[[10, 7, 5, 4, 3]]_learningRate=0.0004_numEpochsInds=964_numRepeats=40_LastEpochsInds=9998_num_of_disribuation_samples=1_numEphocs=10000_batch=410/'
    data, epochs, train_data, test_data, ws, params, epochsInds, normalize_factor, information_each_neuron, loss_train_data, loss_test_data, norms1, \
    norms2, gradients, eigs = get_data(new_name)
    xs_s = np.mean(np.squeeze(data[0, :, :, :, :, :, :]), axis=0)
    ys_s = np.mean(np.squeeze(data[1, :, :, :, :, :, :]), axis=0)

    #for layer in range(xs_s.shape[3]):
    ax1.plot(xs_s[:,:],ys_s[:,:], color='r')

    #arrays = np.mean(np.array(total_arrays), axis=0)
    ax1.set_xlim([0, 12])
    ax1.set_ylim([0, 1.01])
    ax1.set_xlabel('$I(X;T)$')
    ax1.set_ylabel('$I(T;Y)$')

    """"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    max_index =72
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, max_index + 1)]
    #xs = np.mean(np.squeeze(np.array(arrays)[:2, :, 0, :, :]), axis=0)
    #ys = np.mean(np.squeeze(np.array(arrays)[:2, :, 1, :, :]), axis=0)
    xs = np.squeeze(np.array(arrays)[ :, 0, :, :])
    ys = np.squeeze(np.array(arrays)[:, 1, :, :])
    for i in range(max_index):
        x = xs[i, :]
        y = ys[i,:]
        #y = np.mean(np.squeeze(np.array(arrays)[:, i, 1, :, :]), axis=0)

        #x = np.insert(x, 0, 12)
        #y = np.insert(y, 0, 0.9)

        for beta in range(arrays[i]):
            x = arrays[:][i][0]
            y = arrays[:][i][1]
            x = np.array([j[0] for j in x])
            y = np.array([j[0] for j in y])
            x = np.insert(x,0, 12)
            y = np.insert(y,0,0.9)

        ax.plot(x,y, color = colors[i])
        ax.set_xlim([0,12])
        ax.set_ylim([0, 0.91])
    """
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    #cbar_ax = f.add_axes(colorbar_axis)
    bar_font = 20
    cbar = fig1.colorbar(sm, ticks=[])
    cbar.ax.tick_params(labelsize=bar_font)
    cbar.set_label('Training samples', size=bar_font)
    cbar.ax.text(0.5, -0.01, '10%', transform=cbar.ax.transAxes,
                 va='top', ha='center', size=bar_font)
    cbar.ax.text(0.5, 1.0, '100%', transform=cbar.ax.transAxes,
                 va='bottom', ha='center', size=bar_font)


    plt.show()

if __name__ == "__main__":
    main()
    plt.show()
import numpy as np
from sklearn.decomposition import PCA

def roma_ds(fluor):
    """
    Takes a 2-D numpy array of fluorescence time series (in many cells) and returns a downsampled version, following the
        filtering method published by Romaszko (threshold summed time-diff global network activity at 0.02)

    inputs---
        fluor: numpy array of fluorescence time series. rows are cells, columns are time points / frames
    outputs---
        fluor_ds: downsampled numpy array of fluorescence time series.
    """
    thresh1 = 20
    thresh2 = 30000000
    fluordiff = np.diff(fluor, axis=1)
    totF = np.sum(fluordiff, axis=0)
    fluor = fluor[:,np.logical_and(totF>thresh1,totF<thresh2)]
    
    return fluor

def max_ds(fluor, block=100):
    """
    Takes a 2-D numpy array of fluorescence time series (in many cells) and returns a 
    downsampled version by taking the max of every BLOCK frames

    inputs---
        fluor: numpy array of fluorescence time series. rows are cells, columns are time points / frames
        block: size of the max-pooling window
    outputs---
        fluor_ds: downsampled numpy array of fluorescence time series.
    """
    fluor_ds = np.empty((fluor.shape[0],0))

    for i in range(0,fluor.shape[1],block):
        ind = np.min([i+block,fluor.shape[1]-1])
        
        fluor_ds = np.append(fluor_ds,np.max(fluor[:,i:ind], axis=1)[:,None], axis=1)

    return fluor_ds

def mean_ds(fluor, block=100):
    """
    Takes a 2-D numpy array of fluorescence time series (in many cells) and returns a 
    downsampled version by taking the max of every BLOCK frames

    inputs---
        fluor: numpy array of fluorescence time series. rows are cells, columns are time points / frames
        block: size of the max-pooling window
    outputs---
        fluor_ds: downsampled numpy array of fluorescence time series.
    """
    fluor_ds = np.empty((fluor.shape[0],0))

    for i in range(0,fluor.shape[1],block):
        ind = np.min([i+block,fluor.shape[1]-1])
        
        fluor_ds = np.append(fluor_ds,np.mean(fluor[:,i:ind], axis=1)[:,None], axis=1)

    return fluor_ds

def dunn_ds(fluor):
    """
    Takes a 2-D numpy array of fluorescence time series (in many cells) and returns a 
    downsampled version by taking every 100th frame

    inputs---
        fluor: numpy array of fluorescence time series. rows are cells, columns are time points / frames
    outputs---
        fluor_ds: downsampled numpy array of fluorescence time series.
    """
    
    return np.diff(fluor[:,::100])

def smooth(fluor):
    """
    Takes a 2-D numpy array of fluorescence time series (in many cells) and returns a 
    numpy array of fluorescence that has been smoothed by summing up neighboring points

    f(x_t) = f(x_t-1) + f(x_t) + f(x_t+1)

    inputs---
        fluor: numpy array of fluorescence time series. rows are cells, columns are time points / frames
    outputs---
        smoothed numpy array of fluorescence time series.
    """

    smoothfluor = np.zeros((fluor.shape[0],fluor.shape[1]-2))

    cnt = 0
    for j in range(1,fluor.shape[1]-1):
        smoothfluor[:,cnt] = fluor[:,cnt-1] + fluor[:,cnt] + fluor[:,cnt+1]
        cnt += 1

    return smoothfluor

def regularize(fluor):
    """
    Takes a 2-D numpy array of fluorescence time series (in many cells) and returns a 
    numpy array of fluorescence that has been regularized to emphasize periods of low global network activity
    inputs---
        fluor: numpy array of fluorescence time series. rows are cells, columns are time points / frames
    outputs---
        rfluor: regularized numpy array of fluorescence time series.
    """
    rfluor = np.ones(fluor.shape)
    for j in range(rfluor.shape[1]):
        sumj = np.sum(fluor[:,j])
        if sumj > 0:
            rfluor[:,j] = (fluor[:,j]+1)**(1+1/sumj)

    return rfluor

def thresh(fluor, thr = 0.11):
    """
    Takes a 2-D numpy array of fluorescence time series (in many cells) and returns a 
    numpy array of fluorescence that has been thresholded to zero under thr

    inputs---
        fluor: numpy array of fluorescence time series. rows are cells, columns are time points / frames
        thr: value at whic hto threshold to zero
    """
    fluor[fluor<thr] = 0

    return fluor

def unscatter(fluor, positions):
    """
    Takes a 2-D numpy array of fluorescence time series (in many cells) and network position information 
        (spatial position), returns a 2-D numpy array of fluorescence time series with light-scattering effects removed

    inputs---
        fluor: numpy array of fluorescence time series. rows are cells, columns are time points / frames
        positions: 2-D numpy array of spatial positions. Rows are cells, columns are x/y-coordinates
    outputs---
        2-D numpy array of fluorescence time series, scattering effects removed. rows are cells, columns are time points / frames
    """
    lsc = 0.025
    Nn = fluor.shape[0]
    D=np.zeros((Nn,Nn))
    for i in range(Nn):
        for j in range(Nn):
            if i==j:
                D[i,i] = 1
            else:
                D[i,j]=((positions[i,0]-positions[j,0])**2+(positions[i,1]-positions[j,1])**2)**0.5;
                D[i,j]=0.15*np.exp(-(D[i,j]/lsc)**2)

    Dinv = np.linalg.inv(D)

    Xfluor = fluor.copy()

    #Mtest here is time points x neurons
    for j in range(fluor.shape[1]):
        b=Dinv@fluor[:,j]
        Xfluor[:,j] = b

    return Xfluor

def pairwise_poolprep_tuple(fluor_max, fluor_mean, connect, num_images_target=1.2e6):
    """
    Same as pairwise_prep tuple below, but creates a 6-row tensor that includes the max-pooled and
        mean-pooled fluorescence for each pair of cells and also the mean max-pooled and 
        mean mean-pooled activity across the entire network
    
    inputs---
        fluor_max: tuple of 2-D numpy array of max-pooled fluorescence time series. rows are cells, columns are 
            time points / frames
        fluor_mean: tuple of 2-D numpy array of mean-pooled fluorescence time series. rows are cells, columns are 
            time points / frames
        connect: tuple of 2-D numpy array connectivity matrix summarizing all possible pairwise connectivity.
        num_images_target: number of examples we wish to include. The final number is typically less than this, 
            as we aim to include the same number of all each positive example in the dataset (with random time offset).
    outputs---
        fluor_tf: 4-D pairwise numpy array ready for tensorflow
        label_tf: a 1-D numpy array labeling connectivity for each possible pair in the dataset
    """
    num_samples = 330

    # Calculate representation of each positive connected pair
    num_con = 0
    for i in range(len(connect)):
        cons = np.where(connect[i]==1)
        num_con += len(cons[0])
    
    num_con_reps = np.floor(num_images_target/2/num_con).astype('int')

    num_images = num_con_reps*num_con*2
    fluor_tf = np.empty((num_images, 6, num_samples, 1),dtype='float32')
    label_tf = np.zeros((num_images,2),dtype='float32')

    cnt = 0
    for k in range(len(connect)):
        cons = np.where(connect[k]==1)
        num_con = len(cons[0])

        fmax = fluor_max[k]
        fmean = fluor_mean[k]

        raw_samples = fmax.shape[1]

        avg_F_max = np.mean(fmax,axis=0)
        avg_F_mean = np.mean(fmean,axis=0)
    
        # Add connected pairs to tensor
        for i in range(num_con):
            for j in range(num_con_reps):
                startpos = np.random.randint(0,raw_samples-num_samples,1)[0]
                fluor_tf[cnt,:,:,0] = np.vstack((fmax[cons[0][i],startpos:startpos+num_samples],
                                                       fmean[cons[0][i],startpos:startpos+num_samples], 
                                                       fmax[cons[1][i],startpos:startpos+num_samples],
                                                       fmean[cons[1][i],startpos:startpos+num_samples], 
                                                       avg_F_max[startpos:startpos+num_samples],
                                                       avg_F_mean[startpos:startpos+num_samples]))
                label_tf[cnt,0] = 1
                cnt += 1
    
        # Find all non-connected pairs
        # There are typically too many non-connected pairs to have any repetitions in the training set
        noncons = np.where(connect[k]==0)
    
        # Sample randomly from noncons without replacement
        noncons_samp = (np.random.choice(noncons[0],num_images/2,replace=False), 
                        np.random.choice(noncons[1],num_images/2,replace=False))
        
        for i in range(np.ceil(num_images/2/len(connect)).astype('int')):
            if cnt >= fluor_tf.shape[0]:
                break
            startpos = np.random.randint(0,raw_samples-num_samples,1)[0]
            fluor_tf[cnt,:,:,0] = np.vstack((fmax[noncons_samp[0][i],startpos:startpos+num_samples],
                                                   fmean[noncons_samp[0][i],startpos:startpos+num_samples],  
                                                   fmax[noncons_samp[1][i],startpos:startpos+num_samples],
                                                   fmean[noncons_samp[1][i],startpos:startpos+num_samples],  
                                                   avg_F_max[startpos:startpos+num_samples],
                                                   avg_F_mean[startpos:startpos+num_samples]))
            label_tf[cnt,1] = 1
            cnt += 1

    print("target size of processed traces: {}. count var: {}".format(fluor_tf.shape[0],cnt))
                
                
    return fluor_tf, label_tf

def standardize_rows(np_arr):
    """
    Standardizes data row-wise by subtracted the row mean and dividng by the row standard deviation

    inputs---
        np_arr: 2-D numpy array
    outputs---
        standardized 2-D numpy array
    """

    np_arr = np_arr - np.mean(np_arr,axis=1)[:,None]
    np_arr = np_arr/np.std(np_arr,axis=1)[:,None]
    return np_arr


def pairwise_prep_tuple(fluor_, connect, num_images_target=1.2e6):
    """
    Same as pairwise_prep above, but combiend records across files/networks
    
    inputs---
        fluor: tuple of 2-D numpy array of fluorescence time series. rows are cells, columns are time points / frames
        connect: tuple of 2-D numpy array connectivity matrix summarizing all possible pairwise connectivity.
        num_images_target: number of examples we wish to include. The final number is typically less than this, as we aim
            to include the same number of all each positive example in the dataset (with random time offset).
    outputs---
        fluor_tf: 4-D pairwise numpy array ready for tensorflow
        label_tf: a 1-D numpy array labeling connectivity for each possible pair in the dataset
    """
    num_samples = 330

    # Calculate representation of each positive connected pair
    num_con = 0
    for i in range(len(connect)):
        cons = np.where(connect[i]==1)
        num_con += len(cons[0])
    
    num_con_reps = np.floor(num_images_target/2/num_con).astype('int')

    num_images = num_con_reps*num_con*2
    fluor_tf = np.empty((num_images, 3, num_samples, 1),dtype='float32')
    label_tf = np.zeros((num_images,2),dtype='float32')

    cnt = 0
    for k in range(len(connect)):
        cons = np.where(connect[k]==1)
        num_con = len(cons[0])
        fluor = fluor_[k]
        raw_samples = fluor.shape[1]

        avg_F = np.mean(fluor,axis=0)
    
        # Add conncted pairs to tensor
        for i in range(num_con):
            for j in range(num_con_reps):
                startpos = np.random.randint(0,raw_samples-num_samples,1)[0]
                fluor_tf[cnt,:,:,0] = np.vstack((fluor[cons[0][i],startpos:startpos+num_samples], 
                                                       fluor[cons[1][i],startpos:startpos+num_samples], 
                                                       avg_F[startpos:startpos+num_samples]))
                label_tf[cnt,0] = 1
                cnt += 1
    
        # Find all non-connected pairs
        # There are typically too many non-connected pairs to have any repetitions in the training set
        noncons = np.where(connect[k]==0)
    
        # Sample randomly from noncons without replacement
        noncons_samp = (np.random.choice(noncons[0],num_images/2,replace=False), 
                        np.random.choice(noncons[1],num_images/2,replace=False))
        
        for i in range(np.ceil(num_images/2/len(connect)).astype('int')):
            if cnt >= fluor_tf.shape[0]:
                break
            startpos = np.random.randint(0,raw_samples-num_samples,1)[0]
            fluor_tf[cnt,:,:,0] = np.vstack((fluor[noncons_samp[0][i],startpos:startpos+num_samples], 
                                                   fluor[noncons_samp[1][i],startpos:startpos+num_samples], 
                                                   avg_F[startpos:startpos+num_samples]))
            label_tf[cnt,1] = 1
            cnt += 1

    print("target size of processed traces: {}. count var: {}".format(fluor_tf.shape[0],cnt))
                
                
    return fluor_tf, label_tf

            
def valid_eval_tfomics(nnt, val_dat, N=14, fragLen=330):
    """
    Properly calidates current CNN filters by passing filters over retained validation set N number of times and averaging
    the set of predictions for each pair
    
    inputs---
        nnt: tfomics nntrainer object
        val_dat: 2-D numpy array of downsampled fluorescence traces
        N: number of separate start positions for each test fragment to be averaged for each pair
        fragLen: length of trained CNN filter, in time points/samples
    outputs---
        pred_lbl: 1-D numpy array of predicted connectivity
    """

    avg_F = np.mean(val_dat,axis=0)

    startgap = np.ceil((val_dat.shape[1] - fragLen)/N).astype('int')

    pred_lbl = np.zeros((val_dat.shape[0]*val_dat.shape[0],), dtype='float32')
    # Counter for the "pred_lbl" array
    cnt_u = 0
    for a in range(val_dat.shape[0]):
        if a%100 == 0:
            print('\r' + 'X'*(a//100))

        # Create batch array to send thru network
        im_eval = np.empty((N*val_dat.shape[0],3,fragLen,1), dtype='float32')

        # Count the number of traces in each batch
        cnt = 0

        for b in range(val_dat.shape[0]):

            for n in range(0, val_dat.shape[1] - fragLen, startgap):
                try:
                    im_eval[cnt,:,:,0] = np.vstack((val_dat[a,n:n+fragLen],
                                         val_dat[b,n:n+fragLen],
                                         avg_F[n:n+fragLen]))
                except:
                    from IPython.core.debugger import Tracer
                    Tracer()()

                cnt += 1

        #im_eval = np.tile(im_eval,(100,1,1,1))
        # Run batch through network
        test = {'inputs': im_eval, 'keep_prob_dense': 1, 'keep_prob_conv': 1, 'is_training': False}
        pred_stop = nnt.get_activations(test, layer='output')[:,0]
        # Average output over each group of N traces
        for u in range(0, len(pred_stop), N):
            pred_lbl[cnt_u] = np.mean(pred_stop[u:u+N])
            cnt_u += 1  
    return pred_lbl

def get_partial_corr_scores(fluor):
    """
    Uses a partial correlation coefficient workflow to find multivariate scores for each connection
    
    inputs---
        fluor: 2-D numpy array of fluorescence time series. rows are cells, columns are time points / frames
    outputs---
        pred_out: 2-D numpy array of partial correlation coefficients for each neuron pair
    """
    F = smooth(fluor)
    dF = np.diff(F,axis = 1)
    tF = thresh(dF)
    rF = regularize(tF)
    # pca = PCA(whiten=True, n_components = fluor.shape[0]*0.8).fit(rF.T)
    pca = PCA(whiten=True, n_components = 800).fit(rF.T)
    pred = -pca.get_precision()

    return scale(pred)

# These helper functions used for getting partial correlation coefficients out of the precision matrix
def min_diagonal(X):
    np.fill_diagonal(X,X.min())
    return X

def min_max(X):
    X_scale = X.ravel() - X.min()
    X_scale /= X_scale.max()
    return X_scale.reshape(X.shape)

def scale(X):
    return min_max(min_diagonal(X))

def pairwise_prep_tuple_partialcorr(fluor_, connect, pcorr_, num_images_target=1.2e6, represent=50, num_samples = 330):
    """
    Same as pairwise_prep_tuple_partialcorr above, but adds an extra line for the partial correlation coefficient
    
    inputs---
        fluor: tuple of 2-D numpy array of fluorescence time series. rows are cells, columns are time points / frames
        connect: tuple of 2-D numpy array connectivity matrix summarizing all possible pairwise connectivity.
        pcorr_: tuple of 2-D numpy array of partial correlation coefficients for each neuron pair
        num_images_target: number of examples we wish to include. The final number is typically less than this, as we aim
            to include the same number of all each positive example in the dataset (with random time offset).
        represent: int representing the percent representation of positive examples in the final data structure
        num_samples = int. number of samples in extracted from time series for each example in data structure
    outputs---
        fluor_tf: 4-D pairwise numpy array ready for tensorflow
        label_tf: a 1-D numpy array labeling connectivity for each possible pair in the dataset
    """
    


    # Calculate representation of each positive connected pair
    num_con = 0
    for i in range(len(connect)):
        cons = np.where(connect[i]==1)
        num_con += len(cons[0])
    
    num_con_reps = np.floor(represent*num_images_target/100/num_con).astype('int')

    num_images = num_con_reps*num_con + num_con_reps*num_con*(100-represent)/represent
    fluor_tf = np.empty((num_images, 4, num_samples, 1),dtype='float32')
    label_tf = np.zeros((num_images,2),dtype='float32')

    # from IPython.core.debugger import Tracer
    # Tracer()()

    cnt = 0
    for k in range(len(connect)):
        cons = np.where(connect[k]==1)
        num_con = len(cons[0])
        fluor = fluor_[k]
        raw_samples = fluor.shape[1]
        pcorr = pcorr_[k]

        avg_F = np.mean(fluor,axis=0)
    
        # Add conncted pairs to tensor
        for i in range(num_con):
            for j in range(num_con_reps):
                startpos = np.random.randint(0,raw_samples-num_samples,1)[0]
                fluor_tf[cnt,:,:,0] = np.vstack((fluor[cons[0][i],startpos:startpos+num_samples], 
                                                       fluor[cons[1][i],startpos:startpos+num_samples], 
                                                       avg_F[startpos:startpos+num_samples],
                                                       np.tile(pcorr[cons[0][i],cons[1][i]],num_samples)))
                label_tf[cnt,0] = 1
                cnt += 1
    
        # Find all non-connected pairs
        # There are typically too many non-connected pairs to have any repetitions in the training set
        noncons = np.where(connect[k]==0)
    
        # Sample randomly from noncons without replacement
        noncons_samp = (np.random.choice(noncons[0],(100-represent)*num_images/100,replace=False), 
                        np.random.choice(noncons[1],(100-represent)*num_images/100,replace=False))
        
        for i in range(np.ceil((100-represent)*num_images/100/len(connect)).astype('int')):
            if cnt >= fluor_tf.shape[0]:
                break
            startpos = np.random.randint(0,raw_samples-num_samples,1)[0]
            fluor_tf[cnt,:,:,0] = np.vstack((fluor[noncons_samp[0][i],startpos:startpos+num_samples], 
                                                   fluor[noncons_samp[1][i],startpos:startpos+num_samples], 
                                                   avg_F[startpos:startpos+num_samples],
                                                   np.tile(pcorr[noncons_samp[0][i],noncons_samp[1][i]]
                                                    ,num_samples)))
            label_tf[cnt,1] = 1
            cnt += 1

    print("target size of processed traces: {}. count var: {}".format(fluor_tf.shape[0],cnt))
                
                
    return fluor_tf, label_tf

def valid_eval_tfomics_partialcorr(nnt, val_dat, pcorr, N=14, fragLen=330):
    """
    Properly validates current CNN filters by passing filters over retained validation set N number of times and averaging
    the set of predictions for each pair. For data with an extra row for partial correlation
    
    inputs---
        nnt: tfomics nntrainer object
        val_dat: 2-D numpy array of downsampled fluorescence traces
        N: number of separate start positions for each test fragment to be averaged for each pair
        fragLen: length of trained CNN filter, in time points/samples
    outputs---
        pred_lbl: 1-D numpy array of predicted connectivity
    """

    avg_F = np.mean(val_dat,axis=0)

    startgap = np.ceil((val_dat.shape[1] - fragLen)/N).astype('int')

    pred_lbl = np.zeros((val_dat.shape[0]*val_dat.shape[0],), dtype='float32')
    # Counter for the "pred_lbl" array
    cnt_u = 0

    num_cells = val_dat.shape[0]

    # Note that construction of the im_eval matrix can be amortized and loaded from disc in order to speed up evaluation time
    for a in range(val_dat.shape[0]):
        if a%100 == 0:
            print('\r' + 'X'*(a//100))

        # Create batch array to send thru network
        im_eval = np.empty((N*val_dat.shape[0],4,fragLen,1), dtype='float32')

        # Count the number of traces in each batch
        cnt = 0

        for b in range(val_dat.shape[0]):

            for n in range(0, val_dat.shape[1] - fragLen, startgap):
                try:
                    im_eval[cnt,:,:,0] = np.vstack((val_dat[a,n:n+fragLen],
                                         val_dat[b,n:n+fragLen],
                                         avg_F[n:n+fragLen],
                                         np.tile(pcorr[a,b],fragLen)))
                except:
                    from IPython.core.debugger import Tracer
                    Tracer()()

                cnt += 1

        #im_eval = np.tile(im_eval,(100,1,1,1))
        # Run batch through network
        test = {'inputs': im_eval, 'keep_prob_dense': 1, 'keep_prob_conv': 1, 'is_training': False}
        pred_stop = nnt.get_activations(test, layer='output',batch_size=2000)[:,0]
        # Average output over each group of N traces
        rock = len(pred_stop)/14
        pred_lbl_ = np.reshape(pred_stop,(rock,14))
        pred_lbl[a*num_cells:(a+1)*num_cells] = np.mean(pred_lbl_,axis=1)

    return pred_lbl
import numpy as np
import random 
import matplotlib.pyplot as plt
import torch
import train_2D_rt as rt1

    
def get_predicted_PMF(p_list,npdf,position,model,get_ypred_at_RT):
    '''Returns predicted histogram for p given current state of model.'''

    p1 = p_list[position:position+1]
    w_p1 = model.forward(p1)[0]
    predicted_y1 = get_ypred_at_RT(p1[0],npdf,w_p1)
    
    return(predicted_y1)


def plot_PMF(p_list,y_list,npdf,model,get_ypred_at_RT,kld=True):
    '''Plots predicted and true PMF for given parameter, ykerlist and ylist (one p, yker, y in each list)'''
    
    position = 0
    
    y_pred = get_predicted_PMF(p_list=p_list,
                               npdf=npdf,position=position,model=model,get_ypred_at_RT = get_ypred_at_RT)
    
    
    
    kld = rt1.get_metrics(y_pred,y_list,metric = 'kld')
    print('KLD: ',kld.item())
    
    y = y_list.detach().numpy()
    Y = y_pred.detach().numpy().reshape(y.shape)
    

    fig1,ax1=plt.subplots(nrows=1,ncols=3,figsize=(12,4))
    cm='viridis'
    ax1[0].imshow(np.log10(y).T,cmap=cm,aspect='auto')
    ax1[0].invert_yaxis()
    ax1[0].set_title('True log-PMF & basis locations')

    ax1[1].imshow(np.log10(Y).T,cmap=cm,aspect='auto')
    ax1[1].invert_yaxis()
    ax1[1].set_title('Reconstructed log-PMF')

    ax1[2].imshow(np.log10(np.abs(Y-y)).T,cmap=cm,aspect='auto')
    ax1[2].invert_yaxis()
    ax1[2].set_title('Log-absolute difference between PMFs')


def plot_training(e_,t_,npdf=None):
    '''Plots training data'''
    plt.figure(figsize=(9,6))
    plt.plot(range(len(e_)),e_,c='blue',label='Training Data')
    plt.plot(range(len(t_)),t_,c='red',label='Testing Data')
    plt.suptitle(f'Min KLD: {np.min(e_)}')
    if npdf:
        plt.title(f'NPDF = {npdf}')
    plt.xlabel('Epoch')
    plt.ylabel('KL Divergence')
    plt.legend()
    

def plot_CDF(array,xlim=None):
    '''Plots CDF'''
    cdf = np.zeros(len(array))
    array_sorted = np.sort(array)
    for i,value in enumerate(array_sorted):
        cdf[i] = len(array_sorted[array_sorted<value])/len(array_sorted)

    plt.scatter(array_sorted,cdf,s=5)
    plt.xlabel('KL Divergence')
    plt.ylabel('CDF')
   
    if xlim:
        xlow,xhigh=xlim
        plt.xlim(xlow,xhigh)
       
    plt.show()


def plot_histogram(array,bins,xlim=None):
    '''Histogram of bin number of bins, xlim'''
    plt.hist(array,bins = bins)
    if xlim:
        xlow,xhigh = xlim
        plt.xlim(xlow,xhigh)
    plt.title(f'Max KLD: {np.max(array):.4f}, Min KLD: {np.min(array):.4f}')
    plt.xlabel('KL Divergence')
    plt.ylabel('Frequency')


    
def get_parameters_quantile(train_list,model,klds,quantiles = [.95,1.0]):
    '''Returns given percent parameters with the highest klds and klds.'''
    
    parameters,y_list = rt1.load_training_data(train_list)
    
    kld_low = np.quantile(klds,quantiles[0])
    kld_high = np.quantile(klds,quantiles[1])
    
    klds_segment = klds[klds>kld_low]
    params_segment = parameters[klds>kld_low]
    
    klds_segment_2 = klds_segment[klds_segment<kld_high]
    params_segment_2 = params_segment[klds_segment<kld_high]
    
    return(params_segment_2,klds_segment_2)



def plot_param_quantiles(klds,train_list,model):
    
    params_segment_1,klds_segment_1 = get_parameters_quantile(train_list,model,klds,quantiles=[0,.25])
    params_segment_2,klds_segment_2 = get_parameters_quantile(train_list,model,klds,quantiles=[.25,.5])
    params_segment_3,klds_segment_3 = get_parameters_quantile(train_list,model,klds,quantiles=[.5,.75])
    params_segment_4,klds_segment_4 = get_parameters_quantile(train_list,model,klds,quantiles=[.75,.95])
    params_segment_5,klds_segment_5 = get_parameters_quantile(train_list,model,klds,quantiles=[.95,1.])
    
    b_1 = 10**np.array([ p[0] for p in params_segment_1 ])
    beta_1 = 10**np.array([ p[1] for p in params_segment_1  ])
    gamma_1 = 10**np.array([ p[2] for p in params_segment_1  ])

    b_2 = 10**np.array([ p[0] for p in params_segment_2 ])
    beta_2 = 10**np.array([ p[1] for p in params_segment_2  ])
    gamma_2 = 10**np.array([ p[2] for p in params_segment_2  ])

    b_3 = 10**np.array([ p[0] for p in params_segment_3 ])
    beta_3 = 10**np.array([ p[1] for p in params_segment_3  ])
    gamma_3 = 10**np.array([ p[2] for p in params_segment_3  ])

    b_4 = 10**np.array([ p[0] for p in params_segment_4 ])
    beta_4 = 10**np.array([ p[1] for p in params_segment_4  ])
    gamma_4 = 10**np.array([ p[2] for p in params_segment_4  ])

    b_5 = 10**np.array([ p[0] for p in params_segment_5 ])
    beta_5 = 10**np.array([ p[1] for p in params_segment_5  ])
    gamma_5 = 10**np.array([ p[2] for p in params_segment_5  ])
    
    fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(12,5))
    
    
    # some labels
    ax[0].scatter(10,10,c='grey',label = 'Quantile 0-0.25')
    ax[0].scatter(10,10,c='blue',label = 'Quantile 0.25-0.50')
    ax[0].scatter(10,10,c='purple',label = 'Quantile 0.50-0.75')
    ax[0].scatter(10,10,c='green',label = 'Quantile 0.75-0.95')
    ax[0].scatter(10,10,c='red',label = 'Quantile 0.95-1.0')

    ax[0].scatter(b_1,beta_1,c = klds_segment_1, cmap= 'Greys')
    ax[0].scatter(b_2,beta_2,c = klds_segment_2, cmap= 'Blues')
    ax[0].scatter(b_3,beta_3, c = klds_segment_3, cmap= 'Purples')
    ax[0].scatter(b_4,beta_4,c = klds_segment_4, cmap= 'Greens')
    ax[0].scatter(b_5,beta_5,c = klds_segment_5, cmap= 'Reds')
    ax[0].set_xlabel('b')
    ax[0].set_ylabel('beta')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    
    ax[1].scatter(b_1,gamma_1,c = klds_segment_1, cmap= 'Greys')
    ax[1].scatter(b_2,gamma_2, c = klds_segment_2, cmap= 'Blues')
    ax[1].scatter(b_3,gamma_3,c = klds_segment_3, cmap= 'Purples')
    ax[1].scatter(b_4,gamma_4,c = klds_segment_4, cmap= 'Greens')
    ax[1].scatter(b_5,gamma_5,c = klds_segment_5, cmap= 'Reds')
    ax[1].set_xlabel('b')
    ax[1].set_ylabel('gamma')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    
    ax[2].scatter(beta_1,gamma_1,c = klds_segment_1, cmap= 'Greys')
    ax[2].scatter(beta_2,gamma_2, c = klds_segment_2, cmap= 'Blues')
    ax[2].scatter(beta_3,gamma_3,c = klds_segment_3, cmap= 'Purples')
    ax[2].scatter(beta_4,gamma_4,c = klds_segment_4, cmap= 'Greens')
    ax[2].scatter(beta_5,gamma_5,c = klds_segment_5, cmap= 'Reds')
    ax[2].set_xlabel('beta')
    ax[2].set_ylabel('gama')
    ax[2].set_xscale('log')
    ax[2].set_yscale('log')
 
    ax[0].legend()
    fig.tight_layout()
    plt.title('MLP 1 Parameters Colored by KLD Quantile')
    
    
def plot_PMF_grid(file_list,npdf,nrows,ncols,model,get_ypred_at_RT,kld=True):
    
    p_list,y_list = rt1.load_training_data(file_list)
    rand = np.zeros(nrows*ncols)
    
    for i in range(nrows*ncols):
        rand[i] = random.randint(0,len(y_list))
    
    y = []
    Y = []
    
    for r in rand:
        r = int(r)
        y_pred = get_predicted_PMF(p_list=p_list,
                                   npdf=npdf,position=r,model=model,get_ypred_at_RT = get_ypred_at_RT)
        
        y.append(y_list[r])
        Y.append(y_pred)
    
    Y = [Y_.detach().numpy() for Y_ in Y]
    y = [y_.detach().numpy() for y_ in y]
    
    Y = [Y_.reshape(y[i].shape) for i,Y_ in enumerate(Y)]

    fig, ax1 = plt.subplots(nrows=nrows, ncols=2*ncols, figsize=(16, 8))
    k = 0

    j_num = np.arange(0,ncols*2,2)
    
    for i in range(nrows):
        for j in j_num:
            y_ = y[k]
            Y_ = Y[k]
            cm='viridis'
            ax1[i,j].imshow(np.log10(y_).T,cmap=cm,aspect='auto')
            ax1[i,j].invert_yaxis()
            ax1[i,j].set_title('True log-PMF & basis locations')
            ax1[i,j].set_xlabel('mRNA counts')
            ax1[i,j].set_ylabel('nRNA counts')
            
            ax1[i,j+1].imshow(np.log10(Y_).T,cmap=cm,aspect='auto')
            ax1[i,j+1].invert_yaxis()
            ax1[i,j+1].set_title('Reconstructed log-PMF')
            ax1[i,j+1].set_xlabel('mRNA counts')
            ax1[i,j+1].set_ylabel('nRNA counts')
            
            if kld == True:
                kld_ = -np.sum(y_.flatten()*np.log(Y_.flatten()/y_.flatten()))
                ax1[i,j].title.set_text(f'KLD: {kld_:.5f}')
            k = k + 1

        
    fig.tight_layout()
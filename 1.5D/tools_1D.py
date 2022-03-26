import numpy as np
import random 
import matplotlib.pyplot as plt
import torch
import train_MLP1 as tm1

def get_predicted_PMF(p_list,yker_list,y_list,position,model):
    '''Returns predicted histogram for p given current state of model.'''
    model.eval()
    
    p1 = p_list[position:position+1]

    w_p1 = model(p1)[0]
    predicted_y1 = get_probabilities_torch(yker_list[position],y_list[position],w_p1)
    
    return(predicted_y1)

def get_probabilities_torch(yker, y, w):
    ''' Multiplies yker by weights, then reshapes Yker to be shape of y.'''
    # shapes of Yker and w?? 
    Y = torch.matmul(yker,w).reshape(y.shape)
    EPS=1e-8
    Y[Y<EPS]=EPS

    return Y

def plot_PMF_grid(file_list,nrows,ncols,model,kld=True):
    '''Plots predicted and true PMFs for random parameters chosen from file_list.
    Number: nrows*ncols'''
    p_list,yker_list,y_list = tm1.load_training_data(file_list)
    
    npdf = yker_list[0].shape[1]
    
    rand = np.zeros(nrows*ncols)
    
    for i in range(nrows*ncols):
        rand[i] = random.randint(0,len(y_list))
    
    y = []
    Y = []
    
    for r in rand:
        r = int(r)
        y_pred = get_predicted_PMF(p_list=p_list,
                                yker_list=yker_list,y_list=y_list,position=r,model=model)
        
        y.append(y_list[r])
        Y.append(y_pred)
    
    Y = [Y_.detach().numpy() for Y_ in Y]
    y = [y_.detach().numpy() for y_ in y]
    


    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))
    k = 0

    for i in range(nrows):
        for j in range(ncols):
            
            x = np.arange(len(y[k]))
            axs[i,j].plot(x,y[k],'k-',label='True PMF')
            axs[i,j].plot(x,Y[k],'r--',label='Predicted PMF')
        
            axs[i,j].set_xlabel('# mat RNA')
            axs[i,j].set_ylabel('probability')

            axs[i,j].legend()
            
            if kld == True:
                kld_ = -np.sum(y[k]*np.log(Y[k]/y[k]))
                axs[i,j].title.set_text(f'KLD: {kld_:.5f}')
            k = k + 1

        
    fig.tight_layout()
    
def plot_PMF(p_list,yker_list,y_list,model,kld=True):
    '''Plots predicted and true PMF for given parameter, ykerlist and ylist (one p, yker, y in each list)'''
    
    position = 0
    
    y_pred = get_predicted_PMF(p_list=p_list,
                                yker_list=yker_list,y_list=y_list,position=0,model=model)
    
    
    Y = y_pred.detach().numpy() 
    y = y_list[0].detach().numpy()


    x = np.arange(len(y))
    plt.plot(x,y,'k-',label='True PMF')
    plt.plot(x,Y,'r--',label='Predicted PMF')
        
    plt.xlabel('# mat RNA')
    plt.ylabel('probability')

    plt.legend()
            
    if kld == True:
        kld_ = -np.sum(y*np.log(Y/y))
        plt.title(f'KLD: {kld_:.5f}')
    plt.show()


def calculate_test_klds(test_list,model):
    '''Calculates klds for parameters given current state of model'''
    parameters,yker_list,y_list = tm1.load_training_data(test_list)
    metrics = np.zeros(len(parameters))
    
    for i in range(len(parameters)):
        Y = get_predicted_PMF(parameters,yker_list=yker_list,y_list=y_list,
                              position=i,model=model)
        y = y_list[i]
        metric = -torch.sum(y*torch.log(Y/y))
        metrics[i] = metric.detach().numpy()
        
    metrics = np.array(metrics)
    return(metrics,np.mean(metrics))


def plot_CDF(array,npdf=None,epochs=None,xlim=None):
    '''Plots CDF'''
    cdf = np.zeros(len(array))
    array_sorted = np.sort(array)
    for i,value in enumerate(array_sorted):
        cdf[i] = len(array_sorted[array_sorted<value])/len(array_sorted)

    plt.scatter(array_sorted,cdf,s=5)
    plt.xlabel('KL Divergence')
    plt.ylabel('CDF')
    
    if npdf:
        plt.title(f'CDF of KLDs for {epochs} epochs, NPDF = {npdf}')
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
    
def plot_training(e_,t_,npdf=None,batchsize=512):
    '''Plots training data'''
    plt.figure(figsize=(9,6))
    plt.plot(range(len(e_)),e_,c='blue',label='Training Data')
    plt.plot(range(len(t_)),t_,c='red',label='Testing Data')
    plt.suptitle(f'Min KLD: {np.min(e_)}')
    if npdf:
        plt.title(f'NPDF = {npdf}, Batchsize = {batchsize}')
    plt.xlabel('Epoch')
    plt.ylabel('KL Divergence')
    plt.legend()



def plot_high_params(high_params,high_klds,cmap='Purples'):
    '''Plots the params b vs. gamma, b vs. beta and beta vs. gamma'''
    high_b = 10**np.array([ p[0] for p in high_params ])
    high_beta = 10**np.array([ p[1] for p in high_params ])
    high_gamma = 10**np.array([ p[2] for p in high_params ])
    
    fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(10,5))
    
    ax[0].scatter(high_b,high_beta,c = high_klds,cmap=cmap)
    ax[0].set_xlabel('b')
    ax[0].set_ylabel('beta')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    
    ax[1].scatter(high_b,high_gamma,c = high_klds,cmap=cmap)
    ax[1].set_xlabel('b')
    ax[1].set_ylabel('gamma')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    
    ax[2].scatter(high_beta,high_gamma,c = high_klds,cmap=cmap)
    ax[2].set_xlabel('beta')
    ax[2].set_ylabel('gama')
    ax[2].set_xscale('log')
    ax[2].set_yscale('log')
    
    fig.tight_layout()

    
def get_parameters_quantile(train_list,klds,model,quantiles = [.95,1.0]):
    '''Returns given percent parameters with the highest klds and klds.'''
    
    parameters,yker_list,y_list = tm1.load_training_data(train_list)
    
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
    params_segment_5,klds_segment_5 = get_parameters_quantile(train_list,klds,model,quantiles=[.95,1.])
    
    
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



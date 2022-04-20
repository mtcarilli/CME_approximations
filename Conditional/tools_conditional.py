import numpy as np
import random 
import matplotlib.pyplot as plt
import torch
import train_conditional_mc as trc
import scipy


def plot_PMF_grid(data_list,npdf,nrows,ncols,model,get_ypred_at_RT,kld=True,
        normalize = True):
    '''Plots predicted and true PMFs for random parameters chosen from file_list.
    Number: nrows*ncols'''
    
    p_list,y_list = trc.unpack_data(data_list)
    
    rand = np.zeros(nrows*ncols)
    
    for i in range(nrows*ncols):
        rand[i] = random.randint(0,len(y_list))
    
    y = []
    Y = []
    
    for r in rand:
        r = int(r)
        y_pred = trc.get_predicted_PMF(p_list,position=r,model=model,get_ypred_at_RT=get_ypred_at_RT)
        
        y.append(y_list[r])
        Y.append(y_pred)
    
    Y = [Y_.detach().numpy() for Y_ in Y]
    y = [y_.detach().numpy() for y_ in y]

    if normalize == True:
        y = [y_/np.sum(y_) for y_ in y]
        Y = [Y_/np.sum(Y_) for Y_ in Y]
    


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
    
def plot_PMF(p_list,y_list,model,npdf,get_ypred_at_RT,kld=True,rand=False):
    '''Plots predicted and true PMF for given parameter, ykerlist and ylist (one p, yker, y in each list)'''
    
    
    y_pred = trc.get_predicted_PMF(p_list=p_list,
                            position=0,model=model,get_ypred_at_RT = get_ypred_at_RT)
    
    
    Y = y_pred.detach().numpy() 
    y = y_list[0].detach().numpy()


    x = np.arange(len(y))
    plt.plot(x,y,'k-',label='True PMF')
    plt.plot(x,Y,'r--',label='Predicted PMF')
        
    plt.xlabel('# mat RNA')
    plt.ylabel('probability')

    
            
    if kld == True:
        kld_ = -np.sum(y*np.log(Y/y))
        plt.title(f'KLD: {kld_:.5f}')

    
    plt.legend()
    plt.show()


def plot_CDF(array,metric='KLD'):
    '''Plots CDF'''
    cdf = np.zeros(len(array))
    array_sorted = np.sort(array)
    for i,value in enumerate(array_sorted):
        cdf[i] = len(array_sorted[array_sorted<value])/len(array_sorted)

    plt.scatter(array_sorted,cdf,s=5)
    plt.title('metric CDF')
    plt.xlabel(f'{metric}')
    plt.ylabel('CDF')
       
    plt.show()


def plot_histogram(array,bins,metric='kld'):
    '''Histogram of bin number of bins, xlim'''
    plt.hist(array,bins = bins)
    plt.title(f'Average {metric}: {np.mean(array):.4f}')
    plt.xlabel(f'{metric} Divergence')
    plt.ylabel('Frequency')
    
def plot_training(e_,t_,metric='kld'):
    '''Plots training data'''
    plt.figure(figsize=(9,6))
    plt.plot(range(len(e_)),e_,c='blue',label='Training Data')
    plt.plot(range(len(t_)),t_,c='red',label='Testing Data')
    plt.xlabel('Epoch')
    plt.ylabel(f'{metric}')
    plt.title('Loss vs. epoch')
    plt.legend()

def plot_rand(index,npdf,train_list,model,get_ypred_at_RT,hyp=1.5,rand=False,unif=False,ls=False,norm_nnls=False):
    
    p_list,y_list = trc.unpack_data(train_list)
    
    w_rand = np.random.rand(npdf)
    w_rand /= w_rand.sum()
    
    w_unif = np.ones(npdf)/npdf
    
    
    j = index
    

    y_pred = trc.get_predicted_PMF(p_list[j:j+1],0,model,get_ypred_at_RT).detach().numpy()

    if unif == True:
        y_unif = get_ypred_at_RT(torch.tensor(train_list[j][0]),w_unif,hyp,npdf=npdf)
        plt.plot(y_unif,'--',c='orange',label='uniform weights')
    
    if rand == True:
        y_rand = get_ypred_at_RT(torch.tensor(train_list[j][0]),w_rand,hyp,npdf=npdf)
        plt.plot(y_rand,'g--',label='random weights')
    
    if ls == True:
        w_ones = np.ones(npdf)
        y_ones = get_ypred_at_RT(torch.tensor(train_list[j][0]),w_ones,hyp,npdf=npdf,array=True).detach().numpy()
        w_ls,res,rank,s = scipy.linalg.lstsq(y_ones.T,train_list[j][1])
        #y_ls = get_ypred_at_RT(torch.tensor(train_list[j][0]),w_ls,hyp,npdf=npdf).detach().numpy()
        y_ls = np.matmul(w_ls,y_ones)
        plt.plot(y_ls,'.--',c='m',label='least squares',)
    
    if norm_nnls == True:
        w_ones = np.ones(npdf)
        y_ones = get_ypred_at_RT(torch.tensor(train_list[j][0]),w_ones,hyp,npdf=npdf,array=True).detach().numpy()
        w_nnls,res = scipy.optimize.nnls(y_ones.T,train_list[j][1])
        w_norm_nnls = w_nnls/w_nnls.sum()
        #y_norm_nnls = get_ypred_at_RT(torch.tensor(train_list[j][0]),w_norm_nnls,hyp,npdf=npdf).detach().numpy()
        y_norm_nnls = np.matmul(w_norm_nnls,y_ones)
        plt.plot(y_norm_nnls,'.--',c='dodgerblue',label='norm nnls squares')
        
        
        
    plt.plot(train_list[j][1],'k',label='ground truth')
    plt.plot(y_pred,'r--',label='nn predicted')
    
    plt.ylabel('conditional probability')
    plt.xlabel('# mature RNA')
    plt.title('Conditional Probability')
    plt.legend()


def get_parameters_quantile(train_list,model,klds,quantiles = [.95,1.0]):
    '''Returns given percent parameters with the highest klds and klds.'''
    
    parameters,y_list = trc.unpack_data(train_list)
    
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
    


def save_model_and_meta(model,model_config,train_config,time,path,name):
    
    torch.save(model.state_dict(),path+name+'_MODEL')
    
    meta = np.array([model_config,train_config,time])
    np.save(path+name+'_meta',meta)
    
    
class Trained_Model():
    
    def __init__(self, path, name):
        
        meta = np.load(path + name + '_meta.npy',allow_pickle=True)
        self.model_config = meta[0]
        self.train_config = meta[1]
        self.time = meta[2]
        
        self.model = my_MLP1(self.model_config['input_dim'],
                             self.model_config['output_dim'], 
                             self.model_config['h1_dim'], 
                             self.model_config['h2_dim'], 
                             norm_type=self.model_config['softmax'])
            
        self.model.load_state_dict(torch.load(path+name+'_MODEL'))
        self.model.eval()
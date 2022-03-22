import numpy as np 
import matplotlib.pyplot as plt
from scipy.fft import irfft2
import scipy
from scipy import integrate, stats
from numpy import linalg
import time
from scipy.special import gammaln
from itertools import repeat
import numdifftools
import torch

import multiprocessing


import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy import optimize

def cme_integrator(p,lm,method,fixed_quad_T=10,quad_order=60,quad_vec_T=np.inf):
    b,bet,gam = p
    u = []
    mx = np.copy(lm)

    #initialize the generating function evaluation points
    mx[-1] = mx[-1]//2 + 1
    
    for i in range(len(mx)):
        l = np.arange(mx[i])
        u_ = np.exp(-2j*np.pi*l/lm[i])-1
        u.append(u_)
    g = np.meshgrid(*[u_ for u_ in u], indexing='ij')
    for i in range(len(mx)):
        g[i] = g[i].flatten()[:,np.newaxis]

    #define function to integrate by quadrature.
    fun = lambda x: INTFUN(x,g,b,bet,gam)
    if method=='quad_vec':
        T = quad_vec_T*(1/bet+1/gam+1)
        gf = scipy.integrate.quad_vec(fun,0,T)[0]
    if method=='fixed_quad':
        T = fixed_quad_T*(1/bet+1/gam+1)
        gf = scipy.integrate.fixed_quad(fun,0,T,n=quad_order)[0]

    #convert back to the probability domain, renormalize to ensure non-negativity.
    gf = np.exp(gf) #gf can be multiplied by k in the argument, but this is not relevant for the 3-parameter input.
    gf = gf.reshape(tuple(mx))
    Pss = irfft2(gf, s=tuple(lm)) 
    EPS=1e-16
    Pss[Pss<EPS]=EPS
    Pss = np.abs(Pss)/np.sum(np.abs(Pss)) #always has to be positive...
    return Pss

def INTFUN(x,g,b,bet,gam):
    """
    Computes the Singh-Bokes integrand at time x. Used for numerical quadrature in cme_integrator.
    """
    if not np.isclose(bet,gam): #compute weights for the ODE solution.
        f = bet/(bet-gam)
        U = b*(np.exp(-bet*x)*(g[0]-g[1]*f)+np.exp(-gam*x)*g[1]*f)
    else:
        g[1] *= (b*gam)
        g[0] *= b
        U = np.exp(-bet*x)*(g[0] + bet * g[1]* x)
    return U/(1-U)

def INTFUN_old(x,g,b,bet,gam):
    """
    Computes the Singh-Bokes integrand at time x. Used for numerical quadrature in cme_integrator.
    """
    if not np.isclose(bet,gam): #compute weights for the ODE solution.
        f = b*bet/(bet-gam)
        g[1] *= f
        g[0] *= b
        g[0] -= g[1]
        U = np.exp(-bet*x)*g[0]+np.exp(-gam*x)*g[1]
    else:
        g[1] *= (b*gam)
        g[0] *= b
        U = np.exp(-bet*x)*(g[0] + bet * g[1]* x)
    return U/(1-U)

def get_moments(p):
    b,beta,gamma=p
    
    r = torch.tensor([1/beta, 1/gamma])
    MU = b*r
    VAR = MU*torch.tensor([1+b,1+b*beta/(beta+gamma)])
    STD = torch.sqrt(VAR)
    xmax = torch.ceil(MU)
    xmax = torch.ceil(xmax + 4*STD)
    xmax = torch.clip(xmax,30,np.inf).int()
    return MU, VAR, STD, xmax


def calculate_exact_cme(p,method,xmax_fun):
    
    '''Given parameter vector p, calculate the exact probabilites using CME integrator.'''
    p1 = torch.from_numpy(p).float()
    p1 = 10**p1
    
    MU, VAR, STD, xmax = get_moments(p1)
    
    xmaxc = xmax_fun(xmax)
    
    xmaxc = np.array([int(xmaxc[0]),int(xmaxc[1])])
    
    y = cme_integrator(np.array(p1),xmaxc+1,method=method)
    y_save = y[0:xmax[0]+1,0:xmax[1]+1]
    
    return([p,np.array(y_save)])


def create_file_paths(set_size,num_files,path_to_directory):
    '''Creates file paths for a certain set size. Stores in t'''
    file_paths = []
    
    for i in range(num_files):
        file_paths.append(path_to_directory+str(set_size)+'_'+str(i))
        
    return(file_paths)

def prepare_set(position,size,param_vectors,method,xmax_fun):
    '''Outputs parameters and exact CME a set size amount of parameters.'''
    
    params_ = param_vectors[position:position+size]
    
    set_list = []
    
    for i in range(size):
        param_ = params_[i]
        p_,y_ = calculate_exact_cme(param_,method,xmax_fun)
        set_list.append([p_,y_])

    return(set_list)


def prepare_set_pcme(position,size,param_vectors,method,xmax_fun,NCOR):
    
    params = param_vectors[position:size+position]
     
    print('Starting parallelization...')
    pool=multiprocessing.Pool(processes=NCOR)
    data_tuples = zip(params,repeat(method),repeat(xmax_fun))
    set_list = pool.starmap(calculate_exact_cme, data_tuples)
    pool.close()
    pool.join()
    print('Parallelization done!')
    
    return(set_list)
    
def generate_sets(set_size,num_files,param_vectors,method,xmax_fun,NCOR,path_to_directory):
    '''Generates kernel and true histograms for params in param_vectors
    Saves them in sets of set_size, in directory path_to_directory.'''
    
    file_paths = create_file_paths(set_size,num_files,path_to_directory)
    
    for i,file in enumerate(file_paths):
        print(i)
        position = i*set_size
        set_list = prepare_set(position,set_size,param_vectors,method,xmax_fun,NCOR)

        np.save(file,set_list)
        
def generate_sets_pcme(set_size,num_files,param_vectors,method,xmax_fun,NCOR,path_to_directory):
    '''Generates kernel and true histograms for params in param_vectors
    Saves them in sets of set_size, in directory path_to_directory.'''
    
    file_paths = create_file_paths(set_size,num_files,path_to_directory)
    
    for i,file in enumerate(file_paths):
        print(i)
        position = i*set_size
        set_list = prepare_set_pcme(position,set_size,param_vectors,method,xmax_fun,NCOR)

        np.save(file,set_list)
        
def generate_sets_pset(set_size,num_files,param_vectors,method,xmax_fun,NCOR,path_to_directory):
    
    file_paths = create_file_paths(set_size,num_files,path_to_directory)
    
    params_split = np.split(param_vectors,num_files)
    
    data_tuples = zip(params_split,file_paths,repeat(0),repeat(set_size),repeat(method),repeat(xmax_fun))
    
    print('Starting parallelization...')
    pool=multiprocessing.Pool(processes=NCOR)
    pool.starmap(parallel_prepare_set, data_tuples)
    pool.close()
    pool.join()
    print('Parallelization done!')
    
    
     
def prepare_set_pset(param_vectors,file_path,position,set_size,method,xmax_fun):
    
    set_list = prepare_set(position,set_size,param_vectors,method,xmax_fun)
    
    np.save(file_path,set_list)
    
        
def generate_param_vectors(N):
    '''Generates N parameter vectors randomly spaced in logspace between bounds.'''
    
    logbnd = np.log10([[1,300],[0.05,50],[0.05,50]])
    dbnd = logbnd[:,1]-logbnd[:,0]
    lbnd = logbnd[:,0]
    param_vectors = np.zeros((N,3))


    i = 0
    a = 0
    while i<N:
        a += 1 
        th = np.random.rand(3)*dbnd+lbnd
        MU, VAR, STD, xmax=get_moments(10**th)

        if xmax[0] > 1e3:
            continue
        if xmax[1] > 1e3:
            continue
        if th[1] == th[2]:
            continue
        else:
            param_vectors[i,:] = np.float32(th)
            i+=1
            
    return(param_vectors)
    
class my_MLP1(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input = nn.Linear(input_dim, 256)
        self.hidden1 = nn.Linear(256, 256)
        self.hidden2 = nn.Linear(256, 128)
        self.hidden3 = nn.Linear(128, 128)
        self.output = nn.Linear(128, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, parameters):

        # pass parameters to the first layer of neurons 
        l_1 = self.input(parameters)

        # pass through a sigmoid function to second layer of neurons
        l_2 = torch.sigmoid(self.hidden1(l_1))
        
        # pass to a second layer of neurons 
        l_3 = torch.sigmoid((self.hidden2(l_2)))
        
        # pass to third layer of neurons 
        l_4 = torch.sigmoid(self.hidden3(l_3))

        # pass out to output dimensions (predicted weights), averaged to sum to 1 with softmax
        w_pred = self.softmax(self.output(l_4))

        return w_pred
    

class my_MLP2(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input = nn.Linear(input_dim, 128)
        self.hidden1 = nn.Linear(128, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.hidden3 = nn.Linear(128, 128)
        self.output = nn.Linear(128, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, parameters):

        # pass parameters to the first layer of neurons 
        l_1 = self.input(parameters)

        # pass through a sigmoid function to second layer of neurons
        l_2 = F.relu(self.hidden1(l_1))
        
        # pass to a second layer of neurons 
        l_3 = F.relu((self.hidden2(l_2)))
        
        # pass to third layer of neurons 
        l_4 = F.relu(self.hidden3(l_3))

        # pass out to output dimensions (predicted weights), averaged to sum to 1 with softmax
        w_pred = self.softmax(self.output(l_4))

        return w_pred
    

def load_data_list(file_path):
    full_file_list = list(np.load(file_path,allow_pickle=True))
    return(full_file_list)

def shuffle_training_data(full_file_list):
    '''Load .npy file, returns tensor for parameters, unweighted kernal functions, and ground truth histograms'''
    
    random.shuffle(full_file_list)
    parameters = np.array([ a[0] for a in full_file_list ])
    parameters_tensor = torch.from_numpy(parameters).float()
    y_tensor = [ torch.tensor(a[1]).float() for a in full_file_list ]
    
    return(parameters_tensor,y_tensor)

def load_training_data(full_file_list):
    '''Load .npy file, returns tensor for parameters and ground truth histograms'''
 
    parameters = np.array([ a[0] for a in full_file_list ])
    parameters_tensor = torch.from_numpy(parameters).float()
    y_tensor = [ torch.tensor(a[1]).float() for a in full_file_list ]
    
    return(parameters_tensor,y_tensor)


def get_metrics(pred,y,metric):
    '''Calculates desired metric between predicted Y and y.'''
    

    y = torch.flatten(y)
    pred = torch.flatten(pred)

    if metric=='kld':
        return -torch.sum(y*torch.log(pred/y))
    if metric=='totalse':
        return torch.sum((pred-y)**2)
    if metric=='mse':
        return torch.mean((pred-y)**2)
    if metric=='maxabsdev':
        return torch.max(torch.abs(pred-y))
    if metric=='maxabsdevlog':
        return torch.max(torch.abs(torch.log(pred)-torch.log(y)))
    if metric=='mselog':
        return torch.mean((torch.log(pred)-torch.log(y))**2)
    
def get_predicted_PMF(p_list,npdf,position,model,get_ypred_at_RT):
    '''Returns predicted histogram for p given current state of model.'''
    model.eval()

    p1 = p_list[position:position+1]
    w_p1 = model(p1)[0]
    p1 = p1[0]
    predicted_y1 = get_ypred_at_RT(p1,npdf,w_p1)
    
    return(predicted_y1)

def loss_fn(p_list,y_list,npdf,w,batchsize,get_ypred_at_RT,metric):
    '''Calculates average metval over batch between predicted Y and y.
    yker_list and y_list are actually lists of tensor histograms with first dimension batchsize'''
    
    metval = torch.tensor(0.0)
    
    for b in range(batchsize):
        
        y_ = y_list[b]
        p_ = p_list[b]
        w_ = w[b]
        
    
        yker_ = get_ypred_at_RT(p_,npdf,w_)
        
        
        met_ = get_metrics(yker_,y_,metric)
    
        
        metval += met_
    
    return(metval/batchsize)

def calculate_test_metrics(test_list,npdf,model,get_ypred_at_RT,metric):
    parameters,y_list = shuffle_training_data(test_list)
    metrics = np.zeros(len(parameters))
    
    for i in range(len(parameters)):
        Y = get_predicted_PMF(parameters,npdf,i,model,get_ypred_at_RT)
        y = y_list[i]
        y = y.flatten()
        Y = Y.flatten()
        metric_ = get_metrics(Y,y,metric)
        metrics[i] = metric_.detach().numpy()
        
    metrics = np.array(metrics)
    return(metrics,np.mean(metrics))

def train(parameters_tensor,y_list,npdf,model,optimizer,batchsize,get_ypred_at_RT,metric):
    '''Trains the model for given input tensors and list of tensors. Divides training data into groups of 
    batchsizes. If the number of input parameters cannot be divided by batchsize, ignores remainder...'''
    
    metvals = []
    trials = int(np.floor(parameters_tensor.size()[0] / batchsize ))
    model.train()  # can this model be accessed inside the function ????? 
    
    for j in range(trials):
        i = j * batchsize
        p = parameters_tensor[i:i+batchsize]
        y = y_list[i:i+batchsize]

        # Zero the gradients
        optimizer.zero_grad()
        
        # Perform forward pass
        w_pred = model(p)

        # Compute loss
        loss = loss_fn(p,y,npdf,w_pred,batchsize,get_ypred_at_RT,metric)
        
        metvals.append(loss.item())

        # Perform backward pass
        loss.backward()
      
        # Perform optimization
        optimizer.step()
    
    return(metvals)

def run_epoch_and_test(train_list,test_list,number_of_epochs,npdf,model,optimizer,get_ypred_at_RT,
                       batchsize,
                       metric):

    epoch_metrics = np.zeros(number_of_epochs)
    test_metrics = []
    batch_metrics_all = []

    for e in range(number_of_epochs):
        print('Epoch Number:',e)


        model.train()
        batch_metrics = []

        parameters,y_list = shuffle_training_data(train_list)
        metric_ = train(parameters,y_list,npdf,model,optimizer,batchsize,get_ypred_at_RT,metric)
        batch_metrics.append(metric_)
        batch_metrics_all.append(metric_)

        batch_metric_array = np.array(batch_metrics).flatten()
        epoch_metric_ = np.mean(batch_metric_array)

        epoch_metrics[e] = epoch_metric_


            # test by evaluating the model
        test_metric_list_,test_metric_ = calculate_test_metrics(test_list,npdf,model,get_ypred_at_RT,metric)
        test_metrics.append(test_metric_)
    

    return(epoch_metrics,np.array(batch_metrics_all).flatten(),test_metrics)

def get_data(set_size,number_of_training_files,number_of_testing_files,total_files,file_path):
    train_list = []
    test_list = []
    
    for i in range(number_of_training_files):
        num = i
        file_list_ = load_data_list(file_path+f'{set_size}_{i}.npy')
        train_list = train_list + file_list_

    for i in range(number_of_testing_files):
        num = total_files-i-1
        file_list_ = load_data_list(file_path+f'{set_size}_{i}.npy')
        test_list = test_list + file_list_

    return(train_list,test_list)

def train_MLP(train_list,test_list,num_epochs,npdf,batchsize,get_ypred_at_RT,metric,learning_rate=1e-3,MLP=1):
    print(MLP)
    if MLP == 1:
        model = my_MLP1(3,npdf[0]*npdf[1])      # define model 
        
    if MLP == 2:
        model = my_MLP1(3,npdf[0]*npdf[1])      # define model 
        
    #elif:
        #print('using pre-loaded model')
        #model = MLP
        #model.train()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # optimizer to use 

    e_,b_,t_ = run_epoch_and_test(train_list,test_list,num_epochs,npdf,model,optimizer,get_ypred_at_RT,
                       batchsize,
                       metric)
    
    return(e_,b_,t_,model)
import numpy as np 
import matplotlib.pyplot as plt

from scipy.fft import irfft
import scipy
from scipy import integrate, stats
from numpy import linalg
from scipy.special import gammaln
from numpy.random import normal

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy 
from scipy import optimize



# place holder
NORM = 1 


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
    '''Load .npy file, returns tensor for parameters, unweighted kernal functions, and ground truth histograms'''
 
    parameters = np.array([ a[0] for a in full_file_list ])
    parameters_tensor = torch.from_numpy(parameters).float()
    y_tensor = [ torch.tensor(a[1]).float() for a in full_file_list ]
    
    return(parameters_tensor,y_tensor)

def get_data(npdf,number_of_training_files,number_of_testing_files,number_total_files,path):
    train_list = []
    test_list = []
    
    ker = npdf
    
    for i in range(number_of_training_files):
        num = i
        file_list_ = load_data_list(f'{path}5120_{num}_ker{ker}.npy')
        train_list = train_list + file_list_

    for i in range(number_of_testing_files):
        num = number_total_files - i - 1
        file_list_ = load_data_list(f'{path}5120_{num}_ker{ker}.npy')
        test_list = test_list + file_list_

    return(train_list,test_list)


def get_metrics(pred,y,metric = 'kld'):
    '''Calculates desired metric between predicted Y and y.'''

    pred = pred.flatten()
    y = y.flatten()
    if metric=='kld':
        return -torch.sum(y*torch.log(pred/y))
    
    
    if metric=='totalse':
        return torch.sum((pred-y)**2)
    if metric=='mse':
        return torch.mean((pred-y)**2)
    if metric=='maxabsdev':
        return torch.max(torch.abs(pred-y))
    
    
def get_moments(p):
    b,beta,gamma=p
    r = torch.tensor([1/beta, 1/gamma])
    MU = b*r
    VAR = MU*torch.tensor([1+b,1+b*beta/(beta+gamma)])
    STD = torch.sqrt(VAR)
    xmax = torch.ceil(MU+4*STD)
    xmax = torch.maximum(torch.tensor(15),xmax).int()
    xmax = xmax[1]
    MU=MU[1]
    VAR=VAR[1]
    STD=STD[1]
    return MU, VAR, STD, xmax


    return translin


    
def loss_fn(p_list,y_list,w,npdf,batchsize,get_ypred_at_RT,metric='kld'):
    '''Calculates average metval over batch between predicted Y and y.
    p_list and y_list are actually lists of TENSORS.''' 
    metval = torch.tensor(0.0)

    for b in range(batchsize):
        p = p_list[b]
        y_ = y_list[b]
        w_ = w[b]
        Y_ = get_ypred_at_RT(p,npdf,w_)
        met_ = get_metrics(Y_,y_,metric=metric)
    
        
        metval += met_
    
    return(metval/batchsize)


# TESTING FUNCTIONS

def get_predicted_PMF(p_list,y_list,npdf,position,model,get_ypred_at_RT):
    '''Returns predicted histogram for p given current state of model.'''
    model.eval()

    p1 = p_list[position:position+1]
    w_p1 = model(p1)[0]
    predicted_y1 = get_ypred_at_RT(p1[0],npdf,w_p1)
    
    return(predicted_y1)


def calculate_test_klds(test_list,npdf,model,get_ypred_at_RT):
    parameters,y_list = shuffle_training_data(test_list)
    metrics = np.zeros(len(parameters))
    
    for i in range(len(parameters)):
        Y = get_predicted_PMF(parameters,y_list=y_list,npdf=npdf,position=i,model=model,get_ypred_at_RT=get_ypred_at_RT)
        y = y_list[i]
        metric = -torch.sum(y*torch.log(Y/y))
        metrics[i] = metric.detach().numpy()
        
    metrics = np.array(metrics)
    return(metrics,np.mean(metrics))



def train(parameters_tensor,y_list,npdf,model,optimizer,batchsize,get_ypred_at_RT,metric = 'kld'):
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
        loss = loss_fn(p_list=p,y_list=y,w=w_pred,npdf=npdf,batchsize=batchsize,get_ypred_at_RT =get_ypred_at_RT, metric=metric)
        
        metvals.append(loss.item())

        # Perform backward pass
        loss.backward()
      
        # Perform optimization
        optimizer.step()
    
    return(metvals)

def run_epoch_and_test(train_list,test_list,number_of_epochs,npdf,model,optimizer,batchsize,get_ypred_at_RT,
                       metric = 'kld'):

    epoch_metrics = np.zeros(number_of_epochs)
    test_metrics = []
    batch_metrics_all = []
    
    for e in range(number_of_epochs):
        print('Epoch Number:',e)


        model.train()
        batch_metrics = []

        parameters,y_list = shuffle_training_data(train_list)

        

        metric_ = train(parameters,y_list,npdf=npdf,
                        model=model,optimizer=optimizer,batchsize=batchsize,get_ypred_at_RT = get_ypred_at_RT,metric = metric)
        batch_metrics.append(metric_)
        batch_metrics_all.append(metric_)

        batch_metric_array = np.array(batch_metrics).flatten()
        epoch_metric_ = np.mean(batch_metric_array)

        epoch_metrics[e] = epoch_metric_


            # test by evaluating the model
        test_metric_list_,test_metric_ = calculate_test_klds(test_list,npdf,model,get_ypred_at_RT)
        test_metrics.append(test_metric_)
    

    return(epoch_metrics,np.array(batch_metrics_all).flatten(),test_metrics)



def train_MLP2(train_list,test_list,num_epochs,npdf,batchsize,get_ypred_at_RT,learning_rate=1e-3,
                     metric='kld'):
    

    model = my_MLP2(3,npdf)      # define model 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # optimizer to use 

    e_,b_,t_ = run_epoch_and_test(train_list,test_list,
                                  num_epochs,npdf=npdf,optimizer=optimizer,batchsize=batchsize,get_ypred_at_RT=get_ypred_at_RT,
                                  model=model)
    
    return(e_,b_,t_,model)
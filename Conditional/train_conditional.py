# load in modules

import random
import numpy as np
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

# get file paths
def get_file_paths(set_size,number_of_training_files,number_of_testing_files,total_files,file_path):
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


# load in data
def load_data_list(file_path):
    full_file_list = list(np.load(file_path,allow_pickle=True))
    return(full_file_list)

# unpack data
def unpack_data(data_list):
    '''Load .npy file, returns tensor for parameters and ground truth histograms'''
 
    ps = np.array([ a[0] for a in data_list ])
    p_tensor = torch.from_numpy(ps).float()
    y_tensor = [ torch.tensor(a[1]).float() for a in data_list ]
    
    return(p_tensor,y_tensor)


# shuffle data 
def shuffle_data(data_list):
    '''shuffles the pre-loaded data list (keeps param vectors with y values)'''
    
    random.shuffle(data_list)
    parameters = np.array([ a[0] for a in data_list ])
    parameters_tensor = torch.from_numpy(parameters).float()
    y_tensor = [ torch.tensor(a[1]).float() for a in data_list ]
    
    return(parameters_tensor,y_tensor)

# get moments
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




# get metrics
def get_metrics(ypred,y,metric):
    '''Calculates desired metric between predicted probability and y.'''
    

    y = torch.flatten(y)
    ypred = torch.flatten(ypred)

    if metric=='kld':
        return -torch.sum(y*torch.log(ypred/y))
    if metric=='kld_normalized':
        return -torch.sum(y*torch.log(ypred/y))/y.size(0)
    if metric=='totalse':
        return torch.sum((ypred-y)**2)
    if metric=='mse':
        return torch.mean((ypred-y)**2)
    if metric=='maxabsdev':
        return torch.max(torch.abs(ypred-y))
    if metric=='maxabsdevlog':
        return torch.max(torch.abs(torch.log(ypred)-torch.log(y)))
    if metric=='mselog':
        return torch.mean((torch.log(ypred)-torch.log(y))**2)


# calculate test klds 
def calculate_test_metrics(test_list,model,get_ypred_at_RT,metric):
    model.eval()
    p_list,y_list = unpack_data(test_list)
    metrics = np.zeros(len(p_list))
    
    for i in range(len(p_list)):
        y = y_list[i].flatten()

        ypred = get_predicted_PMF(p_list,i,model,get_ypred_at_RT)
        ypred = ypred.flatten()

        metric_ = get_metrics(ypred,y,metric)
        metrics[i] = metric_.detach().numpy()
        
    return(metrics,np.mean(metrics))

# get predicted PMF
def get_predicted_PMF(p_list,position,model,get_ypred_at_RT):
    '''Returns predicted histogram for p given current state of model.'''
    model.eval()

    p_ = p_list[position:position+1]
    w_,hyp_,w_un_ = model(p_)

    p = p_[0]
    w = w_[0]
    hyp = hyp_[0]

    ypred = get_ypred_at_RT(p,w,hyp)
    
    return ypred 


# define loss function
def loss_fn(ps,ys,w,hyp,get_ypred_at_RT,metric):
    '''Calculates average metval over batch between predicted Y and y.
    yker_list and y_list are actually lists of tensor histograms with first dimension batchsize'''
    
    metval = torch.tensor(0.0)
    
    batchsize = len(ps)
 

    for b in range(batchsize):
       
        y_ = ys[b]
        p_ = ps[b]
        w_ = w[b]
        hyp_ = hyp[b]
        
        ypred_ = get_ypred_at_RT(p_,w_,hyp_)
        
        met_ = get_metrics(ypred_,y_,metric)
        if torch.isnan(met_) == True:
            print('nan!',p_)

            print('batch',b)
            print('p right before',ps[b-1])
            break 
        
        metval += met_
        #print('training metval: ',metval)


    return metval/batchsize

# define model
class my_MLP1(nn.Module):

    def __init__(self, input_dim, output_dim, h1_dim, h2_dim, norm_type='softmax'):
        super().__init__()

        self.input = nn.Linear(input_dim, h1_dim)
        self.hidden = nn.Linear(h1_dim, h2_dim)
        self.output = nn.Linear(h2_dim, output_dim)

        self.hyp = nn.Linear(h2_dim,1)

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = torch.sigmoid

        
        self.norm_type = norm_type
        

    def forward(self, inputs):

        # pass inputs to first layer, apply sigmoid
        l_1 = self.sigmoid(self.input(inputs))
        #l_1 = F.relu(self.input(inputs))

        # pass to second layer, apply sigmoid
        l_2 = self.sigmoid(self.hidden(l_1))
        #l_2 = F.relu(self.hidden(l_1))
        
        # pass inputs out (unnormalized)
        w_un = (self.output(l_2))
        
        # pass out hyperparameter, sigmoid so it is between 0 and 1, then scale between 1 and 6
        hyp = self.sigmoid(self.hyp(l_2))*5+1 
    

        # apply normalization or softmax
        if self.norm_type == 'softmax':
            w_pred = self.softmax(w_un)
      
        elif self.norm_type == 'normalize':
            # before normalization, apply sigmoid to ensure all weights are positive
            w_pred_ = self.sigmoid(w_un)
            w_pred = (w_un/w_un.sum(axis=0)).sum(axis=0)

        else:
            # just take absolute value of output weights
            w_pred = torch.abs(self.output(w_un)) #no softmax at all
    

        # return unnormalized weights in case you want to apply l1 regularization 
        return w_pred, hyp, w_un 





def run_epoch(p_list,y_list,model,optimizer,batchsize,get_ypred_at_RT,metric):

    model.train()

    # number of batches (data/batchsize)
    trials = int(np.floor(p_list.size()[0] / batchsize ))

    # should this be an array or a tensor ???? 
    metvals = torch.zeros(trials)

    for j in range(trials):
        i = j * batchsize
        ps = p_list[i:i+batchsize]
        ys = y_list[i:i+batchsize]

        # Zero the gradients
        optimizer.zero_grad()
        
        # Perform forward pass
        w, hyp, w_un  = model(ps)

        # Compute loss
    
        loss = loss_fn(ps,ys,w,hyp,get_ypred_at_RT,metric)
        
        # average metric for the batch j
        metvals[j] = loss.item()

        # Perform backward pass
        loss.backward()

        #nn.utils.clip_grad_value_(model.parameters(), clip_value=.1)
      
        # Perform optimization
        optimizer.step()

        for p in model.parameters():
            if torch.isnan(torch.sum(p.grad)):
                print('trial',j)
                print('parameter',p)
                print('gradients',p.grad)
                
    
    # calculate the average metric over the epoch 
    av_metval = torch.mean(metvals)

    return(av_metval)


def train(train_list,test_list,model_config,train_config,get_ypred_at_RT):
    
    train_list_ = train_list
    # define model configurations
    npdf = model_config['npdf']
    input_dim = model_config['input_dim']
    h1_dim = model_config['h1_dim']
    h2_dim = model_config['h2_dim']
    norm_type = model_config['norm_type']
    

    # define model
    model = my_MLP1(input_dim=input_dim, output_dim=npdf,
     h1_dim=h1_dim, h2_dim=h2_dim, norm_type=norm_type)
        

    # define training configurations
    num_epochs = train_config['num_epochs']
    lr = train_config['lr']
    batchsize = train_config['batchsize']
    metric = train_config['metric']

    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 

    
    epoch_metvals = np.zeros(num_epochs)
    test_metvals = np.zeros(num_epochs)
    
    for e in range(num_epochs):
        print('Epoch Number:',e)

     
        batch_metrics_ = []

        p_list,y_list = shuffle_data(train_list_)
        

        # REDEFINE
        metval_ = run_epoch(p_list,y_list,model,optimizer,batchsize,get_ypred_at_RT,metric)
       
        # store epoch metric
        epoch_metvals[e] = metval_


        # test by evaluating the model
        
        test_metval_list_,test_metval_ = calculate_test_metrics(test_list,model,get_ypred_at_RT,metric)
        
        # store test metric
        test_metvals[e] = test_metval_
    

    return(epoch_metvals, test_metvals, model)
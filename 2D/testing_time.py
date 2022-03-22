import numpy as np
import time
import torch

import scipy.stats as stats
from scipy.special import gammaln


import train_2D_rt as tr

set_size = 1
num_files = 15
N = num_files*set_size


params = tr.generate_param_vectors(N)

def get_moments(p,N):
    b,beta,gamma=p
    
    r = torch.tensor([1/beta, 1/gamma])
    MU = b*r
    VAR = MU*torch.tensor([1+b,1+b*beta/(beta+gamma)])
    STD = torch.sqrt(VAR)
    xmax = torch.ceil(MU)
    xmax = torch.ceil(xmax + N*STD)
    xmax = torch.clip(xmax,30,np.inf).int()
    return MU, VAR, STD, xmax

def calculate_exact_cme(p,method,N):
    
    '''Given parameter vector p, calculate the exact probabilites using CME integrator.'''
    p1 = torch.from_numpy(p).float()
    p1 = 10**p1
    
    MU, VAR, STD, xmaxc = get_moments(p1,N)

    
    xmaxc = np.array([int(xmaxc[0]),int(xmaxc[1])])
    
    y = tr.cme_integrator(np.array(p1),xmaxc+1,method=method)
    
    return(xmaxc[0]*xmaxc[1])

# FIXED QUAD
P = 15 

sigmas = [1,2,3,5,10,15,25,50]
state_spaces = []

time_sigmas_fixedquad = []

for sig in sigmas:
    print(sig)
    t1 = time.time()

    state_spaces_ = np.zeros(P)

    for i in range(P):

        s_ = calculate_exact_cme(params[i], method = 'fixed_quad',N=sig)
        state_spaces_[i] = s_
    
    state_spaces.append(state_spaces_)
    t2 = time.time()
    
    time_sigmas_fixedquad.append(t2-t1)
    
# QUAD VEC




time_sigmas_quadvec = []

for sig in sigmas:
    print(sig)
    t1 = time.time()



    for i in range(P):

        s_ = calculate_exact_cme(params[i], method = 'quad_vec',N=sig)

    
    t2 = time.time()
    
    time_sigmas_quadvec.append(t2-t1)
    
    
    
    
# NN

def generate_grid(npdf,VAR,MU,quantiles=None):
    if quantiles=='PRESET':
        logstd = torch.sqrt(np.log((VAR/MU**2)+1))
        logmean = torch.log(MU**2/np.sqrt(VAR+MU**2))
        translin_0 = torch.exp(logmean[0]+logstd[0]*NORM_nas)
        translin_1 = torch.exp(logmean[1]+logstd[1]*NORM_mat)
        return translin_0,translin_1
        return(translin)

def get_ypred_at_RT(p,npdf,w,N,hyp=2.4,quantiles='PRESET',
                   first_special=False,special_std='tail_prob'):
    p = 10**p
    MU, VAR, STD, xmax = get_moments(p,N)
    
    #two separate variables. a bit ugly and leaves room for error. 
    grid_nas,grid_mat = generate_grid(npdf,VAR,MU,quantiles=quantiles) 
    # no zs implementation yet. not sure i want to implement it.

    s_nas = torch.zeros(npdf[0])
    s_mat = torch.zeros(npdf[1])

    spec = 0 if first_special else -1
    if first_special:
        s_nas[1:] = torch.diff(grid_nas)
        s_mat[1:] = torch.diff(grid_mat)
    else: #last special... for now
        s_nas[:-1] = torch.diff(grid_nas)
        s_mat[:-1] = torch.diff(grid_mat)
    
    if special_std == 'mean':
        s_nas[spec] = grid_nas[spec]
        s_mat[spec] = grid_mat[spec]
    elif special_std == 'neighbor': #assign_neighbor_to_special
        s_nas[spec] = s_nas[1] if first_special else s_nas[-2]
        s_mat[spec] = s_mat[1] if first_special else s_mat[-2]
    elif special_std == 'tail_prob':
        if first_special:
            print('If you are using this setting, you are doing something wrong.')
        t_max = torch.log(p[1]/p[2])/(p[1] - p[2])
        f = (torch.exp(-p[2]*t_max) - torch.exp(-p[1]*t_max)) * p[1]/(p[1] - p[2]) * p[0]
        tailratio = 1/(1+1/f) #the mature tail ratio
        s_mat[spec] = torch.sqrt(grid_mat[spec] / (1-tailratio))
        tailratio = p[0]/(1+p[0]) #the nascent tail ratio
        s_nas[spec] = torch.sqrt(grid_nas[spec] / (1-tailratio))
    else:
        print('did not specify a standard deviation convention!')
    
    s_nas *= hyp
    s_mat *= hyp
    v_nas = s_nas**2
    v_mat = s_mat**2

    r_nas = grid_nas**2/(v_nas-grid_nas)
    p_nas = 1-grid_nas/v_nas 
    r_mat = grid_mat**2/(v_mat-grid_mat)
    p_mat = 1-grid_mat/v_mat 
    
    xgrid_nas = torch.arange(xmax[0]+1)
    xgrid_mat = torch.arange(xmax[1]+1)
    
    gammaln_xgrid_nas = lnfactorial[1:(xmax[0]+2)]
    gammaln_xgrid_mat = lnfactorial[1:(xmax[1]+2)] 

    Y = torch.zeros((xmax[0]+1,xmax[1]+1))
    
    for i in range(npdf[0]):
        lnas = -grid_nas[i] + xgrid_nas * torch.log(grid_nas[i]) - gammaln_xgrid_nas
        if p_nas[i] > 1e-10:
            lnas += torch.special.gammaln(xgrid_nas+r_nas[i]) - torch.special.gammaln(r_nas[i]) \
                - xgrid_nas*torch.log(r_nas[i] + grid_nas[i]) + grid_nas[i] \
                + r_nas[i]*torch.log(1-p_nas[i])
        for j in range(npdf[1]):
            lmat =  - grid_mat[j] + xgrid_mat * torch.log(grid_mat[j]) - gammaln_xgrid_mat
            if p_mat[j] > 1e-10:
                lmat += torch.special.gammaln(xgrid_mat+r_mat[j]) - torch.special.gammaln(r_mat[j]) \
                - xgrid_mat*torch.log(r_mat[j] + grid_mat[j]) + grid_mat[j] \
                + r_mat[j]*torch.log(1-p_mat[j]) #wasteful: we're recomputing a lot of stuff.
            Y += w[i*npdf[1] + j] * torch.exp(lnas[:,None] + lmat[None,:])
            #note convention change. Y = the predicted PMF is now returned in the same shape as the original histogram.
            #this is fine bc Y is flattened anyway later on down the line.
    return Y




# define NORM and YPRED_FUN

def NORM_function(npdf):
    if npdf[0] == npdf[1]:
        n = np.arange(npdf[0])
        q = np.flip((np.cos((2*(n+1)-1)/(2*npdf)*np.pi)+1)/2)
        NORM = stats.norm.ppf(q)
        NORM_nas = torch.tensor(NORM)
        NORM_mat = NORM_nas
    else:
        n = np.arange(npdf[0])
        q = np.flip((np.cos((2*(n+1)-1)/(2*npdf[0])*np.pi)+1)/2)
        #print(q)
        NORM_nas = torch.tensor(stats.norm.ppf(q))
        n = np.arange(npdf[1])
        q = np.flip((np.cos((2*(n+1)-1)/(2*npdf[1])*np.pi)+1)/2)
        #print(q)
        NORM_mat = torch.tensor(stats.norm.ppf(q))
    

    n_n = np.linspace(0,1,npdf[0]+2)[1:-1]
    n_m = np.linspace(0,1,npdf[1]+2)[1:-1]
    NORM_nas = stats.norm.ppf(n_n)
    NORM_mat = stats.norm.ppf(n_m)
    #print(NORM_nas)
    return(NORM_nas,NORM_mat)

lnfactorial = torch.special.gammaln(torch.arange(10000000))
    

YPRED_FUN = lambda p, npdf, w, N: get_ypred_at_RT(p=p,npdf=npdf,w=w,N=N,hyp=2.4,
                                               quantiles='PRESET')



def get_predicted_PMF(p_list,npdf,N,position,model,get_ypred_at_RT):
    '''Returns predicted histogram for p given current state of model.'''
    model.eval()

    p1 = p_list[position:position+1]
    w_p1 = model(p1)[0]
    p1 = p1[0]
    predicted_y1 = get_ypred_at_RT(p1,npdf,w_p1,N)
    
    return(predicted_y1)



npdf = [10,11]
model_10 = tr.my_MLP1(3,npdf[0]*npdf[1])
model_10.load_state_dict(torch.load('./quadvec_models/10npdf_256params_qlin_MODEL'))
model_10.eval()


npdf = [20,21]
# pre-loaded model
model_20 = tr.my_MLP1(3,npdf[0]*npdf[1])
model_20.load_state_dict(torch.load('./quadvec_models/07032022_20npdf_1train_qlin_15epochs_MODEL'))
model_20.eval()


npdf = [30,31]
# pre-loaded model
model_30 = tr.my_MLP1(3,npdf[0]*npdf[1])
model_30.load_state_dict(torch.load('./quadvec_models/30npdf_256params_qlin_MODEL'))
model_30.eval()



params_tensor = torch.from_numpy(params).float()


npdf = [10,11]
time_sigmas_NN_10 = []

NORM_nas,NORM_mat = NORM_function(np.array(npdf))

for sig in sigmas:
    print(sig)
    t1 = time.time()


    for i in range(P):

        s_ = get_predicted_PMF(params_tensor[i:i+1],npdf,sig,0,model_10,
                               YPRED_FUN)
    t2 = time.time()
    
    time_sigmas_NN_10.append(t2-t1)
    
    
    
npdf = [20,21]
time_sigmas_NN_20 = []

NORM_nas,NORM_mat = NORM_function(np.array(npdf))

for sig in sigmas:
    print(sig)
    t1 = time.time()


    for i in range(P):

        s_ = get_predicted_PMF(params_tensor[i:i+1],npdf,sig,0,model_20,
                               YPRED_FUN)
    t2 = time.time()
    
    time_sigmas_NN_20.append(t2-t1)
    
 


 
npdf = [30,31]
time_sigmas_NN_30 = []

NORM_nas,NORM_mat = NORM_function(np.array(npdf))

for sig in sigmas:
    print(sig)
    t1 = time.time()


    for i in range(P):

        s_ = get_predicted_PMF(params_tensor[i:i+1],npdf,sig,0,model_30,
                               YPRED_FUN)
    t2 = time.time()
    
    time_sigmas_NN_30.append(t2-t1)
    
    
    
    
sigma_state_space = np.array([np.sum(a) for a in state_spaces])




save_list =  [np.array(sigmas),
              sigma_state_space,
              np.array(time_sigmas_fixedquad),np.array(time_sigmas_quadvec),
              np.array(time_sigmas_NN_10),
              np.array(time_sigmas_NN_20),
              np.array(time_sigmas_NN_30)]


np.save('./plots/03172022_testing_time',save_list)
  
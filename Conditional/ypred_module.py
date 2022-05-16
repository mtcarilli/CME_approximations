import numpy as np
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F


# DEFINE MODEL PATH! 
model_path = './models/20220422_cheb_hyp_MODEL'

# define MLP
class my_MLP1(nn.Module):

    def __init__(self, input_dim, npdf, h1_dim, h2_dim, norm_type='softmax'):
        super().__init__()

        self.input = nn.Linear(input_dim, h1_dim)
        self.hidden = nn.Linear(h1_dim, h2_dim)
        self.output = nn.Linear(h2_dim, npdf)
        

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
        hyp = self.sigmoid(self.hyp(l_2))
    

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
        
        
       
        return w_pred,hyp

        
        
        
npdf = 10

# load in model
model = my_MLP1(7,10,256,256,'softmax')
model.load_state_dict(torch.load(model_path))
model.eval()


# precalculate lngammas
lnfactorial = torch.special.gammaln(torch.arange(1003))


def get_NORM(npdf,quantiles):
    if quantiles == 'lin':
        q = np.linspace(0,1,npdf+2)[1:-1]
        NORM = stats.norm.ppf(q)
        NORM = torch.tensor(NORM)
        return NORM
    if quantiles == 'cheb':
        n = np.arange(npdf)
        q = np.flip((np.cos((2*(n+1)-1)/(2*npdf)*np.pi)+1)/2)

        NORM = stats.norm.ppf(q)
        NORM = torch.tensor(NORM)
        return NORM


NORM = get_NORM(npdf,quantiles='cheb')

def get_moments(p):
    b,beta,gamma=p
    
    r = np.array([1/beta, 1/gamma])
    MU = b*r
    VAR = MU*np.array([1+b,1+b*beta/(beta+gamma)])
    STD = np.sqrt(VAR)
    xmax = np.ceil(MU)
    xmax = np.ceil(xmax + 4*STD)
    xmax = np.clip(xmax,30,np.inf)
    xmax_m = int(xmax[1])
    return MU, VAR, STD, xmax_m

def get_conditional_moments(MU, VAR, STD, COV, nas_range):
    
    logvar = np.log((VAR/MU**2)+1)
    logstd = np.sqrt(logvar)
    logmean = np.log(MU**2/np.sqrt(VAR+MU**2))

    logcov = np.log(COV * np.exp(-(logmean.sum()+logvar.sum()/2)) +1 ) 
    logcorr = logcov/np.sqrt(logvar.prod())

    logmean_cond = logmean[1] + logcorr * logstd[1]/logstd[0] * (np.log(nas_range+1) - logmean[0])
    logstd_cond = logstd[1] * np.sqrt(1-logcorr**2)   
   
    # convert to tensors
    logstd_cond = torch.tensor([logstd_cond],dtype=torch.float).repeat(len(nas_range),1)
    logmean_cond = torch.tensor(logmean_cond,dtype=torch.float)

    return(logmean_cond,logstd_cond)


def generate_grid(logmean_cond,logstd_cond):
    

    logmean_cond = torch.reshape(logmean_cond,(-1,1))
    logstd_cond = torch.reshape(logstd_cond,(-1,1))
    translin = torch.exp(torch.add(logmean_cond,logstd_cond*NORM))
    
    return translin


npdf = 10

def get_NORM(npdf,quantiles='cheb'):
    if quantiles == 'lin':
        q = np.linspace(0,1,npdf+2)[1:-1]
        NORM = stats.norm.ppf(q)
        NORM = torch.tensor(NORM)
        return NORM
    if quantiles == 'cheb':
        n = np.arange(npdf)
        q = np.flip((np.cos((2*(n+1)-1)/(2*npdf)*np.pi)+1)/2)

        NORM = stats.norm.ppf(q)
        NORM = torch.tensor(NORM)
        return NORM

def generate_grid(logmean_cond,logstd_cond):
    

    logmean_cond = torch.reshape(logmean_cond,(-1,1))
    logstd_cond = torch.reshape(logstd_cond,(-1,1))
    translin = torch.exp(torch.add(logmean_cond,logstd_cond*NORM))
    return translin


def get_ypred_at_RT(p,w,hyp,n_range,m_range,pos,npdf=npdf):
    '''Given a parameter vector (tensor) and weights (tensor), and hyperparameter,
    calculates ypred (Y) at runtime.'''
    
        
    p_vec = 10**p[:,0:3]
    logmean_cond = p[:,3]
    logstd_cond = p[:,4]
    
    hyp = hyp*5+1

    grid = generate_grid(logmean_cond,logstd_cond)
    s = torch.zeros((len(n_range),npdf))
    s[:,:-1] = torch.diff(grid,axis=1)
    s *= hyp
    s[:,-1] = torch.sqrt(grid[:,-1])
   

    
    v = s**2
    r = grid**2/(v-grid)
    p_nb = 1-grid/v
    
    xgrid = torch.tensor(m_range).repeat(len(n_range),1)
    Y = torch.zeros((len(n_range),len(m_range)))
    index = torch.tensor(m_range+1,dtype=torch.long)
    GAMMALN_XGRID = torch.index_select(lnfactorial, 0, index).repeat(len(n_range),1)
    
    for i in range(npdf):
        grid_i = grid[:,i].reshape((-1,1))


        r_i = r[:,i].reshape((-1,1))
        w_i = w[:,i].reshape((-1,1))
        p_nb_i = p_nb[:,i]
        
        l = -grid_i + torch.mul(xgrid,torch.log(grid_i )) - GAMMALN_XGRID

        if (p_nb_i > 1e-10).any():

            index = [p_nb_i > 1e-10]
            l[index] += torch.special.gammaln(xgrid[index]+r_i[index]) - torch.special.gammaln(r_i[index]) \
                - xgrid[index]*torch.log(r_i[index] + grid_i[index]) + grid_i[index] \
                + r_i[index]*torch.log(r_i[index]/(r_i[index]+grid_i[index]))

        Y += torch.mul(w_i,torch.exp(l))
    

    EPS = 1e-40
    Y[Y<EPS]=EPS
    return Y


NORM = get_NORM(npdf,quantiles='cheb')

# precalculate lngammas
lnfactorial = torch.special.gammaln(torch.arange(1003))


def get_prob(p_in,nas_range,mat_range):
    
    # calculate overall moments
    pv = 10**p_in
    MU, VAR, STD, xmax_m = get_moments(pv)
    COV = pv[0]**2/(pv[1] + pv[2])
    
    # calculate negative binomial P(n) 
    b = pv[0]
    beta = pv[1]
    n = 1/beta
    p = 1/(b+1)

    prob_n = stats.nbinom.pmf(k=nas_range, n=n, p=p)
 
    
    # calculate conditional moments
    logmean_cond,logstd_cond = get_conditional_moments(MU, VAR, STD, COV, nas_range)
    
    
    # now convert to tensors
    mat_range = torch.tensor(mat_range)
    nas_range = torch.tensor(nas_range)
    p_in_array = torch.tensor(p_in,dtype=torch.float).repeat(len(nas_range),1)
    xmax_m = torch.tensor(xmax_m,dtype=torch.float).repeat(len(nas_range),1)

    # and stack for model
    p_array = torch.column_stack((p_in_array,logmean_cond,logstd_cond,xmax_m,nas_range))
    
    # run through model
    w_,hyp_= model(p_array)
    
    # get conditional probabilites
    ypred_cond = get_ypred_at_RT(p_array,w_,hyp_,nas_range,mat_range,0)
    
    
    # multiply conditionals P(m|n) by P(n)
    predicted = prob_n.reshape((-1,1))* ypred_cond.detach().numpy()
    
    
    return(predicted)
    
    
    
    
    
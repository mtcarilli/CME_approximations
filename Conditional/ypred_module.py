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
        
        
       
        return w_pred,hyp,w_un

        
        
        
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

def get_conditional_moments(MU, VAR, STD, xmax, COV, n):
    
    logvar = np.log((VAR/MU**2)+1)
    logstd = np.sqrt(logvar)
    logmean = np.log(MU**2/np.sqrt(VAR+MU**2))

    logcov = np.log(COV * np.exp(-(logmean.sum()+logvar.sum()/2)) +1 ) 
    logcorr = logcov/np.sqrt(logvar.prod())

    logmean_cond = logmean[1] + logcorr * logstd[1]/logstd[0] * (np.log(n+1) - logmean[0])
    logstd_cond = logstd[1] * np.sqrt(1-logcorr**2)   
    
    
    return(logmean_cond,logstd_cond,xmax)

def generate_grid(logmean_cond,logstd_cond,NORM):
    
    translin = torch.exp(logmean_cond+logstd_cond*NORM)
    return translin



def get_ypred_at_RT(p,w,hyp,mat,npdf=npdf,NORM=NORM):
    '''Given a parameter vector (tensor) and weights (tensor), and hyperparameter,
    calculates ypred (Y) at runtime.'''
        
    p_vec = 10**p[0:3]
    logmean_cond = p[3]
    logstd_cond = p[4]
    

    hyp = hyp*5+1
    
    
    xmax_m = p[5].int()

    grid = generate_grid(logmean_cond,logstd_cond,NORM)

    s = torch.zeros(npdf)

    s[:-1] = torch.diff(grid)
    s *= hyp
    s[-1] = torch.sqrt(grid[-1])
    
    v = s**2

    r = grid**2/(v-grid)
    p_nb = 1-grid/v

    GAMMALN_XGRID = lnfactorial[mat+1]
    
    Y = 0 
    
    for i in range(npdf):
        l = -grid[i] + mat * torch.log(grid[i]) - GAMMALN_XGRID
        if (p_nb[i] >1e-10):
            l += torch.special.gammaln(mat+r[i]) - torch.special.gammaln(r[i]) \
                - mat*torch.log(r[i] + grid[i]) + grid[i] \
                + r[i]*torch.log(r[i]/(r[i]+grid[i]))
        Y += w[i]*torch.exp(l)

    return Y.detach().numpy()


def get_prob(p_in,nas,mat):
    p_vec = 10**p_in
    b = p_vec[0]
    beta = p_vec[1]
    n = 1/beta
    p = 1/(b+1)
    
    MU, VAR, STD, xmax = get_moments(p_vec)
    COV = p_vec[0]**2/(p_vec[1] + p_vec[2])
    
    prob_n = stats.nbinom.pmf(k=nas, n=n, p=p)

    logmean_cond,logstd_cond,xmax_m = get_conditional_moments(MU, VAR, STD, xmax, COV, nas)
    
    p_ = torch.tensor([[p_in[0],p_in[1],p_in[2],logmean_cond,logstd_cond,xmax_m,nas]],dtype=torch.float)
    
    w_,hyp_,w_un_ = model(p_)
    
    w = w_[0]
    hyp = hyp_[0]
    
    cond_prob = get_ypred_at_RT(p_[0],w,hyp,mat)
    
    
    prob_m_n = prob_n*cond_prob
    
    return(prob_m_n)
    
    
    
    
    
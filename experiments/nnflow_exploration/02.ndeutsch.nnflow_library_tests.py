#!/usr/bin/env python
# coding: utf-8

# In[5]:


import torch
from math import pi,sqrt,log,e
from time import time
from matplotlib import pyplot
import numpy as np
from matplotlib import pyplot as plt
from tqdm.autonotebook import tqdm
dtype = torch.float
device = torch.device("cuda:1")
#device = torch.device("cpu")


# # Testing jacobians for simple transforms

# In[11]:


from src.models.flows.sampling import FactorizedFlowPrior
from src.models.flows.backprop_jacobian_flows.simple_backprop_flows import LinearFlow,SigmoidFlow


# ## Linear transform

# In[23]:


prior_mu =  torch.tensor(0.).to(device)
prior_sig =  torch.tensor(100.).to(device)
prior = torch.distributions.normal.Normal(prior_mu,prior_sig)

sampler = FactorizedFlowPrior(d=2,prior_1d=prior)
nnflow = LinearFlow(d=2)

nnflow.cuda(device=device)

nbatch=100000
zj = sampler(nbatch)
xlj = nnflow(zj)
j = torch.exp(xlj[:,-1])
v,r=torch.var_mean(j)
v=v.detach().cpu().item()
r=r.detach().cpu().item()
print("calculated Jacobian: ",torch.mean(xlj[:,-1]-(zj[:,-1])).detach().cpu().item())
print("analytic Jacobian  : ",torch.log(torch.abs(torch.det(nnflow.flow.weight.data))).cpu().item())


# ## Sigmoid transform

# In[21]:


prior_mu =  torch.tensor(0.).to(device)
prior_sig =  torch.tensor(3.).to(device)
prior = torch.distributions.normal.Normal(prior_mu,prior_sig)

sampler = FactorizedFlowPrior(d=2,prior_1d=prior)
nnflow = SigmoidFlow(d=2)

nnflow.cuda(device=device)

nbatch=10000
zj = sampler(nbatch)
xlj = nnflow(zj)
j = torch.exp(xlj[:,-1])
v,r=torch.var_mean(j)
v=v.detach().cpu().item()
r=r.detach().cpu().item()
print("Integral of 1 sampled non-uniformly: {}+/-{}".format(r,sqrt(v/nbatch)))
sj1=torch.exp(xlj[:,-1]-zj[:,-1])
sj2=xlj[:,0]*(1-xlj[:,0])*xlj[:,1]*(1-xlj[:,1])
print("computed Jacobian: ",sj1)
print("analytic Jacobian: ",sj2)
print("All relative differences < .1%: ",torch.all(torch.abs(sj1-sj2)/sj1 < 1.e-3).cpu().item())


# ## Combining two transforms

# In[45]:


sig=1.
prior_mu =  torch.tensor(0.).to(device)
prior_sig =  torch.tensor(sig).to(device)
prior = torch.distributions.normal.Normal(prior_mu,prior_sig)

sampler = FactorizedFlowPrior(d=2,prior_1d=prior)
lflow = LinearFlow(d=2)
lflow.weight_init_identity_(0.001)
sflow = SigmoidFlow(d=2)

sampler.cuda(device=device)
lflow.cuda(device=device)
sflow.cuda(device=device)

nbatch=10000000

zj = sampler(nbatch)
yj = lflow(zj)
xj = sflow(yj)
j = torch.exp(xj[:,-1])
v,r=torch.var_mean(j)
v=v.detach().cpu().item()
r=r.detach().cpu().item()
print("Integral of 1 sampled non-uniformly: {}+/-{}".format(r,sqrt(v/nbatch)))


# In[47]:


# Checking gradients
def normal(x,s):
    return(torch.exp(-(x/s)**2/2.)/sqrt(2.*pi*s**2))

print(torch.exp(yj[:,-1] - zj[:,-1]))
print(torch.abs(torch.det(lflow.flow.weight.data)))

print(torch.exp(xj[:,-1] - yj[:,-1]))
print(xj[:,0]*(1-xj[:,0])*xj[:,1]*(1-xj[:,1]))

print(torch.exp(zj[:,-1]))
print(torch.abs(1/(normal(zj[:,0],sig)*normal(zj[:,1],sig))))


# In[ ]:





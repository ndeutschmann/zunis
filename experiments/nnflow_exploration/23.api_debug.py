#!/usr/bin/env python
# coding: utf-8

# # Debugging the user facing API
# 
# Starting revno: 777cc59c2b06ceeb0a22d670b982342240bc00e9
# fixed bugs with: 00b99d096a8db08b986fa188e878a1901fb8d9ea
# 

# In[1]:


import torch
import numpy as np
from matplotlib import pyplot as plt
from functools import partial

from src.integration.default_integrator import Integrator
from src import setup_std_stream_logger


# In[2]:


setup_std_stream_logger(debug=True)
device=torch.device("cuda:7")


# In[3]:


def nonzerocos(x):
    return torch.cos(4*(x[:,0]+x[:,1]))**2


# In[51]:


integrator = Integrator(d=2,f=nonzerocos,loss="variance",
                        trainer_options={"n_epochs":10,"minibatch_size":20000,"optim":partial(torch.optim.Adam,lr=1.e-3)},
                        flow_options={"cell_params":{"n_bins":300}},
                        verbosity=3,trainer_verbosity=1,device=device)


# In[52]:


result = integrator.integrate(50,1)


# In[53]:


x,px,fx = integrator.sample_survey()
varf,meanf = torch.var_mean(fx/px)
print(f"flat: {meanf} with var {varf}")
x,px,fx = integrator.sample_refine()
var,mean = torch.var_mean(fx/px)
print(f"model: {mean} with var {var}")
print(f"speed-up: {varf/var}")
print("\n")

x,px,fx=integrator.sample_refine(n_points=100000)
fig, axs = plt.subplots(1,3)
fig.set_size_inches((12,4))
n=30
h2=axs[0].hist2d(x[:,0].cpu().numpy(),x[:,1].cpu().numpy(),bins=n)
axs[1].imshow(nonzerocos(torch.cartesian_prod(torch.arange(0,1,1/n),torch.arange(0,1,1/n))).reshape(n,n).numpy(),origin="lower",)
axs[2].imshow(h2[0],origin="lower")
plt.show()


# In[58]:


# This is still a problem in 00b99d096a8db08b986fa188e878a1901fb8d9ea

integrator = Integrator(d=2,f=nonzerocos,flow="realnvp",loss="variance",
                        trainer_options={"n_epochs":10,"minibatch_size":20000,"optim":partial(torch.optim.Adam,lr=1.e-6)},
                        verbosity=3,trainer_verbosity=1,device=device)

result = integrator.integrate(1,1)


# In[7]:


# The problem is with the variance loss, not with the model
integrator = Integrator(d=2,f=nonzerocos,flow="realnvp",loss="dkl",
                        trainer_options={"n_epochs":100,"minibatch_size":20000,"optim":partial(torch.optim.Adam,lr=1.e-3)},
                        flow_options={"cell_params":{"n_hidden":16}},
                        verbosity=3,trainer_verbosity=1,device=device)

result = integrator.integrate(5,1)


# In[8]:


x,px,fx = integrator.sample_survey()
varf,meanf = torch.var_mean(fx/px)
print(f"flat: {meanf} with var {varf}")
x,px,fx = integrator.sample_refine()
var,mean = torch.var_mean(fx/px)
print(f"model: {mean} with var {var}")
print(f"speed-up: {varf/var}")
print("\n")

x,px,fx=integrator.sample_refine(n_points=100000)
fig, axs = plt.subplots(1,3)
fig.set_size_inches((12,4))
n=30
h2=axs[0].hist2d(x[:,0].cpu().numpy(),x[:,1].cpu().numpy(),bins=n)
axs[1].imshow(nonzerocos(torch.cartesian_prod(torch.arange(0,1,1/n),torch.arange(0,1,1/n))).reshape(n,n).numpy(),origin="lower",)
axs[2].imshow(h2[0],origin="lower")
plt.show()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from matplotlib import pyplot
import numpy as np
from matplotlib import pyplot as plt
device = torch.device("cuda:7")


# In[2]:


from src import setup_std_stream_logger
setup_std_stream_logger(debug=False)


# In[3]:


from src.integration import DefaultIntegrator


# In[4]:


# Defining a camel function
def f(x):
    return torch.exp( - torch.sum(((x-0.25)/0.1)**2,axis=-1)) + torch.exp( - torch.sum(((x-0.75)/0.1)**2,axis=-1))


# In[5]:


integrator=DefaultIntegrator(f=f, d=2, device=device, n_epochs=20, minibatch_size=10000,lr=1.e-4, model_params={"repetitions":3})


# In[6]:


result=integrator.integrate(2,10)


# In[8]:


x=integrator.model_trainer.sample_forward(100000).cpu().numpy()
plt.figure(figsize=(5,5))
plt.hist2d(x[:,0],x[:,1],bins=30)
plt.show()


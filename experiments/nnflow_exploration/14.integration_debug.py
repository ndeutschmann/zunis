#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from math import pi,sqrt,log,e,exp
from time import time
from matplotlib import pyplot
import numpy as np
from matplotlib import pyplot as plt
from tqdm.autonotebook import tqdm
#torch.set_default_dtype(torch.float64)
device = torch.device("cuda:7")


# In[2]:


from src.models.flows.coupling_cells.real_nvp import RealNVP
from src.models.flows.sampling import FactorizedGaussianSampler, UniformSampler
from src.models.flows.analytic_flows.element_wise import InvertibleAnalyticSigmoid
from src.models.flows.sequential import InvertibleSequentialFlow
from src.training.weighted_dataset.dkl_training import BasicStatefulDKLTrainer
from src import setup_std_stream_rootlogger
from src.integration.flat_refine_integrator import FlatSurveySamplingIntegrator


# In[3]:


setup_std_stream_rootlogger(debug=False)


# In[4]:


def f(x):
    return torch.exp(-10*(2*x[:,1]-torch.cos(4*pi*x[:,0])-1)**2)


# In[5]:


posterior=UniformSampler(d=2,low=0.,high=1.,device=device)
prior=FactorizedGaussianSampler(d=2,device=device)


# In[6]:


try:
    del model
except:
    pass


model  = InvertibleSequentialFlow(2,[
        RealNVP(d=2,
              mask=[True,False],
              d_hidden=256,
              n_hidden=16,).to(device),
        RealNVP(d=2,
              mask=[False,True],
              d_hidden=256,
              n_hidden=16,).to(device),
        RealNVP(d=2,
              mask=[True,False],
              d_hidden=256,
              n_hidden=16,).to(device),
        RealNVP(d=2,
              mask=[False,True],
              d_hidden=256,
              n_hidden=16,).to(device), 
    InvertibleAnalyticSigmoid(d=2),
])

optim = torch.optim.Adam(model.parameters(),lr=1.e-4)


# In[7]:


trainer = BasicStatefulDKLTrainer(flow=model,latent_prior=prior)


# In[8]:


trainer.set_config(n_epochs=100, minibatch_size=20000, optim=torch.optim.Adam(model.parameters()))


# In[9]:


integrator=FlatSurveySamplingIntegrator(f,trainer,2,device=device)


# In[10]:


result=integrator.integrate(1,10)


# In[11]:


integrator.integration_history.loc[0]


# In[12]:


x=trainer.sample_forward(100000).cpu().numpy()
plt.figure(figsize=(5,5))
plt.hist2d(x[:,0],x[:,1],bins=30)
plt.show()


# In[13]:


x,px,fx=integrator.sample_refine()


# In[14]:


torch.mean(fx/px)


# In[15]:


torch.var(fx/px)


# In[16]:


x,px,fx=trainer.generate_target_batch_from_posterior(10000,f,posterior)


# In[17]:


torch.mean(fx/px)


# In[18]:


torch.var(fx/px)


# In[28]:


result


# In[27]:


((result[2][result[2]["phase"] == "refine"]["error"]**2).mean()/ len(result[2][result[2]["phase"] == "refine"]))**.5


# In[30]:


((result[2][result[2]["phase"] == "refine"]["error"]**2).mean())**0.5


# In[31]:


((result[2][result[2]["phase"] == "refine"]["error"]).mean())


# In[ ]:





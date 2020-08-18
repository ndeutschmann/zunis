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
device = torch.device("cuda:6")


# In[2]:


from src.models.flows.coupling_cells.real_nvp import RealNVP
from src.models.flows.sampling import FactorizedGaussianSampler, UniformSampler
from src.models.flows.analytic_flows.element_wise import InvertibleAnalyticSigmoid
from src.models.flows.sequential import InvertibleSequentialFlow
from src.training.weighted_dataset.dkl_training import BasicStatefulDKLTrainer
from src import setup_std_stream_logger
from src.integration.dkltrainer_integrator import DKLAdaptiveSurveyIntegrator


# In[3]:


setup_std_stream_logger(debug=True)


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


rm test_checkpoint.h5


# In[8]:


try:
    del trainer
except:
    pass


# In[9]:


trainer = BasicStatefulDKLTrainer(flow=model,latent_prior=prior,checkpoint="test_checkpoint.h5",max_reloads=5)


# In[10]:


trainer.set_config(n_epochs=30, minibatch_size=20000, optim=optim)


# In[11]:


try:
    del integrator
except:
    pass


# In[12]:


integrator=DKLAdaptiveSurveyIntegrator(f,trainer,2,device=device,trainer_verbosity=3)


# In[13]:


trainer.config


# In[14]:


result=integrator.integrate(10,10)


# In[15]:


x=trainer.sample_forward(100000).cpu().numpy()
plt.figure(figsize=(5,5))
plt.hist2d(x[:,0],x[:,1],bins=30)
plt.show()


# In[16]:


refines = integrator.integration_history.loc[(integrator.integration_history["phase"]=="refine")]


# In[17]:


refines["integral"].mean()


# In[18]:


np.sqrt((refines["error"]**2 / len(refines["error"])**2).sum())


# In[ ]:





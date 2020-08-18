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
import pandas as pd
#torch.set_default_dtype(torch.float64)
device = torch.device("cuda:6")


# In[2]:


from src.models.flows.coupling_cells.real_nvp import RealNVP
from src.models.flows.sampling import FactorizedGaussianSampler, UniformSampler
from src.models.flows.analytic_flows.element_wise import InvertibleAnalyticSigmoid
from src.models.flows.sequential import InvertibleSequentialFlow
from src.training.weighted_dataset.dkl_training import BasicStatefulDKLTrainer
from src import setup_std_stream_logger
from src.integration.flat_survey_integrator import FlatSurveySamplingIntegrator


# In[3]:


setup_std_stream_logger(debug=True)


# In[4]:


def f(x):
    return torch.exp( - torch.sum(((x-0.25)/0.1)**2,axis=-1))+torch.exp( - torch.sum(((x-0.75)/0.1)**2,axis=-1))


# In[5]:


posterior=UniformSampler(d=2,low=0.,high=1.,device=device)
prior=FactorizedGaussianSampler(d=2,device=device)


# In[12]:


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
    InvertibleAnalyticSigmoid(d=2),
])

optim = torch.optim.Adam(model.parameters(),lr=1.e-3)


# In[13]:


trainer = BasicStatefulDKLTrainer(flow=model,latent_prior=prior)


# In[14]:


trainer.set_config(n_epochs=10, minibatch_size=20000, optim=optim)


# In[15]:


integrator=FlatSurveySamplingIntegrator(f,trainer,2,device=device)


# In[16]:


result=integrator.integrate(10,10,verbosity=3, trainer_verbosity=3)


# In[17]:


x=trainer.sample_forward(100000).cpu().numpy()
plt.figure(figsize=(5,5))
plt.hist2d(x[:,0],x[:,1],bins=30)
plt.show()


# In[68]:


OneDimGaussian = 0.0354491


# In[125]:


def build_checker_integrator(d,reps=2,lr=1.e-4,f=f,epochs=10):
    layers = []
    for rep in range(reps):
        for i in range(2):
            mask = [j % 2 == i for j in range(d)]
            layers.append(
                RealNVP(d=d,
                  mask=mask,
                  d_hidden=256,
                  n_hidden=16,).to(device),
            )
    layers.append(InvertibleAnalyticSigmoid(d=2))
    
    model = InvertibleSequentialFlow(d,layers)
    optim = torch.optim.Adam(model.parameters(),lr=lr)
    prior=FactorizedGaussianSampler(d=d,device=device)
    
    trainer = BasicStatefulDKLTrainer(flow=model,latent_prior=prior)
    trainer.set_config(n_epochs=epochs, minibatch_size=20000, optim=torch.optim.Adam(model.parameters()))
    integrator=FlatSurveySamplingIntegrator(f,trainer,d,device=device)
    
    return integrator


# In[128]:


results = pd.DataFrame({
    "d":pd.Series([],dtype="int"),
    "integral": pd.Series([],dtype="float"),
    "error": pd.Series([],dtype="float"),
    "nsig":pd.Series([],dtype="float"),
    "speedup": pd.Series([],dtype="float"),
})


# In[129]:


d=2
integrator = build_checker_integrator(d=d,lr=1.e-5)
result = integrator.integrate(5,10)
speedup = (result[2].loc[result[2]["phase"] == "survey"]["error"].mean()/result[2].loc[result[2]["phase"] == "refine"]["error"].mean())**2
results = results.append({"integral": result[0], "error": result[1], "nsig":abs(result[0] - OneDimGaussian**d)/result[1],"speedup": speedup, "d":d},ignore_index=True)
integrator = build_checker_integrator(d=d,lr=1.e-5)
result = integrator.integrate(5,10)
speedup = (result[2].loc[result[2]["phase"] == "survey"]["error"].mean()/result[2].loc[result[2]["phase"] == "refine"]["error"].mean())**2
results = results.append({"integral": result[0], "error": result[1], "nsig":abs(result[0] - OneDimGaussian**d)/result[1],"speedup": speedup, "d":d},ignore_index=True)


# In[130]:


d=3
integrator = build_checker_integrator(d=d,lr=1.e-5)
result = integrator.integrate(5,10)
speedup = (result[2].loc[result[2]["phase"] == "survey"]["error"].mean()/result[2].loc[result[2]["phase"] == "refine"]["error"].mean())**2
results = results.append({"integral": result[0], "error": result[1], "nsig":abs(result[0] - OneDimGaussian**d)/result[1],"speedup": speedup, "d":d},ignore_index=True)
integrator = build_checker_integrator(d=d,lr=1.e-5)
result = integrator.integrate(5,10)
speedup = (result[2].loc[result[2]["phase"] == "survey"]["error"].mean()/result[2].loc[result[2]["phase"] == "refine"]["error"].mean())**2
results = results.append({"integral": result[0], "error": result[1], "nsig":abs(result[0] - OneDimGaussian**d)/result[1],"speedup": speedup, "d":d},ignore_index=True)


# In[131]:


d=4
integrator = build_checker_integrator(d=d,lr=1.e-5)
result = integrator.integrate(5,10)
speedup = (result[2].loc[result[2]["phase"] == "survey"]["error"].mean()/result[2].loc[result[2]["phase"] == "refine"]["error"].mean())**2
results = results.append({"integral": result[0], "error": result[1], "nsig":abs(result[0] - OneDimGaussian**d)/result[1],"speedup": speedup, "d":d},ignore_index=True)
integrator = build_checker_integrator(d=d,lr=1.e-5)
result = integrator.integrate(5,10)
speedup = (result[2].loc[result[2]["phase"] == "survey"]["error"].mean()/result[2].loc[result[2]["phase"] == "refine"]["error"].mean())**2
results = results.append({"integral": result[0], "error": result[1], "nsig":abs(result[0] - OneDimGaussian**d)/result[1],"speedup": speedup, "d":d},ignore_index=True)


# In[132]:


results


# In[ ]:





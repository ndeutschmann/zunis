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
device = torch.device("cuda:7")

from src.training.weighted_dataset.dkl_training import BasicStatefulDKLTrainer
from src.training.weighted_dataset.variance_training import BasicStatefulVarTrainer
from src.models.flows.sequential import InvertibleSequentialFlow
from src.models.flows.coupling_cells.real_nvp import RealNVP
from src.models.flows.coupling_cells.piecewise_coupling.piecewise_linear import PWLinearCoupling
from src.models.flows.sampling import UniformSampler,FactorizedGaussianSampler
from src.models.flows.analytic_flows.element_wise import InvertibleAnalyticSigmoid
from src.integration.flat_survey_integrator import FlatSurveySamplingIntegrator
from src.integration.dkltrainer_integrator import DKLAdaptiveSurveyIntegrator
from src import setup_std_stream_logger

setup_std_stream_logger(debug=True)


# # Debugging the DKL training

# In[2]:


posterior=UniformSampler(d=2,low=0.,high=1.,device=device)
prior=UniformSampler(d=2,low=0.,high=1.,device=device)


# In[3]:


# a function with a small amplitude and no zero

def nonzerocos(x):
    return 1+torch.cos(4*(x[:,0]+x[:,1]))**2

n = 30
plt.imshow(nonzerocos(torch.cartesian_prod(torch.arange(0,1,1/n),torch.arange(0,1,1/n))).reshape(n,n).numpy(),origin="lower")
plt.show()


# In[4]:


# We add a layer for the neural net that maps the unit hypercube to -1,1

class Reshift(torch.nn.Module):
    def forward(self,x):
        return (x-0.5)*2.


# In[42]:


model  = InvertibleSequentialFlow(2,[
        PWLinearCoupling(d=2,
              mask=[True,False],
              d_hidden=256,
              n_hidden=8,
              n_bins=32,
              hidden_activation=torch.nn.LeakyReLU,
              input_activation=Reshift,
              use_batch_norm=False).to(device),
        PWLinearCoupling(d=2,
              mask=[False,True],
              d_hidden=256,
              n_hidden=8,
              n_bins=32,
              hidden_activation=torch.nn.LeakyReLU,
              input_activation=Reshift,
              use_batch_norm=False).to(device), 
])

trainer = BasicStatefulDKLTrainer(flow=model,latent_prior=prior)


optim = torch.optim.Adam(model.parameters(),lr=1.e-3)

trainer.set_config(n_epochs=30, minibatch_size=20000, optim=optim)

integrator=FlatSurveySamplingIntegrator(nonzerocos,trainer,2,device=device, verbosity=3, trainer_verbosity=3)


# In[43]:


result=integrator.integrate(8,1)


# In[44]:


losses = []
for r in result[2].dropna()["training record"]:
    losses+=r["metrics"]["loss"]
plt.plot(losses)
plt.show()


# In[67]:


x,px,fx = integrator.sample_survey()
var,mean = torch.var_mean(fx/px)
print(mean.item(),sqrt(var))
fig, ax = plt.subplots()
fig.set_size_inches((4,4))
n=30
h1=ax.hist2d(x[:,0].cpu().numpy(),x[:,1].cpu().numpy(),bins=n)
plt.show()


# In[78]:


x,px,fx=integrator.sample_refine(n_points=100000)
var,mean = torch.var_mean(fx/px)
print(mean.item(),sqrt(var))
fig, axs = plt.subplots(1,3)
fig.set_size_inches((12,4))
n=30
h2=axs[0].hist2d(x[:,0].cpu().numpy(),x[:,1].cpu().numpy(),bins=n)
axs[1].imshow(nonzerocos(torch.cartesian_prod(torch.arange(0,1,1/n),torch.arange(0,1,1/n))).reshape(n,n).numpy(),origin="lower",)
axs[2].imshow(h2[0],origin="lower")
plt.show()


# In[69]:


func_sample = nonzerocos(torch.cartesian_prod(torch.arange(0,1,1/n),torch.arange(0,1,1/n))).reshape(n,n).numpy()


# In[77]:


fn = func_sample/np.sum(func_sample)
h1n = h1[0]/np.sum(h1[0])
h2n = h2[0]/np.sum(h2[0])
print((fn/h1n).var())
print((fn/h2n).var())


# Quite annoying: the function has visibly been learned, the DKL has significantly decreased but the variance is actually worse!!

# In[73]:


del model, trainer, integrator, x , px ,fx, optim


# ### making checks more systematic

# In[166]:


def test_func(f, epochs=10, n_survey=1):
    model  = InvertibleSequentialFlow(2,[
        PWLinearCoupling(d=2,
              mask=[True,False],
              d_hidden=128,
              n_hidden=8,
              n_bins=30,
              hidden_activation=torch.nn.LeakyReLU,
              input_activation=Reshift,
              use_batch_norm=False).to(device),
        PWLinearCoupling(d=2,
              mask=[False,True],
              d_hidden=128,
              n_hidden=8,
              n_bins=30,
              hidden_activation=torch.nn.LeakyReLU,
              input_activation=Reshift,
              use_batch_norm=False).to(device),
])

    trainer = BasicStatefulDKLTrainer(flow=model,latent_prior=prior)


    optim = torch.optim.Adam(model.parameters(),lr=1.e-3)

    trainer.set_config(n_epochs=epochs, minibatch_size=20000, optim=optim)

    integrator=FlatSurveySamplingIntegrator(f,trainer,2,device=device, verbosity=1, trainer_verbosity=0)
    
    result=integrator.integrate(n_survey,1)
    losses = []
    for r in result[2].dropna()["training record"]:
        losses+=r["metrics"]["loss"]
    plt.plot(losses)
    plt.show()
    
    x,px,fx = integrator.sample_survey()
    varf,meanf = torch.var_mean(fx/px)
    print(f"flat: {meanf} with var {varf}")
    x,px,fx = integrator.sample_refine()
    var,mean = torch.var_mean(fx/px)
    print(f"model: {mean} with var {var}")
    print(f"speed-up: {varf/var}")
    print("\n")


# In[127]:


# Checking: there's a sweet spot: if we train the DKL too much we make things worse
test_func(nonzerocos,2,1)
test_func(nonzerocos,10,1)
test_func(nonzerocos,30,1)


# In[128]:


# Could it be overfitting? Not so far
test_func(nonzerocos,1,2)
test_func(nonzerocos,5,2)
test_func(nonzerocos,8,2)
test_func(nonzerocos,15,2)


# In[129]:


# More aggressive attempt: slightly better
test_func(nonzerocos,1,15)


# In[130]:


# More aggressive attempt: still ends up not good
test_func(nonzerocos,1,20)


# # Is this an issue with the DKL?

# In[5]:


def test_func_var(f, epochs=10, n_survey=1, show_sample=False, lr=1.e-3):
    model  = InvertibleSequentialFlow(2,[
        PWLinearCoupling(d=2,
              mask=[True,False],
              d_hidden=128,
              n_hidden=8,
              n_bins=30,
              hidden_activation=torch.nn.LeakyReLU,
              input_activation=Reshift,
              use_batch_norm=False).to(device),
        PWLinearCoupling(d=2,
              mask=[False,True],
              d_hidden=128,
              n_hidden=8,
              n_bins=30,
              hidden_activation=torch.nn.LeakyReLU,
              input_activation=Reshift,
              use_batch_norm=False).to(device),
])

    trainer = BasicStatefulVarTrainer(flow=model,latent_prior=prior)


    optim = torch.optim.Adam(model.parameters(),lr=lr)

    trainer.set_config(n_epochs=epochs, minibatch_size=20000, optim=optim)

    integrator=FlatSurveySamplingIntegrator(f,trainer,2,device=device, verbosity=1, trainer_verbosity=0)
    
    result=integrator.integrate(n_survey,1)
    losses = []
    for r in result[2].dropna()["training record"]:
        losses+=r["metrics"]["loss"]
    plt.plot(losses)
    plt.show()
    
    x,px,fx = integrator.sample_survey()
    varf,meanf = torch.var_mean(fx/px)
    print(f"flat: {meanf} with var {varf}")
    x,px,fx = integrator.sample_refine()
    var,mean = torch.var_mean(fx/px)
    print(f"model: {mean} with var {var}")
    print(f"speed-up: {varf/var}")
    print("\n")
    
    if show_sample:
        x,px,fx=integrator.sample_refine(n_points=100000)
        fig, axs = plt.subplots(1,3)
        fig.set_size_inches((12,4))
        n=30
        h2=axs[0].hist2d(x[:,0].cpu().numpy(),x[:,1].cpu().numpy(),bins=n)
        axs[1].imshow(f(torch.cartesian_prod(torch.arange(0,1,1/n),torch.arange(0,1,1/n))).reshape(n,n).numpy(),origin="lower",)
        axs[2].imshow(h2[0],origin="lower")
        plt.show()
        
    return varf/var


# In[133]:


test_func_var(nonzerocos,2,1)
test_func_var(nonzerocos,10,1)
test_func_var(nonzerocos,30,1)


# In[137]:


test_func_var(nonzerocos,1,15)
test_func_var(nonzerocos,1,30)
test_func_var(nonzerocos,1,60)


# In[139]:


test_func_var(nonzerocos,1,30)


# OK so it's clearly much better with the variance loss than the DKL loss to optimize the variance. Interesting because they have the same minimum but probably approaching it in a less ideal direction?
# 
# Let's try to validate with other functions
# 
# ## Same function but actually goes to 0

# In[6]:


def poscos(x):
    return torch.cos(4*(x[:,0]+x[:,1]))**2


# In[169]:


test_func_var(poscos,5,1,False)
test_func_var(poscos,10,1,False)
test_func_var(poscos,20,1,False)
test_func_var(poscos,40,1,False)


# There seems to be a sweet spot with a single sample set. Let's go overboard the other way

# In[170]:


test_func_var(poscos,1,5,False)
test_func_var(poscos,1,10,False)
test_func_var(poscos,1,20,False)
test_func_var(poscos,1,40,False)
test_func_var(poscos,1,80,False)
test_func_var(poscos,1,160,False)


# In[171]:


test_func_var(poscos,1,80,True)


# ## Camel function

# In[7]:


def camel(x):
    return torch.exp(- torch.sum(((x-.25)/0.1)**2,dim=-1)  )+torch.exp(- torch.sum(((x-.75)/0.1)**2,dim=-1)  )


# In[179]:


test_func_var(camel,40,1,True)


# In[182]:


test_func_var(camel,1,5,True)


# In[183]:


test_func_var(camel,1,40,True)


# So for short times we're learning "VEGAS-style" but then we create "wrong" peaks

# In[185]:


test_func_var(camel,1,200,True)


# In[186]:


test_func_var(camel,1,200,True,lr=1.e-4)


# ## Is it a problem with zero?

# In[6]:


def nonzerocamel(reg):
    def camel(x):
        return reg+torch.exp(- torch.sum(((x-.25)/0.1)**2,dim=-1)  )+torch.exp(- torch.sum(((x-.75)/0.1)**2,dim=-1)  )
    return camel


# In[199]:


test_func_var(nonzerocamel(1.),1,80,True)
test_func_var(nonzerocamel(.5),1,80,True)
test_func_var(nonzerocamel(.1),1,80,True)
test_func_var(nonzerocamel(.01),1,80,True)


# Yep, seems like it. Niklas is getting the opposite behavior, is it due to the training mode?

# In[ ]:





# # Testing with the inverted model

# In[7]:


test_func_var(nonzerocamel(1.),1,80,True)
test_func_var(nonzerocamel(.5),1,80,True)
test_func_var(nonzerocamel(.1),1,80,True)
test_func_var(nonzerocamel(.01),1,80,True)


# In[ ]:





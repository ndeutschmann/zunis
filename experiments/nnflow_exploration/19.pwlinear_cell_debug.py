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
#dtype = torch.float
device = torch.device("cuda:7")
#torch.set_default_dtype(torch.float64)

#torch.autograd.set_detect_anomaly(True)

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


# In[2]:


setup_std_stream_logger(debug=True)


# In[3]:


posterior=UniformSampler(d=2,low=0.,high=1.,device=device)
prior=UniformSampler(d=2,low=0.,high=1.,device=device)
prior2=FactorizedGaussianSampler(d=2,device=device)


# In[4]:


def one(x):
    return torch.ones(x.shape[0],device=x.device)

def nonzerocos(x):
    return torch.cos(4*(x[:,0]+x[:,1]))**2

def twogauss(x):
    return 1+10*torch.exp(- torch.sum(((x-.25)/0.1)**2,dim=-1)  )+10*torch.exp(- torch.sum(((x-.75)/0.1)**2,dim=-1)  )

def circle(x):
    
    return torch.exp( - ((torch.sqrt(torch.sum((x-.5)**2,dim=-1)) - 0.3)/0.3)**2 )

f = nonzerocos 


# In[5]:


n = 30
plt.imshow(f(torch.cartesian_prod(torch.arange(0,1,1/n),torch.arange(0,1,1/n))).reshape(n,n).numpy(),origin="lower")
plt.show()


# In[6]:


class Reshift(torch.nn.Module):
    def forward(self,x):
        return (x-0.5)*2.
    
class Sine(torch.nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)
    
try:
    del model
except:
    pass

model  = InvertibleSequentialFlow(2,[
        PWLinearCoupling(d=2,
              mask=[True,False],
              d_hidden=32,
              n_hidden=4,
              n_bins=10,
              hidden_activation=torch.nn.LeakyReLU,
              input_activation=Reshift,
              use_batch_norm=False).to(device),
        PWLinearCoupling(d=2,
              mask=[False,True],
              d_hidden=32,
              n_hidden=4,
              n_bins=10,
              hidden_activation=torch.nn.LeakyReLU,
              input_activation=Reshift,
              use_batch_norm=False).to(device),
])

# model2 = InvertibleSequentialFlow(2,
#                                 [
#                                     RealNVP(d=2,
#                                     mask=[False,True],
#                                     d_hidden=512,
#                                     n_hidden=8
#                                     ).to(device),
                                    
#                                    RealNVP(d=2,
#                                     mask=[True,False],
#                                     d_hidden=512,
#                                     n_hidden=8
#                                     ).to(device),
#                                     RealNVP(d=2,
#                                     mask=[False,True],
#                                     d_hidden=512,
#                                     n_hidden=8
#                                     ).to(device),
                                    
#                                    RealNVP(d=2,
#                                     mask=[True,False],
#                                     d_hidden=512,
#                                     n_hidden=8
#                                     ).to(device),                                    
                                    
#                                     InvertibleAnalyticSigmoid(d=2).to(device)
#                                 ])


optim = torch.optim.Adam(model.parameters(),lr=1.e-3)


# In[7]:


trainer = BasicStatefulDKLTrainer(flow=model,latent_prior=prior)

trainer.set_config(n_epochs=30, minibatch_size=20000, optim=optim)

integrator=DKLAdaptiveSurveyIntegrator(f,trainer,2,device=device, verbosity=3, trainer_verbosity=3)

result=integrator.integrate(2,1)


# In[8]:


model.inverse


# In[9]:


xj = prior(100)
yj = model(xj)
xj=xj.detach().cpu().numpy()
yj=yj.detach().cpu().numpy()


# In[10]:


plt.scatter(xj[:,1],yj[:,1])


# In[11]:


plt.scatter(xj[:,0],yj[:,0])


# In[12]:


plt.scatter(xj[:,1],yj[:,2])


# In[13]:


fig, axs = plt.subplots(2,1)
fig.set_size_inches(5,10)
for i in range(len(axs)):
    with torch.no_grad():
        xj = prior(1000000)
        yj = model(xj)
        xj=xj.detach().cpu().numpy()
        yj=yj.detach().cpu().numpy()
        h=axs[i].hist2d(yj[:,0],yj[:,1],bins=30)
 
plt.show()


# In[44]:


distrib=h[0]/np.sum(h[0])

n=30
ftab=f(torch.cartesian_prod(torch.arange(0,1,1/n),torch.arange(0,1,1/n))).reshape(n,n).numpy()
ftabnorm=ftab/np.sum(ftab)


# In[ ]:





# In[45]:


print((distrib/ftabnorm).min())
print((distrib/ftabnorm).max())
print((distrib/ftabnorm).mean())
print((distrib/ftabnorm).std())


# In[19]:


plt.imshow(distrib,interpolation=None,origin="lower",vmin=0.,vmax=3.e-3)
plt.show()
plt.imshow(ftabnorm,interpolation=None,origin="lower",vmin=0.,vmax=3.e-3)
plt.show()
fig,ax = plt.subplots()
plt.imshow(np.abs(ftabnorm-distrib)/(ftabnorm+distrib),interpolation=None,origin="lower",vmin=0.,vmax=1)
plt.show()
plt.imshow(ftabnorm/distrib,interpolation=None,origin="lower",vmin=0,vmax=10)
plt.show()


# In[20]:


fig, ax = plt.subplots()
fig.set_size_inches(5,5)
n=30
plt.imshow(f(torch.cartesian_prod(torch.arange(0,1,1/n),torch.arange(0,1,1/n))).reshape(n,n).numpy(),interpolation=None,origin="lower")
plt.show()


# In[14]:


del xj,yj


# In[15]:


xj = torch.arange(0.01,0.99,0.01)
xj=torch.stack((torch.ones_like(xj)/2.,xj,torch.zeros_like(xj)),-1).to(device)
yj = model(xj).detach().cpu()
xj = xj.detach().cpu()


# In[16]:


plt.scatter(xj[:,1],yj[:,0])


# In[17]:


t1=model.flows[0].T


# In[18]:


qt=(t1(torch.zeros(1000,1).uniform_().to(device))).detach().to("cpu")
qs = torch.nn.Softmax(dim=2)(qt).numpy()


# In[19]:


qt.shape


# In[20]:


plt.hist(qs[:,:,1])
plt.show()
plt.hist(qs[:,:,0])
plt.show()


# In[21]:


plt.hist(qs[:,:,1])


# In[22]:


yj


# In[ ]:





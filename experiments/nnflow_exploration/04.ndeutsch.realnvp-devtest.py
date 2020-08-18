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
dtype = torch.float
device = torch.device("cuda:0")


# # Validating fakerealnvp

# In[2]:


from src.models.flows.coupling_cells.real_nvp import FakeRealNVP
from src.models.flows.sampling import FactorizedFlowPrior


# In[3]:


prior_mu =  torch.tensor(0.).to(device)
prior_sig =  torch.tensor(1.).to(device)
prior = torch.distributions.normal.Normal(prior_mu,prior_sig)

sampler = FactorizedFlowPrior(d=2,prior_1d=prior)


# In[4]:


oxy=FakeRealNVP(d=2,mask=[True,False],s=2.,t=1)


# In[5]:


x = sampler(5)
y = oxy(x)
print(x)
print(y)


# In[7]:


y[:,0] - x[:,0]


# In[9]:


y[:,1] - (x[:,1]*torch.exp(torch.tensor(2.))+1)


# In[10]:


y[:,2] - x[:,2]


# This was validated using the code that now constitutes b7392752e3397e1604e5d3878db84869836557c0

# # Checking the real realNVP

# ## 2 variables

# In[18]:


import torch
from math import pi,sqrt,log,e,exp
from time import time
from matplotlib import pyplot
import numpy as np
from matplotlib import pyplot as plt
from tqdm.autonotebook import tqdm
dtype = torch.float
device = torch.device("cuda:0")

from src.models.flows.coupling_cells.real_nvp import RealNVP
from src.models.flows.sampling import FactorizedFlowPrior


# In[19]:


prior_mu =  torch.tensor(0.).to(device)
prior_sig =  torch.tensor(1.).to(device)
prior = torch.distributions.normal.Normal(prior_mu,prior_sig)

sampler = FactorizedFlowPrior(d=2,prior_1d=prior)


# In[20]:


nvp = RealNVP(d=2,
              mask=[True,False],
              d_hidden=10,
              n_hidden=5,).to(device)


# In[21]:


x = sampler(5)
y = nvp(x)


# In[25]:


x1 = x[:,0]
x2 = x[:,1]


# In[40]:


st=nvp.T(x1.unsqueeze(-1))
s = st[...,0].squeeze()
t = st[...,1].squeeze()


# In[48]:


y2,j= nvp.transform(x2.unsqueeze(-1),st)


# In[55]:


y2=y2.squeeze()


# In[62]:


y[:,0]-x1


# In[63]:


y2-y[:,1]


# In[66]:


j - y[:,2] + x[:,2]


# In[70]:


x2*torch.exp(s)+t - y2


# ## 5 variables

# In[73]:


import torch
from math import pi,sqrt,log,e,exp
from time import time
from matplotlib import pyplot
import numpy as np
from matplotlib import pyplot as plt
from tqdm.autonotebook import tqdm
dtype = torch.float
device = torch.device("cuda:0")

from src.models.flows.coupling_cells.real_nvp import RealNVP
from src.models.flows.sampling import FactorizedFlowPrior

prior_mu =  torch.tensor(0.).to(device)
prior_sig =  torch.tensor(1.).to(device)
prior = torch.distributions.normal.Normal(prior_mu,prior_sig)

sampler = FactorizedFlowPrior(d=5,prior_1d=prior)

nvp = RealNVP(d=5,
              mask=[True,True,True,False,False],
              d_hidden=10,
              n_hidden=5,).to(device)

x = sampler(4)
y = nvp(x)


# In[75]:


print(x.shape,y.shape)


# In[78]:


x1=x[:,[0,1,2]] 
x2=x[:,[3,4]]
jx=x[:,-1]


# In[79]:


x1 - y[:,[0,1,2]]


# In[82]:


st=nvp.T(x1)
s=st[...,0]
t=st[...,1]


# In[84]:


y2,j= nvp.transform(x2,st)


# In[86]:


y2-y[:,[3,4]]


# In[88]:


j - y[:,-1] + x[:,-1]


# In[89]:


x2*torch.exp(s)+t - y2


# In[92]:


j - torch.sum(s,dim=-1)


# In[93]:


j


# In[ ]:





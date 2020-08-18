#!/usr/bin/env python
# coding: utf-8

# In[40]:


import torch
from math import sqrt
from src import setup_std_stream_logger
from src.models.flows.coupling_cells.piecewise_coupling import piecewise_linear as pwl


# In[2]:


pwl.piecewise_linear_transform


# In[28]:


nbatch=100
d=3
b = 4


# In[29]:


x = torch.zeros(nbatch,d).uniform_()
q_tilde = torch.zeros(nbatch,d,b).normal_()


# In[30]:


y,jy = pwl.piecewise_linear_transform(x,q_tilde)


# In[31]:


xx, jx = pwl.piecewise_linear_inverse_transform(y,q_tilde)


# In[32]:


print(torch.max(torch.abs(xx - x)/torch.abs(x)))
print(jx*jy)


# In[33]:


print(torch.min(y),torch.max(y))


# In[46]:


nbatch=1000000
d=3
b = 4

x = torch.zeros(nbatch,d).uniform_()
q_tilde = torch.zeros(nbatch,d,b).normal_()
y,jy = pwl.piecewise_linear_transform(x,q_tilde)

res = torch.mean(jy)
tgt = 1.
unc = torch.std(jy)/sqrt(nbatch)

print(f"Result: {res} +/- {unc}, {torch.abs(res-tgt)/unc} sigmas")


# In[ ]:





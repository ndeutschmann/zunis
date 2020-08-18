#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from math import pi,sqrt,log,e
from time import time
from matplotlib import pyplot
import numpy as np
from matplotlib import pyplot as plt
from tqdm.autonotebook import tqdm
dtype = torch.float
device = torch.device("cuda:0")


# In[2]:


from src.models.flows.sampling import FactorizedFlowPrior
from src.models.flows.backprop_jacobian_flows.nnflows import NNFlow
from src.models.flows.backprop_jacobian_flows.simple_backprop_flows import SigmoidFlow


# In[ ]:





# In[3]:


prior_mu =  torch.tensor(0.).to(device)
prior_sig =  torch.tensor(1.).to(device)
prior = torch.distributions.normal.Normal(prior_mu,prior_sig)

sampler = FactorizedFlowPrior(d=2,prior_1d=prior)

nnflow = NNFlow(d=2, nh=5, dh=10,batch_norm=True).to(device)
sflow = SigmoidFlow(d=2)


# In[4]:


prior_mu =  torch.tensor(0.).to(device)
prior_sig =  torch.tensor(1.).to(device)
prior = torch.distributions.normal.Normal(prior_mu,prior_sig)

sampler = FactorizedFlowPrior(d=2,prior_1d=prior)

sflow = SigmoidFlow(d=2)

mins = []
fig = plt.figure(figsize=(10,10))
for i in range(16):
    nnflow = NNFlow(d=2, nh=5, dh=10,batch_norm=True).to(device)

    z = sampler(10000)
    x = sflow(nnflow(z)*0.+z)
    fig.add_subplot(4,4,i+1)
    hist,_,_,_=plt.hist2d(x[:,0].detach().cpu().numpy(),x[:,1].detach().cpu().numpy())
    
    mins.append(np.min(hist))
plt.show()


# In[5]:


print(mins)


# In[7]:


prior_mu =  torch.tensor(0.).to(device)
prior_sig =  torch.tensor(1.).to(device)
prior = torch.distributions.normal.Normal(prior_mu,prior_sig)

sampler = FactorizedFlowPrior(d=2,prior_1d=prior)

nnflow = NNFlow(d=2, nh=5, dh=10,batch_norm=False).to(device)
nnflow.weight_init_identity_()
sflow = SigmoidFlow(d=2)

batch_size=10000

z = sampler(batch_size)
x = sflow(nnflow(z))
j = torch.exp(x[:,-1])

var,one = torch.var_mean(j)
print("{}+/-{}".format(one.cpu().item(),torch.sqrt(var/batch_size).cpu().item()))


# In[ ]:


plt.figure(figsize=(5,5))
plt.hist2d(x[:,0].detach().cpu().numpy(),x[:,1].detach().cpu().numpy())
plt.show()


# In[98]:


plt.hist((j).detach().cpu())


# In[99]:


torch.max(j),torch.argmax(j)


# In[100]:


x[825898]


# In[101]:


np.exp(1.0238e+01)


# In[102]:


from numpy import exp,arange
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show

 
x = arange(-10,10.0,0.1).astype(np.float32)
y = arange(-10.0,10.0,0.1).astype(np.float32)
X,Y = meshgrid(x, y) # grid of point
X = torch.tensor(X)
Y=torch.tensor(Y)


# In[103]:


XY=torch.cat((X.unsqueeze(-1),Y.unsqueeze(-1)),-1)


# In[104]:


J=(-torch.sum(sampler.prior.log_prob(XY),axis=-1))
print(J.shape)
print(XY.shape)


# In[119]:


XYJ = torch.stack((X,Y,J),-1).to(device)
out=sflow((XYJ.reshape(200*200,3)+0.*nnflow(XYJ.reshape(200*200,3)))).detach().reshape(200,200,3).cpu()
Z=(out[:,:,-1]).numpy()


# In[120]:


extx = torch.min(out[:,:,0]).item(),torch.max(out[:,:,0]).item()
exty = torch.min(out[:,:,1]).item(),torch.max(out[:,:,1]).item()
ext = (*extx,*exty)


# In[121]:


im = imshow(Z,cmap=cm.RdBu,extent=ext) # drawing the function
# adding the Contour lines with labels
#cset = contour(Z,arange(-1,1.5,0.2),linewidths=2,cmap=cm.Set2)
#clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
colorbar(im) # adding the colobar on the right
# latex fashion title
#title('$z=(1-x^2+y^3) e^{-(x^2+y^2)/2}$')
show()


# In[111]:


(nnflow(torch.tensor([[0.,1.,1.],[1.,0.,1.]]).to(device)))


# In[ ]:





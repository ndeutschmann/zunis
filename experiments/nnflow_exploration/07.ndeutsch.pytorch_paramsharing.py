#!/usr/bin/env python
# coding: utf-8

# In preparation for implementing inverses, let's check what happens when we try to share weights between models

# In[1]:


import torch


# In[35]:


class lin(torch.nn.Module):
    def __init__(self):
        super(lin,self).__init__()
        self.layer = torch.nn.Linear(5,5)
    def create_another(self):
        other = lin()
        lin.layer = self.layer
        return other
    def forward(self,x):
        return self.layer(x)


# In[66]:


l1=lin()


# In[67]:


l2=l1.create_another()


# In[68]:


x = torch.zeros(2,5).normal_()
y = torch.zeros(2,5).normal_()


# In[69]:


with torch.no_grad():
    l2.layer.weight += 1


# In[65]:


l1.layer.weight


# In[48]:


opt = torch.optim.SGD(l2.parameters(),lr=0.1)


# In[60]:


print(l2.layer.weight)
print(l1.layer.weight)
opt.zero_grad()
L = torch.mean( (y-l2(x))**2 )
L.backward()
opt.step()
print(l2.layer.weight)
print(l1.layer.weight)


# In[64]:


list(l2.parameters())


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[9]:


import torch
import torch.nn as nn
torch.manual_seed(42)

# Fake PIV displacement field: 100 points, each with (u, v)
uv_true = torch.randn(100, 2)

# Predicted displacements (slightly off)
uv_pred = uv_true + 0.2 * torch.randn(100, 2)

def endpoint_error_loss(pred, true):
    return torch.sqrt(((pred - true) ** 2).sum(dim=1)).mean()

loss = endpoint_error_loss(uv_pred, uv_true)
print(loss)

class EndpointErrorLoss(nn.Module):
    def forward(self, pred, true):
        return torch.sqrt(((pred - true) ** 2).sum(dim=1)).mean()

# Test it
criterion = EndpointErrorLoss()
loss = criterion(uv_pred, uv_true)
print(loss)


# In[ ]:





# In[ ]:





# In[ ]:





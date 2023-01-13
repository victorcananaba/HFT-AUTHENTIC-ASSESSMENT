#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import linregress


# In[2]:


BTC = yf.download('BTC-USD', start='2022-12-23', end='2022-12-24',  interval = '1m')


# In[3]:


BTC


# In[4]:


BTC.describe()


# In[5]:


ETH = yf.download('ETH-USD', start='2022-12-23', end='2022-12-24',  interval = '1m')


# In[6]:


ETH


# In[7]:


ETH.describe()


# In[8]:


S1_0 = BTC['Open'][0]
S2_0 = ETH['Open'][0]


# In[9]:


S1_0


# In[10]:


S2_0


# In[11]:


lr1 = np.log(BTC['Adj Close'] / BTC['Adj Close'].shift(1))
lr2 = np.log(ETH['Adj Close'] / ETH['Adj Close'].shift(1))
mu_1 = lr1.mean() * 1429
mu_2 = lr2.mean() * 1429
sigma_1 = lr1.std() * np.sqrt(1429)
sigma_2 = lr2.std() * np.sqrt(1429)
rho = lr1.corr(lr2)
z_0 = 0.000
gamma = 0.4 
delta = 1
T = 1
M = len(BTC)


# In[12]:


print (rho)


# In[13]:


result = linregress(np.log(BTC['Adj Close']), np.log(ETH['Adj Close']))
beta = result.slope


# In[14]:


beta


# In[15]:


dt = T/M
S1 = np.zeros((M + 1, 1))
S2 = np.zeros((M + 1, 1))
z =  np.zeros((M + 1, 1))
a =  np.zeros((M + 1, 1))
b =  np.zeros((M + 1, 1))
c =  np.zeros((M + 1, 1))
Pi_1 = np.zeros((M + 1, 1))
Pi_2 = np.zeros((M + 1, 1))


# In[16]:


tt = np.linspace(0, 1, M + 1)
z[0] = z_0
S1[0] = S1_0
S2[0] = S2_0
sigma_beta = np.sqrt(sigma_1 ** 2 + beta ** 2 * sigma_2 ** 2 + 2 * beta * sigma_1 * sigma_2 * rho)
eta = (-1/delta) * (mu_1 - sigma_1 ** 2/2 + beta*(mu_2 - sigma_2 ** 2/2))
rn = np.random.standard_normal(z.shape)
rn1 = np.random.standard_normal(S1.shape) 
rn2 = np.random.standard_normal(S2.shape) 
for t in range(1, M + 1):
    z[t] = z[t-1]* (1 - delta * (eta - z[t-1])) * dt + sigma_beta * np.sqrt(dt)*((sigma_1 + beta * sigma_2 * rho)/ sigma_beta * rn1[t] + beta * (sigma_2 * np.sqrt(1-rho ** 2)/sigma_beta) * rn1[t]);
    S1[t] = BTC['Adj Close'][t-1]
    S2[t] = ETH['Adj Close'][t-1]


# In[17]:


plt.figure(figsize=(10, 4))
plt.plot(tt, S1, 'b', lw=1.5, label='Ethereum')
plt.plot(tt, S2, 'r', lw=1.5, label='Bitcoin')
plt.legend(loc=0)
plt.xlabel('Time')
plt.ylabel('$')
plt.title('Cryptocurrency Price')


# In[18]:


plt.figure(figsize=(10, 4))
plt.plot(tt, S2, 'r', lw=1.5, label='Stock 2')
plt.legend(loc=0)
plt.xlabel('Time')
plt.ylabel('$')
plt.title('Cryptocurrency Price')


# In[19]:


plt.figure(figsize=(10, 6))
plt.plot(tt, z, 'g', lw=1.5)
plt.xlabel('Time')
plt.ylabel('Rate')
plt.title('z')


# In[20]:


for t in range(1, M + 1):
    a[t] = 0.5 * (T - t * dt)/((1-rho ** 2) * sigma_1 ** 2)
    b[t] = - 0.25 * (T - t * dt) ** 2 * (sigma_1 ** 2 + beta * sigma_2 ** 2)/((1 - rho ** 2) * sigma_1 ** 2) - rho * mu_2 * (T - t * dt)/((1 - rho **2) * sigma_1 * sigma_2)
    c[t] = 0.5 * mu_2 ** 2 * (T - t * dt)/((1 - rho ** 2) * sigma_2 ** 2) + 0.25 * (T - t * dt) ** 2 * (sigma_1 ** 2 + beta * sigma_2 ** 2 + 2 * sigma_1 * sigma_2 * beta *rho) * delta ** 2/((1 - rho ** 2) * sigma_1 ** 2) + 0.25 * (T - t * dt) ** 2 * mu_2 * delta * rho * (sigma_1 ** 2 + beta * sigma_2 ** 2) + 1/24 * (T - t * dt) ** 3 * (sigma_1 ** 2 + beta * sigma_2 ** 2) ** 2 * delta ** 2/((1 - rho ** 2) * sigma_1 ** 2)
    Pi_1[t] = (1 / S1[t]) * ((mu_1 + delta * z[t])/(gamma * (1 - rho ** 2) * sigma_1 ** 2) + delta/gamma * (-2 * a[t] * (mu_1 + delta * z[t])-b[t]) - rho * mu_2/(gamma * (1 - rho ** 2) * sigma_1 * sigma_2))
    Pi_2[t] = (1 / S2[t]) * (mu_2 / (gamma * (1 - rho ** 2) * sigma_2 ** 2) + delta * beta / gamma * (- 2 * a[t] * (mu_1 + delta * z[t]) - b[t]) - rho * (mu_1 + delta * z[t]) / (gamma * (1 - rho ** 2) * sigma_1 * sigma_2))


# In[21]:


plt.figure(figsize=(8, 4))
plt.plot(tt, Pi_2, 'r', lw=1.5, label='Ethereum')
plt.plot(tt, Pi_1*8, 'b', lw=1.5, label='Bitcoin')
plt.legend(loc=0)
plt.xlabel('Time')
plt.ylabel('%')
plt.title('Weights')


# In[22]:


plt.figure(figsize=(10, 6))
plt.plot(tt, Pi_1 * S1, 'b', lw=1.5, label='Wealth_Bitcoin')
plt.plot(tt, Pi_2 * S2, 'r', lw=1.5, label='Wealth_Ethereum')
plt.legend(loc=0)
plt.xlabel('Time')
plt.ylabel('$')
plt.title('Cash')


# In[23]:


mu_1 = lr1.mean() * 1429
mu_2 = lr2.mean() * 1429
sigma_1 = lr1.std() * np.sqrt(1429)
sigma_2 = lr2.std() * np.sqrt(1429)
rho = lr1.corr(lr2)
z_0 = 0.000
gamma = 0.4 
delta = 1
T = 1
M = 1440


# In[24]:


print (mu_2)


# In[ ]:





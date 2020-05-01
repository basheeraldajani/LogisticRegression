#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# In[2]:


df = pd.read_csv("C:/Users/bashe/Desktop/Npeaksplits.csv") #read dataset


# In[3]:


df.head() #visualize dataset


# In[4]:


peakSplits = np.zeros(5, dtype=int) #initialize x-values
CF810 = np.zeros(5, dtype=float) #initialize y1-values
CF830 = np.zeros(5, dtype=float) #initialize y2-values
CF8M10 = np.zeros(5, dtype=float) #initialize y3-values
CF8M30 =np.zeros(5, dtype=float) #initialize y4-values


# In[5]:


#convert x-values to array
for i in range(0,5):
    peakSplits[i] = df["T/N Peak Splits"][i]


# In[6]:


#convert y1-values to array
for i in range(0,5):
    CF810[i] = df["CF8 10,000 hours [per Spot]"][i]


# In[7]:


#convert y2-values to array
for i in range(0,5):
    CF830[i] = df["CF8 30,000 hours [per Spot]"][i]


# In[8]:


#convert y3-values to array
for i in range(0,5):
    CF8M10[i] = df["CF8M 10,000 hours [per Spot]"][i]


# In[9]:


#convert y4-values to array
for i in range(0,5):
    CF8M30[i] = df["CF8M 30,000 hours [per Spot]"][i]


# In[10]:


#inspect arrays
print(peakSplits)
print(CF810)
print(CF830)
print(CF8M10)
print(CF8M30)


# In[11]:


#plot all data points
plt.plot(peakSplits, CF810, marker = 'o', markerfacecolor = 'blue', markersize = 4, color = "skyblue", linewidth = 2, label = 'CF8 10hrs')
plt.plot(peakSplits, CF830, marker = 'o', markerfacecolor = 'red', markersize = 4, color = "pink", linewidth = 2, label = 'CF8 30hrs')
plt.plot(peakSplits, CF8M10, marker = 'o', markerfacecolor = 'green', markersize = 4, color = "olive", linewidth = 2, label = 'CF8M 10hrs')
plt.plot(peakSplits, CF8M30, marker = 'o', markerfacecolor = 'black', markersize = 4, color = "gray", linewidth = 2, label = 'CF8M 30hrs')
plt.title("Logistic Regression")
plt.xlabel("Peak Splits")
plt.ylabel("Probability")
plt.legend(loc="best")
plt.show()


# In[12]:


#reshape x-values to 2D to feed into logistic regression model
peakSplits = peakSplits.reshape(-1, 1)
print(peakSplits)


# In[13]:


#create, define, and train logistic regression model for x, y1
model = LogisticRegression(solver='liblinear', random_state=0) #create model
model = model.fit(peakSplits, CF810) #train model
CF810_pred = model.predict(peakSplits) #predict values of y1
score = model.score(peakSplits, CF810) #find model accuracy
con_mat = confusion_matrix(CF810, model.predict(peakSplits)) #confusion matrix

print(score)
print(con_mat)


# In[ ]:





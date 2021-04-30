# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 16:04:39 2018

@author: Payam Aminpour
         Michigan State University
         aminpour@msu.edu
"""

# In[1]:

import __init__ as init
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt

import random
import xlrd
import pandas as pd
import numpy as np
import math
import networkx as nx



# In[2]:

file_location = init.file_location

workbook = xlrd.open_workbook(file_location)
sheet = workbook.sheet_by_index(0)

n_concepts = sheet.nrows-1

Adj_matrix = np.zeros((n_concepts,n_concepts))
activation_vec = np.ones(n_concepts)
node_name = {}



# In[3]:

Noise_Threshold = 0

for i in range (1,n_concepts+1):
    for j in range (1,n_concepts+1):
        if abs(sheet.cell_value(i,j))<=Noise_Threshold:
            Adj_matrix[i-1,j-1]=0
        else:
            Adj_matrix[i-1,j-1]=sheet.cell_value(i,j)


# In[4]:

Concepts_matrix = []
for i in range (1,n_concepts+1):
    Concepts_matrix.append(sheet.cell_value(0,i))


# In[5]:

G = nx.DiGraph(Adj_matrix)


# In[6]:

for nod in G.nodes():
    node_name[nod] = sheet.cell_value(nod+1,0)


# In[7]:

def TransformFunc (x, n, f_type,landa= init.Lambda):
    
    if f_type == "sig":
        x_new = np.zeros(n)
        for i in range (n):
            x_new[i]= 1/(1+math.exp(-landa*x[i]))
            
        return x_new    
    
    if f_type == "tanh":
        x_new = np.zeros(n)
        for i in range (n):
            x_new[i]= math.tanh(landa*x[i])
        
        return x_new
    
    if f_type == "bivalent":
        x_new = np.zeros(n)
        for i in range (n):
            if x[i]> 0:
                x_new[i]= 1
            else:
                x_new[i]= 0
        
        return x_new
    
    if f_type == "trivalent":
        x_new = np.zeros(n)
        for i in range (n):
            if x[i]> 0:
                x_new[i] = 1
            elif x[i]==0:
                x_new[i] = 0
            else:
                x_new[i] = -1
        
        return x_new


# In[8]:
    
def infer_steady (init_vec = activation_vec , AdjmT = Adj_matrix.T 
                  , n =n_concepts , f_type="sig", infer_rule ="mk"):
    act_vec_old= init_vec
    
    resid = 1
    while resid > 0.00001:
        x = np.zeros(n)
        
        if infer_rule == "k":
            x = np.matmul(AdjmT, act_vec_old)
        if infer_rule == "mk":
            x = act_vec_old + np.matmul(AdjmT, act_vec_old)
        if infer_rule == "r":
            x = (2*act_vec_old-np.ones(n)) + np.matmul(AdjmT, (2*act_vec_old-np.ones(n)))
            
        act_vec_new = TransformFunc (x ,n, f_type)
        resid = max(abs(act_vec_new - act_vec_old))
        if resid < 0.00001:
            break
        
        act_vec_old = act_vec_new
    return act_vec_new


# In[9]:

def infer_scenario (Scenario_concepts,zeros, init_vec = activation_vec , AdjmT = Adj_matrix.T
                    , n =n_concepts , f_type="sig", infer_rule ="mk" ):
    act_vec_old= init_vec
    
    my_random ={}
    for rC in  Scenario_concepts:
        PN = random.choice([-1,1])
        my_random[rC] = random.random() * PN
        
    resid = 1
    while resid > 0.00001:
        act_vec_new = np.zeros(n)
        x = np.zeros(n)
        
        if infer_rule == "k":
            x = np.matmul(AdjmT, act_vec_old)
        if infer_rule == "mk":
            x = act_vec_old + np.matmul(AdjmT, act_vec_old)
        if infer_rule == "r":
            x = (2*act_vec_old-np.ones(n)) + np.matmul(AdjmT, (2*act_vec_old-np.ones(n)))
            
        act_vec_new = TransformFunc (x ,n, f_type)
        
        for z in zeros:
            act_vec_new[z] = 0
        for c in  Scenario_concepts:
            
            act_vec_new[c] = my_random[c]
        
            
        resid = max(abs(act_vec_new - act_vec_old))
        #if resid < 0.0001:
            #break
        
        act_vec_old = act_vec_new
    return act_vec_new


# In[10]:

def combinations(iterable, r):
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)


# In[11]:

Principles= init.Principles

prin_concepts_index = []
for nod in node_name.keys():
    if node_name[nod] in Principles:
        prin_concepts_index.append(nod)

listPossibleNodes=[]
for nod in G.nodes():
    if G.in_degree(nbunch=None, weight=None)[nod] <= init.Thresh and Concepts_matrix[nod] not in Principles:
        listPossibleNodes.append(nod)
        
# In[13]:

function_type = init.function_type
infer_rule = init.infer_rule

SteadyState = infer_steady (f_type = function_type , infer_rule =infer_rule)

change_in_principles ={}
for pr in prin_concepts_index:
        change_in_principles[pr]=[]

iteration = 0

for iter in range (init.n_iteration): # You can increas the number of times you repeat the random process of input vector generation
    rand = random.randint(1,len(listPossibleNodes))
    com = random.sample(listPossibleNodes , rand)
  
    iteration += 1

    Scenario_concepts = com 
    ScenarioState = infer_scenario (Scenario_concepts,listPossibleNodes, f_type=function_type, infer_rule = infer_rule)
    changes = ScenarioState - SteadyState
    
    for pr in prin_concepts_index:
        change_in_principles[pr].append(changes[pr])
    
iteration


# In[ ]:

df = pd.DataFrame()
df["IDS"] = list(range(iteration))
for pr in prin_concepts_index:
    df[node_name[pr]]=change_in_principles[pr]

# In[ ]:

from math import pi


# number of variable
categories=list(df)[1:]
N = len(categories)
 
# We are going to plot the first line of the data frame.
# But we need to repeat the first value to close the circular graph:


plt.figure(figsize=(10,10))
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
 
# Initialise the spider plot
ax = plt.subplot(111, polar=True)
 
# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories, color='black', size=9)
 
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([-1,-0.5,0,0.5,1], 
            ["-1","-0.5","0","0.5","1"], color="red", size=10)
#plt.ylim(-1,1)

for i in range(int(iteration/10)):
    values=df.loc[i*10].drop('IDS').values.flatten().tolist() 
    values += values[:1]
  
    # Plot data
    ax.plot(angles, values, linewidth=0.1, color="black" , alpha = 0.1,  linestyle='-')

# Fill area
#ax.fill(angles, values, 'b', alpha=0.1)
plt.savefig('Uncertainty_Analysis_Results.pdf')
plt.show()

# In[ ]:




# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 16:04:39 2018

@author: Payam Aminpour
         Michigan State University
         aminpour@msu.edu
"""

# In[1]:


import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt

import random
import xlrd
import pandas as pd
import numpy as np
import math
import networkx as nx



# In[2]:

''' the file location is the path of your project file in your computer
# this file should be an excel file with .xlsx extention
# Please see the Adjacency_Matrix_Example file to check how your matrix should look like'''

file_location = "C:/Paym Computer/...../...../Adjacency_Matrix_Example.xlsx"

workbook = xlrd.open_workbook(file_location)
sheet = workbook.sheet_by_index(0)

n_concepts = sheet.nrows-1

Adj_matrix = np.zeros((n_concepts,n_concepts))
activation_vec = np.ones(n_concepts)
node_name = {}



# In[3]:
''' sometimes you need to remove the links with significantly low weights to avoid messiness
Noise_Threshold defines a boundary below which all links will be removed from the FCM
E.g. Noise_Threshold = 0.15 means that all the edges with weight <= 0.15 will be removed from FCM ''' 

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
''' TransformFunc is the squashing function: 
    
   Four transformation functions:
   (Bivalent: 'biv', Trivalent: 'triv', Sigmoid: 'sig' or Hyperbolic tangent: 'tanh')

   where Lambda is a real positive number (λ>0) which determines the steepness of 
   the continuous function f and x is the value Ai(k) on the equilibrium point'''

def TransformFunc (x, n, f_type,landa=2):
    
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
''' Every concept in the FCM graph has a value Ai that expresses the quantity of 
its corresponding physical value and it is derived by the transformation of 
the fuzzy values assigned by the experts to numerical values. The value Ai 
of each concept Ci is calculated during each simulation step, computing the 
influence of other concepts to the specific concept by selecting one of the 
following equations (inference rules):
    
    k = Kasko
    mk = Modified Kasko
    r = Rescaled Kasko  '''
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
    ''' in each FCM you have some variables which are more important and
considered to be the main principles of the system. For example, in one FCM my
main variables are "water pollution" and "CO2 emission". These are the system 
indicators. By defining these principles you would be able to build an additional list
for keeping track of changes in these principles. The only thing you need to do
is to put their name in the list below "Principles". you can add as
many principles as you want.  '''

Principles=["principal_1_Name","principal_2_Name","principal_3_Name"]

prin_concepts_index = []
for nod in node_name.keys():
    if node_name[nod] in Principles:
        prin_concepts_index.append(nod)

listPossibleNodes=[]
for nod in G.nodes():
    if G.in_degree(nbunch=None, weight=None)[nod] <= 1 and Concepts_matrix[nod] not in Principles:
        listPossibleNodes.append(nod)
        
len (listPossibleNodes)


# In[13]:


function_type = input("What is the type of Threshold function (sig , tanh , bivalent, trivalent)?  ")
infer_rule = input("What is the Inference Rule (k , mk , r)?  ")


SteadyState = infer_steady (f_type = function_type , infer_rule =infer_rule)

change_in_principles ={}
for pr in prin_concepts_index:
        change_in_principles[pr]=[]
iteration = 0

for iter in range (20): # You can increas the number of times you repeat the random process of input vector generation
    for com in combinations (listPossibleNodes,5): # you can change the size of input vector 
  
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

for i in range(int(iteration/20)):
    values=df.loc[i*20].drop('IDS').values.flatten().tolist() 
    values += values[:1]
  
    # Plot data
    ax.plot(angles, values, linewidth=0.05, color="black" , alpha = 0.03,  linestyle='-')

# Fill area
#ax.fill(angles, values, 'b', alpha=0.1)
plt.show()

# In[ ]:




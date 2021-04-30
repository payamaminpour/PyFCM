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

import xlrd
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

Noise_Threshold = init.Noise_Threshold

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

def TransformFunc (x, n, f_type,landa=init.Lambda):
    
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
        act_vec_new = np.zeros(n)
        x = np.zeros(n)
        
        if infer_rule == "k":
            x = np.matmul(AdjmT, act_vec_old)
        if infer_rule == "mk":
            x = act_vec_old + np.matmul(AdjmT, act_vec_old)
        if infer_rule == "r":
            x = (2*act_vec_old-np.ones(n)) + np.matmul(AdjmT, (2*act_vec_old-np.ones(n)))
            
        act_vec_new = TransformFunc (x ,n, f_type)
        resid = max(abs(act_vec_new - act_vec_old))
        
        act_vec_old = act_vec_new
    return act_vec_new


# In[9]:

def infer_scenario (Scenario_concept,init_vec = activation_vec , AdjmT = Adj_matrix.T
                    , n =n_concepts , f_type="sig", infer_rule ="mk" , changeLevel = 1):
    act_vec_old= init_vec
    
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
        act_vec_new [Scenario_concept] = changeLevel
        #act_vec_new [what] = 1
        resid = max(abs(act_vec_new - act_vec_old))
        #if resid < 0.0001:
            #break
        
        act_vec_old = act_vec_new
    return act_vec_new


# In[10]:

Principles = init.Principles

prin_concepts_index = []
for nod in node_name.keys():
    if node_name[nod] in Principles:
        prin_concepts_index.append(nod)


# In[11]:


list_of_consepts_to_run = init.list_of_consepts_to_run

                            
# In[12]:

function_type = init.function_type
infer_rule = init.infer_rule

#what = Concepts_matrix.index(list_of_consepts_to_run[1])
SteadyState = infer_steady (f_type = function_type , infer_rule =infer_rule)

#Scenario 
for name in list_of_consepts_to_run:
    Sce_Con_name =name
   
    Scenario_concept = Concepts_matrix.index(Sce_Con_name)
    change_levels = np.linspace(0,1,21)
    
    change_in_principles ={}
    for pr in prin_concepts_index:
        change_in_principles[pr]=[]
    

    for c in change_levels:
    
        ScenarioState = infer_scenario (Scenario_concept,f_type=function_type, infer_rule = infer_rule , changeLevel=c)
        changes = ScenarioState - SteadyState
        
        for pr in prin_concepts_index:
            change_in_principles[pr].append(changes[pr])


    fig = plt
    fig.clf()   # Clear figure
    for pr in prin_concepts_index:
        fig.plot(change_levels,change_in_principles[pr], '-o' ,markersize=3 ,label=node_name[pr])
        fig.legend(fontsize=8 )
        plt.xlabel("activation state of {}".format(Sce_Con_name))
        plt.ylabel('State of system principles')

        fig.savefig('{}.pdf'.format(Sce_Con_name))
    plt.show()


# In[ ]:




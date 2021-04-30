# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 13:23:02 2018

@author: Payam Aminpour
         Michigan State University
         aminpour@msu.edu
"""
import __init__ as init
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
import xlrd
import numpy as np
import math
import networkx as nx


#______________________________________________________________________________

file_location = init.file_location
workbook = xlrd.open_workbook(file_location)
sheet = workbook.sheet_by_index(0)

n_concepts = sheet.nrows-1

Adj_matrix = np.zeros((n_concepts,n_concepts))
activation_vec = np.ones(n_concepts)
node_name = {}
#______________________________________________________________________________

Noise_Threshold = init.Noise_Threshold

for i in range (1,n_concepts+1):
    for j in range (1,n_concepts+1):
        if abs(sheet.cell_value(i,j))<=Noise_Threshold:
            Adj_matrix[i-1,j-1]=0
        else:
            Adj_matrix[i-1,j-1]=sheet.cell_value(i,j)

#______________________________________________________________________________

# Generating a python NetworkX graph using our Adjacancy Matrix
# Concepts_matrix is a list to keep concept names
Concepts_matrix = []
for i in range (1,n_concepts+1):
    Concepts_matrix.append(sheet.cell_value(0,i))

G = nx.DiGraph(Adj_matrix)
for nod in G.nodes():
    node_name[nod] = sheet.cell_value(nod+1,0)
#______________________________________________________________________________

def TransformFunc (x, n, f_type,Lambda=init.Lambda):
    
    if f_type == "sig":
        x_new = np.zeros(n)
        for i in range (n):
            x_new[i]= 1/(1+math.exp(-Lambda*x[i]))
            
        return x_new    

    
    if f_type == "tanh":
        x_new = np.zeros(n)
        for i in range (n):
            x_new[i]= math.tanh(Lambda*x[i])
        
        return x_new
    
    if f_type == "biv":
        x_new = np.zeros(n)
        for i in range (n):
            if x[i]> 0:
                x_new[i]= 1
            else:
                x_new[i]= 0
        
        return x_new
    
    if f_type == "triv":
        x_new = np.zeros(n)
        for i in range (n):
            if x[i]> 0:
                x_new[i] = 1
            elif x[i]==0:
                x_new[i] = 0
            else:
                x_new[i] = -1
        
        return x_new
#______________________________________________________________________________

def infer_steady (init_vec = activation_vec , AdjmT = Adj_matrix.T \
                  , n =n_concepts , f_type="sig", infer_rule ="mk"):
        
    act_vec_old= init_vec
    
    resid = 1
    while resid > 0.00001:  # here you have to define the stoping rule for steady state calculation
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
#______________________________________________________________________________
def infer_scenario (Scenario_concepts,change_level, init_vec = activation_vec , AdjmT = Adj_matrix.T \
                    , n =n_concepts , f_type="sig", infer_rule ="mk" ):
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
        
        for c in  Scenario_concepts:
            
            act_vec_new[c] = change_level[c]
        
            
        resid = max(abs(act_vec_new - act_vec_old))
        
        act_vec_old = act_vec_new
    return act_vec_new
#______________________________________________________________________________


Principles=init.Principles


prin_concepts_index = []
for nod in node_name.keys():
    if node_name[nod] in Principles:
        prin_concepts_index.append(nod)


list_of_consepts_to_run = init.list_of_consepts_to_run

#______________________________________________________________________________
function_type = init.function_type
infer_rule = init.infer_rule
change_level = init.change_level

change_level_by_index = {} 
for name in change_level.keys():
    change_level_by_index[Concepts_matrix.index(name)] = change_level[name]

Scenario_concepts = [] 
for name in list_of_consepts_to_run:
    Sce_Con_name =name
    Scenario_concepts.append(Concepts_matrix.index(Sce_Con_name))

    
change_IN_PRINCIPLES = []
    

SteadyState = infer_steady (f_type=function_type, infer_rule = infer_rule)
ScenarioState = infer_scenario (Scenario_concepts,change_level_by_index ,f_type=function_type, infer_rule = infer_rule)
change_IN_ALL = ScenarioState - SteadyState

for c in Scenario_concepts:
    change_IN_ALL[c] = 0

for i in range (len(prin_concepts_index)): 
    change_IN_PRINCIPLES.append(change_IN_ALL[prin_concepts_index[i]])



What_to_show = input("You want to see the results in All (Type: 'A') or only Principles (Type: 'P')?  ")

if What_to_show =="A":
    changes = change_IN_ALL
    a = 50
    plt.figure(figsize=(a,5))
    plt.bar(np.arange(len(changes)), changes, align='center', alpha=1 ,color='g')
    plt.xticks(np.arange(len(changes)) , Concepts_matrix , rotation='vertical' )

else:
    changes = change_IN_PRINCIPLES
    a = 10
    plt.figure(figsize=(a,3))
    plt.bar(np.arange(len(changes)), changes, align='center', alpha=1 ,color='b')
    plt.xticks(np.arange(len(changes)) , Principles , rotation='vertical' )


#plt.ylim(0,1)
#plt.ylabel('changes')
plt.title("changes in variables")
ax = plt.axes()        
ax.xaxis.grid() # vertical lines
plt.savefig('Scenario_Results.pdf')
plt.show()


changes_dic ={}
for nod in G.nodes():
    changes_dic[node_name[nod]] = change_IN_ALL[nod]
    
with open('Changes_In_All_Concepts.csv', 'w') as f:
    [f.write('{0},{1}\n'.format(key, value)) for key, value in changes_dic.items()]
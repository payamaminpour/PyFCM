# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 13:23:02 2018

@author: Payam Aminpour
         Michigan State University
         aminpour@msu.edu
"""

import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
import xlrd
import numpy as np
import math
import networkx as nx


#______________________________________________________________________________

''' the file location is the path of your project file in your computer
# this file should be an excel file with .xlsx extention
# Please see the Adjacency_Matrix_Example file to check how your matrix should look like'''

file_location = "C:/Paym Computer/project/...../.../Project_File_Name.xlsx" # change the path
workbook = xlrd.open_workbook(file_location)
sheet = workbook.sheet_by_index(0)

n_concepts = sheet.nrows-1

Adj_matrix = np.zeros((n_concepts,n_concepts))
activation_vec = np.ones(n_concepts)
node_name = {}
#______________________________________________________________________________

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

''' TransformFunc is the squashing function: 
    
   Four transformation functions:
   (Bivalent: 'biv', Trivalent: 'triv', Sigmoid: 'sig' or Hyperbolic tangent: 'tanh')

   where Lambda is a real positive number (λ>0) which determines the steepness of 
   the continuous function f and x is the value Ai(k) on the equilibrium point'''

def TransformFunc (x, n, f_type,Lambda=1):
    
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

            
''' Every concept in the FCM graph has a value Ai that expresses the quantity of 
its corresponding physical value and it is derived by the transformation of 
the fuzzy values assigned by the experts to numerical values. The value Ai 
of each concept Ci is calculated during each simulation step, computing the 
influence of other concepts to the specific concept by selecting one of the 
following equations (inference rules):
    
    k = Kasko
    mk = Modified Kasko
    r = Rescaled Kasko  '''
    
    
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
            
            act_vec_new[c] = change_level
        
            
        resid = max(abs(act_vec_new - act_vec_old))
        
        act_vec_old = act_vec_new
    return act_vec_new
#______________________________________________________________________________

''' in each FCM you have some variables which are more important and
considered to be the main principles of the system. For example, in one FCM my
main variables are "water pollution" and "CO2 emission". These are the system 
indicators. By defining these principles you would be able to build an additional list
for keeping track of changes in only these principles not all of the concepts. The only 
thing you need to do is to put their name in the list below "Principles". you can add as
many principles as you want  '''

Principles=["principal_1_Name","principal_2_Name","principal_3_Name"]


prin_concepts_index = []
for nod in node_name.keys():
    if node_name[nod] in Principles:
        prin_concepts_index.append(nod)


''' to run a scenario, you need to activate one or some of the variables in the map
to see the changes in other variables. Here you just need to put their name in the list below.
You can add more than one concept (separate them by comma). You can also have only one concept.
For example: list_of_consepts_to_run = ["chemical fertilizer" , "technology"] '''


list_of_consepts_to_run = ["concept_1_name", "concept_2_name"] 

#______________________________________________________________________________
function_type = input("What is the type of Threshold function (sig , tanh , biv, triv)?  ")
infer_rule = input("What is the Inference Rule (k , mk , r)?  ")
change_level = input("What is the level of the activation for running scenario [0,1]?  ")


Scenario_concepts = [] 
for name in list_of_consepts_to_run:
    Sce_Con_name =name
    Scenario_concepts.append(Concepts_matrix.index(Sce_Con_name))
    
change_IN_PRINCIPLES = []
    

SteadyState = infer_steady (f_type=function_type, infer_rule = infer_rule)
ScenarioState = infer_scenario (Scenario_concepts,change_level ,f_type=function_type, infer_rule = infer_rule)
change_IN_ALL = ScenarioState - SteadyState

for c in Scenario_concepts:
    change_IN_ALL[c] = 0

for i in range (len(prin_concepts_index)): 
    change_IN_PRINCIPLES.append(change_IN_ALL[prin_concepts_index[i]])



What_to_show = input(" All or Principles ")

if What_to_show =="All":
    changes = change_IN_ALL
    a = 100
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
plt.show()


changes_dic ={}
for nod in G.nodes():
    changes_dic[node_name[nod]] = change_IN_ALL[nod]
    
with open('my_file.csv', 'w') as f:
    [f.write('{0},{1}\n'.format(key, value)) for key, value in changes_dic.items()]
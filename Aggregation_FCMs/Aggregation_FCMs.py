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
import networkx as nx



# In[2]:

file_location = init.file_location
workbook = xlrd.open_workbook(file_location)
sheet = workbook.sheet_by_index(0)

n_concepts = sheet.nrows-1


# In[3]:

# Agregating FCMs

adj = np.zeros((n_concepts,n_concepts))
count = np.zeros((n_concepts,n_concepts))
adj_ag = np.zeros((n_concepts,n_concepts))
All_ADJs =[]
p=0
for i in range(0,workbook.nsheets):
    p+=1
    sheet = workbook.sheet_by_index(i)
    Adj_matrix = np.zeros((n_concepts,n_concepts))

    for i in range (1,n_concepts+1):
        for j in range (1,n_concepts+1):
            Adj_matrix[i-1,j-1]=sheet.cell_value(i,j)
            if sheet.cell_value(i,j) != 0:
                count[i-1,j-1] += 1
    
    All_ADJs.append(Adj_matrix)
    adj += Adj_matrix
    

adj_copy = np.copy(adj)    


if init.Aggregation_technique == "AMX":
    for i in range (n_concepts):
        for j in range (n_concepts):
            if count[i,j] == 0:
                adj_ag[i,j] = 0
            else:
                adj_ag[i,j] = adj_copy[i,j]/count[i,j]

if init.Aggregation_technique == "AMI":
    from statistics import mean as mean
    for i in range (n_concepts):
        for j in range (n_concepts):
            a = [ind[i,j] for ind in All_ADJs]
            adj_ag[i,j] = mean(a)

if init.Aggregation_technique == "MED":
    from statistics import median as med
    for i in range (n_concepts):
        for j in range (n_concepts):
            a = [ind[i,j] for ind in All_ADJs]
            adj_ag[i,j] = med(a)
         
if init.Aggregation_technique == "GM":
    import scipy
    for i in range (n_concepts):
        for j in range (n_concepts):
            a = [ind[i,j] for ind in All_ADJs if ind[i,j] !=0]
            adj_ag[i,j] = float(scipy.stats.mstats.gmean(np.array(a)))


Adj_aggregated_FCM = adj_ag


# In[4]:

G = nx.DiGraph(Adj_aggregated_FCM)


plt.figure(figsize=(10,10));

everylarge=[(u,v) for (u,v,d) in G.edges(data=True) if abs(d['weight']) >=0.75]
elarge=[(u,v) for (u,v,d) in G.edges(data=True) if abs(d['weight']) >0.5 and abs(d['weight']) <0.75]
esmall=[(u,v) for (u,v,d) in G.edges(data=True) if abs(d['weight']) <=0.5 and abs(d['weight']) >0.25]
everysmall=[(u,v) for (u,v,d) in G.edges(data=True) if abs(d['weight']) <=0.25]


#################Centrality#####################################################################
label= {}
for nod in G.nodes():
    label[nod] = sheet.cell_value(nod+1,0)

#pos = nx.random_layout(G)
pos=nx.spring_layout(G,dim=2, k=0.75)
#########################Visualization##############################################################
nx.draw_networkx(G,pos,labels=label,font_size=7, node_size= 200 ,node_color='lightgreen' ,alpha=0.6)
nx.draw_networkx_edges(G,pos,edgelist=everylarge, width=2,alpha=0.5,edge_color='gold')
nx.draw_networkx_edges(G,pos,edgelist=elarge, width=1,alpha=0.5,edge_color='g',style='dashed')
nx.draw_networkx_edges(G,pos,edgelist=esmall, width=0.5,alpha=0.5,edge_color='lightcoral',style='dashed')
nx.draw_networkx_edges(G,pos,edgelist=everysmall, width=0.25,alpha=0.5,edge_color='lightgray',style='dashed')

plt.show()

#######################################################################################################


nx.write_edgelist(G, "aggregated_edg.csv")
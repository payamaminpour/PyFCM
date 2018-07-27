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

import xlrd
import numpy as np
import networkx as nx



# In[2]:
''' the file location is the path of your project file in your computer
# this file should be an excel file with .xlsx extention
# Please see the "AllParticipants_Adjacency_Matrix_Example" file to check how your matrix should look like'''

file_location = "C:/Paym Computer/........../......../AllParticipants_Adjacency_Matrix_Example.xlsx"
workbook = xlrd.open_workbook(file_location)
sheet = workbook.sheet_by_index(0)

n_concepts = sheet.nrows-1


# In[3]:

# Agregating FCMs

adj = np.zeros((n_concepts,n_concepts))
count = np.zeros((n_concepts,n_concepts))
adj_ag = np.zeros((n_concepts,n_concepts))
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
    
    adj += Adj_matrix

adj_copy = np.copy(adj)    

Zeros = input("What is the type of aggregation (Ex_Included , Included)?  ")

if Zeros == 'Ex_Included':
    for i in range (n_concepts):
        for j in range (n_concepts):
            if count[i,j] == 0:
                adj_ag[i,j] = 0
            else:
                adj_ag[i,j] = adj_copy[i,j]/count[i,j]

else:
    adj_ag = adj_copy/p

Adj_aggregated_FCM = adj_ag


# In[4]:

Aggregated_FCM = nx.DiGraph(Adj_aggregated_FCM)
nx.write_edgelist(Aggregated_FCM, "aggregated_edg.csv")
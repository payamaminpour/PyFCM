# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 16:04:39 2018

@author: Payam Aminpour
         Michigan State University
         aminpour@msu.edu
"""

import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
import xlrd
import numpy as np
import networkx as nx


#____________________________________________________________________________________
''' the file location is the path of your project file in your computer
# this file should be an excel file with .xlsx extention
# Please see the "AllParticipants_Adjacency_Matrix_Example" file to check how your matrix should look like'''


file_location = "C:/Paym Computer/........../......../AllParticipants_Adjacency_Matrix_Example.xlsx"

workbook = xlrd.open_workbook(file_location)
sheet = workbook.sheet_by_index(0)
n_concepts = sheet.nrows-1
n_participants = workbook.nsheets
#____________________________________________________________________________________
########## Creat a dictionary keys = name of participants;  values = Adj Matrix 
Allparticipants={}
IDs = []  # each participant has a unique name or ID
for i in range(0,n_participants):
   
    sheet = workbook.sheet_by_index(i)
    Adj_matrix = np.zeros((n_concepts,n_concepts))
    for row in range (1,n_concepts+1):
        for col in range (1,n_concepts+1):
            Adj_matrix[row-1,col-1]=sheet.cell_value(row,col)
    IDs.append(sheet.cell_value(0,0))

    Allparticipants[sheet.cell_value(0,0)]=Adj_matrix


#____________________________________________________________________________________
def FCM(ID):
    '''Generate an FCM in networkx format'''
    
    adj = Allparticipants[ID]
    FCM = nx.DiGraph(adj)
         
    return FCM 


def similarity (agent,FCM_Reference):
    ''' how similar the FCM is to the FCM Reference'''   
    def select_k(spectrum, minimum_energy = 0.9):
        running_total = 0.0
        total = sum(spectrum)
        if total == 0.0:
            return len(spectrum)
        for i in range(len(spectrum)):
            running_total += spectrum[i]
            if running_total / total >= minimum_energy:
                return i + 1
        return len(spectrum)
    
    laplacian1 = nx.spectrum.laplacian_spectrum(agent.FCM.to_undirected())
    laplacian2 = nx.spectrum.laplacian_spectrum(FCM_Reference.to_undirected())
    k1 = select_k(laplacian1)
    k2 = select_k(laplacian2)
    k = min(k1, k2)
    similarity = sum((laplacian1[:k] - laplacian2[:k])**2)
            
    return similarity

########### A class of agents with FCMs and IDs############################
class Agents (object):
    
    def __init__ (self,ID):
        self.ID = ID
        self.FCM = FCM(self.ID)

#____________________________________________________________________________________
'''Here you generate n agents and give each agent an FCM'''

agents=[]
n = 218
for Id in IDs:
    a = Agents(ID=Id)
    agents.append(a)
#____________________________________________________________________________________
'''This Function is generating the reference FCM '''

def Fcm_Reference(How):
    '''there are several ways to generate Reference_FCM
        # FCM_Reference is the average of all FCMs (including zeros) 
        # FCM_Reference is the average of all FCMs (excluding zeros) 
        # FCM_Reference is a n*n zeros matrix                        
        # FCM_Reference is a n*n ones matrix '''
                       
    if How == "ave_in_zeros":
        adj=np.zeros((n_concepts,n_concepts))
        for ag in agents:
            adj+=nx.to_numpy_matrix(ag.FCM)
    
        FCM_Reference = nx.DiGraph(adj/13)
        
    if How == "ave_ex_zeros":
        adj = np.zeros((n_concepts,n_concepts))
        count = np.zeros((n_concepts,n_concepts))
        adj_ag = np.zeros((n_concepts,n_concepts))

        for ag in agents:
    
            Adj_matrix = np.zeros((n_concepts,n_concepts))

            for i in range (0,n_concepts):
                for j in range (0,n_concepts):
                    Adj_matrix[i,j]=nx.to_numpy_matrix(ag.FCM)[i,j]
                    if nx.to_numpy_matrix(ag.FCM)[i,j] != 0:
                        count[i,j] += 1
    
    
            adj += Adj_matrix
            adj_copy = np.copy(adj)
            for i in range (n_concepts):
                for j in range (n_concepts):
                    if count[i,j] == 0:
                        adj_ag[i,j] = 0
                    else:
                        adj_ag[i,j] = adj_copy[i,j]/count[i,j]

        FCM_Reference = nx.DiGraph(adj_ag)
        
    if How == "ones":
        FCM_Reference = nx.DiGraph(np.ones((n_concepts,n_concepts)))
        
    if How == "zeros":
        FCM_Reference = nx.DiGraph(np.zeros((n_concepts,n_concepts)))
        
    return FCM_Reference
#____________________________________________________________________________________
######## You have to choose one way to generate a Reference FCM ###########
FCM_Reference = Fcm_Reference("ave_ex_zeros")

# a dictionary with keys = agent.ID and values = simil index of the agent's FCM
simil = {}
for agent in agents:
    simil[agent.ID] = similarity (agent,FCM_Reference)

#____________________________________________________________________________________
################## K-Mean clustering ######################################
from sklearn.cluster import KMeans
X = np.array(list(simil.values()))
n_clusters = 4
km = KMeans(n_clusters=n_clusters)
km.fit(X.reshape(-1,1)) 
Indiv_Clusters = list(zip(list(simil.keys()),km.labels_))

clusters= {}
for i in range(n_clusters ):
    clusters[i] = []
    
for i in Indiv_Clusters:
    print (i[0] , "is in   cluster {}".format(i[1]))
    clusters[i[1]].append(simil[i[0]])

       
plt.figure(figsize=(5,2))    
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=0)

for cl in range(n_clusters):  
    plt.plot(clusters[cl], np.zeros_like(clusters[cl]),
             'x' , markersize = '6' , label=cl)

plt.legend()
plt.show()

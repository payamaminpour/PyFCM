# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 16:04:39 2018

@author: Payam Aminpour
         Michigan State University
         aminpour@msu.edu
"""
import __init__ as init
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
import xlrd
import numpy as np
import networkx as nx
import math
import random

#____________________________________________________________________________________

file_location = init.file_location

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



    
    #-------------------------------------------
activation_vec = np.ones(n_concepts)
def TransformFunc (x, n, f_type,landa=1):
    
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
def infer_steady ( Adj , init_vec = activation_vec \
                  , n =n_concepts , f_type="tanh", infer_rule ="k"):
    
    act_vec_old= init_vec
    AdjmT = Adj.T
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

def infer_scenario (Scenario_concepts,level, Adj , init_vec = activation_vec \
                    , n =n_concepts , f_type="tanh", infer_rule ="k" ):
    
    act_vec_old= init_vec
    AdjmT = Adj.T

        
    resid = 1
    while resid > 0.0001:
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
            
            act_vec_new[c] = level[c]
        
            
        resid = max(abs(act_vec_new - act_vec_old))
        #if resid < 0.0001:
            #break
        
        act_vec_old = act_vec_new
    return act_vec_new


    #-------------------------------------------

def dynamic (agent,FCM_Reference,f_type,infer_rule ):
    M = 0
    W=[]
    SState = infer_steady (Allparticipants[agent.ID], f_type = f_type, infer_rule = infer_rule)
    SState_ref = infer_steady (FCM_Reference, f_type = f_type, infer_rule= infer_rule)
    iteration = 0
    for iter in range(10):
        for iter in range (100):
            rand = random.randint(1,n_concepts)
            com = random.sample(agent.FCM.nodes() , rand)
            Scenario_concepts = com 
            my_random ={}
            for rC in  Scenario_concepts:
                PN = random.choice([-1,1])
                my_random[rC] = random.random() * PN
            iteration += 1
            ScenarioState = infer_scenario (Scenario_concepts,my_random,Allparticipants[agent.ID],
                                            f_type= f_type, infer_rule= infer_rule)
            ScenarioState_ref = infer_scenario (Scenario_concepts,my_random,FCM_Reference,
                                            f_type= f_type, infer_rule= infer_rule)
        
        
            Change = ScenarioState - SState
            Change_ref = ScenarioState_ref - SState_ref
            M += sum((Change[:] - Change_ref[:])**2)
        M = (math.sqrt(M))/iteration
        W.append(M)
    return np.mean(W)

########### A class of agents with FCMs and IDs############################
class Agents (object):
    
    def __init__ (self,ID):
        self.ID = ID
        self.FCM = FCM(self.ID)

#____________________________________________________________________________________
'''Here you generate n agents and give each agent an FCM'''

agents=[]
n = n_participants
for Id in IDs:
    a = Agents(ID=Id)
    agents.append(a)
#____________________________________________________________________________________
'''This Function is generating the reference FCM '''

def Fcm_Reference(How):

                       
    if How == "AI":
        adj=np.zeros((n_concepts,n_concepts))
        for ag in agents:
            adj+=nx.to_numpy_matrix(ag.FCM)
    
        FCM_Reference = adj/n_participants
        
    if How == "AX":
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

        FCM_Reference = adj_ag
        
    if How == "O":
        FCM_Reference = np.ones((n_concepts,n_concepts))
        
    if How == "Z":
        FCM_Reference = np.zeros((n_concepts,n_concepts))
        
    return FCM_Reference
#____________________________________________________________________________________
######## You have to choose one way to generate a Reference FCM ###########
FCM_Reference = Fcm_Reference(init.Aggregation_technique)

# a dictionary with keys = agent.ID and values = simil index of the agent's FCM
simil = {}
if init.clustering_method == "D":
    f_type = input("What is the type of Squashing function (sig , tanh , bivalent, trivalent)?  ")
    infer_rule = input("What is the Inference Rule (k , mk , r)?  ")
    for agent in agents:
        simil[agent.ID] = dynamic (agent,FCM_Reference,f_type,infer_rule)

if init.clustering_method == "S":
    for agent in agents:
        simil[agent.ID] = similarity (agent,nx.DiGraph(FCM_Reference))

#____________________________________________________________________________________
################## K-Mean clustering ######################################
from sklearn.cluster import KMeans
X = np.array(list(simil.values()))
n_clusters = init.n_clusters
km = KMeans(n_clusters=n_clusters)
km.fit(X.reshape(-1,1)) 
Indiv_Clusters = list(zip(list(simil.keys()),km.labels_))

clusters= {}
for i in range(n_clusters ):
    clusters[i] = []
    
for i in Indiv_Clusters:
    print (i[0] , "is in   cluster {}".format(i[1]))
    clusters[i[1]].append(simil[i[0]])

       
plt.figure(figsize=(10,3))    
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=0)

for cl in range(n_clusters):  
    plt.plot(clusters[cl], np.zeros_like(clusters[cl]),
             'x' , markersize = '8' , label=cl)

plt.legend()
plt.savefig('Clusters.pdf')
plt.show()

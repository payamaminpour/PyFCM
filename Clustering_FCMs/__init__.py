# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 15:19:59 2018

@author: Payam Aminpour
"""

name = "FCM_Scenario_Analysis"

print ("\n","The file location is the path of your project file in your computer.","\n", 
       "For Example: C:/Paym Computer/Safety Project/All_Adjacency_matrix.xlsx","\n", 
       "This file should be an excel file with .xlsx extention","\n",
       "Please see the AllParticipants_Adjacency_Matrix_Example file to check how your matrix should look like")

print("\n")
file_location= input("copy your project file path here:   ")


print ("\n")
print (    '''There are several ways to generate Reference_FCM
       
        # FCM_Reference is the average of all FCMs (including zeros) -> Type: AI
        # FCM_Reference is the average of all FCMs (excluding zeros) -> Type: AX
        # FCM_Reference is a n*n zeros matrix                        -> Type: Z                      
        # FCM_Reference is a n*n ones matrix                         -> Type: O
        ''')

Aggregation_technique = input("what is the method to generate Reference FCM?   ")


clustering_method = input("what is the clusterign criterion? Structure:S, Dynamics:D ->   ")

print ("\n")
n_clusters = int(input("Hom Mnay Clusters?   "))
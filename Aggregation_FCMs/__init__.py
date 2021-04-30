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
       
        # Arithmatic Mean of all FCMs (Including edges with weight = 0)   --> Type: AMI
        # Arithmatic Mean of all FCMs (Excluding edges with weight = 0)   --> Type: AMX
        # Median of all FCMs                                              --> Type: MED                      
        # Geometric Mean of all FCMs                                      --> Type: GM
        # Weighted Mean of all FCMs                                       --> Type: WM
        ''')

print ("\n")
Aggregation_technique = input("what is the method to aggregate all FCMs?   ")


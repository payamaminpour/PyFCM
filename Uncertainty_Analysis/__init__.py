# # -*- coding: utf-8 -*-
# """
# Created on Sun Aug  5 15:19:59 2018

# @author: Payam Aminpour
# """

# name = "FCM_Scenario_Analysis"

# print(
#     "\n",
#     "The file location is the path of your project file in your computer.",
#     "\n",
#     "For Example: C:/Paym Computer/Safety Project/Adjacency_matrix.xlsx",
#     "\n",
#     "This file should be an excel file with .xlsx extention",
#     "\n",
#     "Please see the Adjacency_Matrix_Example file to check how your matrix should look like",
# )

# print("\n")
# file_location = input("copy your project file path here:   ")


# print(
#     "\n",
#     """Sometimes you need to remove the links with significantly low weights to avoid messiness.
# Noise_Threshold is a number in [0,1] which defines a boundary below which all links will be removed from the FCM.
# E.g. Noise_Threshold = 0.15 means that all edges with weight <= 0.15 will be removed from FCM. """,
# )

# print("\n")
# Noise_Threshold = float(input("What is the Noise_Threshold:   "))

# print(
#     "\n",
#     """ Every concept in the FCM graph has a value Ai that expresses the quantity of
# its corresponding physical value and it is derived by the transformation of
# the fuzzy values assigned by who developed the FCM to numerical values.
# The value Ai of each concept Ci is calculated during each simulation step,
# computing the influence of other concepts to the specific concept by selecting one of the
# following equations (inference rules):

#     k = Kasko
#     mk = Modified Kasko
#     r = Rescaled Kasko  """,
# )

# print("\n")
# infer_rule = input("What is the Inference Rule (k , mk , r)?   ")

# print(
#     "\n",
#     "There are several squashing function:",
#     "\n",
#     "\n",
#     "Bivalent: 'biv'",
#     "\n",
#     "Trivalent: 'triv'",
#     "\n",
#     "Sigmoid: 'sig'",
#     "\n",
#     "Hyperbolic tangent: 'tanh'",
# )

# print("\n")
# function_type = input("What is the type of Squashing function?   ")
# print("\n")
# Lambda = float(
#     input(
#         "What is the parameter lambda in Squashing function? choose a number between (0,10)   "
#     )
# )


# print(
#     "\n",
#     """ In each FCM you have some variables which are more important and
# considered to be the main principles of the system. For example, in one FCM my
# main variables are "water pollution" and "CO2 emission". These are the system
# indicators. By defining these principles you would be able to build an additional list
# for keeping track of changes in only these principles not all of the concepts. The only
# thing you need to do is to put their name one by one. you can add as
# many principles as you want  """,
# )

# n_princ = int(input("How many Principles?  "))
# Principles = []
# for i in range(n_princ):
#     Principles.append(input("The name of Principle {} =  ".format(i + 1)))

# print("\n")
# Thresh = int(
#     input(
#         "what is the Maximum Indegree for a Concept to be in list of possible nodes to be activated?  "
#     )
# )

# print("\n")
# n_iteration = int(input("How many iterations?  "))

# print("\n", " Filter the ploting ")

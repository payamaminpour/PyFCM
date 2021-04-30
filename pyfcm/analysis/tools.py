# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 13:40:00 2021

@author: Corey White
         North Carolina State University
         ctwhite@ncsu.edu
"""

import numpy as np
import math


def _transform(x, n, f_type, landa):
    """
    Squashing function applied to FCM

    Parameters
      ----------
      x : numpy.ndarray
          Activation vector after inference rule is applied.
      n : int
          The number of concepts in the adjacency matrix.
      f_type : str
          Sigmoid = "sig", Hyperbolic Tangent = "tanh", Bivalent = "biv", Trivalent = "triv"
      landa : int
          The lambda threshold value used in the squashing fuciton between 0 - 10

      Returns
          -------
          Activation Vector : numpy.ndarray
    """
    if f_type == "sig":
        x_new = np.zeros(n)
        for i in range(n):
            x_new[i] = 1 / (1 + math.exp(-landa * x[i]))
        return x_new

    if f_type == "tanh":
        x_new = np.zeros(n)
        for i in range(n):
            x_new[i] = math.tanh(landa * x[i])
        return x_new

    if f_type == "biv":
        x_new = np.zeros(n)
        for i in range(n):
            if x[i] > 0:
                x_new[i] = 1
            else:
                x_new[i] = 0
        return x_new

    if f_type == "triv":
        x_new = np.zeros(n)
        for i in range(n):
            if x[i] > 0:
                x_new[i] = 1
            elif x[i] == 0:
                x_new[i] = 0
            else:
                x_new[i] = -1
        return x_new

import numpy as np
from numpy.random import seed
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import rcParams
#set the plot figure size
rcParams["figure.figsize"] = 10, 5
%matplotlib inline

class Perceptron(object):
    ""
    Perceptron Classifierself.
    Parameters
    ----------
    eta : float
        Learning rate (between 0.0 and 1.0)
        n_iter : int
        Passes (epochs) over the training set.
    Attributes
    ----------
    w_ : ld-array
        Weights after fitting.
    errors_ : list
        Number of misclassification in every epoch.
    ""
    def __init__(self, eta=0.01, n_iter=10)

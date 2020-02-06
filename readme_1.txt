# Importing libraries into python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

# Importing from sklearn
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans
from sklearn.random_projection import GaussianRandomProjection
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from keras import regularizers
from keras.layers import Dense
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.layers import Flatten
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier  
from sklearn.metrics import roc_auc_score
from sklearn import metrics


# Download the dataset, the link for the Appliances Energy Prediction Dataset:
https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction

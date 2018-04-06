import os
import numpy as np
import random
import cv2
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from subprocess import call
from sklearn.neighbors.kde import KernelDensity
from mpl_toolkits import mplot3d
from sklearn.decomposition import PCA
import pandas as pd
sns.set(color_codes=True)
from nuclei.utilities import *

# path for the training set directory
stage1_train_path = '/Users/jonathansaragosti/Documents/GitHub/Python/Kaggle/2018 Data Science Bowl/nuclei/data/stage1_train/'
# Check how many images there are in total in the training set
dataset = DataSet(stage1_train_path)

hist = []
for subdir in dataset.subdir:
    this_hist, _= subdir.image.hist()
    hist.append(this_hist.tolist())
hist_a = np.array(hist)

n_components = 4
pca = PCA()
pca.fit(hist_a)


# display the first three components on a 3D scatter plot
hist_a_t = pca.transform(hist_a)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(hist_a_t[:,0],hist_a_t[:,1],hist_a_t[:,2],s=2)
plt.show()
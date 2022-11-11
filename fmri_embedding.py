!pip install nilearn

!pip install umap-learn

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import os
import nilearn.image as image
from umap.umap_ import UMAP
import matplotlib.pyplot as plt
from scipy.signal import detrend
import warnings
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from nilearn.regions import RegionExtractor
from nilearn import image
from nilearn import datasets
from nilearn import plotting

class GHEUMAP(object):

  def __init__(self, cfg):

    self.num_heal = cfg['Num_Heal']
    self.num_pat = cfg['Num_Pat']
    self.num_sub = self.num_heal + self.num_pat
    self.region_num = cfg['region_num']
    self.data_set = cfg['Data_Set']
    self.loading_path = cfg['Loading_Path']

  def load_HFD(self):
    
    sub = np.linspace(1, self.num_sub, num=self.num_sub, dtype=int)
    Total = []

    for i in sub:

      print('Loading Subject {}'.format(i))
      Sub = np.load(os.path.join(self.loading_path, 'GHE_DS{}_Region{}_Sub{}.npy'.format(self.data_set, self.region_num, i)))
      Sub = np.nan_to_num(Sub)
      Total.append(Sub)

    self.region_t = np.asarray(Total)

  def plot_rst(self):

    self.load_HFD()
    embedding = UMAP(n_neighbors=15,
                     metric='cosine',
                     min_dist=0.000,
                     learning_rate=1,
                     set_op_mix_ratio=1,
                     spread=3,
                     a = 100,
                     b = 0.9,
                     random_state = 12345,
                     angular_rp_forest = True,
                     local_connectivity=3).fit_transform(self.region_t)

    plt.figure()
    label = []

    for i in range(self.num_sub):

      if i < self.num_heal :
        label.append(1)
        plt.scatter(embedding[i, 0], embedding[i, 1],  marker='o', color='lime', edgecolors='k')
      else:
        label.append(0)
        plt.scatter(embedding[i, 0], embedding[i, 1], marker='v', color='blue', edgecolors='k')

for i in range(0,34):

  rn = i + 1
  print('Working on region {}'.format(rn))
  cfg = {
        'Num_Pat': 14, 
        'Num_Heal': 14,
        'Data_Set': 1,  
        'Saving_Path': 'fMRI Data/Data/GHE/',
        'Saving_fig_path':'fMRI Data/Data/FigResults/',
        'Loading_Path': 'fMRI Data/Data/GHE/',
        'region_num': rn
        }
  fg = GHEUMAP(cfg)
  fg.plot_rst()

plt.show()

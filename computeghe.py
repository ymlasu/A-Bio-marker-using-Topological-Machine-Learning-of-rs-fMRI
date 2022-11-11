# -*- coding: utf-8 -*-
!pip install umap-learn

!pip install nilearn

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

class MRI_GHE(object):

  def __init__(self, cfg):

        self.num_heal = cfg['Num_Heal']
        self.num_pat = cfg['Num_Pat']
        self.num_sub = self.num_heal + self.num_pat
        self.region_num = cfg['region_num']
        self.data_set = cfg['Data_Set']
        self.saving_path = cfg['Saving_Path']
        self.loading_path = cfg['Loading_Path']
        self.Sub_PKS_path = cfg['Sub_PKS_path']
        self.Sub_CTL_path = cfg['Sub_CTL_path']
        self.Saving_fig_path = cfg['Saving_fig_path']
        self.Path_Find_Coord = cfg['Path_Find_Coord']
        self.GHE_Order = cfg['GHE_Order']

  def genhurst(self, S, q=self.GHE_Order):

    L = len(S)
    if L < 100:
      warnings.warn('Data series very short!')

    H = np.zeros((len(range(5, 20)), 1))
    k = 0

    for Tmax in range(5, 20):

      x = np.arange(1, Tmax + 1, 1)
      mcord = np.zeros((Tmax, 1))

      for tt in range(1, Tmax + 1):

        dV = S[np.arange(tt, L, tt)] - S[np.arange(tt, L, tt) - tt]
        VV = S[np.arange(tt, L + tt, tt) - tt]
        N = len(dV) + 1
        X = np.arange(1, N + 1, dtype=np.float64)
        Y = VV
        mx = np.sum(X) / N
        SSxx = np.sum(X ** 2) - N * mx ** 2
        my = np.sum(Y) / N
        SSxy = np.sum(np.multiply(X, Y)) - N * mx * my
        cc1 = SSxy / SSxx
        cc2 = my - cc1 * mx
        ddVd = dV - cc1
        VVVd = VV - np.multiply(cc1, np.arange(1, N + 1, dtype=np.float64)) - cc2
        mcord[tt - 1] = np.mean(np.abs(ddVd) ** q) / np.mean(np.abs(VVVd) ** q)

      mx = np.mean(np.log10(x))
      SSxx = np.sum(np.log10(x) ** 2) - Tmax * mx ** 2
      my = np.mean(np.log10(mcord))
      SSxy = np.sum(np.multiply(np.log10(x), np.transpose(np.log10(mcord)))) - Tmax * mx * my
      H[k] = SSxy / SSxx
      k = k + 1

    mH = np.mean(H) / q

    return mH

  def find_coordinate(self):

    smith_atlas = datasets.fetch_atlas_smith_2009()
    atlas_networks = smith_atlas.rsn10
    Sub = self.Path_Find_Coord
    extraction = RegionExtractor(atlas_networks, min_region_size=800,
                                     threshold=98, thresholding_strategy='percentile')
    extraction.fit_transform(Sub)
    resample = extraction._resampled_maps_img_
    coord = []

    img = image.index_img(resample, self.region_num)
    coords = plotting.find_xyz_cut_coords(img)
    plotting.plot_stat_map(img, cut_coords=coords, colorbar=False)

    plt.savefig(os.path.join(self.Saving_fig_path, 'Region{}.png'.format(self.region_num)))

    img = image.index_img(resample, self.region_num)
    img = img.get_data()
    x, y, z = np.shape(img)

    for i in range(x):
      for j in range(y):
        for k in range(z):
          q = img[i, j, k]
          if np.abs(q) > 2:
            q1 = [i, j, k]
            coord.append(q1)

    coord = np.asarray(coord)

    return coord

  def load_data(self):

    coord = self.find_coordinate()
    print('***** Start Loading Data *****')
    Sub_ctl = []
    Sub_pks = []
    self.region = coord

    for i in range(self.num_pat):

      num_sub = i + 1
      print('****Loading Patient Subject: {}****'.format(num_sub))
      subpks = os.path.join(self.Sub_PKS_path, 'Sub-PKS{}/wrasub-PKS{}_task-rest_bold.nii'.format(num_sub,num_sub))                                                                                         
      Sub_pks.append(subpks)

    for i in range(self.num_heal):
      num_sub = i + 1
      print('****Loading Control Subject: {}****'.format(num_sub))
      sub = os.path.join(self.Sub_CTL_path, 'Sub-CTL{}/wrasub-CTL{}_task-rest_bold.nii'.format(num_sub, num_sub))
      Sub_ctl.append(sub)                                                                                              

    self.results_ctl = image.smooth_img(Sub_ctl, fwhm=None)
    self.results_pks = image.smooth_img(Sub_pks, fwhm=None)

    print('***** Done For Load Data *****')

  def work(self):

    self.load_data()
    print('***** Start Working on HFD *****')
    cnt = 0
    tt_m = []
    nx, _ = np.shape(self.region)
    self.nx = nx

    for sub in self.results_ctl:

      cnt = cnt + 1
      print("------------Working on subject {}-----------".format(cnt))
      Sub_HFD = 1
      sub = sub.get_data()

      for i in range(nx):

        print("Working on Region {} of Sub: {} Pixel: {}".format(self.region_num, cnt, i))
        xx, yy, zz = self.region[i, :]
        x = sub[xx, yy, zz, :100]
        x = np.asarray(x)
        #x = detrend(x)
        HFD = self.genhurst(x)
        Sub_HFD = np.append(Sub_HFD, HFD)

        Sub_HFD = np.delete(Sub_HFD, 0, 0)
        Sub_HFD = np.nan_to_num(Sub_HFD)
        np.save(os.path.join(self.saving_path,
                                     'GHE_DS{}_Region{}_Sub{}'.format(self.data_set, self.region_num, cnt)), Sub_HFD)

        tt_m.append(Sub_HFD)

    cnt = cnt 

    for sub in self.results_pks:

      cnt = cnt + 1
      print("------------Working on subject {}-----------".format(cnt))
      Sub_HFD = 1
      sub = sub.get_data()

      for i in range(nx):
        print("Working on Region {} of Sub: {} Pixel: {}".format(self.region_num, cnt, i))
        xx, yy, zz = self.region[i, :]
        x = sub[xx, yy, zz, :100]
        x = np.asarray(x)
        #x = detrend(x)
        HFD = self.genhurst(x)
        Sub_HFD = np.append(Sub_HFD, HFD)

        Sub_HFD = np.delete(Sub_HFD, 0, 0)
        Sub_HFD = np.nan_to_num(Sub_HFD)
        np.save(os.path.join(self.saving_path,
                                    'GHE_DS{}_Region{}_Sub{}'.format(self.data_set, self.region_num, cnt)), Sub_HFD)

        tt_m.append(Sub_HFD)

    self.region_t = np.asarray(tt_m)

for i in range(0, 7):

  rn = i + 1
  print('Working on region {}'.format(rn))
  cfg = {

        'Num_Pat': 14, # 16, 24, 24， 24
        'Num_Heal': 14,
        'Data_Set': 1,  # 412 indicates 1: motor region + region 2,3,4,5,6
        'Saving_Path': 'fMRI Data/Data/GHE/',
        'Saving_fig_path':'fMRI Data/Data/FigResults/',
        'Loading_Path': 'fMRI Data/Data/GHE/',
        'Sub_PKS_path': 'fMRI Data/Data/DS1/FunImgARW/',
        'Sub_CTL_path': 'fMRI Data/Data/DS1/FunImgARW/',
        'Path_Find_Coord': 'fMRI Data/Data/DS1/FunImgARW/Sub-PKS1/wrasub-PKS1_task-rest_bold.nii',
        'region_num': rn,
        'GHE_Order': 2
        }
  fg = MRI_GHE(cfg)
  fg.work()

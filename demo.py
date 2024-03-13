# -*- coding: utf-8 -*-
"""demo.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Nz_eXH9mV9QWHBSieeDfXGyNysfFnWvz
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.spatial.distance import cdist
from plotly.subplots import make_subplots
import time
import matplotlib.image as mpimg
import torchvision.transforms.functional as TF
from torchvision.transforms import ToTensor, ToPILImage, Grayscale
import torch
import math
import seaborn as sns
import os
from RASCoPy import recurrence, opti_epsi, symbolic_series

#from google.colab import drive
#drive.mount('/content/drive')
#import sys
#sys.path.append('/content/drive/MyDrive/Recurrence_Analysis/RASCoPy')
#from . import recurrence, opti_epsi, symbolic_series

def demo_LotkaVoltera(method):

  X0 = np.array([1,0.17,0.01])
  sig =np.array([1,1.2,1.6])
  rho = np.array([[1, sig[0]/sig[1]+0.5, sig[0]/sig[2]-0.5],
        [sig[1]/sig[0]-0.5, 1, sig[1]/sig[2]+0.5],
        [sig[2]/sig[0]+0.5, sig[2]/sig[1]-0.5, 1]])
  #T = 500
  T=280
  deltaT = 0.1429
  X1 = np.zeros(T)
  X2 = np.zeros(T)
  X3 = np.zeros(T)
  times = np.linspace(0,deltaT*T, T)
  X1[0]=X0[0]
  X2[0]=X0[1]
  X3[0]=X0[2]

  for i in range(T-1):
      X1[i+1] = X1[i] + deltaT*(X1[i]*(sig[0]-rho[0][0]*X1[i]-rho[0][1]*X2[i]-rho[0][2]*X3[i]))
      X2[i+1] = X2[i] + deltaT*(X2[i]*(sig[1]-rho[1][0]*X1[i]-rho[1][1]*X2[i]-rho[1][2]*X3[i]))
      X3[i+1] = X3[i] + deltaT*(X3[i]*(sig[2]-rho[2][0]*X1[i]-rho[2][1]*X2[i]-rho[2][2]*X3[i]))

  y=np.vstack((X1,X2,X3))
  y = np.squeeze(y)
  y = y.T

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot(y[:, 0], y[:, 1], y[:, 2], color='blue', label='Trajectory')
  plt.title('LotkaVoltera trajectory in 3D')
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  plt.show(block=False)
  rep = input("\nDo you want to save this plot ? (Y/n): ")
  if rep.lower() == 'y':
    while True:
      name_file = input("Please, give a name to your plot: ")
      if not os.path.exists(f'{name_file}.png'):
        break
      else:
          rep2 = input(f"The file '{name_file}.png' already exists. Do you want to replace it ? (Y/n): ")
      if rep2.lower() == 'y':
        break
    plt.savefig(f'{name_file}.png')
    print("Plot has been successfully saved")

  plt.figure()
  plt.plot(times, y[:,0], label='X coordinate')
  plt.plot(times, y[:,1], label='Y coordinate')
  plt.plot(times, y[:,2], label='Z coordinate')

  figu = go.Figure(data=[go.Scatter3d(x=y[:, 0] , y=y[:,1], z=y[:,2], mode='markers', marker=dict(size=5))])
  figu.update_layout(scene=dict(xaxis_title='X1', yaxis_title='X2', zaxis_title='X3'))
  figu.update_layout(scene=dict(aspectmode="cube"))
  plt.title('LotkaVoltera 3 coordinates function of time')
  plt.xlabel('Time')
  plt.ylabel('X,Y,Z coordinates')
  plt.legend()
  plt.show(block=False)
  rep = input("\nDo you want to save this plot ? (Y/n): ")
  if rep.lower() == 'y':
    while True:
      name_file = input("Please, give a name to your plot: ")
      if not os.path.exists(f'{name_file}.png'):
        break
      else:
          rep2 = input(f"The file '{name_file}.png' already exists. Do you want to replace it ? (Y/n): ")
      if rep2.lower() == 'y':
        break
    plt.savefig(f'{name_file}.png')
    print("Plot has been successfully saved")

  recurrence.anim_traj(y)

  step = 0.001
  test2=1
  if method == 0:
    epsi = opti_epsi.epsi_entropy(y, step)
    test=1
  elif method == 1:
    epsi = opti_epsi.epsi_utility(y, step)
    test=1
  elif method == 2:
    entropy = opti_epsi.epsi_entropy(y, step)
    utility = opti_epsi.epsi_utility(y, step)
    epsi = (entropy+utility)/2
    test=1
  elif method == 4:
    test=0
  elif method == 5:
    epsi = opti_epsi.opti_epsi_phi(y, step, 1)
    test=1

  while test==0:
    epsilon = input("Please enter your epsilon value: ")
    epsi = float(epsilon)
    R = recurrence.rec_mat(y, epsi)
    recurrence.rec_plt(R)

    serie = symbolic_series.symbolic_serie(R,1)
    symbolic_series.colored_sym_serie(serie,y)
    symbolic_series.plot_col_traj(serie,y)
    recurrence.col_rec_plt(serie, R)

    C_alphabet_size, C_nbr_words, C_LZ = symbolic_series.complexity(serie,1)
    shuf = input("\nDo you want to analyse the complexity ? (Y/n): ")
    if shuf.lower() == 'y':
      symbolic_series.complexity_shuffle(y)

    ans = input("\nAre these results satisfying ? (Y/n): ")
    if ans.lower() == 'y':
      test=1
      test2=0

  if test == 1 and test2==1:
    if epsi != None:
      R = recurrence.rec_mat(y, epsi)
      recurrence.rec_plt(R)

      serie = symbolic_series.symbolic_serie(R,1)
      symbolic_series.colored_sym_serie(serie,y)
      symbolic_series.plot_col_traj(serie,y)
      recurrence.col_rec_plt(serie, R)

      C_alphabet_size, C_nbr_words, C_LZ = symbolic_series.complexity(serie,1)
    else :
      print("epsi is None")

  shuf = input("\nDo you want to analyse the complexity ? (Y/n): ")
  if shuf.lower() == 'y':
    nbr = input("How many tests ? ")
    symbolic_series.complexity_shuffle(y, count=int(nbr))





def demo_Lorenz(method):

  T=2000
  deltaT = 0.01
  times = np.linspace(0, deltaT*T, T)

  SIGMA = 10.0
  RHO = 28.0
  BETA = 8.0/3.0

  y = np.zeros((3,T))
  y0 = np.array([20, 5, -5]).flatten()
  y[:,0]=y0

  for i in range(T-1):
      a=np.array([-SIGMA*y[0,i]+SIGMA*y[1,i], (RHO-y[2,i])*y[0,i]-1*y[1,i], y[1,i]*y[0,i]-BETA*y[2,i]])
      b= deltaT
      y[:,i+1] = y[:,i] + b*a

  y = np.squeeze(y)
  y = y.T

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot(y[:, 0], y[:, 1], y[:, 2], color='blue', label='Trajectory')
  plt.title('Lorenz attractor trajectory in 3D')
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  plt.show(block=False)
  rep = input("\nDo you want to save this plot ? (Y/n): ")
  if rep.lower() == 'y':
    while True:
      name_file = input("Please, give a name to your plot: ")
      if not os.path.exists(f'{name_file}.png'):
        break
      else:
          rep2 = input(f"The file '{name_file}.png' already exists. Do you want to replace it ? (Y/n): ")
      if rep2.lower() == 'y':
        break
    plt.savefig(f'{name_file}.png')
    print("Plot has been successfully saved")

  plt.figure()
  plt.plot(times, y[:,0], label='X coordinate')
  plt.plot(times, y[:,1], label='Y coordinate')
  plt.plot(times, y[:,2], label='Z coordinate')

  figu = go.Figure(data=[go.Scatter3d(x=y[:, 0] , y=y[:,1], z=y[:,2], mode='markers', marker=dict(size=5))])
  figu.update_layout(scene=dict(xaxis_title='X1', yaxis_title='X2', zaxis_title='X3'))
  figu.update_layout(scene=dict(aspectmode="cube"))
  plt.title('Lorenz attractor 3 coordinates function of time')
  plt.xlabel('Time')
  plt.ylabel('X,Y,Z coordinates')
  plt.legend()
  plt.show(block=False)
  rep = input("\nDo you want to save this plot ? (Y/n): ")
  if rep.lower() == 'y':
    while True:
      name_file = input("Please, give a name to your plot: ")
      if not os.path.exists(f'{name_file}.png'):
        break
      else:
          rep2 = input(f"The file '{name_file}.png' already exists. Do you want to replace it ? (Y/n): ")
      if rep2.lower() == 'y':
        break
    plt.savefig(f'{name_file}.png')
    print("Plot has been successfully saved")

  recurrence.anim_traj(y)

  step = 0.1
  test2 = 1
  if method == 0:
    epsi = opti_epsi.epsi_entropy(y, step, 1)
    test=1
  elif method == 1:
    epsi = opti_epsi.epsi_utility(y, step, 1)
    test=1
  elif method == 2:
    entropy = opti_epsi.epsi_entropy(y, step, 1)
    utility = opti_epsi.epsi_utility(y, step, 1)
    epsi = (entropy+utility)/2
    test=1
  elif method == 4:
    test=0

  while test==0:
    epsilon = input("Please enter your epsilon value: ")
    epsi = float(epsilon)
    R = recurrence.rec_mat(y, epsi)
    recurrence.rec_plt(R)

    serie = symbolic_series.symbolic_serie(R, 1)
    symbolic_series.colored_sym_serie(serie,y)
    symbolic_series.plot_col_traj(serie,y)
    recurrence.col_rec_plt(serie, R)

    C_alphabet_size, C_nbr_words, C_LZ = symbolic_series.complexity(serie,1)

    ans = input("\nAre these results satisfying ? (Y/n): ")
    if ans.lower() == 'y':
      test=1
      test2=0

  if test == 1 and test2==1:
    R = recurrence.rec_mat(y, epsi)
    recurrence.rec_plt(R)

    serie = symbolic_series.symbolic_serie(R, 1)
    symbolic_series.colored_sym_serie(serie,y)
    symbolic_series.plot_col_traj(serie,y)
    recurrence.col_rec_plt(serie, R)

    C_alphabet_size, C_nbr_words, C_LZ = symbolic_series.complexity(serie,1)




def dataset(a, method):

  import joblib
  data = joblib.load("skeletons_labels.json")
  num_frames = len(data['vid_name'])
  diag = data['Diag']
  vidname = data['vid_name']

  video_name = []
  video_name_init = []
  video_name_diag = []

  name = vidname[0]
  video_name.append(name)
  video_name_init.append(0)
  video_name_diag.append(diag[0])
  counter = 0
  print("#%d vdeoo_name=%s video_name_init=%d"%(counter,video_name[counter],video_name_init[counter]))

  for i in range(num_frames-1):
      if vidname[i+1]!=vidname[i]:
          counter += 1
          video_name.append(vidname[i+1])
          video_name_init.append(i+1)
          video_name_diag.append(diag[i+1])
          print("#%d vdeoo_name=%s video_name_init=%d duration of previous video:%d diagnose:%d"%(\
              counter,\
              video_name[counter],video_name_init[counter],video_name_init[counter]-video_name_init[counter-1],\
              video_name_diag[counter]))

  video_initial = []
  video_duration = []
  video_num = counter+1
  video_initial.append(0)
  for k in range(video_num-1):
      if k>0:
          video_initial.append(video_name_init[k])
      video_duration.append(video_name_init[k+1]-video_name_init[k])
  video_duration.append(num_frames-video_name_init[video_num-1])

  video_list_joints = []
  k=3 ## number of video
  video_diagnosis = video_name_diag[k]
  initial = video_initial[k]
  duration = video_duration[k]
  video_list_joints.append(data['joints3D'][initial:initial+duration,:,:])
  print(video_list_joints[0][:,0,0])
  print(np.shape(video_list_joints))
  data_final = video_list_joints[0]
  #quit()

  shape__ = np.shape(video_list_joints)
  num_time = shape__[1]


  #t = np.array([0.1*np.ones(10)*i for i in range(100)]).flatten()
  t = np.linspace(0,num_time,num_time)
  b = np.zeros((num_time,3))

  num_joints_shown = 25
  x = np.zeros((num_joints_shown,num_time,3))
  joint = np.arange(0,0+num_joints_shown,1)
  #joint_num = range(num_joints_shown)#for k in range(num_joints_shown):
  fig = plt.figure(figsize=(8,8))
  for k in range(num_joints_shown):
      x[k,:,0]=data_final[:,joint[k],0]
      x[k,:,1]=data_final[:,joint[k],2]
      x[k,:,2]=data_final[:,joint[k],1]

  y = x[a,:,:]
  T=y.shape[0]
  times = np.linspace(0,T,T)

  plt.figure()
  plt.plot(times, y[:,0], color='red', label='X coordinate')
  plt.plot(times, y[:,1], color='blue', label = 'Y coordinate')
  plt.plot(times, y[:,2], color='green', label = 'Z coordinate')
  plt.title("Joint mouvement on each coordinate")
  plt.xlabel('Time')
  plt.ylabel('X,Y,Z coordinates')
  plt.legend()
  plt.show(block=False)
  rep = input("\nDo you want to save this plot ? (Y/n): ")
  if rep.lower() == 'y':
    while True:
      name_file = input("Please, give a name to your plot: ")
      if not os.path.exists(f'{name_file}.png'):
        break
      else:
          rep2 = input(f"The file '{name_file}.png' already exists. Do you want to replace it ? (Y/n): ")
      if rep2.lower() == 'y':
        break
    plt.savefig(f'{name_file}.png')
    print("Plot has been successfully saved")

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot(y[:, 0], y[:, 1], y[:, 2], color='blue', label='Trajectory')
  plt.title('Trajectory in 3D')
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  plt.show(block=False)
  rep = input("\nDo you want to save this plot ? (Y/n): ")
  if rep.lower() == 'y':
    while True:
      name_file = input("Please, give a name to your plot: ")
      if not os.path.exists(f'{name_file}.png'):
        break
      else:
          rep2 = input(f"The file '{name_file}.png' already exists. Do you want to replace it ? (Y/n): ")
      if rep2.lower() == 'y':
        break
    plt.savefig(f'{name_file}.png')
    print("Plot has been successfully saved")

  '''figu = go.Figure(data=[go.Scatter3d(x=y[:, 0] , y=y[:,1], z=y[:,2], mode='markers', marker=dict(size=5))])
  figu.update_layout(scene=dict(xaxis_title='X1', yaxis_title='X2', zaxis_title='X3'))
  figu.update_layout(scene=dict(aspectmode="cube"))
  plt.title('3 coordinates function of time')
  plt.xlabel('Time')
  plt.ylabel('X,Y,Z coordinates')
  plt.show(block=False)'''

  recurrence.anim_traj(y)

  step = 0.001
  test2=1
  if method == 0:
    epsi = opti_epsi.epsi_entropy(y, step,1)
    test=1
  elif method == 1:
    epsi = opti_epsi.epsi_utility(y, step,1)
    test=1
  elif method == 2:
    entropy = opti_epsi.epsi_entropy(y, step,1)
    utility = opti_epsi.epsi_utility(y, step,1)
    epsi = (entropy+utility)/2
    test=1
  elif method == 4:
    test=0

  while test==0:
    epsilon = input("Please enter your epsilon value: ")
    epsi = float(epsilon)
    R = recurrence.rec_mat(y, epsi)
    recurrence.rec_plt(R)

    serie = symbolic_series.symbolic_serie(R,1)
    symbolic_series.colored_sym_serie(serie,y)
    symbolic_series.plot_col_traj(serie,y)
    recurrence.col_rec_plt(serie, R)

    C_alphabet_size, C_nbr_words, C_LZ = symbolic_series.complexity(serie,1)

    ans = input("\nAre these results satisfying ? (Y/n): ")
    if ans.lower() == 'y':
      test=1
      test2=0

  if test == 1 and test2==1:
    R = recurrence.rec_mat(y, epsi)
    recurrence.rec_plt(R)

    serie = symbolic_series.symbolic_serie(R,1)
    symbolic_series.colored_sym_serie(serie,y)
    symbolic_series.plot_col_traj(serie,y)
    recurrence.col_rec_plt(serie, R)

    C_alphabet_size, C_nbr_words, C_LZ = symbolic_series.complexity(serie,1)

  shuf = input("\nDo you want to analyse the complexity ? (Y/n): ")
  if shuf.lower() == 'y':
    nbr = input("How many tests ? ")
    symbolic_series.complexity_shuffle(y, count=int(nbr))


def demo_square():
  y=np.array([[0,0,0], [0,0,0], [0,0,0.1], [0,0,0.2], [0,0,0.3], [0,0,0.5], [0,0,2], [0,0,3.5], [0,0,5], [0,0,8], [0,0,10], [0,1,10], [0,0.5,10.5], [0,1,9.5], [0,0.5,10], [0,1,10], [0,3,10], [0,5,10], [0,7,10], [0,9.8,10], [0,9.8,10], [0,9.9,10], [0,9.95,10], [0,10,10], [0,10.1,10], [0,10,10], [0,10,9.9], [0,10,9.8], [0,10,9.5], [0,10,7.5], [0,10,6], [0,10,4], [0,10,2], [0,10,0.5], [0,10,0.2], [0,10,0.1], [0,10,0.05], [0,10,0], [0,9.8,0], [0,9.5,0], [0,8,0], [0,5,0], [0,3,0], [0,0.5,0], [0,0,0], [0,0,0], [0,0,0.1], [0,0,0.2], [0,0,0.3], [0,0,0.5], [0,0,0.5], [0,0,0.2], [0,0,2], [0,0,3.5], [0,0,5], [0,0,8], [0,0,10], [0,1,10], [0,0.5,10.5], [0,1,9.5], [0,0.5,10], [0,1,10], [0,3,10], [0,5,11], [0,7,10], [0,9.5,10], [0,9.8,10], [0,9.9,10], [0,9.95,10], [0,10,10], [0,10.1,10], [0,10,10], [0,10,9.8], [0,10,9.5], [0,10,8], [0,10,6], [0,10,4], [0,10,2], [0,10,0.5], [0,10,0.2], [0,10,0.1], [0,10,0.05], [0,10,0], [0,9.8,0], [0,9.5,0], [0,8,0], [0,5,0], [0,3,0], [0,0.5,0], [0,0,0], [0,0,0], [0,0,0],[0,0,0], [0,0,0], [0,0,0.1], [0,0,0.2], [0,0,0.3], [0,0,0.5], [0,0,2], [0,0,3.5], [0,0,5]])
  times = np.linspace(0,y.shape[0]-1,y.shape[0])
  print(times)
  epsi = 1.3
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot(y[:, 0], y[:, 1], y[:, 2], marker='.', color='blue', label='Trajectory')
  plt.title('Square trajectory in 3D')
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  plt.show(block=False)
  rep = input("\nDo you want to save this plot ? (Y/n): ")
  if rep.lower() == 'y':
    while True:
      name_file = input("Please, give a name to your plot: ")
      if not os.path.exists(f'{name_file}.png'):
        break
      else:
          rep2 = input(f"The file '{name_file}.png' already exists. Do you want to replace it ? (Y/n): ")
      if rep2.lower() == 'y':
        break
    plt.savefig(f'{name_file}.png')
    print("Plot has been successfully saved")

  plt.figure()
  plt.plot(times, y[:,0], label='X coordinate')
  plt.plot(times, y[:,1], label='Y coordinate')
  plt.plot(times, y[:,2], label='Z coordinate')

  figu = go.Figure(data=[go.Scatter3d(x=y[:, 0] , y=y[:,1], z=y[:,2], mode='markers', marker=dict(size=5))])
  figu.update_layout(scene=dict(xaxis_title='X1', yaxis_title='X2', zaxis_title='X3'))
  figu.update_layout(scene=dict(aspectmode="cube"))
  plt.title('Square 3 coordinates function of time')
  plt.xlabel('Time')
  plt.ylabel('X,Y,Z coordinates')
  plt.legend()
  plt.show(block=False)
  rep = input("\nDo you want to save this plot ? (Y/n): ")
  if rep.lower() == 'y':
    while True:
      name_file = input("Please, give a name to your plot: ")
      if not os.path.exists(f'{name_file}.png'):
        break
      else:
          rep2 = input(f"The file '{name_file}.png' already exists. Do you want to replace it ? (Y/n): ")
      if rep2.lower() == 'y':
        break
    plt.savefig(f'{name_file}.png')
    print("Plot has been successfully saved")

  recurrence.anim_traj(y)

  R = recurrence.rec_mat(y, epsi)
  recurrence.rec_plt(R)

  serie = symbolic_series.symbolic_serie(R,1)
  symbolic_series.colored_sym_serie(serie,y)
  symbolic_series.plot_col_traj(serie,y)
  recurrence.col_rec_plt(serie, R)

  C_alphabet_size, C_nbr_words, C_LZ = symbolic_series.complexity(serie,1)

  shuf = input("\nDo you want to analyse the complexity ? (Y/n): ")
  if shuf.lower() == 'y':
    nbr = input("How many tests ? ")
    symbolic_series.complexity_shuffle(y, count=int(nbr))


















def demo_HD():

  X0 = np.array([1,0.17,0.01])
  sig =np.array([1,1.2,1.6])
  rho = np.array([[1, sig[0]/sig[1]+0.5, sig[0]/sig[2]-0.5],
        [sig[1]/sig[0]-0.5, 1, sig[1]/sig[2]+0.5],
        [sig[2]/sig[0]+0.5, sig[2]/sig[1]-0.5, 1]])
  #T = 500
  T=280
  deltaT = 0.1429
  X1 = np.zeros(T)
  X2 = np.zeros(T)
  X3 = np.zeros(T)
  times = np.linspace(0,deltaT*T, T)
  X1[0]=X0[0]
  X2[0]=X0[1]
  X3[0]=X0[2]

  for i in range(T-1):
      X1[i+1] = X1[i] + deltaT*(X1[i]*(sig[0]-rho[0][0]*X1[i]-rho[0][1]*X2[i]-rho[0][2]*X3[i]))
      X2[i+1] = X2[i] + deltaT*(X2[i]*(sig[1]-rho[1][0]*X1[i]-rho[1][1]*X2[i]-rho[1][2]*X3[i]))
      X3[i+1] = X3[i] + deltaT*(X3[i]*(sig[2]-rho[2][0]*X1[i]-rho[2][1]*X2[i]-rho[2][2]*X3[i]))

  y=np.vstack((X1,X2,X3))
  y = np.squeeze(y)
  y = y.T

  dossier = os.path.dirname(os.path.abspath(__file__))
  path_img1 = os.path.join(dossier, 'darkvador.jpg')
  path_img2 = os.path.join(dossier, 'luke.jpg')
  path_img3 = os.path.join(dossier, 'yoda.jpg')

  # trajectoire animée avec image

  image1 = plt.imread(path_img1)
  image2 = plt.imread(path_img2)
  image3 = plt.imread(path_img3)

  # Convertir les tableaux NumPy en tenseurs PyTorch
  image1_tensor = ToTensor()(image1)
  image2_tensor = ToTensor()(image2)
  image3_tensor = ToTensor()(image3)

  # Convertir le tenseur image3 en image PIL
  image3_pil = ToPILImage()(image3_tensor)

  # Redimensionner toutes les images à la même taille
  image_size = (200, 200)
  image1_res = TF.resize(image1_tensor, image_size)
  image2_res = TF.resize(image2_tensor, image_size)
  image3_res = TF.resize(image3_tensor, image_size)

  # Convertir les images en niveaux de gris
  image1_gray = image1_res[0, :, :]
  image2_gray = image2_res[0, :, :]
  image3_gray = image3_res[0, :, :]

  # Affichage
  fig = plt.figure(figsize=(15, 5))

  # Afficher les images en niveaux de gris
  fig.add_subplot(131); plt.imshow(ToPILImage()(image1_gray), cmap='gray')
  fig.add_subplot(132); plt.imshow(ToPILImage()(image2_gray), cmap='gray')
  fig.add_subplot(133); plt.imshow(ToPILImage()(image3_gray), cmap='gray')

  plt.show()

  s=np.zeros((X1.shape[0], *image_size))
  for i in range(X1.shape[0]):
    s[i] = (X1[i]*image1_gray + X2[i]*image2_gray + X3[i]*image3_gray)
  s_normalized = (s - np.min(s)) / (np.max(s) - np.min(s))

  fig1 = plt.figure()
  for i in range(X1.shape[0]):
    print(i,'/',X1.shape[0]-1)
    s_normalized_uint8 = (s_normalized[i] * 255).astype(np.uint8)
    plt.imshow(ToPILImage()(s_normalized_uint8), cmap='gray')
    plt.show()

  #Recurrence plot High Dimensions
  DistHD = np.zeros([s.shape[0],s.shape[0]])
  yHD = (s_normalized* 255).astype(np.uint8)

  for i in range(s.shape[0]):
    for j in range(s.shape[0]):
      DistHD[i, j] = np.linalg.norm(yHD[i] - yHD[j], 'fro')

  step = 10
  epsi = 0
  nbr_epsi = int(math.ceil(np.max(DistHD))/step)
  Entropy = np.empty(nbr_epsi)
  Epsi = np.empty(nbr_epsi)

  for q in range(nbr_epsi):
    epsi = epsi + step
    R = np.array(DistHD<epsi)
    R=R.astype(int)



    #------------------------------------------------------------------------------
    #Rewritting grammar
    Serie=np.zeros((R.shape[1]))

    for i in range(R.shape[1]):
      Serie[i]=i+1

    for i in range(R.shape[0]):
      Indx = np.where(R[R.shape[0]-1-i, :]!=0)
      Valmin = int(Serie[np.min(Indx[0])])
      for j in Indx[0]:
          if Valmin < Serie[j]:
            Serie[Serie==Serie[j]] = Valmin
    #-----------------------------------------------------------------------------
    #Writing zeros
    newSerie = np.array(Serie)
    for i in range(Serie.shape[0]):
      if i != 0 and i != Serie.shape[0]-1:
        if Serie[i-1]!=Serie[i] and Serie[i]!=Serie[i+1]:
          newSerie[i]=0
      if i==0:
        if Serie[i]!=Serie[i+1] and Serie[i]!=Serie[Serie.shape[0]-1]:
          newSerie[i]=0
      if i==Serie.shape[0]-1:
        if Serie[i]!=Serie[i-1] and Serie[i]!=Serie[0]:
          newSerie[i]=0

    #-----------------------------------------------------------------------------
    #Wrinting continuous numbers
    sort = 0
    Ser = np.sort(newSerie)
    S = np.unique(Ser)
    for i in S:
      newSerie = np.where(newSerie == i, sort, newSerie)
      sort = sort+1

    #----------------------------------------------------------------------------
    #Entropy
    Serie=newSerie
    Serie=Serie.astype(int)
    p = np.array(np.unique(Serie).shape[0])
    H=0

    occurrences = np.bincount(Serie)
    for valeur, nb_occurrences in enumerate(occurrences):
      if nb_occurrences > 0:
        pi = nb_occurrences/Serie.shape[0]
        H=H+pi*np.log2(pi)

    Hneg = -H
    Entropy[q]= Hneg
    Epsi[q] = epsi

  Hmax = np.max(Entropy)
  IndxHmax = np.argmax(Entropy)
  EpsiOpti = Epsi[IndxHmax]

  print('Entropy=',round(Hmax, 3),'     Epsilon=', round(EpsiOpti, 3))

  plt.plot(Epsi,Entropy)

  epsi = EpsiOpti
  #epsi=13000
  R = np.array(DistHD<epsi)

  #------------------------------------------------------------------------------
  #plot recurrence plot
  x, y1 = np.where(R == 1)
  fig=plt.figure()
  plt.scatter(x, y1, c='black', marker='o')

  #------------------------------------------------------------------------------
  #Rewritting grammar
  #R=np.tril(R)
  R=R.astype(int)
  print(R)
  print('')

  Serie=np.zeros((R.shape[1]))

  for i in range(R.shape[1]):
    Serie[i]=i+1

  for i in range(R.shape[0]):
    Indx = np.where(R[R.shape[0]-1-i, :]!=0)
    Valmin = int(Serie[np.min(Indx[0])])
    for j in Indx[0]:
        if Valmin < Serie[j]:
          Serie[Serie==Serie[j]] = Valmin

  #--------------------------------------------------------------------------------
  #Writing zeros

  c=0

  newSerie = np.array(Serie)
  for i in range(Serie.shape[0]):
    if i != 0 and i != Serie.shape[0]-1:
      if Serie[i-1]!=Serie[i] and Serie[i]!=Serie[i+1]:
        newSerie[i]=0

    if i==0:
      if Serie[i]!=Serie[i+1] and Serie[i]!=Serie[Serie.shape[0]-1]:
        newSerie[i]=0
    if i==Serie.shape[0]-1:
      if Serie[i]!=Serie[i-1] and Serie[i]!=Serie[0]:
        newSerie[i]=0

  #------------------------------------------------------------------------------
  #Sorting

  sort = 0
  Ser = np.sort(newSerie)
  S = np.unique(Ser)
  for i in S:
    newSerie = np.where(newSerie == i, sort, newSerie)
    sort = sort+1
  #
  print('Final HD symbolic series = ', newSerie)

  #plot colored serie

  position = 0
  palette = ['red', 'blue', 'yellow', 'green', 'purple', 'orange', 'black', 'pink', 'brown', 'gray', 'turquoise', 'indigo', 'beige', 'olive', 'cyan', 'magenta', 'gold', 'silver', 'coral', 'lavender', 'chartreuse', 'orangered', 'aquamarine', 'skyblue', 'pumpkin', 'emerald']

  fig, ax = plt.subplots()
  for couleur in newSerie:
      ax.barh(0, 1, color=palette[int(couleur)], height=0.2, left=position)
      position += 1

  ax.set_ylim(-0.5, 0.5)
  ax.axis('off')
  plt.show()




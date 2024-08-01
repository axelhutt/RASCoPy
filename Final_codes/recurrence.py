# -*- coding: utf-8 -*-
"""Recurrence.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XtOsL9eNxHAjmHGAP_U27d5nOFWac70N
"""

import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

def rec_mat(y, epsilon, visu=None, back_file=None):
  y = np.array(y)
  D = cdist(y, y, 'euclidean')
  R = np.array(D<epsilon)
  R=R.astype(int)

  if visu is not None:
    print('\n-----------Recurrence matrix-----------')
    print(R)
    #R=np.tril(R)

    if back_file is not None:
      with open(back_file, 'a') as fichier:
        fichier.write('\n-----------Recurrence matrix-----------' + '\n')
        np.savetxt(fichier, R, fmt='%d', delimiter='\t', newline='\n', header='', footer='', comments='')
      print(f"Recurrence matrix has been successfully saved in {back_file}")
    else:
      while(True):
        rep = input("\nDo you want to save this recurrence matrix ? (Y/n): ")
        if rep.lower() == 'y':
          while True:
            name_file = input("Please, give a name to your backup file: ")
            if not os.path.exists(f'{name_file}'):
              a=0
              break
            else:
              rep3 = input(f"The file '{name_file}' already exists. Do you want to write your Recurrence matrix inside? (Y/n): ")
              if rep3.lower() == 'y':
                a=1
                break
              else:
                rep4 = input(f"Do you want to replace '{name_file}'? (Y/n): ")
                if rep4.lower() == 'y':
                  a=2
                  break
          break
        elif rep.lower() == 'n':
          break

        if a == 0 or a == 2:
          with open(name_file, 'w') as fichier:
            fichier.write('\n-----------Recurrence matrix-----------' + '\n')
            np.savetxt(fichier, R, fmt='%d', delimiter='\t', newline='\n', header='', footer='', comments='')
        if a == 1:
          with open(name_file, 'a') as fichier:
            fichier.write('\n-----------Recurrence matrix-----------' + '\n')
            np.savetxt(fichier, R, fmt='%d', delimiter='\t', newline='\n', header='', footer='', comments='')

        print(f"Recurrence matrix has been successfully saved in {name_file}")


  return R

def rec_plt(R):
  R = np.array(R)
  x, y = np.where(R == 1)
  print('\n-----------Recurrence plot-----------')
  fig=plt.figure()
  plt.scatter(x, y, c='black', marker='.')
  plt.title("Recurrence plot")
  plt.xlabel('Samples')
  plt.ylabel('Samples')
  plt.show(block=False)

  while(True):
    rep = input("Do you want to save this recurrence plot ? (Y/n): ")
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
      print("Recurrence plot has been successfully saved")
      break
    elif rep.lower() == 'n':
      break


def anim_traj(y):

  y = np.array(y)

  fig = plt.figure()
  plt.axis('off')
  plt.title("Evolution of trajectory over time")
  if y.shape[1] == 3:
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter([], [], [], c='r', marker='o')
    ax.plot(y[:, 0], y[:, 1], y[:, 2], c='b')

    ax.set_xlim(np.min(y[:, 0]), np.max(y[:, 0]))
    ax.set_ylim(np.min(y[:, 1]), np.max(y[:, 1]))
    ax.set_zlim(np.min(y[:, 2]), np.max(y[:, 2]))

    def init():
        sc._offsets3d = (y[1, 0], y[1, 1], y[1, 2])
        return sc,

    def update(frame, sc, ax, text):
      label1 = str(frame)
      text.set_text('Frame: %s' % label1) 
      alphas = 1.0 - np.arange(len(y)) / len(y)
      sc._offsets3d = (y[frame-4:frame, 0], y[frame-4:frame, 1], y[frame-4:frame, 2])
      sc.set_array(alphas[frame-4:frame])
      return sc, text

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_axis_on()
    text = ax.text2D(0.5, 0.95, '', transform=ax.transAxes, fontsize=9, ha='center', va='center')

  if y.shape[1] == 2:
    ax = fig.gca()
    sc = ax.scatter(y[:, 0], y[:, 1], c='r', marker='o')
    ax.plot(y[:, 0], y[:, 1], c='b')

    ax.set_xlim(np.min(y[:, 0]), np.max(y[:, 0]))
    ax.set_ylim(np.min(y[:, 1]), np.max(y[:, 1]))

    def init():
        sc.set_offsets(np.c_[y[1, 0], y[1, 1]])
        return sc,

    def update(frame, sc, ax, text):
      label1 = str(frame)
      text.set_text('Frame: %s' % label1) 
      alphas = 1.0 - np.arange(len(y)) / len(y)
      sc.set_offsets(np.c_[y[frame-4:frame+1, 0], y[frame-4:frame+1, 1]])
      sc.set_array(alphas[frame-4:frame+1])
      return sc, text

    ax.set_xlabel('Coordinate 1')
    ax.set_ylabel('Coordinate 2')
    text = ax.text(0.5, 0.95, '', transform=ax.transAxes, fontsize=9, ha='center', va='center')



  ani = animation.FuncAnimation(fig, update, frames=range(len(y)), init_func=init, fargs=(sc, ax, text), interval=100)
  plt.tight_layout()
  plt.show(block=False)

  
  while(True):
    rep = input("\nDo you want to save this animation ? (Y/n): ")
    if rep.lower() == 'y':
      while True:
          name_file = input("Please, give a name to your animation: ")
          if not os.path.exists(f'{name_file}'):
              break
          else:
              rep2 = input(f"The file '{name_file}' already exists. Do you want to replace it ? (Y/n)")
              if rep2.lower() == 'y':
                  break
      ani.save(f'{name_file}.gif', writer='pillow', fps=30)
      print(f"{name_file} has been successfully saved")
      break
    elif rep.lower() == 'n':
      break


def col_rec_plt(serie, R):
  serie = np.array(serie)
  R = np.array(R)
  x, y = np.where(R == 1)

  palette = ['red', 'blue', 'yellow', 'green', 'purple', 'orange', 'black', 'pink', 'brown', 'gray', 'turquoise', 'indigo', 'beige', 'olive', 'cyan', 'magenta', 'gold', 'silver', 'coral', 'lavender', 'chartreuse', 'orangered', 'aquamarine', 'skyblue', 'pumpkin', 'emerald']

  if len(palette) > np.max(serie):
    print('\n-----------Colored Recurrence plot-----------')
    fig=plt.figure()
    plt.title("Colored recurrence plot")
    plt.xlabel('Samples')
    plt.ylabel('Samples')
    plt.text(int(R.shape[0])/2, int(R.shape[0]-1),"Red: transient state / Other colors: metastable states", fontsize=9, ha='center', va='center')
    for row, col in zip(x, y):
          if row>=col:
             plt.scatter(row, col, marker='.', color=palette[int(serie[row])])
          if col>row:
            plt.scatter(row, col, marker='.', color=palette[int(serie[col])]) 
    plt.show(block=False)
    while(True):
      rep = input("Do you want to save this colored recurrence plot ? (Y/n): ")
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
        print("Colored recurrence plot has been successfully saved")
        break
      elif rep.lower() == 'n':
        break
  else:
     print("Your data is too complex to plot the colored recurrence plot")
# -*- coding: utf-8 -*-
"""Symbolic_series.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nqBuBfswdkOvgyHHWYO6FnsTOp-mgbdu
"""

import numpy as np
import matplotlib.pyplot as plt
from RASCoPy import recurrence
import os

def symbolic_serie(R):

  #------------------------------------------------------------------------------
  #Rewriting grammar
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
  #Writing continuous number's sequence
  sort = 0
  Ser = np.sort(newSerie)
  S = np.unique(Ser)
  for i in S:
    newSerie = np.where(newSerie == i, sort, newSerie)
    sort = sort+1

  print('\n-----------Symbolic serie-----------')
  print(newSerie)

  rep = input("Do you want to save this symbolic series ? (Y/n): ")
  if rep.lower() == 'y':
    while True:
      name_file = input("Please, give a name to your backup file: ")
      if not os.path.exists(f'{name_file}'):
        a=0
        break
      else:
        rep3 = input(f"The file '{name_file}' already exists. Do you want to write your Symbolic serie inside? (Y/n): ")
        if rep3.lower() == 'y':
          a=1
          break
        else:
          rep4 = input(f"Do you want to replace '{name_file}'? (Y/n): ")
          if rep4.lower() == 'y':
            a=2
            break

    if a == 0 or a == 2:
      with open(name_file, 'w') as fichier:
        fichier.write('\n-----------Symbolic serie-----------' + '\n')
        np.savetxt(fichier, newSerie, fmt='%d', delimiter='\t', newline='\n', header='', footer='', comments='')
    if a == 1:
      with open(name_file, 'a') as fichier:
        fichier.write('\n-----------Symbolic serie-----------' + '\n')
        np.savetxt(fichier, newSerie, fmt='%d', delimiter='\t', newline='\n', header='', footer='', comments='')

    print(f"Symbolic serie has been successfully saved in {name_file}")


  return newSerie

def colored_sym_serie(serie,y):
  y = np.array(y)
  position = 0
  palette = ['red', 'blue', 'yellow', 'green', 'purple', 'orange', 'black', 'pink', 'brown', 'gray', 'turquoise', 'indigo', 'beige', 'olive', 'cyan', 'magenta', 'gold', 'silver', 'coral', 'lavender', 'chartreuse', 'orangered', 'aquamarine', 'skyblue', 'pumpkin', 'emerald']
  if len(palette) > np.max(serie) :
    fig, ax = plt.subplots()
    for couleur in serie:
        ax.barh(0, 1, color=palette[int(couleur)], height=0.2, left=position)
        position += 1


    ax.set_ylim(-0.5, 0.5)
    ax.axis('off')
    plt.show(block=False)

    rep = input("Do you want to save this colored symbolic serie plot ? (Y/n): ")
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
      print("Colored symbolic serie plot has been successfully saved")
  else:
    print("Your data is too complexe to plot the colored symbolic serie")

def plot_col_traj(serie,y):
  y = np.array(y)

  palette = ['red', 'blue', 'yellow', 'green', 'purple', 'orange', 'black', 'pink', 'brown', 'gray', 'turquoise', 'indigo', 'beige', 'olive', 'cyan', 'magenta', 'gold', 'silver', 'coral', 'lavender', 'chartreuse', 'orangered', 'aquamarine', 'skyblue', 'pumpkin', 'emerald']
  if len(palette) > np.max(serie) :
    figure = plt.figure()
    if y.shape[1] == 3 :
      ax = figure.add_subplot(111, projection='3d')
      for i in range(serie.shape[0]):
        ax.scatter(y[i, 0],y[i,1],y[i,2], color=palette[int(serie[i])], marker='o')
    elif y.shape[1] == 2:
      for i in range(serie.shape[0]):
        plt.scatter(y[i,0], y[i,1], color=palette[int(serie[i])], marker = 'o')
    elif y.shape[1] == 1:
      for i in range(serie.shape[0]):
        plt.scatter(y[i], 0, color=palette[int(serie[i])], marker='o')
    plt.show(block=False)
  else :
    print("Your data is too complexe to color a trajectory")

  rep = input("\nDo you want to save this colored trajectory ? (Y/n): ")
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
    print("Colored trajectory has been successfully saved")

def complexity(serie):
    # Alphabet size
    C_alphabet_size = np.max(serie)+1

    print('\n------------- Complexity with the alphabet size method -------------\n')
    print('Complexity alphabet size = ' + str(C_alphabet_size))

    # Number of words
    a = 0
    b = 0
    W_nw = []

    for i in range(len(serie)-1):
        w = ''
        if serie[i] != serie[i+1]:
            for j in range(a, i, 1):
                w = w + str(serie[j])
            W_nw.append(w)
            b = b + 1
            a = i + 1
    unique = set(W_nw)
    C_nbr_words = len(unique)
    print('\n------------- Complexity with the number of words method -------------\n')
    print('Complexity number of words = ' + str(C_nbr_words))

    # Lempel-Ziv

    def pattern_in_serie(w, s):
      w_str = ''
      s_str = ''
      for a in w:
        w_str=w_str+str(a)
      for b in s:
        s_str = s_str+str(b)
      return s_str in w_str

    C_LZ = 0
    i = 0
    W_LZ = []
    W_LZ.append(serie[i])
    i = 1
    while i < len(serie):
        j = i
        Bool = pattern_in_serie(W_LZ[:j], str(serie[j]))
        if Bool == False:
            W_LZ.append(str(serie[i]))
            i = i + 1
        else:
            while Bool == True:
                j = j + 1
                if j >= len(serie):
                    break
                Bool = pattern_in_serie(W_LZ[:j], serie[i:j + 1])
            W_LZ.append(''.join(map(str, serie[i:j + 1])))
            i = j + 1

    C_LZ = len(W_LZ)

    print('\n------------- Complexity with the Lempel-Ziv method -------------\n')
    print('Complexity Lempel-Ziv = ' + str(C_LZ))
    print('\n')

    rep = input("Do you want to save this complexity values ? (Y/n): ")
    if rep.lower() == 'y':
      while True:
        name_file = input("Please, give a name to your backup file: ")
        if not os.path.exists(f'{name_file}'):
          a=0
          break
        else:
          rep3 = input(f"The file '{name_file}' already exists. Do you want to write your complexity values inside? (Y/n): ")
          if rep3.lower() == 'y':
            a=1
            break
          else:
            rep4 = input(f"Do you want to replace '{name_file}'? (Y/n): ")
            if rep4.lower() == 'y':
              a=2
              break

      if a == 0 or a == 2:
        with open(name_file, 'w') as fichier:
          fichier.write('\n------------- Complexity with the alphabet size method -------------\n')
          fichier.write('Complexity alphabet size = ' + str(C_alphabet_size))
          fichier.write('\n------------- Complexity with the number of words method -------------\n')
          fichier.write('Complexity number of words = ' + str(C_nbr_words))
          fichier.write('\n------------- Complexity with the Lempel-Ziv method -------------\n')
          fichier.write('Complexity Lempel-Ziv = ' + str(C_LZ))
      if a == 1:
        with open(name_file, 'a') as fichier:
          fichier.write('\n------------- Complexity with the alphabet size method -------------\n')
          fichier.write('Complexity alphabet size = ' + str(C_alphabet_size))
          fichier.write('\n------------- Complexity with the number of words method -------------\n')
          fichier.write('Complexity number of words = ' + str(C_nbr_words))
          fichier.write('\n------------- Complexity with the Lempel-Ziv method -------------\n')
          fichier.write('Complexity Lempel-Ziv = ' + str(C_LZ))

      print(f"Complexity values have been successfully saved in {name_file}")

    return C_alphabet_size, C_nbr_words, C_LZ
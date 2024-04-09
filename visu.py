"""import joblib
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from rascopy import recurrence

data = joblib.load("data_20-01.json")
if all(isinstance(valeur, dict) for valeur in data.values()):
    while(True):
        videoname = input("\nEnter the name of the video: ")
        if videoname in data:
            break
        else:
            print("Video doesn't exist")
    vidname = data[videoname]
    data_final=np.array(vidname['joints3D'])
    x = np.zeros((data_final.shape[1], data_final.shape[0], data_final.shape[2]))
    for k in range(data_final.shape[1]):
        #x[k,:,0]=data_final[:,k,2]
        #x[k,:,1]=data_final[:,k,0]
        #x[k,:,2]=data_final[:,k,1]
        x[k,:,0]=data_final[:,k,0]
        x[k,:,1]=data_final[:,k,2]
        x[k,:,2]=data_final[:,k,1]
    x[:, :, 2] *= -1
    #x[:, :, 0] *= -1

    y = x[18,:,1:3]
    y = np.array(y)

    #recurrence.anim_traj(y)

    fig = plt.figure()
    plt.axis('off')
    plt.title("Evolution of trajectory over time")
    

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
            sc.set_offsets(np.c_[y[frame:frame+1, 0], y[frame:frame+1, 1]])
            sc.set_array(alphas[frame:frame+1])
            return sc, text

        ax.set_xlabel('Coordinate 1')
        ax.set_ylabel('Coordinate 2')
        text = ax.text(0.5, 0.95, '', transform=ax.transAxes, fontsize=9, ha='center', va='center')



    ani = animation.FuncAnimation(fig, update, frames=range(len(y)), init_func=init, fargs=(sc, ax, text), interval=100)
    plt.tight_layout()
    plt.show()"""

import joblib
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

data = joblib.load("data_20-01.json")
if all(isinstance(valeur, dict) for valeur in data.values()):
    while(True):
        videoname = input("\nEnter the name of the video: ")
        if videoname in data:
            break
        else:
            print("Video doesn't exist")
    vidname = data[videoname]
    data_final = np.array(vidname['joints3D'])
    x = np.zeros((data_final.shape[1], data_final.shape[0], data_final.shape[2]))
    for k in range(data_final.shape[1]):
        x[k,:,0]=data_final[:,k,2]
        x[k,:,1]=data_final[:,k,0]
        x[k,:,2]=data_final[:,k,1]
    x[:, :, 2] *= -1
    x[:, :, 1] *= -1
    zeroz = min(x[19,:,2])
    y = x[:, :, :]
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            y[i,j,2] = y[i,j,2]-zeroz

    fig = plt.figure()
    plt.axis('off')
    plt.title("Evolution of trajectory over time")

    if y.shape[2] == 3:  # Assuming 25 points in each frame
        ax = fig.add_subplot(111, projection='3d')
        colors = ['k', 'k', 'k', 'k', 'r','r','r','r','g', 'g', 'g', 'g', 'orange','orange','orange','pink','b','b','b','purple','k','brown','brown','y','y']
        sc = ax.scatter([], [], [], marker='o')

        def init():
            sc._offsets3d = (y[0, :, 0], y[0, :, 1], y[0, :, 2])
            return sc,

        def update(frame, sc, ax):
            label1 = str(frame)
            text.set_text('Frame: %s' % label1) 
            alphas = 1.0 - np.arange(len(y)) / len(y)
            sc._offsets3d = (y[:, frame, 0], y[:, frame, 1], y[:, frame, 2])
            sc.set_color(colors)
            return sc,
    
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_axis_on()
        ax.set_xlim(-0.5, 1)  
        ax.set_ylim(-0.5,0.5)  
        ax.set_zlim(0, 2)  
        text = ax.text2D(0.5, 0.95, '', transform=ax.transAxes, fontsize=9, ha='center', va='center')

    ani = animation.FuncAnimation(fig, update, frames=range(y.shape[1]), init_func=init, fargs=(sc, ax), interval=100)
    plt.tight_layout()
    plt.show()

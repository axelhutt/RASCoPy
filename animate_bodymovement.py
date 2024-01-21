import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation



import joblib
data = joblib.load("./data_diwei_2/skeletons_labels.json")
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
    if k%10 == 1:
        kk = k+1
        ax = plt.subplot(1,3,1)
        plt.plot(range(num_time),x[k,:,0])
        ax = plt.subplot(1,3,2)
        plt.plot(range(num_time),x[k,:,1])
        ax = plt.subplot(1,3,3)
        plt.plot(range(num_time),x[k,:,2])
    
#joint1 = 1
#x[0,:,0]=data_final[:,joint1,0]
#x[0,:,1]=data_final[:,joint1,1]
#x[0,:,2]=data_final[:,joint1,2]
#joint2 = 2
#x[1,:,0]=data_final[:,joint2,0]
#x[1,:,1]=data_final[:,joint2,1]
#x[1,:,2]=data_final[:,joint2,2]

xmax = -1000000
xmin = 10000000
ymax = -1000000
ymin = 10000000
zmax = -1000000
zmin = 10000000
for k in range(num_joints_shown):
    if xmax<np.max(x[k,:,0]):
        xmax = np.max(x[k,:,0])
    if xmin>np.min(x[k,:,0]):
        xmin = np.min(x[k,:,0])
    if ymax<np.max(x[k,:,1]):
        ymax = np.max(x[k,:,1])
    if ymin>np.min(x[k,:,1]):
        ymin = np.min(x[k,:,1])
    if zmax<np.max(x[k,:,2]):
        zmax = np.max(x[k,:,2])
    if zmin>np.min(x[k,:,2]):
        zmin = np.min(x[k,:,2])
#       
#xmax=max(np.max(x[0,:,0]),np.max(x[1,:,0]))
#ymax=max(np.max(x[0,:,1]),np.max(x[1,:,1]))
#zmax=max(np.max(x[0,:,2]),np.max(x[1,:,2]))
#xmin=min(np.min(x[0,:,0]),np.min(x[1,:,0]))
#ymin=min(np.min(x[0,:,1]),np.min(x[1,:,1]))
#zmin=min(np.min(x[0,:,2]),np.min(x[1,:,2]))


df_list = []
sc_list = []
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
label = 'subject with diagnosis #%d'%video_diagnosis
title = ax.set_title(label)
for k in range(num_joints_shown):
    df_list.append(pd.DataFrame(x[k,:,:], columns=["x","y","z"]))
    sc_list.append(ax.scatter([],[],[],alpha=0.5))

def update(i):
    range_=1
    label1 = label+'  time step:%d'%i
    title.set_text(label1)
    for k in range(num_joints_shown):
        df = df_list[k]
        sc = sc_list[k]
        if i>range_:
            sc._offsets3d = (df.x.values[i-range_:i], df.y.values[i-range_:i], df.z.values[i-range_:i])
        else:
            sc._offsets3d = (df.x.values[i-range_:i], df.y.values[i-range_:i], df.z.values[i-range_:i])
                
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)
ax.set_zlim(zmin,zmax)

ani = matplotlib.animation.FuncAnimation(fig, update, frames=len(df_list[0]), interval=50)

plt.tight_layout()
plt.show()


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
import sys
sys.path.insert(0, os.getcwd())
import os.path as osp
import joblib
import shutil
import argparse
import subprocess
from mpl_toolkits.mplot3d import Axes3D
import cv2
import pandas as pd

def get_score():
  return [
    'Normal: No problems.',
    'Slight: Independent walking with minor gait impairment.',
    'Mild: Independent walking but with substantial gait impairment.',
    'Moderate: Requires an assistance device for safe walking (walking stick, walker) but not a person. '
  ]

def get_diagnosis():
  return [
    'Healthy',
    'Mild Dementia with Lewy Bodies',
    'Mild Alzheimer’s Disease',
    'Severe Dementia with Lewy Bodies',
    'Severe Alzheimer’s Disease'
  ]


def get_kinectv2_joint_names():
    return [
        'hip',             # 0
        'Spine (H36M)',    # 1
        'neck',            # 2
        'Head (H36M)',     # 3
        'lshoulder',       # 4
        'lelbow',          # 5
        'lwrist',          # 6
        'leftHand',        # 7
        'rshoulder',       # 8
        'relbow',          # 9
        'rwrist',          # 10
        'rightHand',       # 11
        'lhip (SMPL)',     # 12
        'lknee',           # 13
        'lankle',          # 14
        'leftFoot',        # 15
        'rhip (SMPL)',     # 16
        'rknee',           # 17
        'rankle',          # 18
        'rightFoot',       # 19
        'thorax',          # 20
        'leftHandTip',     # 21
        'leftThumb',       # 22
        'rightHandTip',    # 23
        'rightThumb',      # 24
    ]

def get_kinectv2_skeleton():
  # left/right alternatively
  return  np.array(
    [
      [0 , 1],[20, 2],[1 ,20],[2 , 3], # trunk
      [20, 4],[20, 8],[4 , 5],[8 , 9],[5 , 6],[9 ,10], # upper body
      [6 , 7],[10,11],[7 ,21],[11,23],[6 ,22],[10,24], # hands
      [0 ,12],[0 ,16],[12,13],[16,17],[13,14],[17,18], # lower body
      [14,15],[18,19], # feet
    ]
  )
    

def visualize():
    "Visualize the reconstructed skeleton together with the bbox"
    data = joblib.load('data_20-01.json')
    print('downloaded')
    vid_file_format = os.path.join('videos', '{:s}.mp4')
    out_dir = './tmp'
    os.makedirs(out_dir, exist_ok=True)
    # get the skeleton hierarchy for plotting
    skeleton = get_kinectv2_skeleton()
    # color setting
    rcolor = (np.array([215, 48, 39])/255).tolist() # red for right body parts
    lcolor = (np.array([77, 146, 33])/255).tolist() # green for left body parts
    bcolor = [252, 141, 89] # light blue for the bbox
    # enumerate the videos
    for vname, values in data.items():
        bboxes = values['bbox']
        joints3d = values['joints3D']
        vid_file = vid_file_format.format(vname)
        # extract frames from video
        img_folder = os.path.join(out_dir, vname)
        if os.path.isdir(img_folder):
            shutil.rmtree(img_folder)
        os.makedirs(img_folder, exist_ok=True)
        
        # Use Windows Media Player to read the video and extract frames
        command = ['wmic', 'process', 'call', 'create', 'wmplayer', vid_file]
        subprocess.call(command)
        # Wait for the video to finish playing
        input("Press Enter when the video is done playing...")
        # Take screenshots using wmic
        command = ['wmic', 'path', 'Win32_VideoController', 'get', 'DeviceID']
        output = subprocess.check_output(command, universal_newlines=True)
        device_id = output.strip().split('\n')[-1].strip()
        command = ['wmic', 'path', 'Win32_DisplayConfiguration', 'where', f'DeviceID={device_id}', 'call', 'save', 'C:\\Users\\gauth\\Images\\%06d.png']
        subprocess.call(command)
        
        img_list = sorted([os.path.join('C:\\Users\\gauth\\Images', x) for x in os.listdir('C:\\Users\\gauth\\Images') if x.endswith('.png')])
        assert len(img_list) == len(joints3d)
        # initialize the figure for plotting
        fig = plt.figure(f'{vname}', figsize=(10,5), dpi=300)
        ax_img = fig.add_subplot(1,2,1)
        ax_3d = fig.add_subplot(1,2,2, projection='3d')

        for imgpath, joints, bbox in zip(img_list, joints3d, bboxes):
            ax_img.clear()
            ax_img.set_axis_off()
            # add bbox onto the frame
            img = cv2.imread(imgpath)
            img = cv2.rectangle(img, (int(bbox[0]-bbox[2]/2), int(bbox[1]-bbox[2]/2)), \
                    (int(bbox[0]+bbox[2]/2), int(bbox[1]+bbox[2]/2)), bcolor, 2)
            ax_img.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), aspect='equal')
            ax_img.set_title('Press any key to quit ...')
            fig.canvas.draw()
            ax_3d.clear()
            ax_3d.view_init(elev=-77, azim=-90) # set the view angle
            ax_3d.set_xlim3d([-1., 1.])
            ax_3d.set_ylim3d([-1., 1.])
            ax_3d.set_zlim3d([-1., 1.])
            ax_3d.set_xlabel('x')
            ax_3d.set_ylabel('y')
            ax_3d.set_zlabel('z')
            for i,(j1,j2) in enumerate(skeleton):
                color = lcolor if i % 2 == 0 else rcolor
                x, y, z = [joints[[j1,j2], c] for c in range(3)]
                ax_3d.plot(x, y, z, lw=2, c=color)
            fig.canvas.draw()
            plt.show(block=False)
            # REMARK: set block=True to enable the interactive mode
            # in this mode, rotate the 3D skeleton on the right for better visualization
            if plt.waitforbuttonpress(0.01):
                break
        
        plt.close(fig)
        
        shutil.rmtree(img_folder)

def open_json(data=None, videoname=None):
  di=0
  if data == None and videoname == None:
    while(True):
      name = input("\nEnter the name of the json: ")
      if os.path.exists(name+".json"):
        break
      else:
        print("json doesn't exist")
    data = joblib.load(name+".json")

    if all(isinstance(valeur, dict) for valeur in data.values()):
      for vname, video in data.items():
        if 'diag' in video:
          print(vname, ":", video['sex'], " --> ", video['diag'], " | length: ", video['joints3D'].shape[0])
          di=1
      while(True):
        videoname = input("\nEnter the name of the video: ")
        if videoname in data:
          break
        else:
          print("Video doesn't exist")
      vidname = data[videoname]
      #number_frame = input("\nHow many frames to analyze ? ")
      #number_frame=int(number_frame)
      data_final=np.array(vidname['joints3D'])
      x = np.zeros((data_final.shape[1], data_final.shape[0], data_final.shape[2]))
      for k in range(data_final.shape[1]):
        x[k,:,0]=data_final[:,k,2]
        x[k,:,1]=data_final[:,k,0]
        x[k,:,2]=data_final[:,k,1]
      x[:, :, 2] *= -1
      x[:, :, 1] *= -1
      if di==1:
        diagnosis = vidname['diag']
        sex = vidname['sex']
      else:
        donnees = pd.read_excel('label_info_120.xlsx')
        ligne = donnees.loc[donnees['vidname'] == videoname]
        scor = ligne['score'].values[0]
        diagno = ligne['diag'].values[0]
        get_scor = get_score()
        get_diag = get_diagnosis()
        score_disease = get_scor[scor]
        diag_disease = get_diag[diagno]
        diagnosis = diag_disease+' -- '+score_disease
        print(videoname, " : " , diagnosis)
        print("length: ", x.shape[1])
        sex = 'Ukwn'

      print("\nData extracted")

    elif all(not isinstance(valeur, dict) for valeur in data.values()):
      videoname = input("\nEnter the name of the video: ")
      num_frames = len(data[videoname])
      if "diag" in data.keys() or "Diag" in data.keys():
        diag = data["diag"] if "diag" in data.keys() else data["Diag"]
      vidname = data[videoname]

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
        x[k,:,0]=data_final[:,k,2]
        x[k,:,1]=data_final[:,k,0]
        x[k,:,2]=data_final[:,k,1]
      x[:, :, 2] *= -1
      x[:, :, 1] *= -1
      print("Data extracted")

    return x, diagnosis, sex, name, videoname, di

  else:
    if all(isinstance(valeur, dict) for valeur in data.values()):
      for vname, video in data.items():
        if 'diag' in video:
          di=1
          print(vname, ":", video['sex'], " --> ", video['diag'], " | lenghth: ", video['joints3D'].shape[0])
      vidname = data[videoname]
      data_final=np.array(vidname['joints3D'])
      x = np.zeros((data_final.shape[1], data_final.shape[0], data_final.shape[2]))
      for k in range(data_final.shape[1]):
        x[k,:,0]=data_final[:,k,2]
        x[k,:,1]=data_final[:,k,0]
        x[k,:,2]=data_final[:,k,1]
      x[:, :, 2] *= -1
      x[:, :, 1] *= -1
      if di==1:
        diagnosis = vidname['diag']
        sex = vidname['sex']
      else:
        donnees = pd.read_excel('label_info_120.xlsx')
        ligne = donnees.loc[donnees['vidname'] == videoname]
        scor = ligne['score'].values[0]
        diagno = ligne['diag'].values[0]
        get_scor = get_score()
        get_diag = get_diagnosis()
        score_disease = get_scor[scor]
        diag_disease = get_diag[diagno]
        diagnosis = diag_disease+' -- '+score_disease
        print(videoname, " : " , diagnosis)
        print("length: ", x.shape[1])
        sex = 'Ukwn'

      print("Data extracted")

    elif all(not isinstance(valeur, dict) for valeur in data.values()):
      videoname = input("\nEnter the name of the video: ")
      num_frames = len(data[videoname])
      if "diag" in data.keys() or "Diag" in data.keys():
        diag = data["diag"] if "diag" in data.keys() else data["Diag"]
      vidname = data[videoname]

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

      shape__ = np.shape(video_list_joints)
      num_time = shape__[1]

      t = np.linspace(0,num_time,num_time)
      b = np.zeros((num_time,3))

      num_joints_shown = 25
      x = np.zeros((num_joints_shown,num_time,3))
      joint = np.arange(0,0+num_joints_shown,1)
      fig = plt.figure(figsize=(8,8))
      for k in range(num_joints_shown):
        x[k,:,0]=data_final[:,k,2]
        x[k,:,1]=data_final[:,k,0]
        x[k,:,2]=data_final[:,k,1]
      x[:, :, 2] *= -1
      x[:, :, 1] *= -1
      print("Data extracted")

    return x, diagnosis, sex, di




def rascopy():
    fin = ''
    while(fin != 'y'):
        x,diag,sex,name,vid,di = open_json()
        nam = get_kinectv2_joint_names()
        v = ''
        while(v!='n'):
          condition = 0
          ctrl=0
          cond2=0
          stop=''
          ccc=0
          while(ccc==0):
            a = input("\nWhich joints to you want to analyze ? (numbers bewteen 0 and 24 separated by ','): ")
            b = input("\nWhich is your reference joints for 3-steps segmentation ? (14 or 18): ")
            if b == "14" or b == "18":
              ankle_side = int(b)
              cond2 = 0
            else:
              cond2=1
            ctrl=0
            for char in a:
              if (ord(char)<48 or ord(char)>57) and ord(char)!=44:
                ctrl=ctrl+1
              elif (ord(char)>=48 and ord(char)<=57) or ord(char)==44:
                ctrl=ctrl
            if ctrl == 0 and a!='' and a[-1]!=',':
              a_list = []
              cond=0
              for string in a.split(','):
                if int(string)>0 and int(string)<25:
                  cond = cond
                else:
                  cond=cond+1
              if cond == 0 and cond2==0:
                nm=''
                for string in a.split(','):
                  a_list.append(int(string))
                  nm = nm+string
                back_file = name+"_"+vid+"_mean_"+nm
                xyz = x[a_list,:,:]
                T=xyz.shape[1]
                ccc=1
                condition=1
              else:
                print('Error in the number of the joints')
            else:
              print('Error in the number of the joints')

          c=0
          control=0

          while(control!=1):
            c=c+1
            if not os.path.exists(f'{back_file}'):
              with open(back_file, 'w') as fichier:
                fichier.write("Analysis of a "+sex+" people. \nStage : "+diag+".\n\n")
              print(back_file," succesfully created !")
              break
            elif c==1:
                  while(True):
                    rep = input(f"The file '{back_file}' already exists. Do you want to add your new analysis inside? (Y/n): ")
                    if rep.lower() == 'n':
                      rep1 = input(f"Do you want to erase existing '{back_file}'? (Y/n): ")
                      if rep1.lower() == 'y':
                        with open(back_file, 'w') as fichier:
                          fichier.write("Analysis of a "+sex+" people. Stage : "+diag+".")
                          control = 1
                        break
                      elif rep1.lower() == 'n':
                        back_file=back_file+'_'+str(c)
                        break
                    elif rep.lower() == 'y':
                      control = 1
                      break
            else :
              back_file=back_file[:-1]+str(c)

          if condition == 1:
            while(stop!='s'):
              plt.figure()
              times = np.linspace(0,T,T)
              plt.plot(times, x[ankle_side,:,0], color='red', label='X coordinate')
              plt.plot(times, x[ankle_side,:,1], color='blue', label = 'Y coordinate')
              plt.plot(times, x[ankle_side,:,2], color='green', label = 'Z coordinate')
              plt.xlabel('Time')
              plt.ylabel('Position')
              nom=''
              for n in a_list:
                nom = nom+nam[n]+', '
              plt.title(f"Mouvement's mean of {nom} on each coordinate")
              plt.legend()
              plt.show(block=False)
    
              while(True):
                dimensions = input("\nIn which dimensions do you want to analyze ? (2 or 3): ")
                if dimensions.isdigit() and dimensions!="":
                  if int(dimensions)==2 or int(dimensions)==3:
                    dimensions=int(dimensions)
                    break
                  else:
                    print("Dimensions should be 2 or 3.")
                else:
                  print("Dimensions should be 2 or 3.")
              if dimensions == 2:
                verif=1
                while(verif!=0):
                  dim = input("\nWhich axis do you want to analyze ? (x, y, or z separated by a ','): ")
                  ctrl=0
                  for char in dim:
                    if (char != 'x' and char != 'y' and char != 'z' and char != ','):
                      ctrl=ctrl+1
                    elif (char == 'x' or char == 'y' or char == 'z' or char == ','):
                      ctrl=ctrl
                  if ctrl == 0 and dim!='' and dim[-1]!=',' and len(dim)==3 and dim[0]!=',':
                    verif = 0
                    coord_list = []
                    for string in dim.split(','):
                      if string == 'x':
                        coord_list.append(0)
                      elif string == 'y':
                        coord_list.append(1)
                      elif string == 'z':
                        coord_list.append(2)
                    y = xyz[:,coord_list]
                    xy = xyz
                    if os.path.exists(f'{back_file}'):
                      with open(back_file, 'r') as f:
                        for ligne in f:
                          if '------------------ '+str(dim[0])+str(dim[2])+' axis analysis------------------' in ligne:
                            recherche = True
                          else:
                            recherche = False
                      if recherche == False:
                        with open(back_file, 'a') as fichier:
                          fichier.write("\n------------------ "+str(dim[0])+str(dim[2])+" axis analysis------------------\n")
              else:
                y = xyz
                xy = xyz
                with open(back_file, 'a') as fichier:
                  fichier.write("\n------------------ "+str(dimensions)+"D analysis ------------------\n")
              
              start_frame = None
              end_frame = None
              fr = None
              maxmin = ""
              ypeak = x[ankle_side,:,:]
              maxp,minp,locp = opti_epsi.nbr_peaks(ypeak)
              first = min(min(locp['max']),min(locp['min'])) 
              if first in locp['max']:
                maxmin = 'max'
              if first in locp['min']:
                maxmin='min'

              while(True):
                fr = input("\nEnter the 3-steps segmentation n° you want to analyze: (first segmentation = 1): ")
                fr=int(fr)
                if fr < len(locp[maxmin])-2:
                  start_frame = locp[maxmin][fr-1]
                  end_frame = locp[maxmin][fr+2]
                  break
                else:
                  print("Error in segmentation number")
                  
              
              y=xyz[:,start_frame:end_frame,:]
              xy=xyz[:,start_frame:end_frame,:]
              
              while(True):
                method = input("\nWhich optimal epsilon method (number between 0 and 5) ? ")
                if method.isdigit() == True and int(method)<6 and int(method)>=0:
                  break
                else: 
                  print("method should be a number between 0 and 5")
              method = int(method)
              visu = input("\nEnter the numbers of the features you want to visualize (separated by ','): ")
              visu_list = []
              for string in visu.split(','):
                visu_list.append(string)
              
              if("1" in visu_list):
                plt.figure()
                Ty=y.shape[1]
                times = np.linspace(0,Ty,Ty)
                if y.shape[2] == 3:
                  plt.plot(times, x[ankle_side,start_frame:end_frame,0], color='red', label='X coordinate')
                  plt.plot(times, x[ankle_side,start_frame:end_frame,1], color='blue', label = 'Y coordinate')
                  plt.plot(times, x[ankle_side,start_frame:end_frame,2], color='green', label = 'Z coordinate')
                if y.shape[2] == 2:
                  plt.plot(times, x[ankle_side,start_frame:end_frame,0], color='red', label=f'{dim[0]} coordinate')
                  plt.plot(times, x[ankle_side,start_frame:end_frame,1], color='blue', label = f'{dim[2]} coordinate')
                if y.shape[2] == 1:
                  plt.plot(times, x[ankle_side,start_frame:end_frame,0], color='red', label=f'{dim[0]} coordinate')
                plt.xlabel('Time')
                plt.ylabel('Position')
                nom=''
                for n in a_list:
                  nom = nom+nam[n]+', '
                plt.title(f"Mouvement's mean of {nom} on each coordinate")
                plt.legend()
                plt.show(block=False)
                while(True):
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
                    break
                  elif rep.lower() == 'n':
                    break

              if("2" in visu_list):
                fig = plt.figure()
                if y.shape[2] == 3:
                  ax = fig.add_subplot(111, projection='3d')
                  ax.plot(x[ankle_side,start_frame:end_frame,0], x[ankle_side,start_frame:end_frame,1], x[ankle_side,start_frame:end_frame,2], color='blue', label='Trajectory')
                  ax.set_xlabel('X')
                  ax.set_ylabel('Y')
                  ax.set_zlabel('Z')
                  plt.title('Trajectory in 3D')
                if y.shape[2] == 2:
                  plt.plot(x[ankle_side,start_frame:end_frame,0], x[ankle_side,start_frame:end_frame,1], color='blue', label='Trajectory')
                  plt.xlabel(dim[0])
                  plt.ylabel(dim[2])
                  plt.title('Trajectory in 2D')
                plt.show(block=False)
                while(True):
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
                    break
                  elif rep.lower() == 'n':
                    break

              if("3" in visu_list):
                recurrence.anim_traj(x[ankle_side,:,:])

              step = 0.001
              test2=1
              y = np.reshape(y.transpose(1, 0, 2), (y.shape[1], y.shape[0]*y.shape[2]))

              if("4" in visu_list):
                if method == 0:
                  
                  epsi = opti_epsi.epsi_entropy(y, step, 1, back_file)
                  test=1
                elif method == 1:
                  epsi = opti_epsi.epsi_utility(y, step,1, back_file)
                  test=1
                elif method == 2:
                  entropy = opti_epsi.epsi_entropy(y, step,1, back_file)
                  utility = opti_epsi.epsi_utility(y, step,1, back_file)
                  epsi = (entropy+utility)/2
                  test=1
                elif method == 3:
                  epsi = opti_epsi.opti_epsi_phifct(y,step,1,back_file)
                  test=1
                elif method == 4:
                  epsi = opti_epsi.epsi_entropy_n(y, step, 1, back_file)
                  test=1
                elif method == 5:
                  epsi, alpha_des, word_des, Lz_des_min, Lz_des_max = opti_epsi.opti_epsi_phi(y, xy, y.shape[2], step, 1, back_file)
                  test=1
                else:
                  print("Method doesn't exist")
                  test=2
                  test2=2

              else:
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
                elif method == 3:
                  epsi = opti_epsi.opti_epsi_phifct(y,step)
                  test=1
                elif method == 4:
                  epsi = opti_epsi.epsi_entropy_n(y, step)
                  test=1
                elif method == 5:
                  test=0
                else:
                  print("Method doesn't exist")
                  test=2
                  test2=2

              while test==0:
                while(True):
                  epsilon = input("\nPlease enter your epsilon value: ")
                  try:
                    float(epsilon)
                    tr = True
                  except ValueError:
                    tr = False
                  if tr == True:
                    break
                  else:
                    print("Epsilon should be a float")
                    
                epsi = float(epsilon)
                if("5" in visu_list):
                  R = recurrence.rec_mat(y, epsi,1,back_file)
                else:
                  R = recurrence.rec_mat(y, epsi)
                if("6" in visu_list):
                  recurrence.rec_plt(R)
                if("7" in visu_list):
                  serie = symbolic_series.symbolic_serie(R,1, back_file)
                else:
                  serie = symbolic_series.symbolic_serie(R)
                if("8" in visu_list):
                  symbolic_series.colored_sym_serie(serie,y)
                if("9" in visu_list):
                  recurrence.col_rec_plt(serie, R)
                if("10" in visu_list):
                  symbolic_series.plot_col_traj(serie,x[ankle_side,:,:])                 

                ans = input("\nAre these results satisfying ? (Y/n): ")
                if ans.lower() == 'y':
                  test=1
                  test2=0

              if epsi is not None:
                if test == 1 and test2==1:
                  if("5" in visu_list):
                    R = recurrence.rec_mat(y, epsi,1,back_file)
                  else:
                    R = recurrence.rec_mat(y, epsi)
                  if("6" in visu_list):
                    recurrence.rec_plt(R)
                  if("7" in visu_list):
                    serie = symbolic_series.symbolic_serie(R,1,back_file)
                  else:
                    serie = symbolic_series.symbolic_serie(R)
                  if("8" in visu_list):
                    symbolic_series.colored_sym_serie(serie,y)
                  if("9" in visu_list):
                    recurrence.col_rec_plt(serie, R)
                  if("10" in visu_list):
                    symbolic_series.plot_col_traj(serie,x[ankle_side,:,:])

                  cplx = input("Do you want to analyse the complexity ? (Y/n)")

                  if cplx.lower() == 'y':
                    stance = []
                    swing = []
                    if first in locp['max']:
                      if len(locp['min'])==len(locp['max']) or len(locp['min'])==len(locp['max'])-1:
                          for i in range(len(locp['min'][fr-1:fr+2])):
                              if (locp['min'][i+fr-1] != locp['min'][-1] or locp['min'][-1] < locp['max'][-1]) and (locp['max'][i+fr-1] != locp['max'][0]):
                                  stance.append(abs(locp['min'][i+fr-1]-locp['max'][i+fr-1]))
                              if (len(locp['min'])==len(locp['max']) and i!=len(locp['min'])-1) or (len(locp['min'])==len(locp['max'])-1):
                                  if locp['max'][i+fr] != locp['max'][-1] or locp['max'][-1] < locp['min'][-1]:
                                      swing.append(abs(locp['max'][i+fr]-locp['min'][i+fr-1]))
                    if first in locp['min']:
                        if len(locp['min'])==len(locp['min']) or len(locp['max'])==len(locp['min'])-1:
                            for i in range(len(locp['max'][fr-1:fr+2])):
                                if (len(locp['max'])==len(locp['min']) and i!=len(locp['max'])-1) or (len(locp['max'])==len(locp['min'])-1):
                                    if locp['min'][i+fr] != locp['min'][-1] or locp['min'][-1] < locp['max'][-1]:
                                        stance.append(abs(locp['max'][i+fr-1]-locp['min'][i+fr]))
                                if (locp['max'][i+fr-1] != locp['max'][-1] or locp['max'][-1] < locp['min'][-1]) and (locp['min'][i+fr-1]!=locp['min'][0]):
                                    swing.append(abs(locp['min'][i+fr-1]-locp['max'][i+fr-1]))
                    C_swing = np.std(swing)
                    C_stance = np.std(stance)
                    print("")
                    print("C_swing= ",C_swing)
                    print("C_stance = ",C_stance)
                    print("")

                    leg_length = math.sqrt((x[12,0,0]-x[13,0,0])**2+(x[12,0,1]-x[13,0,1])**2+(x[12,0,2]-x[13,0,2])**2)+math.sqrt((x[13,0,0]-x[14,0,0])**2+(x[13,0,1]-x[14,0,1])**2+(x[13,0,2]-x[14,0,2])**2)
                    # C_stepperiod
                    C_stepperiod = (y.shape[0]/3)/leg_length
                    print("Leg length = ",round(round(leg_length,4)*100,2),"cm")
                    print("")
                    print("C_stepperiod = ",C_stepperiod)

                    # C_aplhabet-size
                    alphabet_size = len(np.unique(serie))
                    print("C_aplhabet-size = ",alphabet_size)

                    # -------------------------------- C_Number-of-words -----------------------------------------
                    a = 0
                    b = 0
                    W_nw = []
                    for i in range(len(serie)-1):
                        w = ''
                        if serie[i] != serie[i+1]:
                            for j in range(a, i+1, 1):
                                w = w + str(int(serie[j]))
                            W_nw.append(w)
                            a = i + 1
                        if i == len(serie)-2:
                            for j in range(a, i+2, 1):
                                w = w + str(int(serie[j]))
                            W_nw.append(w)
                    unique = set(W_nw)
                    C_nbr_words = len(unique)
                    print("C_Number-of-words = ",C_nbr_words)
                    # -------------------------------------------------------------------------------------------

                    # ----------------------------------------- C_arms ------------------------------------------
                    if (6 in a_list or 10 in a_list) and 2 not in a_list: # If there are only arms joints in the analysis
                        ideb = 0
                        amplitude = []
                        if 6 in a_list: # Default: take the left wrist into account
                            side = 6
                        elif 10 in a_list: # Otherwise, take the right wrist
                            side = 10
                        # Searching for max amplitude in each step
                        if first in locp['min']:
                            for ifin in locp['min'][fr:fr+3]:
                                amplitude.append(abs(max(x[side,ideb:ifin,0])-min(x[side,ideb:ifin,0])))
                                ideb = ifin
                        elif first in locp['max']:
                            for ifin in locp['max'][fr:fr+3]:
                                amplitude.append(abs(max(x[side,ideb:ifin,0])-min(x[side,ideb:ifin,0])))
                                ideb = ifin
                        aberrant = max(amplitude) #Removing the maximum amplitude to avoid data errors due to noise 
                        s_amplitude = 0
                        cnt=0
                        for i in amplitude:
                            if i != aberrant:
                                s_amplitude = s_amplitude+i
                                cnt = cnt+1
                        C_arms = s_amplitude/cnt # Mean maximum amplitude of each step
                        print("C_arms = ",C_arms)
                    # -------------------------------------------------------------------------------------------

                    stock=[]
                    countarg={}
                    if 0 in serie:
                        k=np.zeros(len(np.unique(serie)))
                    else:
                        k=np.zeros(len(np.unique(serie))+1)
                    k=k.astype(int)
                    serie = serie.astype(int)
                    stockarg = {} # This contains the center frame position of each metastable states appearance

                    temp = serie[0]
                    indic_zero=[]
                    c_z=0
                    c_s1=0
                    c_s2=0
                    aaa=0
                    
                    for i in range (len(serie)-1):
                        if serie[i]==0 and serie[i+1]==temp and temp!=0:
                            aaa=1
                        if serie[i]==0:
                            c_z = c_z+1
                        if serie[i]!=0 and aaa==0:
                            c_s1=c_s1+1
                        if serie[i]!=0 and aaa==1:
                            c_s2=c_s2+1
                        if aaa==1 and serie[i]==temp and serie[i+1]!=serie[i]:
                            if c_z>=0.8*(C_stepperiod*leg_length -(c_s1+c_s2)):
                                indic_zero.append(1)
                            else:
                                indic_zero.append(0)
                            c_z=0
                            c_s1=c_s2
                            aaa=0
                            c_s2=0
                        if aaa==0 and serie[i+1]!=0 and serie[i+1]!=serie[i] and serie[i+1]!=temp:
                            temp=serie[i+1]
                            c_z=0
                            c_s1=0
                            c_s2=0
                        if i == len(serie)-2:
                            if aaa==1 and serie[i+1]==temp:
                                c_s2=c_s2+1
                                if c_z>=0.8*(C_stepperiod*leg_length -(c_s1+c_s2)):
                                    indic_zero.append(1)
                                else:
                                    indic_zero.append(0)

                    for i in range (1,len(serie)):
                        if serie[i-1] in stockarg:
                            if 0 <= k[serie[i-1]] < len(stockarg[serie[i-1]]):
                                stockarg[serie[i-1]][k[serie[i-1]]]=stockarg[serie[i-1]][k[serie[i-1]]]+i-1
                                countarg[serie[i-1]][k[serie[i-1]]]=countarg[serie[i-1]][k[serie[i-1]]]+1
                            else:
                                stockarg[serie[i-1]].append(i-1)
                                countarg[serie[i-1]].append(1)    
                        else:
                            stockarg[serie[i-1]]=[]
                            stockarg[serie[i-1]].append(i-1)
                            countarg[serie[i-1]]=[]
                            countarg[serie[i-1]].append(1)

                        if serie[i]!=serie[i-1]:
                            stock.append(serie[i-1])
                            if i == len(serie)-1:
                                stock.append(serie[i])
                                if serie[i] in stockarg:
                                    stockarg[serie[i]].append(i)
                                    countarg[serie[i]].append(1)
                                else:
                                    stockarg[serie[i]]=[]
                                    stockarg[serie[i]].append(i)
                                    countarg[serie[i]]=[]
                                    countarg[serie[i]].append(1)
                            if serie[i-1] in stock:
                                k[serie[i-1]]=k[serie[i-1]]+1

                        if i == len(serie)-1 and serie[i]==serie[i-1]:
                            stock.append(serie[i])
                            stockarg[serie[i]][k[serie[i]]]=stockarg[serie[i]][k[serie[i]]]+i
                            countarg[serie[i]][k[serie[i]]]=countarg[serie[i]][k[serie[i]]]+1

                    if 0 in serie:
                        k=np.zeros(len(np.unique(stock)))
                    else:
                        k=np.zeros(len(np.unique(stock))+1)
                    k=k.astype(int)
                    iz=0
                    for i in range(len(stock)-2):
                        k[stock[i]] = k[stock[i]]+1
                        if stock[i]!=0 and stock[i]==stock[i+2] and stock[i+1] == 0:
                            if indic_zero[iz]==0:
                                stockarg[stock[i]][k[stock[i]]-1] = stockarg[stock[i]][k[stock[i]]-1] + stockarg[stock[i]][k[stock[i]]]
                                stockarg[stock[i]].pop(k[stock[i]])
                                countarg[stock[i]][k[stock[i]]-1] = countarg[stock[i]][k[stock[i]]-1] + countarg[stock[i]][k[stock[i]]]
                                countarg[stock[i]].pop(k[stock[i]])
                                k[stock[i]]=k[stock[i]]-1
                            iz=iz+1

                    for key in stockarg:
                        for val in range(len(stockarg[key])):
                            stockarg[key][val] = stockarg[key][val]/countarg[key][val]
            # ------------------------------------------------------------------------------------------------------------------------------------------
                    
                    # Calculation of the frame distance between each metastable state
                    distance={} 
                    for key in stockarg: 
                        if key!=0:
                            for val in range(len(stockarg[key])-1):
                                dist=stockarg[key][val+1]-stockarg[key][val]
                                if key in distance:
                                    distance[key].append(dist)
                                else:
                                    distance[key]=[]
                                    distance[key].append(dist)

                    # Calculate the number of state's appearances
                    f=[]
                    for key in stockarg:
                        if key!=0:
                            f.append(len(stockarg[key]))
                     
                    # ----------------------------------------------- C_appearance ---------------------------------------------------
                    erreur_rec = []
                    for i in f:
                        if i<3:
                            err = abs(i-3)/3
                        elif i>4:
                            err = abs(i-4)/4
                        else:
                            err = 0
                        erreur_rec.append(err)
                    #print(erreur_rec)
                    C_appearance = 0
                    for i in erreur_rec:
                        C_appearance = C_appearance+i
                    print("C_appearance = ",C_appearance)
                    # ----------------------------------------------------------------------------------------------------------------

            # ---------------------------------------------------- Symbolic Sequence segmentation step by step ------------------------------------------------------
                    serie=np.array(serie)
                    serie_seg = []
                    if first in locp['max']:
                        ideb = locp['max'][fr-1]
                        for ifin in locp['max'][fr-1+1:fr-1+4]:
                            s_s = serie[ideb-locp['max'][fr-1]:ifin-locp['max'][fr-1]]
                            se_se = []
                            for k in s_s:
                                if k not in se_se:
                                    se_se.append(k)
                                elif k!=se_se[-1]:
                                    se_se.append(k)
                            s_s = se_se
                            if ideb!=locp['max'][fr-1] and serie_seg != [] and s_s[0] == serie_seg[-1][-1]:
                                if (serie_seg[-1][-1] == serie_seg[-1][0]) or (serie_seg[-1][-1] == serie_seg[-1][1] and serie_seg[-1][0]==0):
                                    serie_seg[-1].pop(-1)
                                    serie_seg.append(s_s)
                                elif s_s[1:]!=[]:
                                    serie_seg.append(s_s[1:])
                            else:
                                serie_seg.append(s_s)
                            ideb=ifin
                    else:
                        ideb = locp['min'][fr-1]
                        for ifin in locp['min'][fr-1+1:fr-1+4]:
                            s_s = serie[ideb-locp['min'][fr-1]:ifin-locp['min'][fr-1]]
                            se_se = []
                            for k in s_s:
                                if k not in se_se:
                                    se_se.append(k)
                                elif k!=se_se[-1]:
                                    se_se.append(k)
                            s_s = se_se
                            if ideb!=locp['min'][fr-1] and serie_seg != [] and s_s[0] == serie_seg[-1][-1]:
                                if (serie_seg[-1][-1] == serie_seg[-1][0]) or (serie_seg[-1][-1] == serie_seg[-1][1] and serie_seg[-1][0]==0):
                                    serie_seg[-1].pop(-1)
                                    serie_seg.append(s_s)
                                elif s_s[1:]!=[]:
                                    serie_seg.append(s_s[1:])
                            else:
                                serie_seg.append(s_s)
                            ideb=ifin
            # -------------------------------------------------------------------------------------------------------------------------------------------------

                    # Removing useless first states if necessary 
                    for i in serie_seg:
                        for j in serie_seg:
                            if j != i and len(i)>1 and len(j)>0 and i[-1]==j[0]:
                                i[-1]=0

            # ----------------------------------- Searching for states that appears exactly 3 times in the symbolic sequence ----------------------------------
                    f1={}
                    for indice, i in enumerate(serie_seg):
                        for j in i:
                            ok = 0
                            if i.count(j)==2 and j!=0:
                                for m in range(len(i)-2):
                                    if i[m] == j and i[m+1] == 0 and i[m+2] == j:
                                        ok = 1
                                        i[m+2] = 0
                                if i[0]==j and i[-1]==j:
                                    ok=1
                                    if i!=serie_seg[-1] and serie_seg[indice+1][0]!=j:
                                        serie_seg[indice+1].insert(0,j)
                                    i[-1]==0
                                if (i[0]==0 and i[1]==j) and (i[-1]==j or (i[-1]==0 and i[-2]==j)) or (i[0] == j and (i[-2] == j and i[-1] == 0)):
                                    ok = 1
                                    if i[-2]==j:
                                        if j in f1:
                                            f1[j] = f1[j]+1
                                        else:
                                            f1[j]=1
                                    elif i[-1]==j and i!=serie_seg[-1] and serie_seg[indice+1][0]!=j:
                                        serie_seg[indice+1].insert(0,j)
                                    premier_indice = i.index(j)
                                    deuxieme_indice = i.index(j, premier_indice + 1)
                                    i[deuxieme_indice] = 0
                            if i.count(j)==3 and j!=0:
                                for m in range(len(i)-4):
                                    if i[m] == j and i[m+1] == 0 and i[m+2] == j and i[m+3] == 0 and i[m+4] == j:                                               
                                        ok = 1
                                        i[m+2] = 0
                                        i[m+4] = 0

                            if j!=0 and (i.count(j)==1 or ok==1):
                                if j in f1:
                                    f1[j] = f1[j]+1
                                else:
                                    f1[j]=1
                            else:
                                f1[j] = 1000
                            for k in f1:
                                if k not in i and f1[k] <= indice:
                                    f1[k] = 1000
            # -----------------------------------------------------------------------------------------------------------------------------------

            # -------------------------------------------------- C_deviation --------------------------------------------------------------------
                    etype={}
                    for h in f1:
                        if f1[h] >= 3 and f1[h]<5 and h<len(distance):
                            ecart_type = np.std(distance[h])
                            etype[h] = ecart_type
                        elif f1[h]==5 and f[h] < 5:
                            ecart_type = np.std(distance[h])
                            etype[h] = ecart_type
                    if len(etype)!=0:
                        C_deviation=0
                        for key in etype:
                            C_deviation = C_deviation+etype[key]
                        C_deviation = C_deviation/len(etype)
                        print("C_deviation = ",C_deviation)
                    else:
                        C_deviation = 100
                        print("C_deviation = Impossible")
            # -----------------------------------------------------------------------------------------------------------------------------------

                    # ------------------------------------- Symbolic serie with states merged in a unique symbol --------------------------------
                    serie = np.array(serie)
                    serie = serie.astype(int)
                    order = [] # Order = Symbolic serie with states merged in 1 symbol
                    for i in range(len(serie)-1):
                        if i == 0:
                            order.append(serie[i])
                        if serie[i]!=serie[i+1]:
                            order.append(serie[i+1])
                    print("")
                    print("New symbolic sequence = ",order)
                    print("")
                    # --------------------------------------------------------------------------------------------------------------------------

                    # -------------------------------------------------------------- C_Lempel-Ziv ----------------------------------------------

                    # Search if a symbolic pattern is in a symbolic sequence
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
                    W_LZ.append(order[i])
                    i = 1
                    while i < len(order):
                        j = i
                        Bool = pattern_in_serie(W_LZ[:j], str(order[j]))
                        if Bool == False:
                            W_LZ.append(str(order[i]))
                            i = i + 1
                        else:
                            while Bool == True:
                                j = j + 1
                                if j >= len(order):
                                    break
                                Bool = pattern_in_serie(W_LZ[:j], order[i:j + 1])
                            W_LZ.append(''.join(map(str, order[i:j + 1])))
                            i = j + 1
                    C_LZ = len(W_LZ)
                    C_LZ_n = C_LZ/len(order)
                    print("C_Lempel-Ziv = ",C_LZ)
                    # --------------------------------------------------------------------------------------------------------------------------

                    #----------------------------------------------------- C_2 -----------------------------------------------------------------
                    a=0
                    C_2 = 0
                    ser = order.copy()
                    w_ser = ser.copy()
                    # Searching for maximum length's recurrent serie
                    while(len(ser)!=0): 
                        max_serie = ''
                        best_max_serie = ''
                        j=0
                        i=0
                        i2=0
                        while i < len(ser):
                            if str(ser[i]) != '0' and str(ser[i]) not in max_serie:
                                max_serie = max_serie+str(ser[i])
                                ser_str = ''
                                for p in ser:
                                    ser_str=ser_str+str(p)
                                count=0 
                                rec=0
                                if max_serie in ser_str:
                                    rec=ser_str.count(max_serie)
                                if len(max_serie)>len(best_max_serie) and rec>1 and len(max_serie)<=len(ser)/2:
                                    best_max_serie = max_serie
                                    i=j
                                elif len(max_serie)>=len(ser)/2 and len(max_serie)>len(best_max_serie) and rec==1:
                                    best_max_serie = max_serie
                                    i=j
                            elif str(ser[i]) == '0':
                                max_serie = max_serie+str(ser[i])
                                ser_str = ''
                                for p in ser:
                                    ser_str=ser_str+str(p)
                                count=0 
                                rec=0
                                if max_serie in ser_str:
                                    rec=ser_str.count(max_serie)
                                if len(max_serie)>len(best_max_serie) and rec>1 and len(max_serie)<=len(ser)/2:
                                    best_max_serie = max_serie
                                    i=j
                                elif len(max_serie)>len(ser)/2 and len(max_serie)>len(best_max_serie) and rec==1:
                                    best_max_serie = max_serie
                                    i=j
                            else:
                                max_serie=''
                                i2=i2+1
                                i=i2-1
                                j=i
                            i=i+1
                            j=j+1

                        for i in range(len(w_ser)):
                            c=0
                            ws = ''
                            windx = []
                            for j in range(len(best_max_serie)):
                                if i+j < len(w_ser):
                                    if w_ser[i+j] == int(best_max_serie[j]):
                                        ws=ws+str(w_ser[i+j])
                                        windx.append(i+j)
                                        c = c+1
                            if c==len(best_max_serie):
                                break
                        if windx != []:
                            ind = windx[0]
                        else:
                            ind = 0
                        for i in windx: 
                            w_ser.pop(i)
                            for j in range(len(windx)):
                                windx[j]=windx[j]-1
                        C_2 = C_2+1
                        ws = ''.join(map(str, w_ser))
                        if best_max_serie in ws[:ind]:
                            ws = ws.replace(best_max_serie, '')
                            w_ser = [int(c) for c in ws]
                        if best_max_serie in ws[ind:]:
                            ws = ws.replace(best_max_serie, '')
                            w_ser = [int(c) for c in ws]
                        ser=w_ser
                        a=a+1
                    # --------------------------------------------------------------------------------------------------------------------------

                    #----------------------------------------------------- C_1 -----------------------------------------------------------------
                    C_1 = 0
                    ser = order.copy()
                    store = []
                    store_str = ''
                    i=0
                    while i < len(ser):
                        if str(ser[i]) != '0' and str(ser[i]) not in store_str:
                            store_str = store_str+str(ser[i])
                            ser.pop(i)
                        elif str(ser[i]) == '0':
                            store_str = store_str+str(ser[i])
                            ser.pop(i)
                        else:
                            store.append(store_str)
                            store_str=''
                        if len(ser) == 0:
                            store.append(store_str)
                    conf = 0
                    #print("\nstore = ",store)
                    for i in range(len(store[:-1])):
                        if store[-1] in store[i][:len(store[-1])]:
                            conf = 1
                        if store[i][0] == '0':
                            store[i] = store[i][1:]
                    if conf == 1:
                        store.pop(-1)
                    C_1 = len(np.unique(store))
                    #print("store suppr = ",store)
                    print("C_1 = ",C_1)
                    print("C_2 = ",C_2)
                    print("")
                    print("")
                    # --------------------------------------------------------------------------------------------------------------------------


              while(True):
                s = input("\nContinue with this/these joint(s) ? (Y/n): ")
                if s.lower() == 'y':
                  break
                elif s.lower() == 'n':
                  stop='s'
                  break 

          while(True):
            v = input("\nContinue with this video ? (Y/n): ")
            if v.lower() == 'n':
              break
            elif v.lower() == 'y':
              break  

        while(True):
          finish = input("\nOther analysis ? (Y/n): ")
          if finish.lower() == 'n':
            fin = 'y'
            break
          elif finish.lower() == 'y':
            break
        


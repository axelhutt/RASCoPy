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
        
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/vpare_db/skeletons/test_01-20-2024/data.json')
    parser.add_argument('--vid_dir', type=str, default='data/vpare_db/skeletons/test_01-20-2024/videos')
    
    args = parser.parse_args()
    
    visualize(args) 
'''


def open_json():
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
        print(vname, ":", video['sex'], " --> ", video['diag'], " | lenghth: ", video['joints3D'].shape[0])
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
    diagnosis = vidname['diag']
    sex = vidname['sex']

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
    print("Data extracted")

  return x, diagnosis, sex, name, videoname




def rascopy(start_frame=None, end_frame=None, dimensions=3):
  if end_frame is not None:
    end_frame=int(end_frame)
  if start_frame is not None:
    start_frame=int(start_frame)

  fin = ''

  x,diag,sex,name,videoname = open_json()
  nam = get_kinectv2_joint_names()

  while(fin != 'y'):
    condition = 0
    ctrl=0
    stop=''
    while(condition!=1):
      data = input("\nSingle joint or joint's mean analysis ? (s/m): ")

      if data.lower() == 's':
        cotrl=0
        while(cotrl!=1):
          a = input("\nWhich joint to you want to analyze ? (number bewteen 0 and 24): ")
          ctrl=0
          for char in a:
            if (ord(char)<48 or ord(char)>57) and ord(char)!=44:
              ctrl=ctrl+1
            elif (ord(char)>=48 and ord(char)<=57) or ord(char)==44:
              ctrl=ctrl
          if ctrl == 0 and a!='' and len(a)<=2:
            if int(a)>=0 and int(a)<25:
              back_file = name+"_"+videoname+"_"+a
              a = int(a)
              xy = x[a, start_frame:end_frame, :]
              T=xy.shape[0]
              condition=1
              cotrl=1

      elif data.lower() == 'm':
        while(True):
          a = input("\nWhich joints to you want to analyze ? (numbers bewteen 0 and 24 separated by ','): ")
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
            if cond == 0:
              nm=''
              for string in a.split(','):
                a_list.append(int(string))
                nm = nm+string
              back_file = name+"_"+videoname+"_"+str(dimensions)+"D"+"_mean_"+nm
              xy = np.mean(x[a_list,start_frame:end_frame,:],axis=0)
              T=xy.shape[0]
              condition=1
              break
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
          fichier.write("Analysis of a "+sex+" people. Stage : "+diag+".")
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
      times = np.linspace(0,T,T)
      while(stop!='s'):
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
              y = xy[:,coord_list]
              if os.path.exists(f'{back_file}'):
                with open(back_file, 'r') as f:
                  for ligne in f:
                    if '------------------ '+str(dim[0])+str(dim[2])+' axis ------------------' in ligne:
                      recherche = True
                    else:
                      recherche = False
                if recherche == False:
                  with open(back_file, 'a') as fichier:
                    fichier.write("\n------------------ "+str(dim[0])+str(dim[2])+" axis ------------------\n")
        else:
          y = xy
            
        while(True):
          method = input("\nWhich optimal epsilon method ? ")
          if method.isdigit() == True:
            break
          else: 
            print("method should be a number")
        method = int(method)
        visu = input("\nEnter the numbers of the features you want to visualize (separated by ','): ")
        visu_list = []
        for string in visu.split(','):
          visu_list.append(string)
        
        if("1" in visu_list):
          plt.figure()
          if y.shape[1] == 3:
            plt.plot(times, y[:,0], color='red', label='X coordinate')
            plt.plot(times, y[:,1], color='blue', label = 'Y coordinate')
            plt.plot(times, y[:,2], color='green', label = 'Z coordinate')
          if y.shape[1] == 2:
            plt.plot(times, y[:,0], color='red', label=f'{dim[0]} coordinate')
            plt.plot(times, y[:,1], color='blue', label = f'{dim[2]} coordinate')
          if y.shape[1] == 1:
            plt.plot(times, y[:,0], color='red', label=f'{dim[0]} coordinate')
          plt.xlabel('Time')
          plt.ylabel('Position')
          if data == 's':
            plt.title(f"{nam[a]}'s mouvement on each coordinate")
          elif data == 'm':
            nom=''
            for n in a_list:
              nom = nom+nam[n]+', '
            plt.title(f"Mouvement's mean of {nom} on each coordinate")
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

        if("2" in visu_list):
          fig = plt.figure()
          if y.shape[1] == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(y[:, 0], y[:, 1], y[:, 2], color='blue', label='Trajectory')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.title('Trajectory in 3D')
          if y.shape[1] == 2:
            plt.plot(y[:, 0], y[:, 1], color='blue', label='Trajectory')
            plt.xlabel(dim[0])
            plt.ylabel(dim[2])
            plt.title('Trajectory in 2D')
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

        if("3" in visu_list):
          recurrence.anim_traj(y)

        step = 0.001
        test2=1

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
          elif method == 4:
            test=0
          elif method == 5:
            epsi = opti_epsi.opti_epsi_phi(y, y.shape[0], step, 1, back_file)
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
          elif method == 4:
            test=0
          elif method == 5:
            epsi = opti_epsi.opti_epsi_phi(y, y.shape[0], step)
            test=1
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
            symbolic_series.plot_col_traj(serie,y)
          if("11" in visu_list):
            C_alphabet_size, C_nbr_words, C_LZ = symbolic_series.complexity(serie,1,back_file)

          ans = input("\nAre these results satisfying ? (Y/n): ")
          if ans.lower() == 'y':
            test=1
            test2=0

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
            uniqSerie = np.unique(serie)
          if("8" in visu_list):
            symbolic_series.colored_sym_serie(serie,y)
          if("9" in visu_list):
            recurrence.col_rec_plt(serie, R)
          if("10" in visu_list):
            symbolic_series.plot_col_traj(serie,y)
          if("11" in visu_list):
            C_alphabet_size, C_nbr_words, C_LZ = symbolic_series.complexity(serie,1,back_file)
        
        if test!=2 and test2!=2:
          while(True):
            shuf = input("\nDo you want to analyse the complexity ? (Y/n): ")
            if shuf.lower() == 'y':
              nbr = input("\nHow many tests ? ")
              if nbr != '' and nbr.isdigit()==True:
                symbolic_series.complexity_shuffle(y, 0.01, count=int(nbr), back_file=back_file)
              else:
                symbolic_series.complexity_shuffle(y, 0.01)
              break
            elif shuf.lower() == 'n':
              break

        while(True):
          s = input("\nContinue with this/these joint(s) ? (Y/n): ")
          if s.lower() == 'y':
            break
          elif s.lower() == 'n':
            stop='s'
            break 

    while(True):
      finish = input("\nOther analysis ? (Y/n): ")
      if finish.lower() == 'n':
        fin = 'y'
        break
      elif finish.lower() == 'y':
        break


        

from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import torch
import random
import scipy.spatial
import scipy.io
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy.random import RandomState
import glob
from tkinter import Tcl
import cv2


def trans_dataset_gen(fraction_val, csv_path, image_path):  #### divides the dataset into train and val data

    #os.chdir(csv_path)
    track_files = []
    meta_files = []
    for file in glob.glob(csv_path + "*_tracks.csv"):
        track_files.append(file)

    track_files.sort()

    for file in glob.glob(csv_path + "*_tracksMeta.csv"):
        meta_files.append(file)

    meta_files.sort()

    train_list = []
    val_list = []
    for i in range(len(track_files)):
        df = pd.read_csv(track_files[i])
        meta = pd.read_csv(meta_files[i])
        size = df.shape[0]
        trans_df = dataset_format(df, meta, image_path)
        train_cutoff = int((1 - fraction_val) * size)
        train = trans_df.head(train_cutoff)
        val = trans_df.tail(size - train_cutoff)
        train_list.append(train)
        val_list.append(val)

    df_train = pd.concat(train_list)
    df_train = df_train.sort_values(["recordingId", "frame", "trackId"], ascending=(True, True, True))
    df_train = df_train.reset_index(drop=True)
    df_train = df_train.drop(['recordingId'], axis=1)

    df_val = pd.concat(val_list)
    df_val = df_val.sort_values(["recordingId", "frame", "trackId"], ascending=(True, True, True))
    df_val = df_val.reset_index(drop=True)
    df_val = df_val.drop(['recordingId'], axis=1)

    # name_train = "train_data_" + str(data_num) + ".csv"
    # name_val = "val_data_" + str(data_num) + ".csv"
    # directory_train_out = os.path.join(dataset_out_folder,dataset_name,"train")
    # if not os.path.exists(directory_train_out):
    # os.makedirs(directory_train_out)
    # directory_val_out = os.path.join(dataset_out_folder,dataset_name,"val")
    # if not os.path.exists(directory_val_out):
    #   os.makedirs(directory_val_out)
    # train.to_csv(directory_train_out + "/" + name_train, index=False)
    # val.to_csv(directory_val_out + "/" + name_val, index=False)
    return df_train, df_val



def dataset_format(df,df2,image_path):                                 
    images = os.listdir(image_path)
    images = Tcl().call('lsort', '-dict', images)

    images_P = []
    for image_name in images:
      images_P.append(image_path + image_name)

    df_img = pd.DataFrame(images_P)
    df3 = df2.transpose()
    df4 = df3.loc[['trackId', 'class']]
    df5 = df4.transpose()
    a = df5.to_dict(orient= 'dict')
    b = df['trackId'].values.tolist()
    classes = np.unique(df2['class'].values.tolist())
    c = []
    for i in b:
        d = a['class'][i]
        c.append(d)
    df6 = pd.DataFrame(c, columns= ['class'])
    P = len(np.unique(c))
    df = df.drop(['trackLifetime','lonVelocity','latVelocity','lonAcceleration','latAcceleration'], axis=1)
    df['class'] = df6
    df = df[['recordingId','frame','trackId','xCenter','yCenter','xVelocity','yVelocity','xAcceleration','yAcceleration','heading','width','length','class']]
    le = LabelEncoder()
    df['class_le']= le.fit_transform(df['class'])
    Y = df[['class_le','class']].values
    enc = OneHotEncoder(categories='auto')
    X = enc.fit_transform(Y).toarray()
    X = X[:,:P]
    df9 = pd.DataFrame(X, columns= classes)
    col_name = df9.columns.values
    col_name= col_name.astype(str)
    df[col_name] = df9
    df['maps'] = df_img
    transformed_data = df.drop(['class','class_le'], axis=1)

    return transformed_data

########################################################## Function to check whether there's need of pre-processing of dataset ###################
def is_data_prepared(save_dir, dataset_info):
    print('Checking if data is already prepared')
    save_file_name = 'info.pt'
  
    for (root, dirs, files) in os.walk(save_dir):
      for file in files:
        if save_file_name == file:
          save_file_path = os.path.join(root, save_file_name)        
          saved_info = torch.load(save_file_path)
          if saved_info == dataset_info:
            return True, root  
    return False, None
###################################################################################################################################################

def get_strided_data_clust(dt, gt_size, horizon, step):
    inp_te = []
    #dtt = dt.astype(np.float32)  #convert dt into float. dt is a dataset file inside the folder
    raw_data = dt
    column_count = len(dt.columns)          # 2 + number of features
    trackId = raw_data.trackId.unique() #removes data that is repeating and common. We get sorted unique values. Returns an array
    frame=[]
    track_ids=[]
    l_maps = []
    for t in trackId:
        for i in range(1+(raw_data[raw_data.trackId == t].shape[0] - gt_size - horizon) // step):
            frame.append(dt[dt.trackId == t].iloc[i * step:i * step + gt_size + horizon, [0]].values.squeeze())
            # print("%i,%i,%i" % (i * 4, i * 4 + gt_size, i * 4 + gt_size + horizon))
            inp_te.append(raw_data[raw_data.trackId == t].iloc[i * step:i * step + gt_size + horizon, 2:column_count-1].values)   #exclude trackId, frame
            map_list = raw_data[raw_data.trackId == t].iloc[i * step:i * step + gt_size, column_count-1:].values
            l_maps.append(map_list)
            #for j in range(len(map_list)):
             # map = map_list[j]
             # l_maps.append(map)
            track_ids.append(t)

    frames=np.stack(frame)
    inp_te_np = np.stack(inp_te)
    track_ids=np.stack(track_ids)
    loc_maps = np.stack(l_maps)

    #inp_relative_pos= inp_te_np-inp_te_np[:,:1,:]
    #inp_speed = np.concatenate((np.zeros((inp_te_np.shape[0],1,2)),inp_te_np[:,1:,0:2] - inp_te_np[:, :-1, 0:2]),1)
    #inp_accel = np.concatenate((np.zeros((inp_te_np.shape[0],1,2)),inp_speed[:,1:,0:2] - inp_speed[:, :-1, 0:2]),1)
    #inp_std = inp_no_start.std(axis=(0, 1))
    #inp_mean = inp_no_start.mean(axis=(0, 1))
    #inp_norm= inp_no_start
    #inp_norm = (inp_no_start - inp_mean) / inp_std

    #vis=inp_te_np[:,1:,2:4]/np.linalg.norm(inp_te_np[:,1:,2:4],2,axis=2)[:,:,np.newaxis]
    #inp_norm=np.concatenate((inp_norm,vis),2)
    inp_norm= inp_te_np
    inp_mean=np.zeros(4)
    inp_std=np.ones(4)

    return inp_norm[:,:gt_size],inp_norm[:,gt_size:],{'mean': inp_mean, 'std': inp_std, 'seq_start': inp_te_np[:, 0:1, :].copy(),'frames':frames,'tracks':track_ids, 'maps':loc_maps}


class IndividualTfDataset(Dataset):
  def __init__(self,data,name):
    super(IndividualTfDataset,self).__init__()

    self.data=data
    self.name=name
    #self.transform = transform
    #self.csv_file = csv_file
    #self.img_dir = img_dir

    #self.mean= mean
    #self.std = std

  def __len__(self):
    return self.data['src'].shape[0]

  def __getitem__(self,index):
    #map_tensor = torch.tensor([cv2.imread(self.data['maps'][index][i][0]) for i in range(len(self.data['maps'][index]))])
    #map_tensor = map_tensor/255.0

    return {'src':torch.Tensor(self.data['src'][index]),
            'trg':torch.Tensor(self.data['trg'][index]),
            'frames':self.data['frames'][index],
            'seq_start':self.data['seq_start'][index],
            'tracks': self.data['tracks'][index],
            'maps':self.get_maps(index).view(self.get_maps(index).shape[0],self.get_maps(index).shape[3],self.get_maps(index).shape[1],self.get_maps(index).shape[2]),
            }

  def get_maps(self, index):
     map_tensor = torch.tensor([cv2.imread(self.data['maps'][index][i][0]) for i in range(len(self.data['maps'][index]))])
     map_tensor = map_tensor/255.0
     return map_tensor

def create_dataset(fraction_val, csv_path, image_path, val_size, gt, horizon, train=True, eval=False, verbose=False, step=1):
    df_train, df_val = trans_dataset_gen(fraction_val, csv_path, image_path)

    if train == True:
        raw_data = df_train
    if train == False and eval == False:
        raw_data = df_val
    data = {}
    data_src = []
    data_trg = []
    data_seq_start = []
    data_frames = []
    data_dt = []
    data_trackId = []
    data_maps = []

    val_src = []
    val_trg = []
    val_seq_start = []
    val_frames = []
    val_dt = []
    val_trackId = []

    if verbose:
        print("start loading dataset")
        print("validation set size -> %i" % (val_size))

    inp, out, info = get_strided_data_clust(raw_data, gt, horizon, 1)

    dt_frames = info['frames']
    dt_seq_start = info['seq_start']
    # dt_dataset=np.array([i_dt]).repeat(inp.shape[0])
    dt_tracks = info['tracks']
    dt_maps = info['maps']

    data_src.append(inp)
    data_trg.append(out)
    data_seq_start.append(dt_seq_start)
    data_frames.append(dt_frames)
    # data_dt.append(dt_dataset)
    data_trackId.append(dt_tracks)
    data_maps.append(dt_maps)

    data['src'] = np.concatenate(data_src, 0)  # the sequence to encoder
    data['trg'] = np.concatenate(data_trg, 0)  # the sequnce to decoder
    data['seq_start'] = np.concatenate(data_seq_start, 0)
    data['frames'] = np.concatenate(data_frames, 0)
    # data['dataset'] = np.concatenate(data_dt, 0)
    data['tracks'] = np.concatenate(data_trackId, 0)
    # data['dataset_name'] = datasets_list
    data['maps'] = np.concatenate(data_maps, 0)
    input_mean= torch.cat((torch.Tensor(data['src']),torch.Tensor(data['trg'])),1).mean((0,1))
    input_std = torch.cat((torch.Tensor(data['src']),torch.Tensor(data['trg'])),1).std((0,1))
    target_mean = input_mean[:2]
    target_std = input_std[:2]
    #print(raw_data.iloc[:,:-1])
    #mean= data['src'].mean((0,1))
    #print(data['maps'][0][2][0])
    #std= data['src'].std((0,1))
    #print(data['maps'])


    #return data['tracks']
    #return inp,out,info
    return IndividualTfDataset(data,"train"), input_mean, input_std, target_mean, target_std #IndividualTfDataset(data_val,"validation",mean,std)


def distance_metrics(gt,preds):
    errors = np.zeros(preds.shape[:-1])
    for i in range(errors.shape[0]):
        for j in range(errors.shape[1]):
            errors[i, j] = scipy.spatial.distance.euclidean(gt[i, j], preds[i, j])
    return errors.mean(),errors[:,-1].mean(),errors

    
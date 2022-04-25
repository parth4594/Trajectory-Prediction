#%matplotlib inline
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import argparse
import glob
import sys
from loguru import logger
from tracks_import import read_from_csv
import skimage.io
from os import makedirs,path


start = time.time()

def create_args():
    parser = argparse.ArgumentParser(description="ParameterOptimizer")
    parser.add_argument('--output_path', default="../Software Package/Traj_Transformer/Local_Maps/",
                        help="Dir with Local Maps", type=str)
    parser.add_argument('--output_dataset_path', default="../Software Package/Traj_Transformer/Datasets_5Hz/",
                        help="Dir with Local Maps", type=str)
    parser.add_argument('--input_path', default="../Local_Map_Generation/datasets/dataset_03/",
                        help="Dir with csv files", type=str)
    parser.add_argument('--recording_name', default="03",
                        help="Choose dataset name.", type=str)
    parser.add_argument('--scale_down_factor', default=12,
                        help="Factor by which the tracks are scaled down to match a scaled down image.",
                        type=float)
    parser.add_argument('--dimension_local', default=8,
                        help="Factor by which the tracks are scaled down to match a scaled down image.",
                        type=float)
    parser.add_argument('--skipped_frames', default=5, help="Factor by which the frames are skipped", type=float)
    parser.add_argument('--generate_reduced_data', type=bool, default=False)
    parsed_configuration = vars(parser.parse_args())
    return parsed_configuration  

def reduced_data(raw_data, skipped_frames, recording_name, output_dataset_path):
    sorted_data = raw_data.sort_values(by=['frame'])
    list_frames = sorted_data['frame']
    list_frames = list_frames.to_numpy()
    unique_frame = np.unique(list_frames)
    loop_count = unique_frame[-1:] + 1

    newd = pd.DataFrame(columns = raw_data.columns)

    for i in range(0,loop_count, extract_freq):
        df2 = raw_data[raw_data.iloc[:,2] == i]
        newd = newd.append(df2)

    newd.to_csv(output_dataset_path + recording_name + '_' + 'tracks.csv', index = False)


def global_gen(tracks_files, static_tracks_files, recording_meta_files ,background_image_path,sdf):

  tracks, static_info, meta_info = read_from_csv(tracks_files[0], static_tracks_files[0], recording_meta_files[0])

  #in_frame = static_info[tr_ID]["initialFrame"]
  #fi_frame = static_info[tr_ID]["finalFrame"] + 1

  maximum_frames = np.max([static_info[track["trackId"]]["finalFrame"] for track in tracks])

  # Create indicies for each frame
  ids_for_frame = {}
  for i_frame in range(maximum_frames):
    indices = [i_track for i_track, track in enumerate(tracks)
                if
                static_info[track["trackId"]]["initialFrame"] <= i_frame <= static_info[track["trackId"]][
                          "finalFrame"]]
    ids_for_frame[i_frame] = indices
  
  # Global image generation
  fig = None
  colors = dict(car="grey", truck="grey", trailer="grey", bicycle="grey", bus="grey", default="grey", motorcycle="grey")
  rect_style = dict(fill=True, edgecolor="k", alpha=0.4, zorder=19)
  background_image = skimage.io.imread(background_image_path)
  image_height = background_image.shape[0]
  image_width = background_image.shape[1]
  

  fig, ax = plt.subplots(1, 1)
  fig.set_size_inches(image_width/100, image_height/100)
  plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
  ax.imshow(background_image)
  figs = []
    
  for current_frame in range(len(ids_for_frame)):
    for track_ind in ids_for_frame[current_frame]:
      track = tracks[track_ind]
      track_id = track["trackId"]
      static_track_information = static_info[track_id]
      initial_frame = static_track_information["initialFrame"]
      current_index = current_frame - initial_frame

      object_class = static_track_information["class"]
      is_vehicle = object_class in ["car", "truck", "bus", "motorcycle", "bicycle", "trailer"]
      bounding_box = track["bboxVis"][current_index] / sdf
      

                
      color = colors[object_class] if object_class in colors else colors["default"]

      if is_vehicle:
      	rect = plt.Polygon(bounding_box, True, facecolor=color, **rect_style)
      	#output_path = os.path.join(output_global_path, 'Global0'+str(current_frame)+'.jpg')       
      	ax.add_patch(rect)

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1]+(3,))
    data = cv2.resize(data, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
    figs.append(data)
    #fig.savefig(output_path)
    b = ax.patches
    b.clear()
    
  return np.asarray(figs)


def convert_global_to_local(img_ar,df,df2,half_w,scale_factor,skip_frames,output_local_path,recording_name):
    images = []
    original = []
    for img_index in range(len(img_ar)):
        img = img_ar[img_index]
        org = img.copy()
        if img is not None:
          images.append(img)
          original.append(org)

    list_frame = df['frame'].to_list()
    list_frame = list(set(list_frame))
    factor_df = df2["orthoPxToMeter"]
    factor = factor_df.to_numpy()
    c = factor[0] * scale_factor

    zipped_data_im = dict(zip(list_frame,images))
    zipped_data_org = dict(zip(list_frame,original))

   	# Initialize the counter to calculate number of skipped frames.
    for key, value in zipped_data_im.items():
    #for key, value in zipped_data_im.items():
        if key%skip_frames == 0:
            centers_x = df.loc[df['frame']==key, ['xCenter']]
            centers_y = df.loc[df['frame']==key, ['yCenter']]
            track_id = df.loc[df['frame']==key, ['trackId']]
            x = centers_x.to_numpy()/c
            y = centers_y.to_numpy()/-c
            trk = track_id.to_numpy()
            a = np.stack((x,y), axis=-1)

            for j in range(len(a)):
                imgs = cv2.rectangle(zipped_data_im[key], (int(a[j][0][0]-half_w), int(a[j][0][1]-half_w)), (int(a[j][0][0]+half_w), int(a[j][0][1]+half_w)), (0, 255, 0), 2)
                ROI = zipped_data_org[key][int(a[j][0][1]-half_w):int(a[j][0][1]+half_w), int(a[j][0][0]-half_w):int(a[j][0][0]+half_w)]
                cv2.imwrite(os.path.join(output_local_path, recording_name + '_' + 'ROI_' + str(key) + '_' + str(trk[j][0]) + '.jpg') , ROI)


if __name__ == '__main__':
    config = create_args()
    recording_name = config["recording_name"]
    input_root_path = config["input_path"]
    sdf = config["scale_down_factor"]
    dimension_bb = config["dimension_local"]
    output_local_path = config["output_path"]
    skip_frames = config["skipped_frames"]
    reduced_data_gen = config["generate_reduced_data"]
    output_data_path = config["output_dataset_path"]

    if output_local_path is None:
        parent_dir = "../Software Package/Traj_Transformer/"
        output_local_path = os.path.join(parent_dir, "Local_Maps/") 
        if not os.path.isdir(output_local_path):
        	os.makedirs(output_local_path)


    tracks_files = glob.glob(input_root_path + recording_name + "*_tracks.csv")
    recording_meta_files = glob.glob(input_root_path + recording_name + "*_recordingMeta.csv")
    static_tracks_files = glob.glob(input_root_path + recording_name + "*_tracksMeta.csv")
    background_image_path = input_root_path + recording_name + "_b1segment.jpg"

    half_w = dimension_bb/2
    df = pd.read_csv(tracks_files[0])
    df2 = pd.read_csv(recording_meta_files[0])

    if reduced_data_gen == True:
        reduced_data(df,skip_frames,recording_name,output_data_path)

    global_fig = global_gen(tracks_files,static_tracks_files,recording_meta_files,background_image_path,sdf)
    convert_global_to_local(global_fig,df,df2,half_w,sdf,skip_frames,output_local_path,recording_name)

    end = time.time()
    print(end-start)
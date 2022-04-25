import argparse
import baselineUtils_transformed
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from datetime import datetime
from transformer.batch import subsequent_mask
from torch.optim import Adam,SGD, RMSprop, Adagrad
from transformer.noam_opt import NoamOpt
import numpy as np
import scipy.io
import json
import pickle
import ast
import math
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from baselineUtils_transformed import trans_dataset_gen
from baselineUtils_transformed import is_data_prepared
import cv2

def means_and_stds(train_dataset, feature_count):
    input_means=[]
    input_stds=[]
    target_means=[]
    target_stds=[]
    
    for i in np.unique(train_dataset[:]['dataset']):            
        ind=train_dataset[:]['dataset']==i

        # take feature_count "velocity" values    
        input_src = train_dataset[:]['src'][ind, :, 0:feature_count]               ##change
        input_trg = train_dataset[:]['trg'][ind, :, 0:feature_count]                ##change

        # calculate mean and std over the features
        input_src_mean = torch.cat((input_src, input_trg), 1).mean((0, 1))       #find mean across zero axis first then find the mean across axis 1
        input_src_std = torch.cat((input_src, input_trg), 1).std((0, 1))
    
        # safe mean and std values of this dataset
        input_means.append(input_src_mean)
        input_stds.append(input_src_std)

        # safe only coordinate velocities mean and std
        target_means.append(input_src_mean[:2])
        target_stds.append(input_src_std[:2])

     # calculate the mean and std of our datasets
    if len(input_means) is 1: # if it is just one dataset
        # all input features
        input_mean=input_means[0]
        input_std=input_stds[0]        
        # target coordinate distances
        target_mean=target_means[0]
        target_std=target_stds[0]
    else: 
        input_mean=torch.stack(input_means).mean(0)
        input_std=torch.stack(input_stds).std(0)        

        target_mean=torch.stack(target_means).mean(0)
        target_std=torch.stack(target_stds).std(0)

    return input_mean, input_std, target_mean, target_std


def main():
    parser=argparse.ArgumentParser(description='Train the individual Transformer model')
    parser.add_argument('--obs',type=int,default=8)
    parser.add_argument('--preds',type=int,default=12)
    parser.add_argument('--emb_size',type=int,default=512)
    parser.add_argument('--heads',type=int, default=8)
    parser.add_argument('--layers',type=int,default=6)
    parser.add_argument('--dropout',type=float,default=0.1)
    parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--val_size',type=int, default=0)
    parser.add_argument('--verbose',action='store_true')
    parser.add_argument('--max_epoch',type=int, default=1500)
    parser.add_argument('--batch_size',type=int,default=70)
    parser.add_argument('--validation_epoch_start', type=int, default=30)
    parser.add_argument('--resume_train',action='store_true')
    parser.add_argument('--delim',type=str,default='\t')
    parser.add_argument('--name', type=str, default="zara1")
    parser.add_argument('--factor', type=float, default=1.)
    parser.add_argument('--save_step', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--evaluate', type=bool, default=True)
    parser.add_argument('--model_pth', type=str)
    parser.add_argument('--dff', type=int, default=2048)
    parser.add_argument('--dec_in', type=int, default=3)
    parser.add_argument('--dec_out', type=int, default=3)
    parser.add_argument('--beta1', type=float ,default=0.9)
    parser.add_argument('--beta2', type=float, default=0.98)
    parser.add_argument('--epsilon', type=float, default=1e-9)
    parser.add_argument('--check_freq', type=int, default=50)
    parser.add_argument('--val_thresh', type=float, default=0.01)
    parser.add_argument('--fraction_val', type=float, default=0.3)
    parser.add_argument('--csv_path', type=str, default= '../Traj_Transformer/Datasets_5Hz/')
    #parser.add_argument('--dataset_folder', type=str, default='../traffic_data_fussion/Traj_Transformer/Datasets/Trasformed Datasets/')
    parser.add_argument('--image_path', type=str, default= '../Traj_Transformer/Local Maps/Local_Maps_8x8/')
    parser.add_argument('--dataset_name', type=str, default='RounD_trained')
    parser.add_argument('--data_num', type=str, default="03")
    parser.add_argument('--img_path', type=str)
    parser.add_argument('--pretrained', type=bool, default=False)

    args=parser.parse_args()
    model_name=args.name

    try:
        os.mkdir('models')
    except:
        pass
    try:
        os.mkdir('output')
    except:
        pass
    try:
        os.mkdir('output/Individual')
    except:
        pass
    try:
        os.mkdir(f'models/Individual')
    except:
        pass

    try:
        os.mkdir(f'output/Individual/{args.name}')
    except:
        pass

    try:
        os.mkdir(f'models/Individual/{args.name}')
    except:
        pass

    try:
        os.makedirs(f'{pytorch_data_save}/{args.dataset_name}')
    except:
        pass

    dataset_info  = {
    'dataset_name': args.dataset_name,
    'obs': args.obs,
    'preds': args.preds}

    pytorch_data_save = 'pytorch_data_save'

    log=SummaryWriter('logs/Ind_%s'%model_name)

    log.add_scalar('eval/mad', 0, 0)
    log.add_scalar('eval/fad', 0, 0)
    device=torch.device("cuda")

    if args.cpu or not torch.cuda.is_available():
        device=torch.device("cpu")

    args.verbose=True
	
    #trans_dataset_gen(0.3,'/home/parth/traffic_data_fussion/Traj_Transformer/Datasets/Original Datasets/dataset_03_red/','/home/parth/traffic_data_fussion/Traj_Transformer/Datasets/Transformed Datasets/',"RounD_new_img_xyvelo", "03", '/home/parth/traffic_data_fussion/Local Map Generation/Local_Maps_traj/')

######## Below is the script for creating the csv files as tensors so as to save time from pre-processing ##################

    now = datetime.now()
    save_dir_name = now.strftime("%d-%m-%Y_%Ss-%Mm-%Hh")
    available, path = baselineUtils_transformed.is_data_prepared(pytorch_data_save, dataset_info)

    if available:    
      train_dataset = torch.load(os.path.join(path, "train", 'train.pt'))    
      val_dataset = torch.load(os.path.join(path, "val", 'val.pt'))
      input_mean = torch.load(os.path.join(path, "inp_m", 'inp_m.pt')) 
      input_std =  torch.load(os.path.join(path, "inp_s", 'inp_s.pt'))
      target_mean = torch.load(os.path.join(path, "trg_m", 'trg_m.pt'))
      target_std = torch.load(os.path.join(path, "trg_s", 'trg_s.pt'))
      print(f'Loaded prepared data with: {dataset_info}')

    else:
      train_dataset,input_mean, input_std, target_mean, target_std = baselineUtils_transformed.create_dataset(args.fraction_val,args.csv_path,args.image_path,0,args.obs,args.preds,train=True, eval= False, verbose=False)
      val_dataset,input_mean_v, input_std_v, target_mean_v, target_std_v = baselineUtils_transformed.create_dataset(args.fraction_val,args.csv_path,args.image_path,0, args.obs,
                                                                  args.preds, train=False, eval= False,
                                                                  verbose=args.verbose)
      dss = [train_dataset, val_dataset,input_mean, input_std, target_mean, target_std]
      labels = ['train', 'val', 'inp_m','inp_s','trg_m','trg_s']
      for ds, label in zip(dss, labels):
        save_dir_path = os.path.join(pytorch_data_save, args.dataset_name, save_dir_name, label)  
        try:
          os.makedirs(save_dir_path)
        except:
          pass
        torch.save(ds, os.path.join(save_dir_path, f'{label}.pt'))

      save_dir_path = os.path.join(pytorch_data_save, args.dataset_name, save_dir_name)
      torch.save(dataset_info, os.path.join(save_dir_path, 'info.pt'))
      print(f'Prepared and saved data with: {dataset_info}')
###################################################################################################################

    #print(train_dataset[0]['maps'])
    #input('stop11')
    feature_count = train_dataset[0]['src'].shape[1]
    #print("feature_count \n", feature_count)
    #input_mean, input_std, target_mean, target_std = means_and_stds(train_dataset, feature_count)
    #scipy.io.savemat(f'models/Individual/{args.name}/norm.mat',{'mean':input_mean.cpu().numpy(),'std':input_std.cpu().numpy()})
    

######### Dataloader preparation for train and val data used for the network ################

    from transformer.individual_TF import IndividualTF
    model=IndividualTF(feature_count, args.dec_in, args.dec_out, N=args.layers, in_channels=3, pretrained=args.pretrained, ##change
                   d_model=args.emb_size, d_ff=args.dff, h=args.heads, dropout=args.dropout,mean=[0,0],std=[0,0]).to(device) #to(device) means to transfer a tensor to a GPU
    if args.resume_train:
        model.load_state_dict(torch.load(f'models/Individual/{args.name}/{args.model_pth}')) #A state_dict is simply a Python dictionary object that maps each layer to its parameter tensor.

    tr_dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    #test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    #optim = SGD(list(a.parameters())+list(model.parameters())+list(generator.parameters()),lr=0.01)
    #sched=torch.optim.lr_scheduler.StepLR(optim,0.0005)
    optim = NoamOpt(args.emb_size, args.factor, len(tr_dl)*args.warmup,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(args.beta1, args.beta2), eps=args.epsilon))

    now = datetime.now()
    save_time = now.strftime("%d-%m-%Y_%Hh-%Mm-%Ss") 

    save_comment = f'{save_time}_he={args.heads}_la={args.layers}_es={args.emb_size}_ba={args.batch_size}'
    train_comment = (f'heads={args.heads} ' +
                    f'layers={args.layers} ' +
                    f'emb_size={args.emb_size} ' +
                    f'batch_size={args.batch_size} ')


    next_row = {"date": save_time, "current_epoch": 0, "max_epoch": args.max_epoch, "training_time": 0.0, "total_loss": 0.0, "agv_loss": 0.0, "mad": 0.0, "fad": 0.0,
                        "layers": args.layers, "emb_size": args.emb_size, "heads": args.heads, "dropout": args.dropout}
      
###########################################################################################################    
    import time
    print(f'Training for: {train_comment}')
    t0 = time.time()

    #optim=Adagrad(list(a.parameters())+list(model.parameters())+list(generator.parameters()),lr=0.01,lr_decay=0.001)
    epoch=0
    epoch_check_freq = args.check_freq # our k
    val_rel_err_thresh = args.val_thresh
    val_err_min = math.inf  
    val_rel_err = math.inf
    val_errs = []
       
############# Training Loop ################################################################################

    while val_rel_err > val_rel_err_thresh and epoch < args.max_epoch:
        epoch_loss=0
        e_t0 = time.time()
        model.train()

        train_batch_len = len(tr_dl)

        for id_b,batch in enumerate(tr_dl):
            optim.optimizer.zero_grad()

            # the input consists of all features
            #print('original0 \n',batch['src'][0].shape)
            #print('input_mean \n',input_mean.shape)
            #print('input_std \n',input_std.shape)
            #print('batch \n', batch['frames'])
            #print('batch_t \n', batch['tracks'])
            #print('std \n', input_std)
            #input("stop0")
            inp=(batch['src'][:,:,0:feature_count].to(device)-input_mean.to(device))/input_std.to(device)
            #print('inp \n',inp)
            inp_img = batch['maps'].to(device)
            #print('img \n', inp_img)
            #input('stop')
            target=(batch['trg'][:, :-1, 0:2].to(device)-target_mean.to(device))/target_std.to(device)       #change  #our target is the coordinates why -1?
            #print('target0 \n',target[0])  
            #input("stop2")
            target_c=torch.zeros((target.shape[0],target.shape[1],1)).to(device)
            #print('target_c \n',target_c[0])
            target=torch.cat((target,target_c),-1)
            #print('target2 \n',target[0])
            #print('target2 \n',target.shape)      
            start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(target.shape[0],1,1).to(device)
            #print('start_of_seq \n',start_of_seq[0])

            dec_inp = torch.cat((start_of_seq, target), 1)
            #print('dec_inp \n',dec_inp[0])
            #print('dec_inp \n',dec_inp.shape)
            #input("stop3")
            src_att = torch.ones((inp.shape[0], 1,inp.shape[1])).to(device)
            #print('src_att \n',src_att[0])
            trg_att=subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0],1,1).to(device)
            #print('trg_att \n',trg_att[0])

            pred=model(inp, inp_img, dec_inp, src_att, trg_att)
            #print('pred \n',pred[0])

            y_pred = pred[:, :,0:2].contiguous().view(-1, 2)      #predicts 3 values. X and Y coordinates and an end token. We only select coordinates
            #print('y_pred \n',y_pred)
            #print('y_pred_size \n',y_pred.shape)
            y_real = ((batch['trg'][:, :, 0:2].to(device)-target_mean.to(device))/target_std.to(device)).contiguous().view(-1, 2).to(device) #change
            #print('y_real \n',y_real)
            #print('y_real_size \n',y_real.shape)
            #input("stop4")              
            loss = F.pairwise_distance(y_pred, y_real).mean() + torch.mean(torch.abs(pred[:,:,2]))
            #print('loss \n',loss)
              
          

            loss.backward()
            optim.step()

            print("train epoch %03i/%03i  batch %04i / %04i loss: %7.4f" % (epoch, args.max_epoch, id_b, len(tr_dl), loss.item()))
            epoch_loss += loss.item()
        #sched.step()
        log.add_scalar('Loss/train', epoch_loss / len(tr_dl), epoch)

############## Validation Loop ########################################################################################

        min_mad = math.inf
        with torch.no_grad():
            model.eval()

            val_loss=0
            step=0
            gt = []
            pr = []
            inp_ = []
            tracks = []
            frames = []
            #dt = []

            for id_b, batch in enumerate(val_dl):
                inp_.append(batch['src'])
                gt.append(batch['trg'][:, :, 0:2])
                frames.append(batch['frames'])
                tracks.append(batch['tracks'])
                #dt.append(batch['dataset'])

                #print('source_inp_val0 \n',batch['src'][0].shape)
                #print('source_inp_val0 \n',batch['src'][0])
                #print('ground_truth_val0 \n',batch['trg'][:,:,0:2][0])
                #input("stop5")
                inp = (batch['src'][:, :, 0:feature_count].to(device) - input_mean.to(device)) / input_std.to(device)  #change
                #print('inp_val \n', inp[0].shape)
                inp_img = batch['maps'].to(device)
                src_att = torch.ones((inp.shape[0], 1, inp.shape[1])).to(device)
                #print('src_att_val \n', src_att[0])
                start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(inp.shape[0], 1, 1).to(device)
                #print('start_of_seq \n', start_of_seq)
                dec_inp = start_of_seq
                #print('dec_inp_val \n', dec_inp[0]) 

                for i in range(args.preds):
                    trg_att = subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0], 1, 1).to(device)
                    #print('trg_att_val2 \n', trg_att)
                    out = model(inp, inp_img, dec_inp, src_att, trg_att)
                    #print('out_val \n', out[0])
                    dec_inp = torch.cat((dec_inp, out[:, -1:, :]), 1)
                    #print('dec_inp_val \n', dec_inp[0])

                preds_tr_b = (dec_inp[:, 1:, 0:2] * target_std.to(device) + target_mean.to(device)).cpu() #.numpy() .cumsum(1) + batch['src'][:, -1:, 0:2].cpu().numpy() #'1'dec_inp means we are leaving out the token
                #print('predicted_val \n', preds_tr_b[0])
                pr.append(preds_tr_b)
                print("val epoch %03i/%03i  batch %04i / %04i" % (epoch, args.max_epoch, id_b, len(val_dl)))
                

            tracks = np.concatenate(tracks, 0)
            frames = np.concatenate(frames, 0)
            #dt = np.concatenate(dt, 0)
            gt = np.concatenate(gt, 0)
            #dt_names = val_dataset.data['dataset_name']
            pr = np.concatenate(pr, 0)
            mad, fad, errs = baselineUtils_transformed.distance_metrics(gt, pr)
            log.add_scalar('validation/MAD', mad, epoch)
            log.add_scalar('validation/FAD', fad, epoch)
            if mad < min_mad:
                min_mad = mad
            val_errs.append(mad)

            train_time = f'{time.time()-e_t0:03.4f}'
            avg_loss = epoch_loss / len(tr_dl)
            update = {"current_epoch": epoch, "training_time": train_time, "total_loss": epoch_loss, "agv_loss": avg_loss, "mad": mad, "fad": fad}
            next_row.update(update)
################################################################################################################################

            #if epoch==1:
                #torch.save(model.state_dict(),f'models/{args.name}/{epoch:05d}.pth')

            if epoch%args.save_step==0:         
                print(f"Epoch: {epoch:03d} Training time: {train_time}  Loss: {epoch_loss:03.4f}  Avg. Loss: {avg_loss:03.4f} MAD: {mad:03.4f} FAD: {fad:03.4f}") 
      
            if epoch%epoch_check_freq==0:
                if epoch > epoch_check_freq:
                    curr_val_err_min = min(val_errs)
                    val_rel_err = (val_err_min - curr_val_err_min)/val_err_min
                    print(f'val_err_min: {val_err_min:03.4f} curr_val_err_min: {curr_val_err_min:03.4f} val_rel_err: {val_rel_err:03.4f}')
        
                val_err_min = min(val_errs)
                val_errs = []        

                if val_rel_err < val_rel_err_thresh:
                    print(f'Reached less than {val_rel_err_thresh} val error change over last {epoch_check_freq} epochs. Stopping training')
                if epoch >= args.max_epoch:
                    print(f'Reached max epoch of {args.max_epoch}. Stopping training.')

        epoch+=1

    total_train_time = time.time()-t0
    print(f"Total training time: {total_train_time:07.4f}")

    

if __name__=='__main__':
    main()

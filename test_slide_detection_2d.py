# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 18:16:03 2021

Code is partly based on https://git.aweirdimagination.net/perelman/slide-detector

@author: Aline Sindel
"""
import shutil

def detect_initial_slide_transition_candidates_resnet2d(net, videofile, base, roi, load_size_roi, out_dir, opt):
    # 清空 log/video_name/ 目录
    output_slide_dir = os.path.join('log', base)
    if os.path.exists(output_slide_dir):
        shutil.rmtree(output_slide_dir)
    os.makedirs(output_slide_dir)

import os
import random
import argparse
from gc import freeze

import numpy as np
import cv2

import torch
import torch.nn as nn

import decord
from decord import VideoReader

from model import *

from data.data_utils import *
from data.test_video_clip_dataset import BasicTransform
from imageSimilarity import ORB_img_similarity
import cv2
import os

def filter_static_slides_with_orb(slide_ids, frame_ids_1, frame_ids_2, vr_fullres, base, orb_sim_thresh=0.75):
    """
    只对 static slide 做 ORB 去重，相似度高的 slide 会被剔除
    """
    filtered_slide_ids = []
    filtered_frame_ids_1 = []
    filtered_frame_ids_2 = []

    prev_mid_img = None
    prev_slide_id = None
    debug_dir = os.path.join("log", base, "filtered_by_orb")
    os.makedirs(debug_dir, exist_ok=True)

    for sid, f1, f2 in zip(slide_ids, frame_ids_1, frame_ids_2):
        if sid == -1:
            continue

        mid_idx = (f1 + f2) // 2 - 1
        if mid_idx < 0 or mid_idx >= len(vr_fullres):
            continue

        frame = vr_fullres[mid_idx]
        if hasattr(frame, 'asnumpy'):
            frame = frame.asnumpy()
        elif hasattr(frame, 'numpy'):
            frame = frame.numpy()
        elif isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()

        if frame.dtype != np.uint8:
            frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
        if frame.ndim == 3 and frame.shape[2] == 1:
            frame = frame[:, :, 0]

        if prev_mid_img is None:
            prev_mid_img = frame
            prev_slide_id = sid
            filtered_slide_ids.append(sid)
            filtered_frame_ids_1.append(f1)
            filtered_frame_ids_2.append(f2)
            continue

        sim = ORB_img_similarity(prev_mid_img, frame)
        print(f"[Post-ORB] Slide {prev_slide_id} vs {sid}: sim={sim:.3f}")

        if sim > orb_sim_thresh:
            print(f"[Filtered] Slide {sid} is too similar to {prev_slide_id}, sim={sim:.3f}")
            cv2.imwrite(os.path.join(debug_dir, f"filtered_prev_{prev_slide_id:03d}.jpg"),
                        cv2.cvtColor(prev_mid_img, cv2.COLOR_GRAY2BGR if prev_mid_img.ndim==2 else cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(debug_dir, f"filtered_curr_{sid:03d}.jpg"),
                        cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR if frame.ndim==2 else cv2.COLOR_RGB2BGR))
            continue

        filtered_slide_ids.append(sid)
        filtered_frame_ids_1.append(f1)
        filtered_frame_ids_2.append(f2)
        prev_mid_img = frame
        prev_slide_id = sid

    return filtered_slide_ids, filtered_frame_ids_1, filtered_frame_ids_2


def detect_initial_slide_transition_candidates_resnet2d(net,videofile, base, roi, load_size_roi, out_dir, opt):
    # 清空 log/video_name/ 目录
    output_slide_dir = os.path.join('log', base)
    if os.path.exists(output_slide_dir):
        shutil.rmtree(output_slide_dir)  # 删除整个目录
    os.makedirs(output_slide_dir)  # 重建空目录
    # load video file
    vr = VideoReader(videofile, width=load_size_roi[1], height=load_size_roi[0])
    # 用于保存高清关键帧（原始分辨率）
    vr_fullres = VideoReader(videofile)

    #determine number of frames
    N_frames = len(vr)
    print("N_frames reported by decord:", len(vr))

    anchor_frame = None
    anchor_frame_idx = -1
    video_frame_idx = None
    prev_video_frame_idx = None
    slide_id = -1
    slide_ids = []
    frame_ids_1 = []
    frame_ids_2 = []
    
    if opt.in_gray:
        data_shape = "2_channel"
        opt.input_nc = 2
    else:
        data_shape = "6_channel"
        opt.input_nc = 6
    my_transform = BasicTransform(data_shape = data_shape) #, blur = opt.blur)
    activation = nn.Sigmoid()
    
    for i in range(0,N_frames,1):
        frame = vr[i]

        imgs = torch.zeros((2,opt.patch_size,opt.patch_size,int(opt.input_nc/2)))
            
        if opt.in_gray: #opencv rgb2gray for torch
            frame = 0.299*frame[...,0]+0.587*frame[...,1]+0.114*frame[...,2]
            frame = frame.unsqueeze(2)
        
        # crop to bounding box region
        frame = crop_frame(frame,roi[0],roi[1],roi[2],roi[3]) 
        #scale to max size (in case patch size changed)
        img_max_size = max(frame.shape[0], frame.shape[1])
        scaling_factor = opt.patch_size / img_max_size
        if scaling_factor != 1:
            if isinstance(frame, torch.Tensor):
                frame = frame.permute(1, 2, 0).cpu().numpy()  # 如果是 CHW 格式
            frame = cv2.resize(frame, (round(frame.shape[1] * scaling_factor), round(frame.shape[0] * scaling_factor)), interpolation = cv2.INTER_NEAREST)
            H,W,C = frame.shape
            imgs[1,:H,:W,:C] = frame
        else:
            H,W,C = frame.shape
            imgs[1,:H,:W,:C] = frame
        
        #set anchor
        if anchor_frame == None:
            imgs[0,:H,:W,:C] = frame
            anchor_frame_idx = i 
            anchor_frame = frame
        else:
            imgs[0,:H,:W,:C] = anchor_frame
            
        imgs = my_transform(imgs)
        
        with torch.no_grad():
            imgs = imgs.cuda()
             
            pred = net(imgs.unsqueeze(0))
            pred = pred.squeeze(1)            
            pred = activation(pred)
            #print(pred)
            if pred<0.5: #transition (class 0)
                if (i - anchor_frame_idx) > opt.slide_thresh: #static frame
                    if video_frame_idx is not None: 
                        if (video_frame_idx - prev_video_frame_idx) > opt.video_thresh:
                            print("video frame {} at {} to {}".format(-1,prev_video_frame_idx+1, video_frame_idx+1))
                            slide_ids.append(-1)
                            frame_ids_1.append(prev_video_frame_idx+1)
                            frame_ids_2.append(video_frame_idx+1) 
                        video_frame_idx = None
                        prev_video_frame_idx = None
                    
                    slide_id += 1
                    print("static slide {} at {} to {}".format(slide_id,anchor_frame_idx+1, i))
                    slide_ids.append(slide_id)
                    frame_ids_1.append(anchor_frame_idx+1)
                    frame_ids_2.append(i)

                else:
                   #video frame or grad transition
                   video_frame_idx = anchor_frame_idx
                   if prev_video_frame_idx is None:
                       prev_video_frame_idx = anchor_frame_idx                  
                #update anchor
                anchor_frame_idx = i
                anchor_frame = frame 
               
    print(len(frame_ids_1)) 
    frame_ids_1 = np.array(frame_ids_1)
    frame_ids_2 = np.array(frame_ids_2)
    # 🔁 ORB 相似度后处理筛选
    slide_ids, frame_ids_1, frame_ids_2 = filter_static_slides_with_orb(
        slide_ids, frame_ids_1, frame_ids_2, vr_fullres, base, orb_sim_thresh=0.5
    )

    #write to file
    logfile_path = os.path.join(out_dir, base + "_results.txt")
    f = open(logfile_path, "w")
    f.write('Slide No, FrameID0, FrameID1\n')
    f.close()    

    for slide_id,frame_id_1,frame_id_2 in zip(slide_ids,frame_ids_1,frame_ids_2):               
        f = open(logfile_path, "a")
        f.write("{}, {}, {}\n".format(slide_id,frame_id_1,frame_id_2))
        f.close()
    output_slide_dir = os.path.join('log', base)
    os.makedirs(output_slide_dir, exist_ok=True)

    slide_index = 1  # 用于生成 slide_001 这样从1开始的编号
    for slide_id, frame_id_1, frame_id_2 in zip(slide_ids, frame_ids_1, frame_ids_2):
        if slide_id == -1:
            continue  # 跳过 video frame 段落

        mid_frame_id = (frame_id_1 + frame_id_2) // 2
        mid_frame_idx = mid_frame_id - 1  # decord 是 0-based

        # 保存高清帧
        if 0 <= mid_frame_idx < len(vr_fullres):
            full_frame = vr_fullres[mid_frame_idx]
            if hasattr(full_frame, 'asnumpy'):
                full_frame = full_frame.asnumpy()
            else:
                full_frame = full_frame.numpy()

            filename = f"slide_{slide_index:03d}_{mid_frame_id}.jpg"
            filepath = os.path.join(output_slide_dir, filename)
            cv2.imwrite(filepath, cv2.cvtColor(full_frame, cv2.COLOR_RGB2BGR))
            slide_index += 1

def test_resnet2d(opt):
    torch.manual_seed(0)
    random.seed(0)

    if os.path.exists(opt.out_dir)==False:
        os.makedirs(opt.out_dir)
    
    ####### Create model
    # --------------------------------------------------------------- 
    net = define_resnet2d(opt)       
    net = net.cuda()
    net = loadNetwork(net, opt.model_path, checkpoint=opt.load_checkpoint, prefix='')
    net.eval()

    #### Create dataloader
    # ---------------------------------------------------------------  
    video_dir = opt.dataset_dir + "/videos/" + opt.phase   

    videoFilenames = []
    videoFilenames.extend(os.path.join(video_dir, x)
                                         for x in sorted(os.listdir(video_dir)) if is_video_file(x))
    
    roi_path = os.path.join(opt.dataset_dir,"videos", opt.phase+'_bounding_box_list.txt')
    rois = read_labels(roi_path)

    decord.bridge.set_bridge('torch')
    
    for k,videofile in enumerate(videoFilenames):
        print("Processing video No. {}: {}".format(k+1, videofile))
        
        base, roi, load_size_roi = determine_load_size_roi(videofile, rois, opt.patch_size)
         
        detect_initial_slide_transition_candidates_resnet2d(net, videofile, base, roi, load_size_roi, opt.out_dir, opt)

            
                  
if __name__ == '__main__':
    parser = argparse.ArgumentParser('slide_detection') 
    parser.add_argument('--dataset_dir', help='path to dataset dir',type=str, default='E:/video_experiment/project/Data/datasets')

    parser.add_argument('--out_dir', help='path to result dir',type=str, default='E:/video_experiment/project/Code/SliTraNet/results/test/SliTraNet-gray-RBG')
    parser.add_argument('--backbone_2D', help='name of backbone (resnet18 or resnet50)',type=str, default='resnet18')
    parser.add_argument('--model_path', help='path of weights',type=str, default='E:/video_experiment/project/Code/SliTraNet/weights/Frame_similarity_ResNet18_gray.pth')
    parser.add_argument('--load_checkpoint', type=bool, default=False, help='model_path is path to checkpoint (True) or path to state dict (False)')
    parser.add_argument('--slide_thresh', type=int, default=8, help='threshold for minimum static slide length')
    parser.add_argument('--video_thresh', type=int, default=13, help='threshold for minimum video length to distinguish from gradual transition')
    parser.add_argument('--patch_size', type=int, default=256, help='network input patch size')
    parser.add_argument('--n_class', type=int, default=1, help='number of classes')
    parser.add_argument('--input_nc', type=int, default=2, help='number of input channels for ResNet: gray:2, RGB:6')
    parser.add_argument('--in_gray', type=bool, default=True, help='run network with grayscale input, else RGB')
    parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')

    opt = parser.parse_args()  

    test_resnet2d(opt)
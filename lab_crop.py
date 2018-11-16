# coding: utf-8
"""
crop人脸并转换landmarks坐标
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
import random

import imgaug as ia
from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split
from lab_utils import drawGaussianHeatmap

import argparse

ia.seed(1)

def parse_args():
    parser = argparse.ArgumentParser(description='data preparing setup')
    
    parser.add_argument('--input-dir', type=str, default='./data/WFLW_images', help='image path')
    parser.add_argument('--output-dir', type=str, default='./data/WFLW_crop_256_npy', help='save path')
    parser.add_argument('--heatmap-size', type=tuple, default=(64, 64), help='')
    parser.add_argument('--anno-save-path', type=str, default='./data/train.txt', help='path for result landmarks annotation')
    parser.add_argument('--anno-path', type=str, default='./data/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt', help='annotation file path')
    
    args = parser.parse_args()
    return args

def read_anno(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    return lines

def save_txt(filename, lst):
    with open(filename, 'w') as f:
        for line in lst:
            f.write(line)
    return True

def resplit(rate):
    """重新分割train test，输入test的比例
    resplit(0.1)
    """
    rate = float(rate)
    train = read_anno('/home/snowcloud/FaceDatasets/WFLW/WFLW_crop_annotations/list_98pt_rect_attr_train_test/train_256.txt')
    test = read_anno('/home/snowcloud/FaceDatasets/WFLW/WFLW_crop_annotations/list_98pt_rect_attr_train_test/test_256.txt')

    train_test = train + test
    new_train, new_test = train_test_split(train_test, test_size=rate, random_state=42)
    
    save_txt('/home/snowcloud/FaceDatasets/WFLW/WFLW_crop_annotations/list_98pt_rect_attr_train_test/resplit_%s_256_train.txt' % rate, new_train)
    save_txt('/home/snowcloud/FaceDatasets/WFLW/WFLW_crop_annotations/list_98pt_rect_attr_train_test/resplit_%s_256_test.txt' % rate, new_test)
    
    return True

def rectangle_to_square_v2(bbox):
    """修正bbox长方形为正方形，若触及边界不用修正，留待padding"""
    x_len = abs(bbox[0]-bbox[2])
    y_len = abs(bbox[1]-bbox[3])
    max_len = max(x_len, y_len)
    
    center_x = (bbox[0]+bbox[2]) / 2.
    center_y = (bbox[1]+bbox[3]) / 2.
    
    return center_x-max_len/2., center_y-max_len/2., center_x+max_len/2., center_y+max_len/2.

def refine_bbox(bbox, landmarks):
    """
    输入原始bbox和landmarks，输出修正过的bbox
    修正方法：求出landmarks的范围，和原始bbox取并集，并取正方形
    """
    land_x = [landmarks[i] for i in range(0, len(landmarks), 2)]
    land_y = [landmarks[i+1] for i in range(0, len(landmarks), 2)]
    
    land_bbox = [min(land_x), min(land_y), max(land_x), max(land_y)]
    
    refined_bbox = [min(bbox[0], land_bbox[0]), min(bbox[1], land_bbox[1]), max(bbox[2], land_bbox[2]), max(bbox[3], land_bbox[3])]
    refined_bbox = rectangle_to_square_v2(refined_bbox)
    
    return refined_bbox

toInt = lambda x: int(eval(x))

def get_anno(anno):
    """从一行标注里获取并返回landmarks, bbox, image_name"""
    anno = anno.split()

    landmarks = anno[:196]
    landmarks = [toInt(e) for e in landmarks]
    bbox = anno[196:200]
    bbox = [toInt(e) for e in bbox]
    image_name = anno[-1]
    
    return landmarks, bbox, image_name

def get_anno_test(anno):
    anno = anno.split()

    landmarks = anno[:196]
    landmarks = [toInt(e) for e in landmarks]
    image_name = anno[-2]
    heatmap_name = anno[-1]
    
    return landmarks, image_name, heatmap_name

def save(filename, images, heatmaps, landmarks):
    """保存images, heatmaps, landmarks至文件"""
    with open(filename, 'w') as f:
        for i, image in enumerate(images):
            image_path = image
            heatmap_path = heatmaps[i]
            landmark = landmarks[i]
            for value in landmark:
                f.write(str(value) + ' ')
            f.write(image_path + ' ' + heatmap_path +'\n')
    return True

def to_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def key_to_list(keypoint):
    """把imgaug的keypoint转为list并返回"""
    result = np.ravel(np.round(keypoint.get_coords_array())).astype(int).tolist()
    return result

def draw_heatmaps(img):
    plt.figure(figsize=(20, 10))
    for i in range(2):
        for j in range(7):
            plt.subplot(2, 7, i*7+j+1)
            plt.imshow(img[i*2+j])
    plt.show()
    
    return True

def draw_heatmaps_v2(img, heatmap):
    one = np.zeros((14, 64, 64), dtype=np.float32)
    img = np.moveaxis(img, 0, -1)
    img = cv2.resize(img, (64, 64))
    one[0] = img
    one[1:] = heatmap[:]
    draw_heatmaps(one)
    
    return True
    
def img_aug(bgr, bbox, landmarks):
    """变换图片，获取crop人脸及变换后的关键点"""
    left = int(bbox[0])
    top = int(bbox[1])
    right = int(bgr.shape[1] - bbox[2])
    bottom = int(bgr.shape[0] - bbox[3])
    
    # 定义一个变换序列，可扩展
    seq=iaa.Sequential([
#         iaa.Grayscale(alpha=1, from_colorspace='BGR'), 
        iaa.CropAndPad(px=(-top, -right, -bottom, -left)), 
        iaa.Scale(256), 
    ])

    seq_det = seq.to_deterministic()
        
    keypoint_lst = [ia.Keypoint(x=landmarks[i], y=landmarks[i+1]) for i in range(0, len(landmarks), 2)]
    keypoints=ia.KeypointsOnImage(keypoint_lst, shape=bgr.shape)
    
    image_aug = seq_det.augment_images([bgr])[0]
    keypoints_aug = seq_det.augment_keypoints([keypoints])[0]
    
    return image_aug, keypoints_aug

def handle_v2(anno):
    """从原始标注获取信息，保存结果图片并返回landmarks和结果图片路径list待保存"""
    """用imgaug变换图片及坐标，改善错位情况"""
    landmarks, bbox, image_name = get_anno(anno)
    refined_bbox = refine_bbox(bbox, landmarks)
    bbox = refined_bbox
    path = os.path.join(args.input_dir, image_name)
    bgr = cv2.imread(path)
    
    image_aug, keypoints_aug = img_aug(bgr, bbox, landmarks)
    keypoints_aug_list = key_to_list(keypoints_aug)
    image_points = keypoints_aug.draw_on_image(image_aug, size=4, copy=True, color=[25, 100, 0]) # 把关键点画在图上
    
    filename = os.path.join(args.output_dir, image_name)
#     filename = '/home/snowcloud/FaceDatasets/WFLW/WFLW_crop_split_256_npy/' + image_name
#     filename = '/home/snowcloud/FaceDatasets/WFLW/WFLW_crop_img/' + image_name
    
    (dst_dir, tempfilename) = os.path.split(filename)
    (shotname, extension) = os.path.splitext(tempfilename)
    
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    # 避免同一张图有多个人脸标注的情况
    i = 1
    original = dst_dir+'/'+shotname+'.npy'
    original_hm = dst_dir+'/'+shotname+'_heatmap.npy'
#     original = dst_dir+'/'+shotname+'.jpg'
    dst_file = original
    dst_hm = original_hm
    while os.path.exists(dst_file):
        dst_file = dst_dir+'/'+shotname+'_%s.npy' % str(i)
        dst_hm = dst_dir+'/'+shotname+'_%s_heatmap.npy' % str(i)
#         dst_file = dst_dir+'/'+shotname+'_%s.jpg' % str(i)
        i += 1
    
    gray_aug = to_gray(image_aug)
    
    hm = drawGaussianHeatmap(gray_aug, keypoints_aug_list)
    hm = cv2.resize(hm, args.heatmap_size)
    hm = np.moveaxis(hm, -1, 0).astype('float32')
    
    gray_aug = gray_aug[np.newaxis, :].astype('float32')
    
    np.save(dst_file, gray_aug)
    np.save(dst_hm, hm)
    
#     img_heatmap = np.concatenate((gray_aug, hm), axis=0).astype('float32')
    
#     cv2.imwrite(dst_file, image_points) 
#     np.save(dst_file, img_heatmap)
    
    return dst_file, dst_hm, keypoints_aug_list

if __name__ == '__main__':
    
    args = parse_args()
    print(args)
    
    images = []
    heatmaps = []
    landmarks = []
    
    with open(args.anno_path) as f:
        lines = f.readlines()
    
    with tqdm(lines[:]) as pbar:
        
        for one in pbar:
            dst_file, dst_hm, keypoints_aug_list = handle_v2(one)
            
            landmarks.append(keypoints_aug_list)
            images.append(dst_file)
            heatmaps.append(dst_hm)
            
    save(args.anno_save_path, images, heatmaps, landmarks)





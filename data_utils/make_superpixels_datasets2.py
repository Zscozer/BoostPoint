#制作一个dataloader方法，pixel2points
import glob
import os
import random

import cv2
import torch
import torchvision
from PIL import Image
from fast_slic.avx2 import SlicAvx2
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import transforms


def crop_image(re_img,new_height,new_width):
    re_img=Image.fromarray(np.uint8(re_img))
    width, height = re_img.size
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    crop_im = re_img.crop((left, top, right, bottom)) #Cropping Image
    crop_im = np.asarray(crop_im)
    return crop_im

transform1 = transforms.Compose([
            transforms.ToTensor(),
        ])
transform2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

def pixel2points(img_path,img_w= 2000,img_h= 2000,new_width=1200,new_height=1200):
    img_path_org = img_path
    flag = 1
    img = cv2.imread(img_path)
    # if img_w != 2000:
    img = cv2.resize(img,(img_w,img_h))
    img = crop_image(cv2.resize(img, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC), new_height=new_height,
                       new_width=new_width)
    slic = SlicAvx2(num_components=2000, compactness=10, min_size_factor=0, convert_to_lab=False)
    assignment = slic.iterate(img)
    data = []
    for i, cluster in enumerate(slic.slic_model.clusters):
        yx = np.array(cluster['yx'])/new_height-0.5
        # print(yx)
        # color = ((np.array(cluster['color'])/255.0)-mean)/std
        color = np.array(cluster['color'])/255.0
        # print("color",color)
        if np.any(color):
            color = (color-mean)/std #归一化
            data_ = np.append(yx, color)
            # data_ = np.append(yx, np.array(cluster['num_members']))
            data.append(np.array(data_))
    pixels = np.unique(np.array(data), axis=0)
    # print(pixels)
    # Center crop
    if pixels.shape[0] < 100:
        flag = 0
        print("Too small or all zeros.")
        return pixels,flag

    if pixels.shape[0]<1024:
        print("smaller than 1024")
        # print("img_path_org", img_path_org)
        print("Current num of points", pixels.shape[0])
        img_h = img_h + 500
        img_w = img_w + 500
        if img_w < 5000:

            pixels,flag = pixel2points(img_path_org,img_h=img_h,img_w=img_w,new_height=int(new_height),new_width=int(new_width))
            print("pixels",pixels.shape)
            # pixels = transform1(np.array(pixels))[0]
            print("Current num of points", pixels.shape[0])
            return pixels,flag




    pixels = transform1(np.array(pixels))[0]
    index = torch.LongTensor(np.linspace(0, pixels.shape[0] - 1, num=1024, dtype=int))
    print("index",index)
    pixels = torch.index_select(pixels, 0, index)
    # print("pixels", pixels.shape)
    return pixels,flag
def get_img(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        imgs_info = f.readlines()
        imgs_info = list(map(lambda x: x.strip().split('\t'), imgs_info))
        return imgs_info


def save_function(img_feature,label,root,save_info):
    classes,name = save_info.split('/')
    print("classes",classes)

    save_path = os.path.join(root,classes)
    name = name[:-4]
    name = name+'.npy'
    save_path = os.path.join(save_path,name)
    img_feature = img_feature.detach().numpy()

    data = {'pixels':img_feature,'label':label}
    # print('DATA',type(data))
    np.save(save_path, data)
    # print(data)
    # print(img_feature)
    # print("save_path", save_path)

if __name__ == '__main__':
    root = 'data/ShapeNet_render/all_list.txt' #root path of rendered images
    data_path = '/'.join(root.split('/')[0:-1])
    # data_root = "/home/zhg/dataset/Cross_ModelNet40"

    data_root = "data/"


    catfile = os.path.join(data_root, 'render_name.txt')
    cat = [line.rstrip() for line in open(catfile)]
    # print(cat)

    render_path = "Superpixels"
    path = os.path.join(data_root, render_path)
    if not os.path.exists(path):
        os.makedirs(path)

    for i in cat:
        class_path = os.path.join(path, i)
        # print(class_path)
        if not os.path.exists(class_path):
            os.makedirs(class_path)
    # all_filepath = load_pic(root)
    # print(all_filepath)
    imgs_info = get_img(root)
    # print(imgs_info)
    for name in imgs_info:
        # print(name)
        path_lst = name[0].split(' ')
        # print(path_lst)
        img_path, label = path_lst
        # print(img_path)
        save_info = img_path
        img_path = os.path.join(data_path, img_path)

        img_features,flag = pixel2points(img_path,2000,2000)
        if flag !=0:

            label = int(label)

            save_path = os.path.join(data_root,'Superpixels')
            # print("save_path",save_path)
            save_function(img_features,label,save_path,save_info)


import os
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm
import glob
import shutil
import random
import pickle
import numpy as np

## [START]: parsing
def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    import re
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]
## [END]: PARSING

## [START]: Get Parent Directory
def get_parent_dir(directory, im_format='.png'):
    sep = directory.split(os.sep)
    _dir = os.path.dirname(os.path.dirname(directory))
    return _dir, sep[-1][:-4] + im_format
## [END]: Get Parent Directory

## [START]: Choose img file Except Non Xml files.
def except_png_files(annot_path, img_path):
    # print("Todo")
    xml_lists = [annot_path for annot_path in glob.iglob(annot_path+'/*.xml')]
    png_list=[]
    for i in xml_lists:
        _, png = get_parent_dir(i)
        png_path = os.path.join(img_path, png)
        if os.path.isfile(png_path):
            png_list.append(png_path)
    return png_list

## [END]: Choose img file Except Non Xml files.
def get_data(input_path, train_r=0.9,test_r=0.1, val_r=0.0):
    
    #Temp path
    all_imgs = []
    classes_count = {}
    class_mapping = {}

    path_list = [path_list for path_list in glob.iglob(input_path+'/**')
                            if os.path.isdir(path_list)]
    
    # KHNP Data
    data_paths = path_list
    data_paths.sort(key=natural_keys)
    print('Parsing annotation files')
    
    data_split = {}
    for data_path in data_paths:
        annot_path = os.path.join(data_path, 'annotation')    
        imgs_path = os.path.join(data_path, 'color')        
        csv_path = os.path.join(data_path, 'csv')
        
        
        imgsets_path = [filename for filename in glob.iglob(imgs_path+'/**/*', recursive=True)
                                if filename.lower().endswith(('.jpg', '.png'))]
        annots_path = [filename for filename in glob.iglob(annot_path+'/**/*.xml', recursive=True)]
        csv_path = [filename for filename in glob.iglob(csv_path+'/**/*.csv', recursive=True)]
        
        imgsets_path.sort(key=natural_keys)
        annots_path.sort(key=natural_keys)
        csv_path.sort(key=natural_keys)
        
        #xml을 기준으로 zip 실시.
        map_img_xml = dict(zip(imgsets_path, annots_path))
        map_xml_img = dict(zip(annots_path, imgsets_path))
        map_xml_csv = dict(zip(annots_path, csv_path))
        
        train_files_csv=[]
        test_files_csv=[]
        val_files_csv=[]
        trainval_files_csv=[]
        
        
        train_files=[]
        train_files_xml=[]
        
        test_files=[]
        test_files_xml=[]

        val_files=[]
        val_files_xml=[]

        trainval_files = []
        trainval_files_xml=[]
        
        dataset_len = [i for i in range(len(annots_path))]
            
        train_split_idx = int(len(dataset_len)*train_r)
        test_split_idx = int(len(dataset_len)*(train_r+test_r))
        
        for idx, i in enumerate(dataset_len):
            # print(dataset_len[i])
            if i < train_split_idx:
                train_files.append(imgsets_path[idx])
                train_files_xml.append(annots_path[idx])
                train_files_csv.append(csv_path[idx])

            elif i < test_split_idx:
                test_files.append(imgsets_path[idx])
                test_files_xml.append(annots_path[idx])
                test_files_csv.append(csv_path[idx])
                
            else:
                val_files.append(imgsets_path[idx])
                val_files_xml.append(annots_path[idx])
                val_files_csv.append(csv_path[idx])
                

        
        annots = tqdm(annots_path)
        for idx, annot in enumerate(annots):
            # try:
            
            # annots.set_description("Processing %s" % annot.split(os.sep)[-1])
            
            et = ET.parse(annot)
            element = et.getroot()

            element_objs = element.findall('object')
            # element_filename = element.find('filename').text + '.jpg'
            element_filename = element.find('filename').text
            element_width = int(element.find('size').find('width').text)
            element_height = int(element.find('size').find('height').text)

            if len(element_objs) > 0:
                annotation_data = {
                                    'filepath': map_xml_img[annot],
                                    'width': element_width,
                                    'height': element_height,
                                    'bboxes': [],
                                    'csvpath':map_xml_csv[annot],
                                    'location':map_xml_csv[annot].split(os.sep)[-3],
                                    'filename':map_xml_csv[annot].split(os.sep)[-1][:-4]
                                    }
                annotation_data['image_id'] = idx
                if annot in trainval_files_xml:
                    annotation_data['imageset'] = 'trainval'
            

                if annot in train_files_xml:
                    annotation_data['imageset'] = 'train'
            

                if annot in val_files_xml:
                    annotation_data['imageset'] = 'val'
            

                if len(test_files) > 0:
                    if annot in test_files_xml:
                        annotation_data['imageset'] = 'test'
            

            
            for element_obj in element_objs:
                class_name = element_obj.find('name').text
                # If class Name Modify & save Please un-comment this two line
                '''
                if class_name =='Motor_1':
                    class_name = 'Motor'
                    element_obj.find('name').text = 'Motor'
                    et.write(annot, encoding='utf-8')
                elif class_name =='Motor_2':
                    class_name = 'Motor'
                    element_obj.find('name').text = 'Motor'
                    et.write(annot, encoding='utf-8')
                '''    
                if class_name not in classes_count:
                    classes_count[class_name] = 1
                else:
                    classes_count[class_name] += 1

                # class mapping 정보 추가
                if class_name not in class_mapping:
                    class_mapping[class_name] = len(class_mapping)  # 마지막 번호로 추가

                obj_bbox = element_obj.find('bndbox')
                x1 = int(round(float(obj_bbox.find('xmin').text)))
                y1 = int(round(float(obj_bbox.find('ymin').text)))
                x2 = int(round(float(obj_bbox.find('xmax').text)))
                y2 = int(round(float(obj_bbox.find('ymax').text)))
                difficulty = int(element_obj.find('difficult').text) == 1
                annotation_data['bboxes'].append(
                    {'class': class_name, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'difficult': difficulty})
            all_imgs.append(annotation_data)
            
    return all_imgs, classes_count, class_mapping

def save_txt(img_list, save_loc, use_for='train'):
    if not os.path.isdir(save_loc):
        os.mkdir(save_loc)
    
    
    for img in img_list:
        bb_list = img['bboxes']
        used_set = set()
        for bb in bb_list:
            cls_path = '{}_{}.txt'.format(os.path.join(save_loc, bb['class']),use_for)
            #Make Each class_used-for.txt
            with open(cls_path, 'a') as f:
                f.writelines('{} -1\n'.format(img['filename']))
            use_path = '{}.txt'.format(os.path.join(save_loc, use_for))
            #Make train, test and tranval.txt 
            text = '{}\n'.format(img['filename'])
            if not text in used_set:
                used_set.add(text)
                with open(use_path, 'a') as f:
                    f.writelines(text)
    
if __name__ == '__main__':
    #PASCAL VOC TRAIN TEST GENERATOR
    all_imgs, cc, cm = get_data(input_path = "../data/VOCdevkit/20_FirsQuarter_readymade_data/", train_r=0, test_r=0, val_r=1)
    train_imgs = [s for s in all_imgs if s['imageset'] == 'train']
    test_imgs = [s for s in all_imgs if s['imageset'] == 'test']
    val_imgs = [s for s in all_imgs if s['imageset'] == 'val']
    print('Total Number:{}'.format(len(all_imgs)))
    print(val_imgs)
    print('Class Mapping:{}'.format(cm)) 
    print('Class Count:{}'.format(cc))
    print('Train Number:{}'.format(len(train_imgs)))
    print('Test Number:{}'.format(len(test_imgs)))
    print('Val Number:{}'.format(len(val_imgs)))
    
    # save_txt(all_imgs, save_loc='../data/VOCdevkit/20_FirsQuarter_readymade_data/ImageSets/Main', use_for='val')
    # save_txt(all_imgs, save_loc='../data/VOCdevkit/20_FirsQuarter_readymade_data/ImageSets/Main', use_for='train')
    # save_txt(all_imgs, save_loc='../data/VOCdevkit/20_FirsQuarter_readymade_data/ImageSets/Main', use_for='test')
        
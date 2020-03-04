import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2
import os
from colorspacious import cspace_converter
from collections import OrderedDict
import copy
import time
import random

'''
    Pallete Adding Algorithm.
    The algorithm works in the following way:
    
    First, if a temperature file (.csv) is given as input it converts from 0 to 1 using the min-max algorithm.

    Second, multiply the data from 0 to 1 by 255 and convert it to a file that can be represented as an image.

    Third, convert the output into a plot that can be used in opencv or matplotplib.

    There are consist of four major colormaps.

    Classes of colormaps

    
    1. Sequential: change in lightness and often saturation of color incrementally, often using a single hue; 
                            should be used for representing information that has ordering.

    2. Diverging: change in lightness and possibly saturation of two different colors that meet in the middle at an unsaturated color; 
                        should be used when the information being plotted has a critical middle value, such as topography or when the data deviates around zero.

    3. Cyclic: change in lightness of two different colors that meet in the middle and beginning/end at an unsaturated color; 
                    should be used for values that wrap around at the endpoints, such as phase angle, wind direction, or time of day.

    4. Qualitative: often are miscellaneous colors; 
                            should be used to represent information which does not have ordering or relationships.
'''

def FunctionName(args):
    pass

def funcname(parameter_list):
    pass

def normalize_thermal(data):
    # normalizedImg = np.zeros((480, 640))
    normalizedImg = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return normalizedImg

def figure_to_array(fig):
    """
    plt.figure를 RGBA로 변환(layer가 4개)
    shape: height, width, layer
    """
    fig.canvas.draw()
    data = np.array(fig.canvas.renderer._renderer)
    data = cv2.cvtColor(data, cv2.COLOR_BGRA2RGB)
    return data

def open_path_array(csv_path, csv=True):
    # Checking if the file exist
    if not os.path.isfile(csv_path):
        print("File {} does not exist!".format(csv_path))
        raise EOFError
        # Reading image as numpy matrix in gray scale (image, color_param)
    if csv:
        data = np.loadtxt(csv_path, delimiter=',')
    return data

def heatMapConvert(data, bboxes=None,specific_cm=None, tool='cv', is_random=False):
    #Deep Copy Original Thermal Data
    ori_data = copy.deepcopy(data)

    if tool == 'cv':
        #Coloring Methods Structure in Opencv 4.1.x
        colormap=OrderedDict()
        colormap['Perceptually Uniform Sequential']=[('magma', cv2.COLORMAP_MAGMA),('plasma', cv2.COLORMAP_PLASMA),
                                                                                    ('inferno', cv2.COLORMAP_INFERNO),('viridis', cv2.COLORMAP_VIRIDIS),
                                                                                    ('cividis', cv2.COLORMAP_CIVIDIS)]
        colormap['Sequential'] = [('autumn',  cv2.COLORMAP_AUTUMN), ('summer', cv2.COLORMAP_SUMMER),
                                                    ('winter', cv2.COLORMAP_WINTER),('parula', cv2.COLORMAP_PARULA),
                                                    ('turbo', cv2.COLORMAP_TURBO)]
        colormap['Miscellaneous'] = [('bone',cv2.COLORMAP_BONE),('rainbow', cv2.COLORMAP_RAINBOW),
                                                        ('jet',cv2.COLORMAP_JET), ('ocean', cv2.COLORMAP_OCEAN),
                                                        ('spring', cv2.COLORMAP_SPRING), ('cool',cv2.COLORMAP_COOL),
                                                        ('pink',cv2.COLORMAP_PINK),('hot',cv2.COLORMAP_HOT)]
        colormap['Cyclic']=[('hsv', cv2.COLORMAP_HSV),('twilight', cv2.COLORMAP_TWILIGHT),('twilight_shifted', cv2.COLORMAP_TWILIGHT_SHIFTED)]
        #Hole Methods
        methods = colormap['Perceptually Uniform Sequential'] + colormap['Sequential'] +\
                    colormap['Miscellaneous'] + colormap['Cyclic']
        #Randomized Methos 
        met_dict = dict(methods)

        try:
            #If the Random Method and Specific Color map is none.
            if is_random is True and specific_cm is None:
                random.seed(time.time())

                specific_cm = random.choice(list(met_dict.values()))
                data = cv2.applyColorMap(data, specific_cm)
            else:
                data = cv2.applyColorMap(data, met_dict[specific_cm])
            
            if bboxes is not None:
                for box in bboxes:
                    xmin = box['x1']
                    ymin = box['y1']
                    xmax = box['x2']
                    ymax = box['y2']
                    bndbox_data = normalize_thermal(ori_data[ymin:ymax, xmin:xmax])
                    if is_random is True:
                        random.seed(time.time())
                        specific_cm = random.choice(list(met_dict.values()))
                    data[ymin:ymax, xmin:xmax] = cv2.applyColorMap(bndbox_data, specific_cm)
            
        except:
            raise ValueError('Colormap is not recognized.\n You can Possible value are: {} only Open_cv{} methods.'
                                        .format(colormap,cv2.__version__))
    elif tool =='matplot':
        colormap=OrderedDict()
        colormap['Perceptually Uniform Sequential']=[ 'viridis', 'plasma', 'inferno', 'magma', 'cividis']
        colormap['Sequential'] = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                                                    'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                                                    'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
        colormap['Sequential (2)'] = ['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
                                                    'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
                                                    'hot', 'afmhot', 'gist_heat', 'copper']
        colormap['Qualitative'] = ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                                                'Dark2', 'Set1', 'Set2', 'Set3',
                                                'tab10', 'tab20', 'tab20b', 'tab20c']
        colormap['Miscellaneous'] = ['flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
                                                    'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
                                                    'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']
        colormap['Cyclic']=['hsv','twilight','twilight_shifted']
        methods = colormap['Perceptually Uniform Sequential']+colormap['Sequential (2)']+\
                         colormap['Qualitative'] + colormap['Miscellaneous'] + colormap['Cyclic']
        # try:
        #Metplotlib Method
        
        img = plt.figure()
        plt.axis("off")
        plt.tight_layout()
        plt.xticks([]), plt.yticks([])
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
        if is_random is True and specific_cm is None:
            random.seed(time.time())
            specific_cm = random.choice(methods)
        plt.imshow(data, cmap=  specific_cm)
        data = figure_to_array(img)
        plt.clf()

        if bboxes is not None:
            for box in bboxes:
                xmin = box['x1']
                ymin = box['y1']
                xmax = box['x2']
                ymax = box['y2']
                x = xmax - xmin
                y = ymax - ymin
                box_img = plt.figure(figsize=(x, y), dpi=1)
                plt.axis("off")
                plt.tight_layout()
                plt.xticks([]), plt.yticks([])
                plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
                bndbox_data = normalize_thermal(ori_data[ymin:ymax, xmin:xmax])
                if is_random is True:
                    random.seed(time.time())
                    specific_cm = random.choice(methods)
                plt.imshow(bndbox_data, cmap=  specific_cm)
                #plt to array
                #Whole data overlap to bounding box img
                data[ymin:ymax, xmin:xmax] = figure_to_array(box_img)
                
                #plt clear
                plt.clf()
        
        
        # except:

        #     raise ValueError('Colormap is not recognized.\n You can Possible value are: {} only matplotlib{} methods.'
        #                                 .format(colormap,mpl.__version__))
    return data

def printAvailMethod():
    '''
        enum  	cv::ColormapTypes {
        cv::COLORMAP_AUTUMN = 0,
        cv::COLORMAP_BONE = 1,
        cv::COLORMAP_JET = 2,
        cv::COLORMAP_WINTER = 3,
        cv::COLORMAP_RAINBOW = 4,
        cv::COLORMAP_OCEAN = 5,
        cv::COLORMAP_SUMMER = 6,
        cv::COLORMAP_SPRING = 7,
        cv::COLORMAP_COOL = 8,
        cv::COLORMAP_HSV = 9,
        cv::COLORMAP_PINK = 10,
        cv::COLORMAP_HOT = 11,
        cv::COLORMAP_PARULA = 12,
        cv::COLORMAP_MAGMA = 13,
        cv::COLORMAP_INFERNO = 14,
        cv::COLORMAP_PLASMA = 15,
        cv::COLORMAP_VIRIDIS = 16,
        cv::COLORMAP_CIVIDIS = 17,
        cv::COLORMAP_TWILIGHT = 18,
        cv::COLORMAP_TWILIGHT_SHIFTED = 19,
        cv::COLORMAP_TURBO = 20
        }
    '''
    print("Hello")

def getHistogram(data, percentile=25):
    t_data = data.copy()
    t_max = int(data.max())
    t_min = int(data.min())

    t = np.percentile(t_data, percentile)
    t_data[t_data>t] = 0

    flatten_data = t_data.flatten()
    #hist, bins = np.histogram(flatten_data, bins=range(0, 255))
    
    # plt.hist(flatten_data, bins=range(t_min, t_max))
    # plt.xlim([t_min,t_max])
    # plt.show()

    #히스토그램 퍼센타일 어떻게 할 것인가..?
    #목적은 해당 객체의 오브젝트를 잘 표현해 보자.
    #1) xmin, ymin, xmax, ymax안의 내용에 따로 heatmap입혀보자. (회전설비나 발열설비가 아니라면 특징점이 배경과 겹치게됨.)
    #2) 다양한 히트맵 입히기
    #3) 히스토그램을 분석해서 percentile을 이용해서 입혀보자.
    return t_data

if __name__ == "__main__":
    import pascal_voc_parser2 as pvp
    all_imgs, cc, cm = pvp.get_data(input_path = "./test/", train_r=1, test_r=0, val_r=0, csv_save=True)
    train_imgs = [s for s in all_imgs if s['imageset'] == 'train']
    test_dict = train_imgs[0]
    csv_path = test_dict['csvpath']
    ############ Input ##########
    data = open_path_array(csv_path)
    #Preprocessing
    
    getHistogram(data, percentile=75)
    normalizedImg = normalize_thermal(data)
    getHistogram(normalizedImg, percentile=75)
    # getHistogram(normalizedImg)
    
    img = heatMapConvert(normalizedImg,bboxes=None, 
                                        specific_cm=None, 
                                        tool='matplot', 
                                        is_random=True)
    img2=heatMapConvert(normalizedImg, bboxes=test_dict['bboxes'],
                                        specific_cm=None,
                                        tool='matplot',
                                        is_random=True)
    print(img2.shape)
    
    
    while(True):
        cv2.imshow('test', img2)
        # cv2.imshow('test2', img2)
        if cv2.waitKey(10) ==27:
            cv2.destroyAllWindows()
            exit()
    
    #OpenCV Method
    # cv2.imwrite('file.png', heatmap_jet)
    # cv2.imshow('cloud cover',heatmap_jet)
    # cv2.waitKey(0)

    
    
    # print(Img.shape)
    # plt.imsave('test.png',normalizedImg, cmap=  'tab20c')
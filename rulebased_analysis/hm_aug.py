import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2
import os
# from colorspacious import cspace_converter
from collections import OrderedDict
import copy
import time
import random
from matplotlib.colors import ListedColormap
import pandas as pd


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

def get_cmap(cmap_name='flirpal', percentile=25):
    
    try:
        path = os.path.join(os.getcwd(), 'pallete', cmap_name+'.csv')
    except:
        print(f"There is no {cmap_name} Color map as follow..")
    cmap = pd.read_csv(path).to_numpy()/255
    length, h = cmap.shape
    
    mid_idx = int(length*0.5)
    
    # exit()
    left_cmap = cv2.resize(cmap[:mid_idx,:], (h, int(mid_idx*(percentile/100))))
    # print(left_cmap.shape)
    right_cmap = cv2.resize(cmap[mid_idx:,:], (h, mid_idx+int(mid_idx*(percentile/100))))
    # print(right_cmap.shape)
    
    cmap = np.vstack([left_cmap, right_cmap])
    
    
    # print(cmap.shape)
    # exit()
    # cmap = cmap[['red', 'green', 'blue']]
    return ListedColormap(cmap)

def FunctionName(args):
    pass

def funcname(parameter_list):
    pass

def normalize_thermal(data):
    # normalizedImg = np.zeros((480, 640))
    if type(data)==list:
        data = np.array(data)
    
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

def open_path_array(path, csv=True):
    csv_path = path#['csvpath']
    # Checking if the file exist
    if not os.path.isfile(csv_path):
        print("File {} does not exist!".format(csv_path))
        raise EOFError
        # Reading image as numpy matrix in gray scale (image, color_param)

    if csv_path.lower().endswith(('.csv')):
        data = np.loadtxt(csv_path, delimiter=',')
    elif csv_path.lower().endswith(('.json')):
        import json
        with open(csv_path, 'r') as f:
            json_data = json.load(f)
        data = json_data['tempData']
            
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
    elif tool=='custom':
        methods = ['flirpal', 'glowbowpal', 'grey10pal', 'grey120pal', 
            'greyredpal', 'hotironpal', 'ironbowpal', 'medicalpal', 'midgreenpal',
            'midgreypal', 'mikronprismpal', 'rainbow1234pal',
            'rainbowpal', 'yellow']

        img = plt.figure()
        plt.axis("off")
        plt.tight_layout()
        plt.xticks([]), plt.yticks([])
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
        if is_random is True and specific_cm is None:
            random.seed(time.time())
            specific_cm = random.choice(methods)
        #This is FLIR Custom map
        specific_cm = get_cmap(specific_cm)
        ###
        # _data = copy.deepcopy(data)
        high = np.percentile(data, 100)
        data[data>high] = high
        low = np.percentile(data, 0)
        data[data<low] = low
        # psm = plt.pcolormesh(_data, cmap=specific_cm,  vmin=22.9, vmax=45.5)
        plt.pcolormesh(data, cmap=specific_cm)
        # plt.colorbar(psm)
        ###
        plt.imshow(data, cmap=  specific_cm)
        # plt.show()
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
                _data[ymin:ymax, xmin:xmax] = figure_to_array(box_img)
                
                #plt clear
                plt.clf()
    return data
def heatMapAllImageSave(data, bboxes=None,specific_cm=None, tool='matplot', fname=None):
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
            if specific_cm is None:
                random.seed(time.time())

                specific_cm = random.choice(list(met_dict.values()))
            data = cv2.applyColorMap(data, specific_cm)
            
            if bboxes is not None:
                for box in bboxes:
                    xmin = box['x1']
                    ymin = box['y1']
                    xmax = box['x2']
                    ymax = box['y2']
                    bndbox_data = normalize_thermal(ori_data[ymin:ymax, xmin:xmax])
                    
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
        colormap['Diverging'] = ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
                                            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
        colormap['Qualitative'] = ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                                                'Dark2', 'Set1', 'Set2', 'Set3',
                                                'tab10', 'tab20', 'tab20b', 'tab20c']
        colormap['Miscellaneous'] = ['flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
                                                    'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
                                                    'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']
        colormap['Cyclic']=['hsv','twilight','twilight_shifted']
        methods = colormap['Perceptually Uniform Sequential']+colormap['Sequential']+colormap['Sequential (2)']+\
                        colormap['Diverging']+ colormap['Qualitative'] + colormap['Miscellaneous'] + colormap['Cyclic']
        # try:
        #Metplotlib Method
        
        for i, method in enumerate(methods):
            plt.clf()
            plt.close()
            img = plt.figure()
            plt.axis("off")
            plt.tight_layout()
            plt.xticks([]), plt.yticks([])
            plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
            # print(method)
            _data = copy.deepcopy(data)
            plt.imshow(_data, cmap=  method)
            _data = figure_to_array(img)
            # plt.savefig('method({0}-{1}).png'.format(i,method), data)
            plt.clf()
            if not os.path.exists(os.path.join(os.getcwd(), fname)):
                os.mkdir(fname)
            cv2.imwrite('./{0}/method({1}-{2}).png'.format(fname, i+1,method), _data)


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

                plt.imshow(bndbox_data, cmap=  specific_cm)
                #plt to array
                #Whole data overlap to bounding box img
                data[ymin:ymax, xmin:xmax] = figure_to_array(box_img)
                
                #plt clear
                plt.clf()
    elif tool=='custom':
        methods = ['flirpal', 'glowbowpal', 'grey10pal', 'grey120pal', 
            'greyredpal', 'hotironpal', 'ironbowpal', 'medicalpal', 'midgreenpal',
            'midgreypal', 'mikronprismpal', 'rainbow1234pal',
            'rainbowpal', 'yellow']

        
        for i, method in enumerate(methods):
            plt.clf()
            plt.close()
            img = plt.figure()
            plt.axis("off")
            plt.tight_layout()
            plt.xticks([]), plt.yticks([])
            plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
            # print(method)
            _data = copy.deepcopy(data)
            
            #This is FLIR Custom map
            
            high = np.percentile(data, 100)
            _data[_data>high] = high
            low = np.percentile(data, 0)
            _data[_data<low] = low

            #This is FLIR Custom map
            specific_cm = get_cmap(method)
            plt.imshow(_data, cmap=  specific_cm)
            _data = figure_to_array(img)
            # plt.savefig('method({0}-{1}).png'.format(i,method), data)
            plt.clf()
            if not os.path.exists(os.path.join(os.getcwd(), fname)):
                os.mkdir(fname)
            cv2.imwrite('./{0}/method({1}-{2}).png'.format(fname, i+1,method), _data)
        
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

                plt.imshow(bndbox_data, cmap=  specific_cm)
                #plt to array
                #Whole data overlap to bounding box img
                data[ymin:ymax, xmin:xmax] = figure_to_array(box_img)
                
                #plt clear
                plt.clf()
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
    if type(t_data) == list:
        t_data = np.array(data)
    

    t = np.percentile(t_data, percentile)
    # print(t)
    t_data[t_data<t] = t
    '''
    t_min = t_data.min()
    t_max = t_data.max()
    #histogram
    flatten_data = t_data.flatten()
    
    hist, bins = np.histogram(flatten_data, bins=range(int(t_min), int(t_max)))
    
    
    # hist, bins = np.histogram(flatten_data, bins=range(t_min+g*(k-1), t_min+g*k))
    # print(t_min+g*(k-1), t_min+g*k)
    # print(hist/hist.sum())
    plt.hist(flatten_data, bins=bins)
    plt.xlim([t_min,t_max])
    plt.show()
    '''
    return t_data


if __name__ == "__main__":
    '''
    import pascal_voc_parser2 as pvp
    all_imgs, cc, cm = pvp.get_data(input_path = "./test/", train_r=1, test_r=0, val_r=0, csv_save=True)
    train_imgs = [s for s in all_imgs if s['imageset'] == 'train']
    print(train_imgs)
    test_dict = train_imgs[0]
    '''
    path = 'E:\\ProJ\\한수원 관련자료\\작동가능\\easy-faster-rcnn.pytorch\\data\\VOCdevkit\\20_FirsQuarter_readymade_data\\saewool1\\csv'
    # save_name = '518-PP04.csv'
    save_name = '811-PT02.csv'
    csv_path = os.path.join(path, save_name)
    
    ############ Input ##########
    data = open_path_array(csv_path)
    #Preprocessing
    
    #getHistogram(data, percentile=75)
    #잠깐##
    # normalizedImg = Normalizer().fit_transform(data)
    normalizedImg = getHistogram(data, percentile=0)
    
    
    ##
    # getHistogram(normalizedImg)

    # Histogram
    
    #Image Generating    
    img = heatMapConvert(normalizedImg,bboxes=None, 
                                        specific_cm='flirpal', 
                                        tool='custom',
                                        is_random=False)
    
    # Bounding BOx Generating
    # img=heatMapConvert(normalizedImg, bboxes=test_dict['bboxes'],
    #                                     specific_cm=None,
    #                                     tool='matplot',
    #                                     is_random=True)
    
    #All Image Saved
    # img = heatMapAllImageSave(normalizedImg,bboxes=None, 
    #                                         specific_cm=None, 
    #                                         tool='custom',  #matplot
    #                                         fname=save_name[:-4])

    img = normalize_thermal(img)
    
    while(True):
        cv2.imshow('test', img)
        # cv2.imshow('test2', img2)
        if cv2.waitKey(10) ==ord('q'):#27: #ESC
            cv2.destroyAllWindows()
            exit()
    
    #OpenCV Method
    # cv2.imwrite('file.png', heatmap_jet)
    # cv2.imshow('cloud cover',heatmap_jet)
    # cv2.waitKey(0)

    
    
    # print(Img.shape)
    # plt.imsave('test.png',normalizedImg, cmap=  'tab20c')
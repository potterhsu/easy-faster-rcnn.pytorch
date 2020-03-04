import cv2
import numpy as np
import copy
import imgaug.augmenters as iaa
from . import pallete_aug as pa

def pallete_augmentation(img, img_data, config):
    if config.pallete:
        csv_path = img_data['csvpath']
        #Exception none value.
        if csv_path is None or '':
            print("CSV path is {}".format(csv_path))
            return img

        data = pa.open_path_array(csv_path)
        #Preprocessing
        
        pa.getHistogram(data, percentile=75)
        normalizedImg = pa.normalize_thermal(data)
        pa.getHistogram(normalizedImg, percentile=75)
        # getHistogram(normalizedImg)
        
        img = pa.heatMapConvert(normalizedImg,bboxes=None, 
                                            specific_cm=None, 
                                            tool='matplot', 
                                            is_random=True)

        # img= pa.heatMapConvert(normalizedImg, bboxes=img_data['bboxes'],
        #                                     specific_cm=None,
        #                                     tool='matplot',
        #                                     is_random=True)
    return img
def augment(img_data, config, augment=True):
    assert 'filepath' in img_data
    assert 'bboxes' in img_data
    assert 'width' in img_data
    assert 'height' in img_data

    img_data_aug = copy.deepcopy(img_data)
    aug_list = []
    img = cv2.imread(img_data_aug['filepath'])

    if augment:
        rows, cols = img.shape[:2]
        #[START] Pallete Augmentation
        pallete_augmentation(img =img, img_data = img_data_aug, config=config)
        #[END] Pallete Augmentation

        if config.use_horizontal_flips and np.random.randint(0, 2) == 0:
            img = cv2.flip(img, 1)
            for bbox in img_data_aug['bboxes']:
                x1 = bbox['x1']
                x2 = bbox['x2']
                bbox['x2'] = cols - x1
                bbox['x1'] = cols - x2

        if config.use_vertical_flips and np.random.randint(0, 2) == 0:
            img = cv2.flip(img, 0)
            for bbox in img_data_aug['bboxes']:
                y1 = bbox['y1']
                y2 = bbox['y2']
                bbox['y2'] = rows - y1
                bbox['y1'] = rows - y2

        if config.rot_90:
            angle = np.random.choice([0,90,180,270],1)[0]
            if angle == 270:
                img = np.transpose(img, (1,0,2))
                img = cv2.flip(img, 0)
            elif angle == 180:
                img = cv2.flip(img, -1)
            elif angle == 90:
                img = np.transpose(img, (1,0,2))
                img = cv2.flip(img, 1)
            elif angle == 0:
                pass

            for bbox in img_data_aug['bboxes']:
                x1 = bbox['x1']
                x2 = bbox['x2']
                y1 = bbox['y1']
                y2 = bbox['y2']
                if angle == 270:
                    bbox['x1'] = y1
                    bbox['x2'] = y2
                    bbox['y1'] = cols - x2
                    bbox['y2'] = cols - x1
                elif angle == 180:
                    bbox['x2'] = cols - x1
                    bbox['x1'] = cols - x2
                    bbox['y2'] = rows - y1
                    bbox['y1'] = rows - y2
                elif angle == 90:
                    bbox['x1'] = rows - y2
                    bbox['x2'] = rows - y1
                    bbox['y1'] = x1
                    bbox['y2'] = x2
                elif angle == 0:
                    pass
        
        if config.color:
            aug_list.append(np.random.choice([iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True),
                                                                    iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
                                                                    iaa.KMeansColorQuantization(),
                                                                    iaa.UniformColorQuantization(),
                                                                    iaa.Grayscale(alpha=(0.0, 1.0))]))

        if config.contrast:
            aug_list.append(np.random.choice([iaa.GammaContrast((0.5, 2.0), per_channel=True),
                                                                    iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6),per_channel=True),
                                                                    iaa.LogContrast(gain=(0.6, 1.4), per_channel=True),
                                                                    iaa.LinearContrast((0.4, 1.6), per_channel=True),
                                                                    iaa.AllChannelsCLAHE(clip_limit=(1, 10), per_channel=True),
                                                                    iaa.AllChannelsHistogramEqualization(),
                                                                    iaa.HistogramEqualization()]))

        ## Augmentation
        aug = iaa.SomeOf((0, None), aug_list, random_order=True)
        seq = iaa.Sequential(aug)
        img = seq.augment_image(img)
        ##
    img_data_aug['width'] = img.shape[1]
    img_data_aug['height'] = img.shape[0]
    return img_data_aug, img


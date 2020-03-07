import os
import parse_xml
from parse_json import parse_json
import copy
import numpy as np
import cv2
import json
import glob
from collections import OrderedDict

class statistics:
    def __init__(self, name):
        self.name = name
        self.count = 0
        self.tmin_sum = 0.0
        self.tmax_sum = 0.0
        self.tmean_sum = 0.0

    def add_data(self, tmin, tmax, tmean):
        self.count += 1
        self.tmin_sum += tmin
        self.tmax_sum += tmax
        self.tmean_sum += tmean

    def result(self):
        ret = OrderedDict()
        ret["count"] = self.count
        ret["tmin"] = round(self.tmin_sum/self.count, 2)
        ret["tmax"] = round(self.tmax_sum/self.count, 2)
        ret["tmean"] = round(self.tmean_sum/self.count, 2)
        return ret

if __name__=="__main__":

    rulebased_statistics = {}

    #data_root_dir = '/home/inseo/Desktop/test_data'
    data_root_dir = '/home/inseo/DATA/fuse'
    data_dir_list = glob.glob(os.path.join(data_root_dir, 'fuse*'))
    prefix_json = 'json_rb'
    
    for data_dir in data_dir_list:
        for json_item in glob.glob(os.path.join(data_dir, prefix_json, '*.json')):
            data = parse_json(json_item)
            for obj in data:
                #unpack data
                class_name = obj['class']
                tmin = obj['tmin']
                tmax = obj['tmax']
                tmean = obj['tmean']

                # create new class if not exist
                if not class_name in rulebased_statistics.keys():
                    rulebased_statistics[class_name] = statistics(class_name)

                #add item to statistics
                rulebased_statistics[class_name].add_data(tmin, tmax, tmean)
    
    # write to json file
    result = OrderedDict()
    for facility_class in rulebased_statistics.keys():
        result[facility_class] = rulebased_statistics[facility_class].result()

    with open(os.path.join(data_root_dir, 'result.json'), 'w', encoding='utf-8') as result_file:
        json.dump(result, result_file, indent='\t')
import os
import sys
try:
    from .import parse_xml
except ImportError as error:
    import parse_xml
import math
import copy
import numpy as np
import cv2
import xmltodict
import json
import glob
from collections import OrderedDict


def calculateMaxSubmission(y, x, csv_arr):
    submission_arr = []
    if x > 0 and y > 0:
        submission_arr.append(csv_arr[y][x]-csv_arr[y-1][x-1])
    if y > 0:
        submission_arr.append(csv_arr[y][x] - csv_arr[y-1][x])
    if x < len(csv_arr[0])-1 and y>0:
        submission_arr.append(csv_arr[y][x] - csv_arr[y-1][x+1])
    if x > 0:
        submission_arr.append(csv_arr[y][x] - csv_arr[y][x-1])
    if x < len(csv_arr[0])-1:
        submission_arr.append(csv_arr[y][x] - csv_arr[y][x+1])
    if x > 0 and y < len(csv_arr)-1:
        submission_arr.append(csv_arr[y][x] - csv_arr[y+1][x-1])
    if y < len(csv_arr)-1:
        submission_arr.append(csv_arr[y][x] - csv_arr[y+1][x])
    if x < len(csv_arr[0])-1 and y < len(csv_arr)-1:
        submission_arr.append(csv_arr[y][x] - csv_arr[y+1][x+1])

    return max(submission_arr)

def writeJason(xmin, ymin, xmax, ymax, tmin, tmax, tmean, object_class, hp_contour, rp_contour):
    object_data = OrderedDict()
    object_data["xmin"] = xmin
    object_data["ymin"] = ymin
    object_data["xmax"] = xmax
    object_data["ymax"] = ymax
    object_data["tmin"] = tmin
    object_data["tmax"] = tmax
    object_data["tmean"] = tmean
    object_data["class"] = object_class
    object_data["hp"] = []

    for i in range(len(hp_contour)):
        temp_contour = []
        for j in range(len(hp_contour[i])):
            temp_contour.append('('+str(hp_contour[i][j][0][0]+xmin)+','+str(hp_contour[i][j][0][1]+ymin)+')')
        object_data["hp"].append(temp_contour)

    object_data["rp"] = []
    for i in range(len(rp_contour)):
        temp_contour = []
        for j in range(len(rp_contour[i])):
            temp_contour.append(
                '(' + str(rp_contour[i][j][0][0] + xmin) + ',' + str(rp_contour[i][j][0][1] + ymin) + ')')
        object_data["rp"].append(temp_contour)
    return object_data

def writeJson2(input_data):
    # print('write "{0}" ...'.format(inp))

    data = OrderedDict()
    # xmin
    data["xmin"] = input_data["xmin"]
    input_data.pop("xmin", None)
    # ymin
    data["ymin"] = input_data["ymin"]
    input_data.pop("ymin", None)
    # xmax
    data["xmax"] = input_data["xmax"]
    input_data.pop("xmax", None)
    # ymax
    data["ymax"] = input_data["ymax"]
    input_data.pop("ymax", None)
    # tmin
    data["tmin"] = input_data["tmin"]
    input_data.pop("tmin", None)
    # tmax
    data["tmax"] = input_data["tmax"]
    input_data.pop("tmax", None)
    # tmean
    data["tmean"] = input_data["tmean"]
    input_data.pop("tmean", None)
    # class
    data["class"] = input_data["class"]
    input_data.pop("class", None)

    # class
    #data["confidence"] = input_data["confidence"]
    data["confidence"] = 1.0
    input_data.pop("confidence", None)
    
    # hp
    data["hp"] = []
    hp_contour = input_data["hp_counter"]
    for i in range(len(hp_contour)):
        temp_contour = []
        for j in range(len(hp_contour[i])):
            temp_contour.append('('+str(hp_contour[i][j][0][0]+data["xmin"])+','+str(hp_contour[i][j][0][1]+data["ymin"])+')')
        data["hp"].append(temp_contour)
    input_data.pop("hp_counter", None)

    # rp
    data["rp"] = []
    rp_contour = input_data["rp_counter"]
    for i in range(len(rp_contour)):
        temp_contour = []
        for j in range(len(rp_contour[i])):
            temp_contour.append(
                '(' + str(rp_contour[i][j][0][0] + data["xmin"]) + ',' + str(rp_contour[i][j][0][1] + data["ymin"]) + ')')
        data["rp"].append(temp_contour)
    input_data.pop("rp_counter", None)

    # tp
    data["top_rate"] = []
    tp_counter = input_data["tp_counter"]
    for i in range(len(tp_counter)):
        temp_contour = []
        for j in range(len(tp_counter[i])):
            temp_contour.append('(' + str(tp_counter[i][j][0][0] + data["xmin"]) + ',' + str(
                tp_counter[i][j][0][1] + data["ymin"]) + ')')
        data["top_rate"].append(temp_contour)
    input_data.pop("tp_counter", None)

    ## Emissivity
    if "Emissivity" in input_data:
        data["Emissivity"] = input_data["Emissivity"]
        input_data.pop("Emissivity", None)
    else:
        data["Emissivity"] = 0.95
    ## Atmospheric Temperature
    if "Atmospheric Temperature" in input_data:
        data["Atmospheric Temperature"] = input_data["Atmospheric Temperature"]
        input_data.pop("Atmospheric Temperature", None)
    else:
        data["Atmospheric Temperature"] = 20
    ## Relative Humidity
    if "Relative Humidity" in input_data:
        data["Relative Humidity"] = input_data["Relative Humidity"]
        input_data.pop("Relative Humidity", None)
    else:
        data["Relative Humidity"] = 40

    ## Point
    if "Point" in input_data:
        data["Point"] = input_data["Point"]
        input_data.pop("Point", None)
    else:
        data["Point"] = "2320-472-M-WV-07PA"

    ## FacilityName
    if "FacilityName" in input_data:
        data["FacilityName"] = input_data["FacilityName"]
        input_data.pop("FacilityName", None)
    else:
        data["FacilityName"] = "Fuse"
    ## FileName
    if "FileName" in input_data:
        data["FileName"] = input_data["FileName"]
        input_data.pop("FileName", None)
    else:
        data["FileName"] = "fuse.jpg"
    ## FacilityClass
    if "FacilityClass" in input_data:
        data["FacilityClass"] = input_data["FacilityClass"]
        input_data.pop("FacilityClass", None)
    else:
        data["FacilityClass"] = "ETC"

    ## FacilityClass_option
    if "FacilityClass_option" in input_data:
        data["FacilityClass"] = str(data["FacilityClass"]) + ", " + str(input_data["FacilityClass_option"])
        input_data.pop("FacilityClass_option", None)
    else:
        data["FacilityClass"] = str(data["FacilityClass"]) + ", " + "A상"
    ## Limit Temperature
    if "Limit Temperature" in input_data:
        data["Limit Temperature"] = input_data["Limit Temperature"]
        input_data.pop("Limit Temperature", None)
    else:
        data["Limit Temperature"] = 25
    ## PointTemperature
    if "PointTemperature" in input_data:
        data["PointTemperature"] = input_data["PointTemperature"]
        input_data.pop("PointTemperature", None)
    else:
        data["PointTemperature"] = 35.9
    ## Over temperature
    if "Over temperature" in input_data:
        data["Over temperature"] = input_data["Over temperature"]
        input_data.pop("Over temperature")
    else:
        data["Over temperature"] = "9.9"

    ## Over temperature_option
    if "Over temperature_option" in input_data:
        data["Over temperature_option"] = str(data["Over temperature_option"]) + ", " + str(input_data["Over temperature_option"])
        input_data.pop("Over temperature_option", None)
    else:
        data["Over temperature"] = str(data["Over temperature"]) + ", " + "정상"

    ## deltaT
    if "deltaT" in input_data:
        data["deltaT"] = input_data["deltaT"]
        input_data.pop("deltaT", None)
    else:
        data["deltaT"] = 0

    ## Cause of Failure
    if "Cause of Failure" in input_data:
        data["Cause of Failure"] = input_data["Cause of Failure"]
        input_data.pop("Cause of Failure", None)
    else:
        data["Cause of Failure"] = "정상"
    ## DiagnosisCode
    if "DiagnosisCode" in input_data:
        data["DiagnosisCode"] = input_data["DiagnosisCode"]
        input_data.pop("DiagnosisCode", None)
    else:
        data["DiagnosisCode"] = "AA"
    ## Diagnosis
    if "Diagnosis" in input_data:
        data["Diagnosis"] = input_data["Diagnosis"]
        input_data.pop("Diagnosis", None)
    else:
        data["Diagnosis"] = "정상임"
    
    print("unused key:")
    for key in input_data.keys():
        print("\t" + key)
    
    return data


def analyze_det(xml, csv):
    #xml = [xmin_arr, xmax_arr, ymin_arr, ymax_arr, name_arr, file_name]
    # analyze = parse_xml.parcingXml(xml)
    analyze = np.array(xml)
    csv_arr = parse_xml.parcingCsv(csv)
    file_data = OrderedDict()
    file_data["facilities"] = []

    print('analyze data: ', analyze)
    
    xmin = int(analyze[0])
    ymin = int(analyze[1])
    xmax = int(analyze[2])
    ymax = int(analyze[3])
    object_class = analyze[4]
    print("confidence: %s.%s" % (analyze[5].split(".")[0], analyze[5].split(".")[1][:2]))
    confidence = analyze[5].split(".")[0] + '.' + analyze[5].split(".")[1][:2]
    confidence = float(confidence)
    fname = analyze[6]
    
    csv_copy = copy.deepcopy(csv_arr)
    csv_crop = csv_copy[ymin:ymax, xmin:xmax]
    csv_flat = csv_crop.flatten()
    csv_flat = np.round_(csv_flat, 1)
    temp_min = csv_flat.min()
    temp_max = csv_flat.max()
    temp_average = np.average(csv_flat)
    temp_average = np.round_(temp_average, 1)

    # find heating points
    csv_copy = copy.deepcopy(csv_arr)
    csv_crop = csv_copy[ymin:ymax, xmin:xmax]
    thresh = np.percentile(csv_crop, 75)
    thresh_arr = np.zeros((len(csv_crop), len(csv_crop[0])), dtype=np.uint8)
    thresh_arr = np.where(csv_crop[:,:]<thresh, 0 ,255)
    thresh_arr = np.array(thresh_arr, dtype=np.uint8)
    hp_contour, _ = cv2.findContours(thresh_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # find reflection points
    csv_copy = copy.deepcopy(csv_arr)
    csv_crop = csv_copy[ymin:ymax, xmin:xmax]
    CRITICAL_GRAD = 0.4
    thresh = np.percentile(csv_crop, 75)
    thresh_arr = np.zeros((len(csv_crop), len(csv_crop[0])), dtype=np.uint8)
    thresh_arr = np.where(csv_crop[:,:]<thresh, 0, 255)
    thresh_arr = np.array(thresh_arr, dtype=np.uint8)

    # find top rate celsius
    csv_copy = copy.deepcopy(csv_arr)
    csv_crop = csv_copy[ymin:ymax, xmin:xmax]
    thresh = np.percentile(csv_crop, 5)
    thresh_arr = np.zeros((len(csv_crop), len(csv_crop[0])), dtype=np.uint8)
    thresh_arr = np.where(csv_crop[:, :] < thresh, 0, 255)
    thresh_arr = np.array(thresh_arr, dtype=np.uint8)
    tp_contour, _ = cv2.findContours(thresh_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    height, width = thresh_arr.shape
    suspected_points = []
    for i in range(height):
        for j in range(width):
            if thresh_arr[i][j] != 0:
                temp = calculateMaxSubmission(i, j, csv_crop)
                if temp > CRITICAL_GRAD:
                    suspected_points.append([j,i])

    masking_img = np.zeros((height, width, 3), dtype=np.uint8)
    for pts in suspected_points:
        xy = np.array(pts)
        cv2.circle(masking_img, (xy[0], xy[1]), 3, (255, 255, 255), -1)
    
    masking_img = masking_img[:,:,0]
    masking_img = masking_img.astype(np.uint8)
    rp_contour, heirachy = cv2.findContours(masking_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # object_data = writeJason(xmin, ymin, xmax, ymax, temp_min, temp_max, temp_average, object_class, hp_contour, rp_contour)
    json_data = {}
    json_data["xmin"] = xmin
    json_data["ymin"] = ymin
    json_data["xmax"] = xmax
    json_data["ymax"] = ymax
    json_data["tmin"] = temp_min
    json_data["tmax"] = temp_max
    json_data["tmean"] = temp_average
    json_data["class"] = object_class
    json_data["confidence"] = confidence
    json_data["hp_counter"] = hp_contour
    json_data["rp_counter"] = rp_contour
    json_data["tp_counter"] = tp_contour

    #rule-base analysis
    rule = DiagnosisRule("./rulebased_analysis/data/diagnosis_rule.json")
    diag_result = rule.diagnose(object_class, temp_max)
    json_data["DiagnosisCode"] = diag_result["code"]
    json_data["Cause of Failure"] = diag_result["cause"]
    json_data["Diagnosis"] = diag_result["action"]
    json_data["Over temperature"] = diag_result["Over Temperature"]
    json_data["FacilityName"] = diag_result["name"]
    json_data["Limit Temperature"] = diag_result["Limit Temperature"]
    json_data["FileName"] = fname
    json_data["PointTemperature"] = json_data["tmax"]
    if diag_result["Over Temperature"] > 0:
        json_data["deltaT"] = json_data["tmax"] / json_data["Limit Temperature"]
        json_data["deltaT"] = round(json_data["deltaT"], 2)

    file_data["facilities"].append(writeJson2(json_data))

    
    # file_data["facilities"].append(object_data)
    
    # # with open('./json_rb/'+fname+'.json', 'w', encoding='utf-8') as make_file:
    # with open(os.path.join(out_dir, (fname+'.json')), 'w', encoding='utf-8') as make_file:
    #     json.dump(file_data, make_file, indent="\t", ensure_ascii=False)
    return file_data

def analyze_with_annotation(xml, csv, img, out_dir, det=False):
    #xml = Annotation file
    item_idx = 0
    fname = os.path.basename(os.path.splitext(xml)[0])
    analyze = parse_xml.parcingXml(xml)
    analyze = np.array(analyze)
    csv_arr = parse_xml.parcingCsv(csv)
    
    for i in range(len(analyze[0])):
        file_data = OrderedDict()
        file_data["facilities"] = []

        xmin = int(analyze[0][i])
        xmax = int(analyze[1][i])
        ymin = int(analyze[2][i])
        ymax = int(analyze[3][i])
        object_class = analyze[4][i]
        csv_copy = copy.deepcopy(csv_arr)
        csv_crop = csv_copy[ymin:ymax, xmin:xmax]
        csv_flat = csv_crop.flatten()
        csv_flat = np.round_(csv_flat, 1)
        temp_min = csv_flat.min()
        temp_max = csv_flat.max()
        temp_average = np.average(csv_flat)
        temp_average = np.round_(temp_average, 1)

        # find heating points
        csv_copy = copy.deepcopy(csv_arr)
        csv_crop = csv_copy[ymin:ymax, xmin:xmax]
        thresh = np.percentile(csv_crop, 75)
        thresh_arr = np.zeros((len(csv_crop), len(csv_crop[0])), dtype=np.uint8)
        thresh_arr = np.where(csv_crop[:,:]<thresh, 0 ,255)
        thresh_arr = np.array(thresh_arr, dtype=np.uint8)
        hp_contour, _ = cv2.findContours(thresh_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # find reflection points
        csv_copy = copy.deepcopy(csv_arr)
        csv_crop = csv_copy[ymin:ymax, xmin:xmax]
        CRITICAL_GRAD = 0.4
        thresh = np.percentile(csv_crop, 75)
        thresh_arr = np.zeros((len(csv_crop), len(csv_crop[0])), dtype=np.uint8)
        thresh_arr = np.where(csv_crop[:,:]<thresh, 0, 255)
        thresh_arr = np.array(thresh_arr, dtype=np.uint8)

        height, width = thresh_arr.shape
        suspected_points = []
        for i in range(height):
            for j in range(width):
                if thresh_arr[i][j] != 0:
                    temp = calculateMaxSubmission(i, j, csv_crop)
                    if temp > CRITICAL_GRAD:
                        suspected_points.append([j,i])
        
        masking_img = np.zeros((height, width, 3), dtype=np.uint8)
        for pts in suspected_points:
            xy = np.array(pts)
            cv2.circle(masking_img, (xy[0], xy[1]), 3, (255, 255, 255), -1)

        masking_img = masking_img[:,:,0]
        masking_img = masking_img.astype(np.uint8)
        rp_contour, heirachy = cv2.findContours(masking_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # object_data = writeJason(xmin, ymin, xmax, ymax, temp_min, temp_max, temp_average, object_class, hp_contour, rp_contour)
        json_data = {}
        json_data["xmin"] = xmin
        json_data["ymin"] = ymin
        json_data["xmax"] = xmax
        json_data["ymax"] = ymax
        json_data["tmin"] = temp_min
        json_data["tmax"] = temp_max
        json_data["tmean"] = temp_average
        json_data["class"] = object_class
        json_data["hp_counter"] = hp_contour
        json_data["rp_counter"] = rp_contour

        #rule-base analysis
        rule = DiagnosisRule("./data/diagnosis_rule.json")
        diag_result = rule.diagnose(object_class, temp_max)
        json_data["DiagnosisCode"] = diag_result["code"]
        json_data["Cause of Failure"] = diag_result["cause"]
        json_data["Diagnosis"] = diag_result["action"]
        json_data["Over temperature"] = diag_result["Over Temperature"]
        json_data["FacilityName"] = diag_result["name"]
        json_data["Limit Temperature"] = diag_result["Limit Temperature"]
        json_data["FileName"] = fname+'.jpg'
        json_data["PointTemperature"] = json_data["tmax"]
        if diag_result["Over Temperature"] > 0:
            json_data["deltaT"] = json_data["tmax"] / json_data["Limit Temperature"]
            json_data["deltaT"] = round(json_data["deltaT"], 2)

        file_data["facilities"].append(writeJson2(json_data))
        
        with open(os.path.join(out_dir, (fname + '_{0}'.format(item_idx) + '.json')), 'w', encoding='utf-8') as make_file:
            json.dump(file_data, make_file, indent="\t", ensure_ascii=False)

        # create image
        img_original = cv2.imread(img)
        cv2.rectangle(img_original, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        
        textLabel = object_class
        (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
        textOrg = (xmin, ymin)
        cv2.rectangle(img_original, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
        cv2.rectangle(img_original, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
        cv2.putText(img_original, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
        
        png_name = fname + '_{0}'.format(item_idx) + '.png'
        cv2.imwrite(os.path.join(out_dir, png_name), img_original)

        item_idx += 1

def analyze(xml, csv, out_dir):
    #xml = Annotation file

    fname = os.path.basename(os.path.splitext(xml)[0])
    analyze = parse_xml.parcingXml(xml)
    analyze = np.array(analyze)
    if csv.endswith('csv'):
        csv_arr = parse_xml.parcingCsv(csv)
    elif csv.endswith('json'):
        with open(csv_path) as f:
            data = json.load(f)   
            csv_arr = np.array(json.loads(data['tempData']))
    file_data = OrderedDict()
    file_data["facilities"] = []

    for i in range(len(analyze[0])):
        xmin = int(analyze[0][i])
        xmax = int(analyze[1][i])
        ymin = int(analyze[2][i])
        ymax = int(analyze[3][i])
        object_class = analyze[4][i]
        csv_copy = copy.deepcopy(csv_arr)
        csv_crop = csv_copy[ymin:ymax, xmin:xmax]
        csv_flat = csv_crop.flatten()
        csv_flat = np.round_(csv_flat, 1)
        temp_min = csv_flat.min()
        temp_max = csv_flat.max()
        temp_average = np.average(csv_flat)
        temp_average = np.round_(temp_average, 1)

        # find heating points
        csv_copy = copy.deepcopy(csv_arr)
        csv_crop = csv_copy[ymin:ymax, xmin:xmax]
        thresh = np.percentile(csv_crop, 75)
        thresh_arr = np.zeros((len(csv_crop), len(csv_crop[0])), dtype=np.uint8)
        thresh_arr = np.where(csv_crop[:,:]<thresh, 0 ,255)
        thresh_arr = np.array(thresh_arr, dtype=np.uint8)
        hp_contour, _ = cv2.findContours(thresh_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # find reflection points
        csv_copy = copy.deepcopy(csv_arr)
        csv_crop = csv_copy[ymin:ymax, xmin:xmax]
        CRITICAL_GRAD = 0.4
        thresh = np.percentile(csv_crop, 75)
        thresh_arr = np.zeros((len(csv_crop), len(csv_crop[0])), dtype=np.uint8)
        thresh_arr = np.where(csv_crop[:,:]<thresh, 0, 255)
        thresh_arr = np.array(thresh_arr, dtype=np.uint8)

        # find top rate celsius
        csv_copy = copy.deepcopy(csv_arr)
        csv_crop = csv_copy[ymin:ymax, xmin:xmax]
        thresh = np.percentile(csv_crop, 5)
        thresh_arr = np.zeros((len(csv_crop), len(csv_crop[0])), dtype=np.uint8)
        thresh_arr = np.where(csv_crop[:, :] < thresh, 0, 255)
        thresh_arr = np.array(thresh_arr, dtype=np.uint8)
        tp_contour, _ = cv2.findContours(thresh_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        height, width = thresh_arr.shape
        suspected_points = []
        for i in range(height):
            for j in range(width):
                if thresh_arr[i][j] != 0:
                    temp = calculateMaxSubmission(i, j, csv_crop)
                    if temp > CRITICAL_GRAD:
                        suspected_points.append([j,i])
        
        masking_img = np.zeros((height, width, 3), dtype=np.uint8)
        for pts in suspected_points:
            xy = np.array(pts)
            cv2.circle(masking_img, (xy[0], xy[1]), 3, (255, 255, 255), -1)

        masking_img = masking_img[:,:,0]
        masking_img = masking_img.astype(np.uint8)
        rp_contour, heirachy = cv2.findContours(masking_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        
        # object_data = writeJason(xmin, ymin, xmax, ymax, temp_min, temp_max, temp_average, object_class, hp_contour, rp_contour)
        json_data = {}
        json_data["xmin"] = xmin
        json_data["ymin"] = ymin
        json_data["xmax"] = xmax
        json_data["ymax"] = ymax
        json_data["tmin"] = temp_min
        json_data["tmax"] = temp_max
        json_data["tmean"] = temp_average
        json_data["class"] = object_class
        json_data["hp_counter"] = hp_contour
        json_data["rp_counter"] = rp_contour
        json_data["tp_counter"] = tp_contour
        json_data["FileName"] = csv.split(os.sep)[-1]
        rule = DiagnosisRule("./data/diagnosis_rule.json")
        rule_dict = rule.diagnose(object_class, temp_average)
        json_data['FacilityName'] = rule_dict['name']
        json_data["Limit Temperature"] = rule_dict["Limit Temperature"]
        file_data["facilities"].append(writeJson2(json_data))

        # file_data["facilities"].append(object_data)
        
    # with open('./json_rb/'+fname+'.json', 'w', encoding='utf-8') as make_file:
    with open(os.path.join(out_dir, (fname+'.json')), 'w', encoding='utf-8') as make_file:
        json.dump(file_data, make_file, indent="\t", ensure_ascii=False)

def analyze_dir_recursive():
    root_data_dir = '/home/inseo/DATA/fuse'
    count = 1
    for data_dir in glob.glob(os.path.join(root_data_dir, 'fuse*')):
        print(count)
        print(data_dir)
        
        # data_dir = '/home/inseo/Desktop/test_data/FLIR3572'
        out_dir = os.path.join(data_dir, 'json_rb')

        prefix_xml = 'annotation'
        prefix_csv = 'csv'
        
        xml_dir = os.path.join(data_dir, prefix_xml)
        csv_dir = os.path.join(data_dir, prefix_csv)

        #create output path
        os.makedirs(out_dir, exist_ok=True)

        for item in glob.glob(os.path.join(xml_dir, '*.xml')):
            item_basename = os.path.basename(os.path.splitext(item)[0])
            csv = os.path.join(csv_dir, (item_basename+'.csv'))
            analyze(item, csv, out_dir)
            pass
        count += 1

def analyze_dir_recursive_1():
    root_data_dir = '/home/inseo/DATA/fuse'
    count = 1
    for data_dir in glob.glob(os.path.join(root_data_dir, 'fuse1')):
        print(count)
        print(data_dir)
        
        # data_dir = '/home/inseo/Desktop/test_data/FLIR3572'
        out_dir = os.path.join(data_dir, 'json_rb')

        prefix_xml = 'annotation'
        prefix_csv = 'csv'
        
        xml_dir = os.path.join(data_dir, prefix_xml)
        csv_dir = os.path.join(data_dir, prefix_csv)

        #create output path
        os.makedirs(out_dir, exist_ok=True)

        for item in glob.glob(os.path.join(xml_dir, '*.xml')):
            item_basename = os.path.basename(os.path.splitext(item)[0])
            csv = os.path.join(csv_dir, (item_basename+'.csv'))
            analyze(item, csv, out_dir)
            pass
        count += 1

def analyze_dir_recursive_2():
    root_data_dir = '/home/inseo/DATA/fuse'
    count = 1
    for data_dir in glob.glob(os.path.join(root_data_dir, 'fuse2')):
        print(count)
        print(data_dir)
        
        # data_dir = '/home/inseo/Desktop/test_data/FLIR3572'
        out_dir = os.path.join(data_dir, 'json_rb')

        prefix_xml = 'annotation'
        prefix_csv = 'csv'
        
        xml_dir = os.path.join(data_dir, prefix_xml)
        csv_dir = os.path.join(data_dir, prefix_csv)

        #create output path
        os.makedirs(out_dir, exist_ok=True)

        for item in glob.glob(os.path.join(xml_dir, '*.xml')):
            item_basename = os.path.basename(os.path.splitext(item)[0])
            csv = os.path.join(csv_dir, (item_basename+'.csv'))
            analyze(item, csv, out_dir)
            pass
        count += 1

class DiagnosisRule():
    def __init__(self, json_filename):
        json_file = open(json_filename, encoding='utf-8')
        self.rule_data = json.load(json_file)
            
    def diagnose(self, f_class, temperature, rule="ITC"):
        if f_class not in self.rule_data[rule].keys():
            f_class_original = f_class
            f_class = "undefined"
        
        name = self.rule_data[rule][f_class]["name"]
        LimitTemp = self.rule_data[rule][f_class]["LimitTemp"]
        
        fail_return = {"code":"NR", "cause":"정상", "action":"정상임", "Over Temperature":0, "name":name, "Limit Temperature": LimitTemp }
        

        for fail in self.rule_data[rule][f_class]["failure"]:
            # print(fail)
            if temperature >= LimitTemp + fail["dT"]:
                fail_return = fail
                fail_return.pop("dT", None)
                fail_return["Over Temperature"] = temperature - LimitTemp
                fail_return["name"] = name
                fail_return["Limit Temperature"] = LimitTemp            
        return fail_return

if __name__=="__main__":
    #example
    data = os.path.join('D:\\2020연구\\1) 한수원\\2분기\\20.04 우선구현 파일(이노팩토리)\\회전설비')
    xml_folder_path =  os.path.join(data,'annotations')
    csv_folder_path = os.path.join(data,'json')
    
    xml_folder = os.listdir(xml_folder_path)
    csv_folder = os.listdir(csv_folder_path)
    
    
    out_dir = os.path.join("D:\\2020연구\\1) 한수원\\2분기\\20.04 우선구현 파일(이노팩토리)\\회전설비", "results")
    for xml, csv in list(zip(xml_folder, csv_folder)):
        xml_path = os.path.join(xml_folder_path, xml)
        csv_path = os.path.join(csv_folder_path, csv)
        print(xml_path, csv_path)

        analyze(xml_path, csv_path, out_dir)
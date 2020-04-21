# --------------------------------------------------------
# RuleBase Diagnosis using Temperature
# Written by SeonWoo Lee
# --------------------------------------------------------

import xml.etree.ElementTree as ET
import os
import _pickle as cPickle
import numpy as np
from analyze_image import DiagnosisRule
import cv2
import json
from tqdm import tqdm
def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                            int(bbox.find('ymin').text),
                            int(bbox.find('xmax').text),
                            int(bbox.find('ymax').text)]
        objects.append(obj_struct)
    return objects
    '''
        온도 범위
        ITC, NAVY, NETA, NMAC의 온도범위에 따른 대응활동과
        대응활동의 권장사항에 대하여 나열하고자 함
        |Diagnosis-------- |ITC--|NAVY--|NETA|NMAC--|Recommanded|
        |Advisory(주의)----|<5C--|10~24C|1~3C|0.3~8C|예방정비 시 점검
        |Intermediate(중급)|N/A--|25~39C|4~15|9~28C-|우선정비(기회가 되는대로)
        |Serious(심각)-----|30~80C|40~69C|N/A-|29~56C|가능한 한 신속 정비작업
        |Immediate(긴급조치)|>80C-|>70---|>15C|>56C--|긴급정비
        전기설비에 대한 시각도 평가 기준
    '''
    
def rulebase_temp(ary): #입력: 관심영역
    TEMP_LIST=[]
    itc_factor = 0.4   #ITC Impact Factor
    navy_factor = 0.3 #NAVY Impact Factor
    neta_factor = 0.2 #NETA Impact Factor
    nmac_factor = 0.1 #NMAC Impact Factor
    ITC_Adversory = np.count_nonzero(np.where(ary<5.0, ary, 0))*itc_factor
    TEMP_LIST.append(('ITC_Adversory',ITC_Adversory))
    ITC_Serious = np.count_nonzero(np.where((ary<=30.0) & (ary>=5.0), ary, 0))*itc_factor
    TEMP_LIST.append(('ITC_Serious', ITC_Serious))
    ITC_Immediate = np.count_nonzero(np.where(ary>80, ary, 0))*itc_factor
    TEMP_LIST.append(('ITC_Immediate', ITC_Immediate)) 
    
    NAVY_Adversory = np.count_nonzero(np.where((ary>=10.0) & (ary<=24.0), ary, 0))*navy_factor
    TEMP_LIST.append(('NAVY_Adversory', NAVY_Adversory))
    NAVY_Intermediate = np.count_nonzero(np.where((ary>=25.0) & (ary<40.0), ary, 0))*navy_factor
    TEMP_LIST.append(('NAVY_Intermediate', NAVY_Intermediate))
    NAVY_Serious = np.count_nonzero(np.where((ary>=40.0) & (ary< 70.0), ary, 0))*navy_factor
    TEMP_LIST.append(('NAVY_Serious', NAVY_Serious))
    NAVY_Immediate = np.count_nonzero(np.where(ary>=70.0, ary, 0))*navy_factor
    TEMP_LIST.append(('NAVY_Immediate', NAVY_Immediate))
    
    NETA_Adversory = np.count_nonzero(np.where((ary>=1) & (ary<4), ary, 0))*neta_factor
    TEMP_LIST.append(('NETA_Adversory', NETA_Adversory))
    NETA_Intermediate = np.count_nonzero(np.where((ary>=9) & (ary<15.0), ary, 0))*neta_factor
    TEMP_LIST.append(('NETA_Intermediate', NETA_Intermediate))
    NETA_Immediate = np.count_nonzero(np.where(ary>=15.0, ary, 0))*neta_factor
    TEMP_LIST.append(('NETA_Immediate', NETA_Immediate))
    
    NMAC_Adversory = np.count_nonzero(np.where((ary>=0.3) & (ary<9.0), ary, 0))*nmac_factor
    TEMP_LIST.append(('NMAC_Adversory', NMAC_Adversory))
    NMAC_Intermediate = np.count_nonzero(np.where((ary>=9) & (ary<28.0), ary, 0))*nmac_factor
    TEMP_LIST.append(('NMAC_Intermediate', NMAC_Intermediate))
    NMAC_Serious = np.count_nonzero(np.where((ary>=29.0 )& (ary<56.0), ary, 0))*nmac_factor
    TEMP_LIST.append(('NMAC_Serious', NMAC_Serious))
    NMAC_Immediate = np.count_nonzero(np.where(ary>=56.0, ary, 0))*nmac_factor
    TEMP_LIST.append(('NMAC_Immediate', NMAC_Immediate))
    
    TEMP_LIST.sort(key=lambda t:t[1], reverse=True)
    result= {'Adversory':0,'Intermediate':0,'Serious': 0, 'Immediate':0}
    top3 = TEMP_LIST[:3]
    for top in top3:
        if top[0].find('Adversory') != -1:
            result['Adversory']+=top[1]
        elif top[0].find('Intermediate') !=-1:
            result['Intermediate'] +=top[1]
        elif top[0].find('Serious')!=-1:
            result['Serious'] +=top[1]
        elif top[0].find('Immediate')!=-1:
            result['Immediate'] +=top[1]
            
    res = list(result.items())
    res.sort(key=lambda t:t[1], reverse=True)
    return res

def rule_base_analysis(filename, obj_list):
    if not filename.endswith('csv'):
        print('{} is not csv file format'.format(filename))
        exit()
    csv_data = np.loadtxt(filename, dtype=float, delimiter=',')
    '''
    data_json= "./data/diagnosis_rule.json"
    if not os.path.exists(data_json):
        print("Json File {} is not Exit!".format(data_json))
        exit()
    rule = DiagnosisRule(data_json)
    '''
    json_list = []
    for obj in obj_list:
        json_data = {}
        object_class = obj['name']
        limit_table={'Motor': 125,
                            'ESWP_B':125,
                            'CB': 60,
                            'Pt_1': 90,
                            'E': 100,
                            'BTCG_TR':85,
                            'BTCG_BUSBAR':100,
                            'TRDR':100,
                            'RT_TR':100,
                            'FUSE':80}
        json_data['filename'] = filename
        json_data['object_name'] = object_class
        json_data["xmin"] = obj['bbox'][0]
        json_data["ymin"] = obj['bbox'][1]
        json_data["xmax"] = obj['bbox'][2]
        json_data["ymax"] = obj['bbox'][3]
        
        roi = csv_data[json_data['ymin']:json_data['ymax'],\
                                json_data['xmin']:json_data['xmax']]
        json_data["tmax"] = np.max(roi)
        json_data["tmin"] = np.min(roi)
        json_data["tmean"] = np.average(roi)
        json_data["class"] = object_class
        try:
            json_data['limit_temp'] = limit_table[object_class]
            # print(object_class)
            # print(obj['bbox'])
            json_data['diagnosis'] = rulebase_temp(roi - limit_table[object_class])
        except:
            print("{} cannot have Limit Class".format(object_class))

        # roi2 = roi.astype(dtype=np.uint8)
        # roi2 = cv2.cvtColor(roi2, cv2.COLOR_GRAY2BGR)
        # json_list.append(json_data)
        # cv2.imshow('test', roi2)
        # cv2.waitKey(0)
        # exit()
        json_list.append(json_data)
    return json_list

if __name__ == '__main__':
    #xml_path = os.path.join('..','data','VOCdevkit','20_FirsQuarter_readymade_data','wulsung3','annotation')
    xml_path = "C:\\Users\\hci-lab01\\Desktop\\새 폴더\\2335-811-E-TR03M(MAIN XFMR) (우선 구현)\\T660\\Annotations"
    #csv_path = os.path.join('..','data','VOCdevkit','20_FirsQuarter_readymade_data','wulsung3', 'csv')
    csv_path = 'C:\\Users\\hci-lab01\\Desktop\\새 폴더\\2335-811-E-TR03M(MAIN XFMR) (우선 구현)\\T660\\Csvs'
    xml_lists = os.listdir(xml_path)
    csv_lists = os.listdir(csv_path)
    json_list = []
    for xml_name, csv_name in tqdm(list(zip(xml_lists, csv_lists))):
        xml = os.path.join(xml_path, xml_name)
        objects = parse_rec(xml)
        
        csv = os.path.join(csv_path, csv_name)
        _json = rule_base_analysis(filename=csv, obj_list=objects)
        json_list.append(_json)
    #with open('./output/result_json.json', 'w') as f:
    with open('C:\\Users\\hci-lab01\\Desktop\\새 폴더\\2335-811-E-TR03M(MAIN XFMR) (우선 구현)\\T660\\Results\\result_json.json', 'w') as f:
        json.dump(json_list, f, indent=4)

    pass
    
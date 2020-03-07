import numpy as np
from xml.dom import minidom

# parameter : fileName, dir path (ex) './imput/') / return : np array (with csv file converted)
def parcingCsv(file):
    csv_array = np.loadtxt(file, delimiter=',')
    return csv_array

# parameter : fileName, dir path (ex) './imput/') / return : xmin, xmax, ymin, ymax list in xml
def parcingXml(file):
    xml_origin = minidom.parse(file)
    xmins = xml_origin.getElementsByTagName('xmin')
    xmaxs = xml_origin.getElementsByTagName('xmax')
    ymins = xml_origin.getElementsByTagName('ymin')
    ymaxs = xml_origin.getElementsByTagName('ymax')
    names = xml_origin.getElementsByTagName('name')
    xmin_arr = []
    xmax_arr = []
    ymin_arr = []
    ymax_arr = []
    name_arr = []
    for i in range(len(xmins)):
        xmin_arr.append(xmins[i].firstChild.data)
        xmax_arr.append(xmaxs[i].firstChild.data)
        ymin_arr.append(ymins[i].firstChild.data)
        ymax_arr.append(ymaxs[i].firstChild.data)
        name_arr.append(names[i].firstChild.data)

    return [xmin_arr, xmax_arr, ymin_arr, ymax_arr, name_arr]
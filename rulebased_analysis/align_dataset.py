import glob
import os
import sys
from os.path import join, basename, splitext

def align_dataset_by_basename(source, target, verbose=False):
    # get source list
    if not os.path.exists(source):
        print('Error : source path "{0}" doesn\'t exist'.format(source))
        sys.exit(1)
    input = join(source,'*.*')
    source_item_list = glob.glob(input)
    source_item_list = [basename(splitext(item)[0]) for item in source_item_list]

    for target_dir in target:
        if not os.path.exists(target_dir):
            print('Warning : target path "{0}"   doesn\'t exist'.format(target_dir))
            continue
        for item in glob.glob(join(target_dir, '*.*')):
            if not basename(splitext(item)[0]) in source_item_list:
                os.remove(item)
        
        if verbose:
            print('Aligned "{0}"'.format(target_dir))
    
    

if __name__ == "__main__":
    
    dataset_dir = "/home/inseo/DATA/fuse/fuse1"
    
    # for flir_dir in glob.glob(join(dataset_dir, '*')):
    #     print(flir_dir)
    #     source = join(flir_dir,'Annotation')
    #     target_list = [join(flir_dir, 'color'), join(flir_dir, 'csv_celsius')]

    #     align_dataset_by_basename(source, target_list, verbose=True)
    source = os.path.join(dataset_dir, 'annotation')
    target_list = [os.path.join(dataset_dir, 'color')]
    align_dataset_by_basename(source, target_list, verbose=True)
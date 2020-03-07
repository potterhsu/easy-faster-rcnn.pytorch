#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import os.path
import glob

if __name__ == "__main__":
    target_path = "/Users/inseo/Downloads/새 폴더 (4)"
    jpg_list = glob.glob(os.path.join(target_path, "**/*.jpg"), recursive=True)
    for jpg in jpg_list:
        print(jpg)
        os.system('python flir_image_extractor.py -c thermal -csv -i "{0}"'.format(jpg))
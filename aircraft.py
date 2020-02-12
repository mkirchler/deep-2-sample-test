import os
from shutil import copyfile

from tqdm import tqdm

BASE = 'PATHTOEXTRACTEDDATA'
PATH_TO_DATA = os.path.join(BASE, 'data/images')
PATH_TO_GT = os.path.join(BASE, 'data/')
PATH_TO_OUTPUT = os.path.join(BASE, 'testing') ## this should equal PATH_TO_PLANES in data.py

def prep_data():
    if not os.path.isdir(PATH_TO_OUTPUT):
        os.mkdir(PATH_TO_OUTPUT)
    
    tv = os.path.join(PATH_TO_GT, 'images_family_trainval.txt')
    t = os.path.join(PATH_TO_GT, 'images_family_test.txt')
    f_trainval = open(tv, 'r')
    f_test = open(t, 'r')

    for line in tqdm(f_trainval):
        split = line.split()
        ID = split[0]
        c = '-'.join(split[1:])
        
        # handle /-characters
        c = c.replace('/', '-')
        class_dir = os.path.join(PATH_TO_OUTPUT, c)
        if not os.path.isdir(class_dir):
            os.mkdir(class_dir)
        filename = ID + '.jpg'
        file_path = os.path.join(PATH_TO_DATA, filename)
        outfile_path = os.path.join(class_dir, filename)
        copyfile(file_path, outfile_path)

    for line in tqdm(f_test):
        split = line.split()
        ID = split[0]
        c = '-'.join(split[1:])
        
        c = c.replace('/', '-')
        class_dir = os.path.join(PATH_TO_OUTPUT, c)
        if not os.path.isdir(class_dir):
            os.mkdir(class_dir)
        filename = ID + '.jpg'
        file_path = os.path.join(PATH_TO_DATA, filename)
        outfile_path = os.path.join(class_dir, filename)
        copyfile(file_path, outfile_path)

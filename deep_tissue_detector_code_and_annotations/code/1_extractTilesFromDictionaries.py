# 1_extractTilesFromDictionaries.py
#
# This file checks the integrity of annotations on all .xml files which contain ROIs for H&E slides
# It creates so-called tile-dictionaries
#
# created by: Marcel Gehrung - 27/11/2018

import glob
import os
import pickle
import sys

import numpy as np
import pyvips as pv
from tqdm import tqdm
from pathlib import Path
sys.path.append('./pathml')
from pathml import slide

verbose = True
magnificationLevel = 0  # 1 = 20(x) or 0 = 40(x)
slidesRootFolder = "../wsis/"
tileDictionariesFileList = glob.glob(
    "../tileDictionaries/*.p")
tilesRootFolder = "../tiles/"

format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

for tileDictionaryFile in tqdm(tileDictionariesFileList, desc="Processing tile dictionaries"):
    if verbose:
        print('Opening file: ' + tileDictionaryFile)
    tileDictionary = pickle.load(open(tileDictionaryFile, "rb"))
    slideFileName = Path(tileDictionaryFile).resolve().stem
    slideFolder = slideFileName


    if os.path.exists(slidesRootFolder + slideFileName+'.svs'):
        heSlide = slide.Slide(
        slidesRootFolder + slideFileName+'.svs', level=magnificationLevel)
    if os.path.exists(slidesRootFolder + slideFileName+'.tif'):
        heSlide = slide.Slide(
        slidesRootFolder + slideFileName+'.tif', level=magnificationLevel)

    for k in range(0,5):
        if 'openslide.level['+str(k)+'].downsample' in heSlide.slideProperties:
            # print(slidePML.slideProperties['openslide.level['+str(k)+'].downsample'])
            # print(round(k))
            if round(float(heSlide.slideProperties['openslide.level['+str(k)+'].downsample'])) == 4:
                downsampledMagnificationLevel = k
    if os.path.exists(slidesRootFolder + slideFileName+'.svs'):
        heSlide = pv.Image.new_from_file(
        slidesRootFolder + slideFileName+'.svs', level=downsampledMagnificationLevel)
    if os.path.exists(slidesRootFolder + slideFileName+'.tif'):
        heSlide = pv.Image.new_from_file(
        slidesRootFolder + slideFileName+'.tif', level=downsampledMagnificationLevel)
    try:
        os.mkdir(tilesRootFolder + slideFolder)
        print("Directory ", tilesRootFolder + slideFolder, " Created ")
    except FileExistsError:
        print("Directory ", tilesRootFolder + slideFolder, " already exists")
        continue
    for key in tileDictionary:
        try:
            os.mkdir(tilesRootFolder + slideFolder + "/" + key)
            print("Directory ", tilesRootFolder + slideFolder + "/" + key, " Created ")
        except FileExistsError:
            print("Directory ", tilesRootFolder + slideFolder + "/" + key, " already exists")
        # Iterate over all tiles, extract them and save
        for tile in tqdm(tileDictionary[key]):
            if not os.path.exists(tilesRootFolder + slideFolder + "/" + key + '/' + slideFolder + '_' + str(tile['x']) + '_' + str(tile['y']) + '_' + str(tile['tileSize']) + '.jpg'):
                # print(tile['x'],tile['y'],tile['tileSize'],tile['tileSize'])
                areaHe = heSlide.extract_area(
                    tile['x'], tile['y'], tile['tileSize'], tile['tileSize'])
                areaHe.write_to_file(tilesRootFolder + slideFolder + "/" + key + '/' + slideFolder + '_' + str(
                    tile['x']) + '_' + str(tile['y']) + '_' + str(tile['tileSize']) + '.jpg', Q=100)
                # areaHeMem = np.ndarray(buffer=areaHe.write_to_memory(), dtype=format_to_dtype[areaHe.format], shape=[areaHe.height, areaHe.width, areaHe.bands])

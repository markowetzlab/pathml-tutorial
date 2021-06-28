# 0_checkAnnotations.py
#
# This file checks the integrity of annotations on all .xml files which contain ROIs for H&E slides
# It creates so-called tile-dictionaries
#

import glob, os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from shapely import geometry
import pyvips as pv
import numpy as np
import math
from tqdm import tqdm
import pickle, sys
sys.path.append('./pathml')
from pathml import slide
from pathlib import Path


tileSize=224 # extract 448 pixel tiles at given magnification level, here its 5x
magnificationLevel = 0 # 1 = 20(x) or 0 = 40(x)
enforceMagnificationLock = True # Only permit slides which were scanned at around 0.25 per micron
annotationCoeverageThreshold = 0.9


verbose = True
slidesRootFolder = "../wsis"
annotationsFileList = glob.glob("../xmls/*.xml")
slideFileList = glob.glob("../wsis/*.svs") + glob.glob("../wsis/*.tif")
classNames = ['artifact', 'background', 'tissue']


for slideFile in tqdm(slideFileList,desc="Processing slides"):
    tileDictionary = {listKey:[] for listKey in classNames}
    slideFileName = Path(slideFile).resolve().stem
    annotationFile = '../xmls/'+slideFileName+'.xml'

    print(slideFileName)
    #if os.path.isfile('data/tileDictionaries-tff3-300px-at-40x/'+slideFileName.replace('.svs','')+'.p'):
    if os.path.isfile('../tileDictionaries/'+slideFileName+'.p'):
        print("Case already processed. Skipping...")
        continue
    slidePML = slide.Slide(slideFile,level=magnificationLevel)

    if float(slidePML.slideProperties['openslide.mpp-x']) > 0.23 and float(slidePML.slideProperties['openslide.mpp-x']) < 0.27:
        slideMagnificationLevel = 40
        tileSize = 448
    elif float(slidePML.slideProperties['openslide.mpp-x']) > 0.48 and float(slidePML.slideProperties['openslide.mpp-x']) < 0.52:
        slideMagnificationLevel = 20
        tileSize = 224

    # ßprint(slideMagnificationLevel)

    for k in range(0,5):
        if 'openslide.level['+str(k)+'].downsample' in slidePML.slideProperties:
            # print(slidePML.slideProperties['openslide.level['+str(k)+'].downsample'])
            # print(round(k))
            if round(float(slidePML.slideProperties['openslide.level['+str(k)+'].downsample'])) == 4:
                downsampledMagnificationLevel = k
    #slidePML = slide.Slide(slideFile,level=downsampledMagnificationLevel)
    # Calculate the pixel size based on provided tile size and magnification level
    rescaledMicronsPerPixel = float(slidePML.slideProperties['openslide.mpp-x'])*float(slidePML.slideProperties['openslide.level['+str(magnificationLevel)+'].downsample'])
    #if verbose: print("Given the properties of this scan, the resulting tile size will correspond to "+str(round(tileSize*rescaledMicronsPerPixel,2))+" μm edge length")

    # Calculate scaling factor for annotations
    annotationScalingFactor = float(slidePML.slideProperties['openslide.level[0].downsample'])/float(slidePML.slideProperties['openslide.level['+str(downsampledMagnificationLevel)+'].downsample'])

    print("Scale "+str(annotationScalingFactor))

    # Extract slide dimensions
    slideWidth = int(slidePML.slideProperties['width'])
    slideHeight = int(slidePML.slideProperties['height'])

    if verbose: print('Opening file: ' + annotationFile)
    tree = ET.parse(annotationFile) # Open .xml file
    root = tree.getroot() # Get root of .xml tree
    if root.tag == "ASAP_Annotations": # Check whether we actually deal with an ASAP .xml file
        if verbose: print('.xml file identified as ASAP annotation collection') # Display number of found annotations
    else:
        raise Warning('Not an ASAP .xml file')
    allAnnotations = root.find('Annotations') # Find all annotations for this slide
    if verbose: print('XML file valid - ' + str(len(allAnnotations)) + ' annotations found.') # Display number of found annotations
    
    # Generate a list of tile coordinates which we can extract
    for annotation in tqdm(allAnnotations,desc="Parsing annotations"):
        annotationTree = annotation.find('Coordinates')
        x = []
        y = []
        polygon = []
        for coordinate in annotationTree:
            info = coordinate.attrib
            polygon.append((float(info['X'])*annotationScalingFactor, float(info['Y'])*annotationScalingFactor))

        polygonNp = np.asarray(polygon)
        polygonNp[:,1] = slideHeight-polygonNp[:,1]
        poly = geometry.Polygon(polygonNp).buffer(0)
        # generate relevant tile stuff
        topLeftCorner = (min(polygonNp[:,0]),max(polygonNp[:,1]))
        bottomRightCorner = (max(polygonNp[:,0]),min(polygonNp[:,1]))
        tilesInXax = math.ceil((bottomRightCorner[0] - topLeftCorner[0])/tileSize)
        tilesInYax = math.ceil((topLeftCorner[1] - bottomRightCorner[1])/tileSize)
        x = poly.exterior.coords.xy[0]
        y = poly.exterior.coords.xy[1]
        ##plt.figure(figsize=(5,5))
        ##plt.plot(x,y,'r')

        if poly.area >1.5*tileSize**2:
            for xTile in range(tilesInXax):
                for yTile in range(tilesInYax):
                    minX = topLeftCorner[0]+tileSize*xTile
                    minY = topLeftCorner[1]-tileSize*yTile
                    maxX = topLeftCorner[0]+tileSize*xTile+tileSize
                    maxY = topLeftCorner[1]-tileSize*yTile-tileSize
                    tileBox = geometry.box(minX,minY,maxX,maxY)
                    #colorThisTile = 1-np.round(np.repeat(poly.intersection(tileBox).area/tileSize**2,3),2)
                    intersectingArea = poly.intersection(tileBox).area/tileSize**2
                    if intersectingArea>annotationCoeverageThreshold:
                        if annotation.attrib['PartOfGroup'] in tileDictionary:
                            tileDictionary[annotation.attrib['PartOfGroup']].append({'x': minX, 'y': slideHeight-minY, 'tileSize': tileSize, 'annotationCoverageArea': round(intersectingArea,4)})
                        else:
                            raise Warning('Annotation ' + annotation.attrib['PartOfGroup'] + ' is not part of a pre-defined group')
                        ##plt.plot(tileBox.exterior.coords.xy[0],tileBox.exterior.coords.xy[1],color=colorThisTile)
        else:
            minX = geometry.Point(poly.centroid).coords.xy[0][0]-tileSize/2
            minY = geometry.Point(poly.centroid).coords.xy[1][0]-tileSize/2
            maxX = geometry.Point(poly.centroid).coords.xy[0][0]+tileSize/2
            maxY = geometry.Point(poly.centroid).coords.xy[1][0]+tileSize/2
            tileBox = geometry.box(minX,minY,maxX,maxY)
            intersectingArea = poly.intersection(tileBox).area/tileSize**2

            #colorThisTile = 1-np.round(np.repeat(poly.intersection(tileBox).area/tileSize**2,3),2)
            #if intersectingArea>0.333: # WHY DID I DO THIS IN THE FIRST PLACE???????????????????????????????????????????????????????????????
            if annotation.attrib['PartOfGroup'] in tileDictionary:
                tileDictionary[annotation.attrib['PartOfGroup']].append({'x': minX, 'y': slideHeight-minY-tileSize, 'tileSize': tileSize, 'annotationCoverageArea': intersectingArea})
            else:
                raise Warning('Annotation is not part of a pre-defined group')

                ##plt.plot(geometry.Point(poly.centroid).coords.xy[0],geometry.Point(poly.centroid).coords.xy[1], marker='o', markersize=3, color="red")
                ##plt.plot(tileBox.exterior.coords.xy[0],tileBox.exterior.coords.xy[1],color=colorThisTile)
        ##plt.xlim(topLeftCorner[0]-tileSize,bottomRightCorner[0]+tileSize)
        ##plt.ylim(bottomRightCorner[1]-tileSize,topLeftCorner[1]+tileSize)
        ##plt.show(block=False)

    pickle.dump(tileDictionary, open('../tileDictionaries/'+slideFileName+'.p', 'wb'))
        #plt.figure(figsize=(5,5))
        ##areaHe = heSlide.extract_area(topLeftCorner[0],slideHeight-topLeftCorner[1],bottomRightCorner[0]-topLeftCorner[0],topLeftCorner[1]-bottomRightCorner[1])
        ##areaHeMem = np.ndarray(buffer=areaHe.write_to_memory(),
        ##                              dtype=format_to_dtype[areaHe.format],
        ##                              shape=[areaHe.height, areaHe.width, areaHe.bands])
        #plt.imshow(areaHeMem)
        #plt.show(block=False)
##plt.show()

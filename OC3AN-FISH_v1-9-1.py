#!/usr/bin/env python3
###
#
# THIS CODE WORKS WITH PYTHON 3.8.0 AND THE SPECIFIED PACKAGE VERSIONS
#
# version 1.9.1
#
### importing libraries
from pathlib import Path # gets path of files (pre-installed)
import os # get parent folder name of images (pre-installed)
import glob # finds all the pathnames matching a specified pattern (pre-installed)
import numpy as np # various calculations (ver. 1.23.1)
import pandas as pd # store measurements in a table and export (ver. 1.4.3)
from skimage.morphology import remove_small_objects, remove_small_holes, disk, ball, erosion, dilation, convex_hull_image # morphological operations (scikit-image ver. 0.19.3)
from skimage import measure, exposure # labeling and segmentation
from skimage.io import imread, imsave # saving image stacks for visualisation
from skimage.filters import threshold_otsu, threshold_li, gaussian, rank # thresholding dapi channel
from skimage.segmentation import watershed, clear_border # separating touching nuclei and removing objects touching the border
from skimage.registration import phase_cross_correlation # image translation registration by cross-correlation
from tifffile import TiffFile # loading image metadata (ver. 2022.8.12)
from tifffile.tifffile import imagej_description_metadata # reading imagej description
from scipy import ndimage as ndi # multidimensional image processing
from scipy.spatial import distance # used for euclidian distance calculation between object centroids
import seaborn as sns # for quick and easy plotting (ver. 0.11.2)
from matplotlib import pyplot as plt # (ver. 3.5.3)
from skimage.color import label2rgb
## the following module can be used for visualisation of stacks with masks ##(uncomment lines 450-471 and/or move the code blocks to any place in the code where you want visualisation)
import napari # 3D viewer for images (ver. 0.4.16)

### ----- SETTINGS ----- ###
# your filename location is the following: fileDir/<image folder with split channels>/ChName+*+dataSuffix
fileDir = '~/Documents/uni_lmu/MSc/images/2022-08-28/Split_Series_bof/' # path to folder where images are located (= work directory)
# all files in 'fileDir' with matching 'dataSuffix' and channel name will be processed! Make sure that no unwanted files are in the provided folder!
dapiChName = 'b_C189' # name prefix of dapi channel
fish1ChName = 'o_C189' # name prefix of first fish channel
fish2ChName = 'f_C189' # name prefix of second fish channel
dataSuffix = '.tif' # suffix of the image files
saveNameMeasure = '220927_C211_SC_BAC-i19_measurements_FISHmask' # savename for measurements table. will be appended with file format (.csv)
saveDir = '~/Documents/uni_lmu/MSc/images/2022-08-28/' # path to folder where data produced by this code will be saved (measurements table, optionally images as well)

## setting for chromatic shift correction
beadsDir = '~/Documents/uni_lmu/MSc/images/beads/Split_20220829/' # path to folder where beads images are located
bb = 'b_' # name of bead channel 1 --> correspondent to DAPI (bb = beads blue)
bo = 'o_' # name of bead channel 2 --> correspondent to FISH Ch 1
bf = 'f_' # name of bead channel 3 --> correspondent to FISH Ch 2

## fallback information in case metadata can't be read
# for beads
b_zSpacing = 0.130 # microns per px
b_xyPXsize = 0.058 # microns per px
# for imgs
zSpacing = 0.250 # microns per px
xyPXsize = 0.025 # microns per px

smallObjSize = 1100 # For dapi imgs: objects with area (px) smaller than this will be removed from each slice
smallObjSizeFish = 100 # Same as above for fish imgs
smallHoleSize = 27000 # For dapi imgs: fills holes smaller than this size (px) in each slice (helps closing the segmentation) # WAS 1950
smallHoleSizeFish = 900 # Same as above for fish imgs
gausSigma = 2 # specifies sigma value for gaussian blur filter
erodeFoot = 15 # footprint size for the minimum_filter (erosion) operation in line 101 --> erosion by replacing each pixel in the image with the smallest pixel value in that pixel's neighborhood
cutoff = 0.925 # cutoff value for solidity coefficient used to determine whether nuclei are separated. If solidity is below cutoff, nuclei are not separated
solidDapi = 0.90 # cutoff value for final dapi mask (quality check). If solidity is below cutoff, mask is deemed bad and the nucleus is not used for calculations

fishSmoothON = True # Turn on/off FISH image smoothing for measurements

### ----- SETTINGS ----- ###

##### Let's start the real thing #####
### initialise pandas dataframe to save all measurements. Each measurement will be added to 'df'
df = pd.DataFrame(columns=['Folder', 'Cell_No', 'FISH_Labels', 'FISH1_Vol', 'FISH2_Vol', 'Total_Vol', 'Overlap_Vol', 'Volume_Comparison', 'Overlap', 'FISH2_in_FISH1'])

print('\nGrabbing the data ...')
pathlistDapi = sorted(glob.glob(fileDir+'**/*'+dapiChName+'*'+dataSuffix, recursive=True)) # contains dapi image paths
pathlistFish1 = sorted(glob.glob(fileDir+'**/*'+fish1ChName+'*'+dataSuffix, recursive=True)) # contains fish1 image paths
pathlistFish2 = sorted(glob.glob(fileDir+'**/*'+fish2ChName+'*'+dataSuffix, recursive=True)) # contains fish2 image paths
### get beads files
pathlistBB = sorted(glob.glob(beadsDir+'**/*'+bb+'*'+dataSuffix, recursive=True)) # contains  image paths
pathlistBO = sorted(glob.glob(beadsDir+'**/*'+bo+'*'+dataSuffix, recursive=True)) # contains  image paths
pathlistBF = sorted(glob.glob(beadsDir+'**/*'+bf+'*'+dataSuffix, recursive=True)) # contains  image paths

# check if there is data available
if (len(pathlistDapi) == 0 or len(pathlistFish1) ==0 or len(pathlistFish2) == 0):
    raise Exception('Data not found, check the provided path and try again.')
if (len(pathlistBB) == 0 or len(pathlistBO) == 0 or len(pathlistBF) == 0):
    raise Exception('Beads not found, check the provided path and try again.')
# check if the sizes of all pathlists match
if (len(pathlistDapi) != len(pathlistFish1) != len(pathlistFish2)):
    raise Exception("List lengths don't match, check your folders for unwanted files. Otherwise, some files might be missing.")

### ----- chromatic shift calculation ----- ###
print('\nCalculating chromatic shift using beads ...')
allShiftBO = np.zeros(shape=(len(pathlistBB),3)) # array to save shift for each bead batch
allShiftBF = np.zeros(shape=(len(pathlistBB),3)) # array to save shift for each bead batch
# function for shift correction
def CScalc(ref, tg, ch):
    """
        ref: reference channel (DAPI) image;                           type: numpy array
        tg:  target channel (FISH) image;                              type: numpy array
        ch:  channels being compared (e. g. 'b-o'), used for printing; type: string
    """
    # Compute pairwise differences between bead channels by measuring phase correlation
    # pixel precision first
    shift, _, _ = phase_cross_correlation(ref, tg, normalization=None)
    shift = shift.astype(int)
    # now subpixel precision (won't be used for shift correction)
    shiftP1, _, _ = phase_cross_correlation(ref, tg, upsample_factor=100, normalization=None)
    # print detected shift as user feedback
    print(f'Detected pixel offset {ch} (z, y, x): {shift}')
    # print detected subpixel shift as user feedback
    print(f'Detected subpixel offset {ch} (z, y, x): {shiftP1}')
    return shift
# process beads first --> loop over beads batches
for batch in range(len(pathlistBB)):
    print('Processing batch', batch+1, ' ...')
    ### load each beads channel
    b1 = imread(pathlistBB[batch]) # DAPI
    b2 = imread(pathlistBO[batch])
    b3 = imread(pathlistBF[batch])
    # read metadata of one channel --> is the same for all channels
    with TiffFile(pathlistBB[batch]) as beadMeta:
        assert beadMeta.is_imagej # check if metadata is written in imagej style (returns 'True')
        if ('unit' in imagej_description_metadata(beadMeta.pages[0].tags['ImageDescription'].value)) == True:
            # get the resolution unit
            resUnit = imagej_description_metadata(beadMeta.pages[0].tags['ImageDescription'].value)['unit']
            assert resUnit == 'micron' or resUnit == '\\u00B5m' # should be micron or µm
            # get the resolution in zyx
            physZ = np.around(beadMeta.imagej_metadata['spacing'], decimals=3) # = physical Z res --> spacing between z-slices in resUnit
            physY, physX = beadMeta.pages[0].resolution # = physical X & Y res in px per resUnit
            # convert xy res to resUnit per px
            physY, physX = np.around(1/physY, decimals=3), np.around(1/physX, decimals=3)
        else:
            physZ = np.around(b_zSpacing, decimals=3) # in microns per px
            physY, physX = np.around(b_xyPXsize, decimals=3), np.around(b_xyPXsize, decimals=3) # in microns per px
    # Compute pairwise differences between bead channels
    shiftBO = CScalc(b1, b2, ch='b-o')
    shiftBF = CScalc(b1, b3, ch='b-f')
    # convert shift in px to µm using metadata
    shiftBO = np.multiply(shiftBO, (physZ, physY, physX))
    shiftBF = np.multiply(shiftBF, (physZ, physY, physX))
    # store calculated shift in array
    allShiftBO[batch] = shiftBO
    allShiftBF[batch] = shiftBF
# calculate and print mean shift
meanShiftBO = [ np.mean(allShiftBO[:,0]), np.mean(allShiftBO[:,1]), np.mean(allShiftBO[:,2]) ] #np.rint()
meanShiftBF = [ np.mean(allShiftBF[:,0]), np.mean(allShiftBF[:,1]), np.mean(allShiftBF[:,2]) ]
print(f'\nMean calculated shift b-o (z, y, x): {np.multiply(meanShiftBO, 1000)} nm')
print(f'Mean calculated shift b-f (z, y, x): {np.multiply(meanShiftBF, 1000)} nm')
### ----- shift calculation done ----- ###

# custom 3D footprints
enh3D = np.stack((np.pad(disk(2), 33), disk(35), np.pad(disk(2), 33))) #np.stack((np.pad(disk(3), 17), disk(20), np.pad(disk(3), 17)))
filt3D = np.stack((np.pad(disk(1), 10), disk(11), np.pad(disk(1), 10))) #np.stack((np.pad(disk(2), 16), np.pad(disk(3), 15), disk(18), np.pad(disk(3), 15), np.pad(disk(2), 16)))

### ----- FISH channel segmentation as a function ----- ###
def fishSegment(fishimg, num):
    """
        --- input:
        fishimg: fish channel image to be segmented;                  type: numpy array
        num:     fish channel number, used for saving images as .tif; type: int
        --- return:
        fishMaskLbl:   segmented and labelled image;                                  type: numpy array
        fishMaskProps: properties of the objects in the segmented and labelled image; type: struct
        segmented:     whether segmentation worked or not;                            type: bool
    """
    # segment each fish channel
    print('Creating FISH mask '+str(num)+' ... ', end="")
    nSlices = fishimg.shape[0]
    # clipping
    vmin, vmax = np.percentile(fishimg, q=(1, 99.5))
    fishImgSmoothClipped = exposure.rescale_intensity(fishimg, in_range=(vmin, vmax), out_range='image')
    # fish channel smoothing
    fishImgSmooth = gaussian(fishImgSmoothClipped, sigma = gausSigma, preserve_range = True).astype('uint8') #median(fishimg, filt3D)
    fishImgSmoothEnh = rank.entropy(fishImgSmooth.astype('uint8'), filt3D, mask=segmentedDapi)
    print('Thresholding ... ', end="")
    # apply calculated DAPI mask
    fishImgSmoothEnhMsk = np.multiply(fishImgSmoothEnh, segmentedDapi)
    # thresholding the FISH imgs inside the DAPI mask
    print(f'Threshold value Entropy: {threshold_otsu(fishImgSmoothEnhMsk[fishImgSmoothEnhMsk != 0])}', end=" ")
    fishMaskConf = fishImgSmoothEnhMsk >= threshold_otsu(fishImgSmoothEnhMsk[fishImgSmoothEnhMsk != 0])#rank.otsu(fishImgSmooth, filt3D, mask=segmentedDapi)#.astype('uint8') #(threshold_otsu(fishImgSmoothMsk[fishImgSmoothMsk != 0])+1) # exclude px outside of DAPI mask (intensity = 0) from threshold calculation
    for slice in range(nSlices):
        img = fishMaskConf[slice,:,:]
        img = dilation(img, footprint=disk(1)) # erode to help remove small objects
        img = remove_small_holes(img, smallHoleSizeFish+300)
        fishMaskConf[slice,:,:] = img
    # thresholding FISH img inside entropy mask
    fishImgSmoothMsk = np.multiply(fishImgSmooth, fishMaskConf)
    fishMask = fishImgSmoothMsk >= threshold_otsu(fishImgSmoothMsk[fishImgSmoothMsk != 0])
    # print threshold values as feedback for user
    print(f'Threshold value FISH otsu: {threshold_otsu(fishImgSmoothMsk[fishImgSmoothMsk != 0])}', end=" ")
    print(f'Threshold value FISH li: {threshold_li(fishImgSmoothMsk[fishImgSmoothMsk != 0])}')
    # clean-up the mask
    for slice in range(nSlices):
        img = fishMask[slice,:,:]
        mask = dilation(segmentedDapi[slice,:,:], footprint=disk(2)) # confine mask again to get objects on edge of the border to overlap with it
        img = erosion(img, footprint=disk(1)) # erode to help remove small objects
        img = remove_small_objects(img, smallObjSizeFish)
        img = remove_small_holes(img, smallHoleSizeFish)
        img = clear_border(img, mask=mask) # use dapi mask instead of image border to remove objects overlapping with nuclear border
        fishMask[slice,:,:] = img
    ### Check for number of objects
    # get 3D objects labelled for FISH
    fishMaskLbl = measure.label(fishMask)
    fishMaskProps = measure.regionprops(fishMaskLbl)
    if len(fishMaskProps) > 2:
        print('More than two objects in FISH mask found, check segmentation ...')
        segmented = True # go on with the next batch
    elif len(fishMaskProps) == 0:
        print('No objects in FISH mask found, check segmentation ...')
        segmented = False # go on with the next batch
    else:
        segmented = True
        print('Number of found objects:', len(fishMaskProps))
        print('FISH mask '+str(num)+' calculated.')

    # save 3D masks for inspection in FIJI
    imsave(saveDir+'3DstackFISH'+str(num)+'mask_'+str(dirs[batch])+'.tif', fishMask.astype(np.uint8), check_contrast=False) # saving stack for visualisation in FIJI
    return fishMaskLbl, fishMaskProps, segmented, fishImgSmooth, fishImgSmoothEnh, fishImgSmoothMsk
### ----- FISH channel segmentation as a function ----- ###

# get parent folders of all images
folders = Path(pathlistDapi[0]).parent.parent
dirs = sorted((f for f in os.listdir(folders) if not f.startswith(".") and not f.endswith(".csv") and not f.endswith(".tif") and not f.endswith(".pdf"))) # save names of folders for labeling of calculations, ignore hidden directories # use ", key=str.lower" to and perform case-insensitive sorting

# read metadata --> assuming its the same for all stacks, read only first image
with TiffFile(pathlistDapi[0]) as imgMeta:
    assert imgMeta.is_imagej, "Wrong Metadata Format ... expected ImageJ" # check if metadata is written in imagej style (returns 'True')
    # check if metadata is available, if not, fall back to user input
    if ('unit' in imagej_description_metadata(imgMeta.pages[0].tags['ImageDescription'].value)) == True:
        # get the resolution unit
        resUnit = imagej_description_metadata(imgMeta.pages[0].tags['ImageDescription'].value)['unit']
        assert resUnit == 'micron' or resUnit == '\\u00B5m', "Wrong or No Resolution Unit ... expected microns" # should be micron
        # get the resolution in zyx
        pZ = np.around(imgMeta.imagej_metadata['spacing'], decimals=3) # = physical Z res --> spacing between z-slices in resUnit
        pY, pX = imgMeta.pages[0].resolution # = physical X & Y res in px per resUnit
        pY, pX = int(pY), int(pX) # use integer instead of float to remove rounding errors
        # convert z res to px per resUnit
        pZ = np.around(1/pZ, decimals=3)
    else:
        pZ = np.around(1/zSpacing, decimals=3) # convert to px per micron
        pY, pX = np.around(1/xyPXsize, decimals=3), np.around(1/xyPXsize, decimals=3) # convert to px per micron
# convert shift in nm to px in fish imgs
meanShiftBO = np.rint(np.multiply(meanShiftBO, (pZ, pY, pX)))
meanShiftBF = np.rint(np.multiply(meanShiftBF, (pZ, pY, pX)))
print(f'\nShift to be applied, b-o (z, y, x): {meanShiftBO} px')
print(f'Shift to be applied, b-f (z, y, x): {meanShiftBF} px')

print("\nYou want to process "+str(len(pathlistDapi))+" batch(es), let's go!")
### loop over batches (individual cells)
for batch in range(len(pathlistDapi)):
    print('\nProcessing batch '+str(batch+1)+'/'+str(len(pathlistDapi))+' ...')
    # get all image stacks
    print('Loading images ...')
    dapiImg = imread(pathlistDapi[batch])
    fish1Img = imread(pathlistFish1[batch])
    fish2Img = imread(pathlistFish2[batch])
    ### correcting shift
    print('Correcting chromatic shift ...')
    # apply shift to images
    fish1Img = ndi.shift(fish1Img, meanShiftBO, cval=int(np.mean(fish1Img))) # missing values are filled with "background" = rounded mean signal intensity in the whole stack
    fish2Img = ndi.shift(fish2Img, meanShiftBF, cval=int(np.mean(fish2Img)))
    imsave(saveDir+'fish1Img_SC_'+str(dirs[batch])+'.tif', fish1Img.astype(np.uint8), check_contrast=False) # saving stack for visualisation in FIJI
    imsave(saveDir+'fish2Img_SC_'+str(dirs[batch])+'.tif', fish2Img.astype(np.uint8), check_contrast=False) # saving stack for visualisation in FIJI

### ----- segment nucleus (DAPI) ----- ###
    print('Segmenting Nucleus ...', end=' ')
    # reduce noise and auto threshold
    dapiImgFilt = gaussian(dapiImg, sigma = gausSigma, preserve_range = True)
    #imsave(saveDir+'3DstackFilt_'+str(dirs[batch])+'.tif', dapiImgFilt.astype(np.uint8), check_contrast=False) # saving stack for visualisation in FIJI
    dapiImgMask = dapiImgFilt > threshold_otsu(dapiImgFilt)
    #imsave(saveDir+'3DstackThr_'+str(dirs[batch])+'.tif', dapiImgMask.astype(np.uint8), check_contrast=False) # saving stack for visualisation in FIJI
    # deal with small artifacts in each slice
    nSlices = dapiImg.shape[0]
    for slice in range(nSlices):
        img = dapiImgMask[slice,:,:]
        img = remove_small_objects(img, smallObjSize)
        img = remove_small_holes(img, smallHoleSize)
        dapiImgMask[slice,:,:] = img
    #imsave(saveDir+'3DstackThrClean_'+str(dirs[batch])+'.tif', dapiImgMask.astype(np.uint8), check_contrast=False) # saving stack for visualisation in FIJI
    ### watershedding
    print('Seeding ...', end=' ')
    # get centres of mass (EDT)
    distTransform = ndi.distance_transform_edt(dapiImgMask)
    distTransform = gaussian(distTransform, sigma=gausSigma)
    #imsave(saveDir+'3DstackDistTrans_'+str(dirs[batch])+'.tif', distTransform.astype(np.uint8), check_contrast=False) # saving stack for visualisation in FIJI
    # define markers -- erode several times
    erodeFootFind = erodeFoot # footprint will be reset to starting value for the next cell
    imgCent = (int(dapiImgMask.shape[1]/2),int(dapiImgMask.shape[2]/2)) # store 2D image center point --> object closest to it will be used as marker for watershedding
    # conditions which will become true if seed is split from other seeds or not found
    split = False
    noSeed = False
    while split == False:
        localMaxs = ndi.minimum_filter(dapiImgMask, size=erodeFootFind)
        ### Check whether nucleus is separated from other nuclei
        # get 2D max intensity projected image and get the objects in it
        lmMax = np.max(localMaxs, axis=0)
        lmMaxLbl = measure.label(lmMax) #[int(nSlices/2),:,:]
        lmProps = measure.regionprops(lmMaxLbl)
        # in case too much of the nucleus was eroded and no object was found, continue with next nucleus
        if len(lmProps) == 0:
            noSeed = True
            split = True # terminate loop
            continue
        # find object closest to image center if there is more than one in the image
        if len(lmProps) > 1:
            nucCentroids = [prop.centroid for prop in lmProps] # get all centroids
            centClosestIdx = distance.cdist([imgCent], nucCentroids).argmin() # get index of object closest to img center
        else:
            centClosestIdx = 0 # in only one object in image, use it
        # Visualisation
        #plt.figure(figsize=(10,10))
        #plt.imshow(label2rgb(lmMaxLbl,image=convex_hull_image(np.isin(lmMaxLbl,centClosestIdx+1).astype(int)), bg_label=0))
        #plt.show()
        # calculate coefficient for evaluation of object shape
        coeffc = lmProps[centClosestIdx].solidity #lmProps[volTop].axis_minor_length/lmProps[volTop].axis_major_length # minor and major axis length of an ellipse fit to the region
        print('Solidity:', "{:.3f}".format(lmProps[centClosestIdx].solidity), end=' ') # area/area_convex
        print('Area:', lmProps[centClosestIdx].area, end=' ')
        print('...', end=' ')
        # If two nuclei are connected, the major axis will be much longer than the minor axis.
        # For perfectly shaped (round) nuclei, the value will approach 1.
        # Hence, we define that nuclei with values below 3/4 are insuffieciently round and adjust the footprint for the minimum filter:
        if coeffc < cutoff:
            erodeFootFind += 5 # increment the footprint by 5 until the shape is right
            print('FP:', erodeFootFind, end=' ')
        else:
            split = True # terminate loop
    # continue with next nucleus if no seed could be determined for watershedding
    if noSeed == True:
        print('\nNo seed found, try adjusting the coefficient ...')
        continue # go on with the next batch
    markers = measure.label(localMaxs)
    #imsave(saveDir+'3DstackDistTransSeed_'+str(dirs[batch])+'.tif', markers.astype(np.uint8), check_contrast=False) # saving stack for visualisation in FIJI
    # compute the watershed segmentation
    print('Watershedding ...', end=' ')
    dapiImgMaskShed = watershed(-distTransform, markers, mask=dapiImgMask) # inverse of distTransform used to make peaks into valleys
    #imsave(saveDir+'3DstackWatershed_'+str(dirs[batch])+'.tif', dapiImgMaskShed.astype(np.uint8), check_contrast=False) # saving stack for visualisation in FIJI
    # remove objects touching borders (2D)
    for slice in range(nSlices):
        img = dapiImgMaskShed[slice,:,:]
        img = clear_border(img)
        dapiImgMaskShed[slice,:,:] = img
    #imsave(saveDir+'3DstackWatershedClear_'+str(batch+1)+'.tif', dapiImgMaskShed.astype(np.uint8), check_contrast=False) # saving stack for visualisation in FIJI
    # confine DAPI mask by two erosions
    dapiImgMaskShed = erosion(dapiImgMaskShed, footprint=ball(3)) # noticed that otsu is rather too soft on most images --> DAPI border is a bit missrepresented due to scattering
    # get number of objects left, if more than one, keep the biggest
    dapiImgMaskShedLbl = measure.label(dapiImgMaskShed) # relabel
    dapiProps = measure.regionprops(dapiImgMaskShedLbl)
    if len(dapiProps) > 1:
        print('Several objects segmented, keeping biggest object ...')
        volTop = sorted( [ (x,i) for (i,x) in enumerate([prop.area for prop in dapiProps]) ], reverse=True )[0][1] # returns biggest volume index
        segmentedDapi = np.isin(dapiImgMaskShedLbl,volTop+1).astype(int) # keep object with matching label
        segmentedDapi = remove_small_holes(segmentedDapi.astype(bool), smallHoleSize)
    elif len(dapiProps) == 1:
        print('Object found ...')
        segmentedDapi = dapiImgMaskShedLbl.astype(bool) # this should now only contain the nucleus of interest as a binary mask
    elif len(dapiProps) == 0:
        print('No nucleus found, check segmentation ...')
        continue # go on with the next batch
    # catching exception due to false labels
    if segmentedDapi.max() == False:
        print('No nucleus found, check labelling ...')
        continue # go on with the next batch
    ## check the final dapi mask solidity --> throw the nucleus if it is below a certain value --> 'solidDapi' 
    segmentedDapiMax = np.max(segmentedDapi, axis=0) # create 2D max projection of nucleus
    # calculate solidity
    segmentedDapiMaxLbl = measure.label(segmentedDapiMax)
    dapiMaxProps = measure.regionprops(segmentedDapiMaxLbl)
    coeffcD = dapiMaxProps[0].solidity # only one object in image, hence idx = [0]
    # if below cutoff, continue with next image stack
    if coeffcD < solidDapi:
        print('\nFound nucleus has bad shape, solidity = '+"{:.3f}".format(coeffcD)+' check segmentation ...')
        print('Close the displayed figure to continue with the next image stack ...')
        # plot the shape and convex hull of dapi mask for visualisation of what is wrong
        plt.figure(figsize=(7,7))
        plt.imshow(label2rgb(segmentedDapiMaxLbl,image=convex_hull_image(segmentedDapiMaxLbl), bg_label=0))
        plt.show()
        imsave(saveDir+'BAD_3DstackNuc_'+str(dirs[batch])+'.tif', segmentedDapi.astype(np.uint8), check_contrast=False) # saving stack for visualisation in FIJI
        continue # go on with the next batch
    # save 3D mask for inspection in FIJI
    imsave(saveDir+'3DstackNuc_'+str(dirs[batch])+'.tif', segmentedDapi.astype(np.uint8), check_contrast=False) # saving stack for visualisation in FIJI
### ----- DAPI segmentation done ----- ### 

### ----- process fish1 and fish2 image stack ----- ###
    print('Processing FISH images ...')
    ### segment the signal
    fish1ImgLbl, fish1ImgProps, segmentedF1, fishImgSmooth1, fishImgSmoothEnh1, fishImgSmoothMsk1 = fishSegment(fish1Img, 1)
    fish2ImgLbl, fish2ImgProps, segmentedF2, fishImgSmooth2, fishImgSmoothEnh2, fishImgSmoothMsk2 = fishSegment(fish2Img, 2)
    #imsave(saveDir+'3DentropyMaskImg1_'+str(dirs[batch])+'.tif', fishImgSmoothMsk1.astype(np.uint8), check_contrast=False) # saving stack for visualisation in FIJI
    #imsave(saveDir+'3DentropyMaskImg2_'+str(dirs[batch])+'.tif', fishImgSmoothMsk2.astype(np.uint8), check_contrast=False) # saving stack for visualisation in FIJI

    # check whether segmentation worked --> go on to next batch
    if segmentedF1 == False or segmentedF2 == False:
        print('Continuing with the next batch ...')
        continue    
    ### Try to filter (in 3D) for the right FISH volumes if more than 2 were found
    # for 2nd FISH signal == Intron
    if len(fish2ImgProps) > 2:
        # get two objects with biggest volume and remove rest
        print('Trying to filter 2nd fish channel, keeping two biggest objects ...', end=" ")
        fish2VolTop1 = sorted( [ (x,i) for (i,x) in enumerate([prop.area for prop in fish2ImgProps]) ], reverse=True )[0][1] # returns biggest volume index
        fish2VolTop2 = sorted( [ (x,i) for (i,x) in enumerate([prop.area for prop in fish2ImgProps]) ], reverse=True )[1][1] # returns 2nd-biggest volume index
        segmentedFish2 = np.isin(fish2ImgLbl,list({fish2VolTop1+1,fish2VolTop2+1})).astype(int) # keep object with matching label
        # re-do the props-calculation for simplicity (not really necessary, just easier to read the code imo)
        fish2ImgProps = measure.regionprops(segmentedFish2)
        print(' Done')
    # for 1st FISH signal == BAC  
    if len(fish1ImgProps) > 2:
        # get objects with biggest volume and remove rest
        print('Trying to filter 1st fish channel, keeping two biggest objects ...', end=" ")
        fish1VolTop1 = sorted( [ (x,i) for (i,x) in enumerate([prop.area for prop in fish1ImgProps]) ], reverse=True )[0][1] # returns biggest volume index
        fish1VolTop2 = sorted( [ (x,i) for (i,x) in enumerate([prop.area for prop in fish1ImgProps]) ], reverse=True )[1][1] # returns 2nd-biggest volume index
        segmentedFish1 = np.isin(fish1ImgLbl,list({fish1VolTop1+1,fish1VolTop2+1})).astype(int) # keep object with matching label
        # re-do the props-calculation for simplicity (not really necessary, just easier to read the code imo)
        fish1ImgProps = measure.regionprops(segmentedFish1)
        print(' Done')
    print('Calculating volumes and overlap ...', end="")
    ### calculate volume comparison
    fish1ImgVol = 0
    fish2ImgVol = 0
    # combine all volumes in fish1ImgProps (=BAC)
    for props in fish1ImgProps:
        vol = props.area
        fish1ImgVol += vol
    # combine all volumes in fish2ImgProps (=intron)
    for props in fish2ImgProps:
        vol = props.area
        fish2ImgVol += vol
    # divide img2_volume / img1_volume (== intron_vol / BAC_vol) to compare volume sizes
    volCompare = fish2ImgVol/fish1ImgVol
    ### calculate the overlap
    # overlay objects
    # make sure all obj in both images have same label (=1)
    fish1lbl = fish1ImgLbl >= 1
    fish2lbl = fish2ImgLbl >= 1
    # add labeled imgs --> overlapping parts will have label = 2
    overlapImg = np.add(fish1lbl.astype('uint8'), fish2lbl.astype('uint8')) # uint8 needed to interpret label correctly (instead of bool)
    # catch mask error if volumes don't overlap --> we are sure they have to
    if np.max(overlapImg) < 2:
        print(" Masks didn't overlap, check segmentation ...")
        continue
    # get volume of all objects together = total volume
    overlapProps = measure.regionprops(overlapImg)
    volTot = overlapProps[0].area + overlapProps[1].area # [0] --> obj with lbl = 1; [1] --> obj with lbl = 2, i. e. overlapping region
    # divide lbl2_volume / total_volume
    overlapRatio = overlapProps[1].area/volTot
    ### get FISH2 to Overlap ratio to check how much of the FISH2 volume is outside of the BAC (=1 means all FISH2 signal is inside BAC)
    f2o = overlapProps[1].area / fish2ImgVol
    print(' Done')
    print(f'Total volume: {volTot:.1f}; Overlap: {overlapRatio:.3f}; Intron in BAC: {(f2o*100):.2f}%')

    ### 3D view of segmented images
    #viewer = napari.Viewer() # initialise the viewer
    #viewer.add_image(fishImgSmooth, colormap='yellow', name='Clipped') #, scale=[1, 10, 1, 1]    
    #viewer.add_image(fishImgSmooth1, colormap='bop blue', name='Pre-filtered BAC') #, scale=[1, 10, 1, 1]
    #viewer.add_image(fishImgSmooth2, colormap='bop orange', name='Pre-filtered Intron', opacity=0.5) #, scale=[1, 10, 1, 1]
    #viewer.add_image(fishImgSmoothEnh1, colormap='bop blue', name='Entropy BAC')
    #viewer.add_image(fishImgSmoothEnh2, colormap='bop orange', name='Entropy Intron', opacity=0.5)
    # add original images
    #viewer.add_image(fish1Img, colormap='magenta', name='Tg BAC', contrast_limits=[5, 140]) #, scale=[1, 10, 1, 1]
    #viewer.add_image(fish2Img, colormap='cyan', name='Intron', opacity=0.5, contrast_limits=[5, 140]) #, scale=[1, 10, 1, 1]
    # add the masks with different colours
    #f1Layer = viewer.add_labels(fish1ImgLbl, name='BAC Lbl', color={1:'blue', 2:'yellow'}, blending='additive') #, color={1:'blue', 2:'yellow'} opacity=0.95,
    #f1ELayer = viewer.add_labels(fishMaskConf, name='Entropy Lbl', blending='additive') #, color={1:'blue', 2:'yellow'} opacity=0.95,
    #f2Layer = viewer.add_labels(fish2ImgLbl, name='Intron Lbl', color={1:'green', 2:'red'}, blending='additive')
    #NucLayer = viewer.add_labels(segmentedDapi.astype(int), name='Nucleus', opacity=0.3, blending='additive')
    # use contours to show segmented masks
    #f1Layer.contour = 2
    #f1ELayer.contour = 1
    #f2Layer.contour = 1
    #NucLayer.contour = 3
    #viewer.grid.enabled = True
    #napari.run() # display the viewer

    # skip batch if intron is less than 80% inside of BAC
    if f2o<0.8:
        print('Too much intron outside of BAC, continue with next batch ...')
        continue
    # add calculations to table as dict
    measuredData = {'Folder': [dirs[batch]], 'Cell_No': [batch+1], 'FISH_Labels': [fish1ChName+'-'+fish2ChName], 'FISH1_Vol': [fish1ImgVol], 'FISH2_Vol': [fish2ImgVol], 'Total_Vol': [volTot], 'Overlap_Vol': [overlapProps[1].area], 'Volume_Ratio': [volCompare], 'Overlap_Ratio': [overlapRatio], 'FISH2_in_FISH1': [f2o]}
    df_measurement = pd.DataFrame.from_dict(measuredData) # pd.concat can only process dataframes, no dicts --> convert dict to df
    df = pd.concat([df, df_measurement], ignore_index=True)
### ----- LOOP OVER BATCHES ---- ###

### plot all volume metrics as boxplots for visual comparison
mdf = pd.melt(df, id_vars=['Cell_No'], value_vars=['Volume_Ratio', 'Overlap_Ratio', 'FISH2_in_FISH1'], var_name='Metric') # melting df into right form
# plotting
sns.set_theme(style="ticks")
# Initialize the figure
f, ax = plt.subplots()
# Plot the metrics as boxplot
sns.boxplot(x='Metric', y="value", data=mdf, whis=[0, 100], width=.6, palette="vlag") #palette="Blues"
# Add in points to show each observation
sns.stripplot(x='Metric', y="value", data=mdf, palette="viridis", hue="Cell_No", size=4, alpha=.46, linewidth=0) #color=".3", dodge=True, zorder=1
# set axis limits and name
ax.set(ylim=(0, 1), yticks=np.arange(0, 1.1, 0.1).tolist(), ylabel="Volume Values")
plt.legend([],[], frameon=False) # remove the legend for a cleaner plot
plt.savefig(saveDir+saveNameMeasure+'_overlap_boxplot.pdf') # save the plot
plt.show() # show the plot to user

# print median values for each metric as feedback for user
sdf = {'BAC Volume':[df['FISH1_Vol'].median()],'Intron Volume':[df['FISH2_Vol'].median()],'Total Volume':[df['Total_Vol'].median()],'Overlap Volume':[df['Overlap_Vol'].median()],'Volume Ratio':[df['Volume_Ratio'].median()],'Overlap Ratio':[df['Overlap_Ratio'].median()],'Intron in BAC':[df['FISH2_in_FISH1'].median()]}
print('\nMedian values:', end=" ")
print('( n =', len(df[(df['FISH1_Vol']!='')]),')') # prints number of measurement points
print(sdf)

### save the measurement table
print('\nSaving measurements ...')
df.to_csv(saveDir+saveNameMeasure+'.csv')
print('All done!')
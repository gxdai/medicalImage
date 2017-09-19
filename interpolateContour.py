import dicom
import sys
import numpy as np
import os
import pickle
#import scipy.spatial.distance.pdist
# Try GIT
# Try GIT new branch
from numpy import linalg as LA

import matplotlib.pyplot as plt

def plotContour(curve1, curve2, curve3):
    curve1 = [float(val) for val in curve1]
    curve2 = [float(val) for val in curve2]
    curve3 = [float(val) for val in curve3]

    plt.plot(curve1[0::3], curve1[1::3], 'r*-')
    plt.plot(curve2[0::3], curve2[1::3], 'g^-')
    plt.plot(curve3[0::3], curve3[1::3], 'k.-')

    plt.show()

def getChangeIndex(diff):

    boolArray = (diff > 0).astype(int)

    diff2_array = np.absolute(boolArray[:-1] - boolArray[1:])

    tuple_aray = np.where(diff2_array == 1)

    return tuple_aray[0][0]

def interpolateContour(dcmfile, annotation):
    # Load the dcm slide
    ds = dicom.read_file(dcmfile)
    print(dcmfile)
    # Get the slice location (z axis value)
    sliceLocation = ds.SliceLocation
    # get the organ number
    class_num = annotation.ROIContourSequence.__len__()

    contourInfo = {}

    for label in range(class_num):
        CS = annotation.ROIContourSequence[label]       # contour sequence
        slidesNum = CS.ContourSequence.__len__()
        heightList = [float(CS.ContourSequence[j].ContourData[2]) for j in range(slidesNum)]
        # Get the height range of ground truth
        bottom = np.amin(heightList)
        top = np.amax(heightList)
        # If slide is not between bottom and top, just continue
        if not ((sliceLocation >= bottom) and (sliceLocation <= top)):
            continue

        signed_diff = sliceLocation - np.array(heightList)
        diff = np.absolute(signed_diff)                                 # check the height difference
        locationIndex = np.where(diff < 1e-6)                          # Get the slide location

        if (locationIndex[0].size == 0):
            print("NEED Interpolation")
            print(heightList)
            print(sliceLocation)
            change_index = getChangeIndex(signed_diff)
            print(change_index)
            print(CS.ContourSequence[change_index].ContourData[:3])
            print(CS.ContourSequence[change_index+1].ContourData[:3])
            print(sliceLocation)

            interpolatedLine = interpolateCoords(CS.ContourSequence[change_index].ContourData, \
                              CS.ContourSequence[change_index+1].ContourData, \
                              sliceLocation)
            contourInfo[label] = interpolatedLine
            print(CS.ContourSequence[change_index].ContourData)
            print(CS.ContourSequence[change_index+1].ContourData)
            print(interpolatedLine)
            plotContour(CS.ContourSequence[change_index].ContourData, CS.ContourSequence[change_index+1].ContourData, interpolatedLine)
            sys.exit()

        elif (locationIndex[0].size > 1):
            contourInfo[label] = [CS.ContourSequence[locationIndex[0][index]].ContourData for index in range(locationIndex[0].size)]


def string2float(coordList):
    # convert string to float
    coords = [float(val) for val in coordList]
    coords = np.reshape(coords, (-1, 3))

    return coords


def interTwoPoints(point1, point2, zLocation):
    """
    :param point1:  1*3 array
    :param point2:  1*3 array
    :param zLocation:   float value
    :return:    1*3 array
    """
    interpCoords = (abs(point1[2] - zLocation) * point2 + abs(point2[2] - zLocation) * point1) / abs(point1[2] - point2[2])

    interpCoords = list(interpCoords)       # convert it to list

    interpCoords = [str(val) for val in interpCoords]       # Convert back to string

    return  interpCoords


def getDistMatrix(locationArray):

    size = locationArray.shape[0]
    distM = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            distM[i, j] = LA.norm(locationArray[i] - locationArray[j])

    return distM


def interpolateCoords(coords1, coords2, zLocation):
    """
    :param coords1: a list of strings
    :param coords2: a list of strings
    :return: list of strings.
    """
    # convert string to float
    if len(coords1) < len(coords2):
            coords1, coords2 = coords2, coords1         # Put the longer list as the first one.
    coords1 = string2float(coords1)
    coords2 = string2float(coords2)

    coordAll = np.concatenate((coords1, coords2), axis=0)
    # distanceMatrix = pdist(coordAll, 'euclidean')
    distanceMatrix = np.zeros((coordAll.shape[0], coordAll.shape[0]))

    distanceMatrix = getDistMatrix(coordAll)

    print(distanceMatrix.shape)
    subMatrix = distanceMatrix[:coords1.shape[0], coords1.shape[0]:]
    print(subMatrix.shape)

    contourList = []
    for i in range(coords1.shape[0]):
        index = np.argmin(subMatrix[i,:])

        coord1 = coords1[i]
        coord2 = coords2[index]
        contourList = contourList + interTwoPoints(coord1, coord2, zLocation)

    return contourList


def batchInterPolation(inputRootDir, outputRootDir):
    sampleList = os.listdir(inputRootDir)
    counter = 0

    stoppedFile = 'C:\Users\gdai\Downloads\HN 01-30\\00007\CT1.2.840.113619.2.55.3.4137941444.252.1471027353.947.163.dcm'
    # stoppedFile = 'nonFile'
    for sample in sampleList:
        if '.DS_Store' == sample:
            continue
        samplePath = os.path.join(inputRootDir, sample)
        print(samplePath)

        # Create output directory
        gtPath = os.path.join(outputRootDir, sample)
        if not os.path.isdir(gtPath):
            os.makedirs(gtPath)

        files = os.listdir(samplePath)


        slides = [slide for slide in files if "struct_set" not in slide]
        contour = [slide for slide in files if "struct_set" in slide]
        print(contour)
        print(type(contour))
        if not len(contour):
            print("No struct_set file")
            break
        annotation = dicom.read_file(os.path.join(samplePath, contour[0]))
        for slide in slides:
            if counter % 100 == 0:
                print(counter)
                print("*****************************")
            print(stoppedFile)
            print(os.path.join(samplePath, slide))
            if stoppedFile != os.path.join(samplePath, slide):
                continue
            print("Interploate************")
            contourInfo = interpolateContour(os.path.join(samplePath, slide), annotation)
            outputfile = os.path.join(gtPath, slide.replace('.dcm', '.pkl'))
            with open(outputfile, 'wb') as f:
                pickle.dump(contourInfo, f, pickle.HIGHEST_PROTOCOL)
            counter += 1




        # if not np.sum(locationIndex[0].astype(int)).astype(bool):
        #     print("NEED interpolation")
        # if (locationIndex[0].size > 0):
        #     for index in range(locationIndex[0].size):
        #         class_label = locationIndex[0][index]
        #         contourInfo[label] = CS.ContourSequence[class_label].ContourData
        #
        # else:
        #     print("Not exist")

        # print(sliceLocation)
        # print(locationIndex[0])
        # #
        # print(np.array(heightList)[locationIndex[0]])


        # if not val:
        #     print("NEED interpolation")
        #     print(float(sliceLocation))
        #     print(valArray)
        # print(sliceLocation)
        # print(heightList)
        # # TO BE continued
        # if float(sliceLocation) >= minval and float(sliceLocation) <= maxval:
        #     if sliceLocation not in heightList:
        #         print("NEED to interpolate")
        #         sys.exit()
        #     # print("The GT include the current slide")
        #     # if sliceLocation in heightList:
        #     #     print("No need interpolate")
        #     # else:
        #     #     print("need to interploate")
        #
        #
        # # for j in range(slidesNum):
        # #     if sliceLocation == CS.ContourSequence[j].ContourData[2]:     # Get the height
        # #         print(sliceLocation)
        #         print('************')




if __name__ == "__main__":

    inputRootDir = 'C:\Users\gdai\Downloads\HN 01-30'
    outputRootDir = 'C:\Users\gdai\Downloads\myGT'
    # inputRootDir = '/Users/GXDAI/Documents/medicalImage/Emory_Yang/HN 1-56'
    # outputRootDir = '/Users/GXDAI/Documents/medicalImage/Emory_Yang/GT'
    batchInterPolation(inputRootDir, outputRootDir)

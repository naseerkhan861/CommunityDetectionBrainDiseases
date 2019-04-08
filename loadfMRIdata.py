import os
import  numpy as np
import matplotlib.pyplot as plt
import  pandas as pd


'function to joing root path and folder path where raw data is'
def joinPath(rootPath,fileName):
    completePath = rootPathOfData + "\\" + regions_200
    return completePath

'function to list all data files absolute path'
def getAllDataFilesPath(filePath):
    files=[]
    for file in os.listdir(filePath):
        files.append(filePath+"\\"+file)
    return files

'function to read a subject data that is in 1D format'
def readFileData(path):
    timeRegions=[]
    with open(path,'r') as data:
        for line in data:
            values=line.strip().split("\t")
            regions=[]
            for val in values:
                if val!="":
                    regions.append(val)
            timeRegions.append(regions)
    return timeRegions




'Path configuration for data'

rootPathOfData="D:\\ABIDE Dataset Complete (1035 patients)\\data\\functionals\cpac\\filt_global"
regions_200='rois_cc200'
completePath=joinPath(rootPathOfData,regions_200)

'listing all files in directory with absoulte path'

dataFilesPath=getAllDataFilesPath(completePath)

subjectData=readFileData(dataFilesPath[0])

for time_points in subjectData:
    print("Length is:",len(time_points))







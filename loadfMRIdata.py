import  numpy as np
import matplotlib.pyplot as plt
import os



def joinPath(rootPath,fileName):
    completePath = rootPathOfData + "\\" + regions_200
    return completePath



'Path configuration for data'

rootPathOfData="D:\\ABIDE Dataset Complete (1035 patients)\\data\\functionals\cpac\\filt_global"
regions_200='rois_cc200'
completePath=joinPath(rootPathOfData,regions_200)

'listing all files in directory with absoulte path'
for file in os.listdir(completePath):
    print(completePath+"\\"+file)






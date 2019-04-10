import os
import  numpy as np
import matplotlib.pyplot as plt
import  pandas as pd


'function to joing root path and folder path where raw data is'
def joinPath(rootPath,fileName):
    completePath = rootPath + "\\" + fileName
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

'Read phenotype file'

def readPhenotypeFile(path):
    with open(path,'r') as data:
        return  pd.read_csv(path)




'Phenotype file path'


phenoRootPath='D:\ABIDE Dataset Complete (1035 patients)\data\phenotypes'
phenoFileName='Phenotypic_V1_0b_preprocessed1.csv'
phenoAbsolutePath=joinPath(phenoRootPath,phenoFileName)
pdPhenoData=readPhenotypeFile(phenoAbsolutePath)

'Path configuration for data'

rootPathOfData="D:\\ABIDE Dataset Complete (1035 patients)\\data\\functionals\cpac\\filt_global"
regions_200='rois_cc200'
completePath=joinPath(rootPathOfData,regions_200)



'listing all files in directory with absoulte path'

site_path_length = {} #Sitewise dictionary for path length
subject_autism={} #Subject Autism wise dictionary
subID=pdPhenoData['SUB_ID']
subAUT=pdPhenoData['DX_GROUP']

'Creating Subject Autism Dictonary'
subject_autism_asso={}
for index in range(len(pdPhenoData)):
    subjectID=pdPhenoData['SUB_ID'][index]
    autismID=pdPhenoData['DX_GROUP'][index]
    subject_autism_asso[subjectID]=autismID


dataFilesPath=getAllDataFilesPath(completePath)


for path in dataFilesPath:
    subjectFileNameTokens=path.split("\\")
    lastTokenSplit=subjectFileNameTokens[len(subjectFileNameTokens)-1].split("_")
    tokenLength=len(lastTokenSplit)
    siteName=lastTokenSplit[0]
    if(tokenLength)==4:
        subjectIDFromFile=lastTokenSplit[1]
    else:
        subjectIDFromFile = lastTokenSplit[2]
    site_path_length[siteName]=tokenLength



'''
count=1
for eachSubject in dataFilesPath:
    print("Subject No:",count," FilePath: ",eachSubject)
    count+=1
    subjectData=readFileData(eachSubject)
    subjectData.pop(0)
    for time_points in subjectData:
        print("Length is:",time_points)
'''






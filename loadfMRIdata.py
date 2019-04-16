import os
import  numpy as np
import matplotlib.pyplot as plt
import  pandas as pd
from sklearn.cluster import AffinityPropagation
from sklearn.manifold import TSNE


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
    count=1
    with open(path,'r') as data:
        next(data)
        for line in data:
            values=line.strip().split("\t")
            regions=[]
            for val in values:
                if val!="":
                    regions.append(val)
            timeRegions.append(regions)
            count+=1
    return timeRegions

'Read phenotype file'

def readPhenotypeFile(path):
    with open(path,'r') as data:
        return  pd.read_csv(path)

'Read subjectID from datafile Path Name'

def getSubjectIDFromDataFilePath(filePath):
    subjectFileNameTokens = filePath.split("\\")
    lastTokenSplit = subjectFileNameTokens[len(subjectFileNameTokens) - 1].split("_")
    tokenLength = len(lastTokenSplit)
    siteName = lastTokenSplit[0]
    subjectIDFromFile=None
    if (tokenLength) == 4:
        subjectIDFromFile = lastTokenSplit[1]
    else:
        subjectIDFromFile = lastTokenSplit[2]
    #site_path_length[siteName] = tokenLength
    return subjectIDFromFile
    # print("Subject ID From DataFile is : ",subjectIDFromFile," Subject Label from Phenotype File is: ",subject_autism_asso[int(subjectIDFromFile)])

def get_autism_healty_distribution(dic):
    tempDic={'autism':0,'healthy':0}
    for val in dic.values():
        if val==1:
            tempDic['autism']+=1
        else:
            tempDic['healthy']+=1
    return  tempDic

def get_site_wise_stats(dic):
    site_wise_dic={}
    for key,val in dic.items():
        if val not in site_wise_dic.keys():
            site_wise_dic[val]=1
        else:
            site_wise_dic[val]+=1
    return site_wise_dic

def getAutismAndHealthyClusteringResults(filePath):
    autListOfClusters=[]
    controlListOfClusters=[]
    for path in filePath:
        fileIDOfSubject = getSubjectIDFromDataFilePath(path)
        subjectfMRIData = readFileData(path)
        timeRowsRegionCols = np.vstack(subjectfMRIData)
        timeRowsRegionCols = timeRowsRegionCols.astype(np.float)
        np.nan_to_num(timeRowsRegionCols,0)
        apClustering = AffinityPropagation().fit(timeRowsRegionCols.transpose())
        if subject_autism_asso[int(fileIDOfSubject)] == 1:
            autListOfClusters.append(apClustering)
        else:
            controlListOfClusters.append(apClustering)
    return autListOfClusters,controlListOfClusters
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

'Creating Subject Autism,Gender and Site Dictionaries'
subject_autism_asso={}
subject_gender_asso={}
subject_site_assoc={}
for index in range(len(pdPhenoData)):
    subjectID=pdPhenoData['SUB_ID'][index]
    autismID=pdPhenoData['DX_GROUP'][index]
    genderID=pdPhenoData['SEX'][index]
    siteInfo=pdPhenoData['SITE_ID'][index]
    subject_autism_asso[subjectID]=autismID   #subject autism dictionary
    subject_gender_asso[subjectID]=genderID   #subject gender dictionary
    subject_site_assoc[subjectID]=siteInfo    #subject site dictionary


dataFilesPath=getAllDataFilesPath(completePath) #Reading all data files in .1D format absolute path

'''
for eachFilePath in dataFilesPath:
    print("Subject ID From File is : ",getSubjectIDFromDataFilePath(eachFilePath))
'''

#print("Subject ID From DataFile is : ",subjectIDFromFile," Subject Label from Phenotype File is: ",subject_autism_asso[int(subjectIDFromFile)])



#Site wise subject statisitcs
site_wise_subjects_dic=get_site_wise_stats(subject_site_assoc)
#plt.plot(*zip(*sorted(site_wise_subjects_dic.items())))
#plt.show()
#plt.savefig('D://images//autism//sitePlot.png')


#subjectData=readFileData(dataFilesPath[2])
#timeRowsRegionCols=np.vstack(subjectData)
#timeRowsRegionCols=timeRowsRegionCols.astype(np.float)
#np.nan_to_num(timeRowsRegionCols,0)    #converting nan to numbers
#reducedDataTSNE=TSNE(n_components=2).fit_transform(timeRowsRegionCols.transpose())
#apClustering = AffinityPropagation().fit(reducedDataTSNE)
#plt.scatter(reducedDataTSNE[:,0],reducedDataTSNE[:,1],c=apClustering.labels_)
#plt.show()
#regionCorrelations=np.corrcoef(timeRowsRegionCols.transpose())
#np.nan_to_num(regionCorrelations,0)    #converting nan to numbers

#subjectData.pop(0)



'''
Applying Affinity propogation technique to 200 x 200 correlation matrix

autCount=1
contCount=1
totalCount=0


for path in dataFilesPath:
    totalCount+=1
    fileIDOfSubject=getSubjectIDFromDataFilePath(path)
    subjectfMRIData=readFileData(path)
    timeRowsRegionCols = np.vstack(subjectfMRIData)
    timeRowsRegionCols = timeRowsRegionCols.astype(np.float)
    #np.nan_to_num(timeRowsRegionCols,0)
    apClustering=AffinityPropagation().fit(timeRowsRegionCols.transpose())
    if subject_autism_asso[int(fileIDOfSubject)]==1:
        print("Autistic Subject ID: ",fileIDOfSubject," Shape of subjectData is:  ",len(apClustering.labels_)," AutCount is: ",autCount)
        autCount+=1
    else:
        print("Control Subject ID: ",fileIDOfSubject," Shape of subjectData is:  ",len(apClustering.labels_)," ContrCount is: ",contCount)
        contCount+=1
'''


autClusters,controlClusters=getAutismAndHealthyClusteringResults(dataFilesPath)


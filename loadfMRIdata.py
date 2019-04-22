import os
import  numpy as np
import matplotlib.pyplot as plt
import  pandas as pd
from sklearn.cluster import AffinityPropagation,KMeans,DBSCAN
from sklearn import  preprocessing
from sklearn.manifold import TSNE


'Site Dictionary'
Site_Info_dic={
            1:'PITT',
            2:'OLIN',
            3:'OHSU',
            4:'SDSU',
            5:'TRINITY',
            6:'UM',
            7:'USM',
            8:'YALE',
            9:'CMU',
            10:'LEUVEN',
            11:'KKI',
            12:'NYU',
            13:'STANFORD',
            14:'UCLA',
            15:'MAXMUN',
            16:'CALTECH',
            17:'SBL'
            }


'Get Site Wise training and testing Data SiteID parameters expects list of site ID'
def getSixteenSitesDataBasedOnOneSite(completePath,siteID):
    trainSiteSubjectDic={}
    testSiteSubjectDic={}
    for site in range(len(Site_Info_dic)):
        siteIndex=site+1
        if siteIndex not in siteID:
            subjectData, subjectLabels, subjectDataInOneSite=getSubjectDataUsingSite(siteIndex,completePath)
            trainSiteSubjectDic[siteIndex]=subjectDataInOneSite
        else:
            subjectData, subjectLabels, subjectDataInOneSite = getSubjectDataUsingSite(siteIndex, completePath)
            testSiteSubjectDic[siteIndex] = subjectDataInOneSite
    return trainSiteSubjectDic,testSiteSubjectDic



def getAutismAndControlDataUsingSite(siteID,completePath):

    filePaths=getAllDataFilesPath(completePath)
    autisticDataDic={}
    controlDataDic={}

    siteInfoDic=Site_Info_dic[siteID]
    for path in filePaths:
        tokens=path.split("\\")
        lastToken=tokens[len(tokens)-1]
        firstToken=lastToken.split("_")[0]
        firstToken=firstToken.upper()
        subjectID,siteInfo=getSubjectIDFromDataFilePath(path)
        siteInfo=siteInfo.upper()
        if siteInfoDic==siteInfo:
            regionTimeData=readFileData(path)
            timeRowsRegionCols = np.vstack(regionTimeData)
            timeRowsRegionCols = timeRowsRegionCols.astype(np.float)
            np.nan_to_num(timeRowsRegionCols, 0)
            condition=subject_autism_asso[int(subjectID)]
            if condition==1:
                autisticDataDic[int(subjectID)]=timeRowsRegionCols
            else:
                controlDataDic[int(subjectID)]=timeRowsRegionCols

    return autisticDataDic,controlDataDic





def getSubjectDataUsingSite(siteID,completePath):

    filePaths=getAllDataFilesPath(completePath)
    subjectLabels=[]
    subjectData=[]
    siteInfoDic=Site_Info_dic[siteID]
    for path in filePaths:
        tokens=path.split("\\")
        lastToken=tokens[len(tokens)-1]
        firstToken=lastToken.split("_")[0]
        firstToken=firstToken.upper()
        subjectID,siteInfo=getSubjectIDFromDataFilePath(path)
        siteInfo=siteInfo.upper()
        if siteInfoDic==siteInfo:
            regionTimeData=readFileData(path)
            timeRowsRegionCols = np.vstack(regionTimeData)
            timeRowsRegionCols = timeRowsRegionCols.astype(np.float)
            np.nan_to_num(timeRowsRegionCols, 0)
            subjectLabels.append(subjectID)
            subjectData.append(timeRowsRegionCols)
    subjectDataInOneSite={}
    for index  in range(len(subjectData)):
        subjectDataInOneSite[int(subjectLabels[index])]=subjectData[index]
    return subjectData,subjectLabels,subjectDataInOneSite


def getSubjectListUsingTimePointsFilteringUpper(lowerTimePoint,completePath):
    siteSubjectDataDic={}
    siteSubjectLabelsDic={}
    for siteID in range(len(Site_Info_dic)):

        subjectSiteData, subjectSiteLabels,subjectDataDicLabels = getSubjectDataUsingSite(siteID + 1, completePath)
        flag=True
        for key , val in subjectDataDicLabels.items():
            if val.shape[0]>=lowerTimePoint:
                if flag:
                    siteSubjectDataDic[siteID + 1] = []
                    siteSubjectLabelsDic[siteID + 1] = []
                    flag=False
                siteSubjectDataDic[siteID + 1].append(val)
                siteSubjectLabelsDic[siteID + 1].append(key)

    return siteSubjectDataDic,siteSubjectLabelsDic












def getSubjectListUsingTimePointsFilteringLower(UpperTimePoint,completePath):
    siteSubjectDataDic={}
    siteSubjectLabelsDic={}
    for siteID in range(len(Site_Info_dic)):

        subjectSiteData, subjectSiteLabels,subjectDataDicLabels = getSubjectDataUsingSite(siteID + 1, completePath)
        flag=True
        for key , val in subjectDataDicLabels.items():
            if val.shape[0]<=UpperTimePoint:
                if flag:
                    siteSubjectDataDic[siteID + 1] = []
                    siteSubjectLabelsDic[siteID + 1] = []
                    flag=False
                siteSubjectDataDic[siteID + 1].append(val)
                siteSubjectLabelsDic[siteID + 1].append(key)

    return siteSubjectDataDic,siteSubjectLabelsDic



def getSubjectListUsingTimePointsFilteringBetween(LowerTimePoint,UpperTimePoint,completePath):
    siteSubjectDataDic = {}
    siteSubjectLabelsDic = {}
    for siteID in range(len(Site_Info_dic)):

        subjectSiteData, subjectSiteLabels, subjectDataDicLabels = getSubjectDataUsingSite(siteID + 1, completePath)
        flag = True
        for key, val in subjectDataDicLabels.items():
            if val.shape[0] <= UpperTimePoint and val.shape[0] >= LowerTimePoint:
                if flag:
                    siteSubjectDataDic[siteID + 1] = []
                    siteSubjectLabelsDic[siteID + 1] = []
                    flag = False
                siteSubjectDataDic[siteID + 1].append(val)
                siteSubjectLabelsDic[siteID + 1].append(key)

    return siteSubjectDataDic, siteSubjectLabelsDic





def getSubjectTimePointsDic(completeFilePath):
    dataFilesPath=getAllDataFilesPath(completePath)
    subjectTimePointsDic = {}
    for path in dataFilesPath:
        subjectfMRIData = readFileData(path)
        subjectID, siteInfo = getSubjectIDFromDataFilePath(path)
        timeRowsRegionCols = np.vstack(subjectfMRIData)
        timeRowsRegionCols = timeRowsRegionCols.astype(np.float)
        subjectTimePointsDic[int(subjectID)] = timeRowsRegionCols.shape
    return subjectTimePointsDic




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

def getSubjectTimePoints(filePath):
    aut_min_max={'MIN':100000000,'MAX':-1}
    control_min_max = {'MIN': 100000000, 'MAX': -1}
    autTimePoints={}
    controlTimePoints={}
    for path in filePath:
        fileIDOfSubject,_ = getSubjectIDFromDataFilePath(path)
        subjectfMRIData = readFileData(path)
        timeRowsRegionCols = np.vstack(subjectfMRIData)
        timeRowsRegionCols = timeRowsRegionCols.astype(np.float)

        if subject_autism_asso[int(fileIDOfSubject)] == 1:
           autTimePoints[fileIDOfSubject]=timeRowsRegionCols.shape[0]
           if aut_min_max['MIN']>autTimePoints[fileIDOfSubject]:
               aut_min_max['MIN']=autTimePoints[fileIDOfSubject]
           if aut_min_max['MAX'] < autTimePoints[fileIDOfSubject]:
               aut_min_max['MAX'] = autTimePoints[fileIDOfSubject]
        else:
            controlTimePoints[fileIDOfSubject] = timeRowsRegionCols.shape[0]
            if control_min_max['MIN'] > controlTimePoints[fileIDOfSubject]:
                control_min_max['MIN'] = controlTimePoints[fileIDOfSubject]
            if control_min_max['MAX'] < controlTimePoints[fileIDOfSubject]:
                control_min_max['MAX'] = controlTimePoints[fileIDOfSubject]
    return autTimePoints,controlTimePoints,aut_min_max,control_min_max
'Read phenotype file'

def readPhenotypeFile(path):
    with open(path,'r') as data:
        return  pd.read_csv(path)


'Read Sitewise Time Points'

def getSiteWiseTimePoints(completePath):
    siteWiseTimePoints={}
    filePaths=getAllDataFilesPath(completePath)
    for path in filePaths:
        fileIDOfSubject,siteInfo = getSubjectIDFromDataFilePath(path)
        siteInfo=siteInfo.upper()
        subjectfMRIData = readFileData(path)
        timeRowsRegionCols = np.vstack(subjectfMRIData)
        timeRowsRegionCols = timeRowsRegionCols.astype(np.float)
        siteWiseTimePoints[siteInfo]=timeRowsRegionCols.shape[0]
    return siteWiseTimePoints

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
    return subjectIDFromFile,siteName
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
        fileIDOfSubject,_ = getSubjectIDFromDataFilePath(path)
        subjectfMRIData = readFileData(path)
        timeRowsRegionCols = np.vstack(subjectfMRIData)
        timeRowsRegionCols = timeRowsRegionCols.astype(np.float)
        #timeRowsRegionCols=preprocessing.normalize(timeRowsRegionCols)
        np.nan_to_num(timeRowsRegionCols,0)
        apClustering = AffinityPropagation().fit(timeRowsRegionCols.transpose())
        if subject_autism_asso[int(fileIDOfSubject)] == 1:
            autListOfClusters.append(apClustering)
        else:
            controlListOfClusters.append(apClustering)
    return autListOfClusters,controlListOfClusters


def getAutismAndHealthyKMeansClusteringResults(filePath,clusters):
    autListOfClusters=[]
    controlListOfClusters=[]
    for path in filePath:
        fileIDOfSubject,_ = getSubjectIDFromDataFilePath(path)
        subjectfMRIData = readFileData(path)
        timeRowsRegionCols = np.vstack(subjectfMRIData)
        timeRowsRegionCols = timeRowsRegionCols.astype(np.float)
        np.nan_to_num(timeRowsRegionCols,0)
        KMEANSClustering = KMeans(n_clusters=clusters).fit(timeRowsRegionCols.transpose())
        if subject_autism_asso[int(fileIDOfSubject)] == 1:
            autListOfClusters.append(KMEANSClustering)
        else:
            controlListOfClusters.append(KMEANSClustering)
    return autListOfClusters,controlListOfClusters



def getClusterSizeDistribution(clusterTechnique):
    subjectClusterDistributions=[]
    subjectClusterSize={}
    for eachCluster in clusterTechnique:
        regionsByClustersDic = {}
        uniqueClusters=np.unique(eachCluster.labels_)
        for eachVal in uniqueClusters:
            regionsByClustersDic[eachVal]=[]
            index=0
        for eachLabel in eachCluster.labels_:
            regionsByClustersDic[eachLabel].append(index)
            index+=1
        subjectClusterDistributions.append((regionsByClustersDic))
    return  subjectClusterDistributions


def getClusterSizeDistributionLengths(clusterDist):
    subjectClustersLength=[]

    for eachCluster in clusterDist:
        clusterLengthDic={}
        for eachKey,eachVal in eachCluster.items():
            clusterLengthDic[eachKey]=len(eachVal)
        subjectClustersLength.append(clusterLengthDic)
    return subjectClustersLength
    

def getSubjectClusterSizeMeanForLinePlot(clusterList):
    subjectClusterSizes=[]
    for eachCluster in clusterList:
        subjectClusters=[]
        for key,val in eachCluster.items():
            subjectClusters.append(val)
        subjectClusterSizes.append(np.mean(subjectClusters))
    return subjectClusterSizes





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

#subjectData,subjectLabels=getSubjectDataUsingSite(1,completePath,subject_autism_asso)


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









# FROM HERE


autClusters,controlClusters=getAutismAndHealthyClusteringResults(dataFilesPath)
autClustersDist=[]
contClustersDist=[]
for clusters in autClusters:
    autClustersDist.append(len(np.unique(clusters.labels_)))
for clusters in controlClusters:
    contClustersDist.append(len(np.unique(clusters.labels_)))

plt.plot(autClustersDist,label='Autism (505 Subjects)')
plt.plot(contClustersDist,label='Controls (530 Subjects')
plt.xlabel("Subjects")
plt.ylabel("Number of Clusters")
plt.title("Clustering Using Affinity Propagation")
plt.legend()
plt.show()



autClustersSize=getClusterSizeDistribution(autClusters)
contClustersSize=getClusterSizeDistribution(controlClusters)

autClusterLengthWiseDist=getClusterSizeDistributionLengths(autClustersSize)
contClusterLengthWiseDist=getClusterSizeDistributionLengths(contClustersSize)

autClusterMeanLength=getSubjectClusterSizeMeanForLinePlot(autClusterLengthWiseDist)
contClusterMeanLength=getSubjectClusterSizeMeanForLinePlot(contClusterLengthWiseDist)

plt.plot(autClusterMeanLength,label='Autism (505 Subjects)')
plt.plot(contClusterMeanLength,label='Controls (530 Subjects')
plt.xlabel("Subjects")
plt.ylabel("Average Size of Cluster")
plt.title("Clustering using Affinity Propogation")
plt.legend()
plt.show()




autTimePoints,controlTimePoints,aut_min_max,cont_min_max=getSubjectTimePoints(dataFilesPath)

plt.plot(list(autTimePoints.values()),label='Autism (505 Subjects)')
plt.plot(list(controlTimePoints.values()),label='Controls (530 Subjects')
plt.xlabel("Subjects")
plt.ylabel("Time Points")
plt.title("TimePoints Plot for Autism and Control")
plt.legend()
plt.show()


#To Here

# Analyzing Distributions of Clusters







'''

'''

#autTimePoints,controlTimePoints,aut_min_max,cont_min_max=getSubjectTimePoints(dataFilesPath)
#siteWiseTimePointsDic=getSiteWiseTimePoints(completePath)
#subjectTimePointsDic=getSiteWiseTimePoints(completePath)
#trainSiteDic,testSiteDic=getSixteenSitesDataBasedOnOneSite(completePath,[14,15,16,17])


#siteSubjectData,siteSubjectLabel=getSubjectListUsingTimePointsFilteringLower(1000,completePath)

def getAutismOrHealthySubjects(completePath):

    autisticSubjects={}
    controlSubjects={}
    filePaths=getAllDataFilesPath(completePath)
    for path in filePaths:
        subjectfMRIData = readFileData(path)
        subjectID, siteInfo = getSubjectIDFromDataFilePath(path)
        timeRowsRegionCols = np.vstack(subjectfMRIData)
        timeRowsRegionCols = timeRowsRegionCols.astype(np.float)
        autismAssoicationID = subject_autism_asso[int(subjectID)]
        if autismAssoicationID==1:
            autisticSubjects[int(subjectID)]=timeRowsRegionCols
        else:
            controlSubjects[int(subjectID)]=timeRowsRegionCols
    return autisticSubjects,controlSubjects



autSubjects,controlSubjects=getAutismOrHealthySubjects(completePath)
plt.plot(autSubjects[51456][:,2])
plt.plot(controlSubjects[51476][:,2])
plt.show()

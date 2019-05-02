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

'Get Clustering Matching Score'
def getClusterRegionMatchScore(clusterOne,clusterTwo):

    clusterOneValues=list(clusterOne.values())[0]
    clusterTwoValues=list(clusterTwo.values())[0]
    unionOfTwoList=list(set(clusterOneValues).union(clusterTwoValues))
    intersectionOfTwoList=list(set(clusterOneValues) & set(clusterTwoValues))
    if len(unionOfTwoList)==0:
        return 0
    else:
        return len(intersectionOfTwoList)/len(unionOfTwoList)



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


def getSubjectListUsingTimePointsFilteringSpecifics(TimePointsList,completePath):
    count=0
    siteSubjectDataDic = {}
    siteSubjectLabelsDic = {}
    for siteID in range(len(Site_Info_dic)):

        subjectSiteData, subjectSiteLabels, subjectDataDicLabels = getSubjectDataUsingSite(siteID + 1, completePath)
        flag = True
        for key, val in subjectDataDicLabels.items():
            if val.shape[0] not in TimePointsList:
                if flag:
                    siteSubjectDataDic[siteID + 1] = []
                    siteSubjectLabelsDic[siteID + 1] = []
                    flag = False
                siteSubjectDataDic[siteID + 1].append(val)
                siteSubjectLabelsDic[siteID + 1].append(key)
            else:
                count+=1
    print("Deleted Time Points: ",count)

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

    autListOfClustersDic = {}
    controlListOfClustersDic = {}

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
            autListOfClustersDic[int(fileIDOfSubject)]=apClustering
        else:
            controlListOfClusters.append(apClustering)
            controlListOfClustersDic[int(fileIDOfSubject)]=apClustering
    return autListOfClusters,controlListOfClusters,autListOfClustersDic,controlListOfClustersDic


def getAutismAndHealthyKMeansClusteringResults(filePath,clusters):
    autListOfClusters=[]
    controlListOfClusters=[]

    autListOfClustersDic = {}
    controlListOfClustersDic = {}

    for path in filePath:
        fileIDOfSubject,_ = getSubjectIDFromDataFilePath(path)
        subjectfMRIData = readFileData(path)
        timeRowsRegionCols = np.vstack(subjectfMRIData)
        timeRowsRegionCols = timeRowsRegionCols.astype(np.float)
        #timeRowsRegionCols=preprocessing.normalize(timeRowsRegionCols)
        np.nan_to_num(timeRowsRegionCols,0)
        KMEANSClustering = KMeans(n_clusters=clusters).fit(timeRowsRegionCols.transpose())
        if subject_autism_asso[int(fileIDOfSubject)] == 1:
            autListOfClusters.append(KMEANSClustering)
            autListOfClustersDic[int(fileIDOfSubject)]=KMEANSClustering
        else:
            controlListOfClusters.append(KMEANSClustering)
            controlListOfClustersDic[int(fileIDOfSubject)]=KMEANSClustering
    return autListOfClusters,controlListOfClusters,autListOfClustersDic,controlListOfClustersDic



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


def getDataFileSiteDic(completePath,SiteDic):

    dataFilesPath = getAllDataFilesPath(completePath)
    siteSubjectDataDic = {}
    dataSiteSubjectsCountDic={}
    count1=0
    count2=0
    for path in dataFilesPath:
        subjectID, siteInfo = getSubjectIDFromDataFilePath(path)
        siteInfo=siteInfo.upper()
        for siteId in range(len(SiteDic)):
            siteId=siteId+1
            if siteInfo==SiteDic[siteId]:
                if siteId not in list(siteSubjectDataDic.keys()):
                    siteSubjectDataDic[siteId]=[]
                    siteSubjectDataDic[siteId].append(int(subjectID))
                else:
                    siteSubjectDataDic[siteId].append(int(subjectID))
    for siteId in range(len(SiteDic)):
        siteName=SiteDic[siteId+1]
        subjectsCount=len(siteSubjectDataDic[siteId+1])
        dataSiteSubjectsCountDic[siteName]=subjectsCount

    return siteSubjectDataDic,dataSiteSubjectsCountDic


def getClusterRegionDistributionFromOneCluster(clusters):
    uniqueLabels=np.unique(clusters)
    labelRegionDic={}
    labelRegionLengthDic={}
    for eachLabel in uniqueLabels:
        labelRegionDic[eachLabel]=[]
    index=0
    for label in clusters:
        labelRegionDic[label].append(index)
        index+=1
    for key ,val in labelRegionDic.items():
        labelRegionLengthDic[key]=len(val)
    return labelRegionDic,labelRegionLengthDic



#Following Code outputs the status of Autism or Healthy on a Subject using a Dictionary mapping
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




def analyzeClustersEvolution(subjectDic,subjectComparedDic,threshold1,threshold2,pathFlag):
    fileName="D:\\Paper_Results\\clusterComparisons"

    if pathFlag==1:
        fileName+="\\Autism"
    elif pathFlag==2:
        fileName += "\\Controls"
    else:
        fileName += "\\Autism_Controls"

    for eachSubjectKey,eachSubjectVal in subjectDic.items():
        folderPath = fileName + "\\" + str(eachSubjectKey)
        comparedSubjects = subjectComparedDic[eachSubjectKey]
        os.mkdir(folderPath)
        flag2=True
        for secondSubject, comparedSubjectValue in comparedSubjects.items():
            secondSubjectPath = folderPath + "\\" + str(secondSubject)
            secondSubjectPath = secondSubjectPath + ".txt"
            with open(secondSubjectPath, 'w') as f:
                for _,eachVal in eachSubjectVal.items():
                    #f.write(str(eachVal))
                    #3f.write("\n")
                    #f.write("\n")
                    for _,clusters in comparedSubjectValue.items():
                        score=getClusterRegionMatchScoreList(eachVal,clusters)
                        if score>=threshold1 and score <= threshold2:
                            f.write(str(clusters)+"\n")






def getemptyfiles(rootdir):
    for root, dirs, files in os.walk(rootdir):
        for d in ['RECYCLER', 'RECYCLED']:
            if d in dirs:
                dirs.remove(d)

        for f in files:
            fullname = os.path.join(root, f)
            try:
                if os.path.getsize(fullname) == 0:
                    #print(fullname)
                    os.remove(fullname)
            except WindowsError:
                continue





def clusterMatchingInWithingSubjects(clustersDic,clusterlengthThreshold,scoreThreshold):
    subjectClusterDic={}
    subjectClusterCompareDic={}
    subjectClusterCompareDicNonEmpty={}

    for subjectOne in list(clustersDic.keys()):
        subjectOnelabelRegionDistDic, subjectOnelabelRegionLengthDic = getClusterRegionDistributionFromOneCluster(
            clustersDic[subjectOne].labels_)
        subjectClusterCompareDic[subjectOne]={}
        flag1 = True
        flag2=False
        for subjectTwo in list(clustersDic.keys()):

            if subjectOne!=subjectTwo:
                subjectTwolabelRegionDistDic, subjectTwolabelRegionLengthDic = getClusterRegionDistributionFromOneCluster(
                    clustersDic[subjectTwo].labels_)
                setOneDic,setTwoDic=compareTwoClustersWithin(subjectOnelabelRegionDistDic,subjectTwolabelRegionDistDic,clusterlengthThreshold,scoreThreshold)
                if setOneDic!={} and flag1:
                    subjectClusterDic[subjectOne] = {}
                    flag1=False
                    flag2=True
                if  flag2:
                    subjectClusterDic[subjectOne].update(setOneDic)
                    if setTwoDic!={}:
                        subjectClusterCompareDic[subjectOne][subjectTwo] = setTwoDic



    for key, val in subjectClusterCompareDic.items():
        if val!={}:
            subjectClusterCompareDicNonEmpty[key]={}
            for againKey, againVal in val.items():
                if againVal != {}:
                    subjectClusterCompareDicNonEmpty[key][againKey]=againVal
    return subjectClusterDic,subjectClusterCompareDicNonEmpty



def clusterMatchingInWithinSubjectsWithoutLengthThreshold(clustersDic,scoreThreshold1,scoreThreshold2):
    subjectClusterDic={}
    subjectClusterCompareDic={}
    subjectClusterCompareDicNonEmpty={}

    for subjectOne in list(clustersDic.keys()):
        subjectOnelabelRegionDistDic, subjectOnelabelRegionLengthDic = getClusterRegionDistributionFromOneCluster(
            clustersDic[subjectOne].labels_)
        subjectClusterCompareDic[subjectOne]={}
        flag1 = True
        flag2=False
        for subjectTwo in list(clustersDic.keys()):

            if subjectOne!=subjectTwo:
                subjectTwolabelRegionDistDic, subjectTwolabelRegionLengthDic = getClusterRegionDistributionFromOneCluster(
                    clustersDic[subjectTwo].labels_)
                setOneDic,setTwoDic=compareTwoClustersWithinWithoutLengthTwoThreshold(subjectOnelabelRegionDistDic,subjectTwolabelRegionDistDic,scoreThreshold1,scoreThreshold2)
                if setOneDic!={} and flag1:
                    subjectClusterDic[subjectOne] = {}
                    flag1=False
                    flag2=True
                if  flag2:
                    subjectClusterDic[subjectOne].update(setOneDic)
                    if setTwoDic!={}:
                        subjectClusterCompareDic[subjectOne][subjectTwo] = setTwoDic



    for key, val in subjectClusterCompareDic.items():
        if val!={}:
            subjectClusterCompareDicNonEmpty[key]={}
            for againKey, againVal in val.items():
                if againVal != {}:
                    subjectClusterCompareDicNonEmpty[key][againKey]=againVal
    return subjectClusterDic,subjectClusterCompareDicNonEmpty





def clusterMatchingInBetweenSubjects(autclustersDic,controlclustersDic,clusterlengthThreshold,scoreThreshold):
    subjectClusterDic={}
    subjectClusterCompareDic={}
    subjectClusterCompareDicNonEmpty={}

    for subjectOne in list(autclustersDic.keys()):
        subjectOnelabelRegionDistDic, subjectOnelabelRegionLengthDic = getClusterRegionDistributionFromOneCluster(
            autclustersDic[subjectOne].labels_)
        subjectClusterCompareDic[subjectOne]={}
        flag1 = True
        flag2=False
        for subjectTwo in list(controlclustersDic.keys()):
            subjectTwolabelRegionDistDic, subjectTwolabelRegionLengthDic = getClusterRegionDistributionFromOneCluster(
                controlclustersDic[subjectTwo].labels_)
            setOneDic,setTwoDic=compareTwoClustersBetween(subjectOnelabelRegionDistDic,subjectTwolabelRegionDistDic,clusterlengthThreshold,scoreThreshold)
            if setOneDic!={} and flag1:
                subjectClusterDic[subjectOne] = {}
                flag1=False
                flag2=True
            if  flag2:
                subjectClusterDic[subjectOne].update(setOneDic)
                if setTwoDic!={}:
                    subjectClusterCompareDic[subjectOne][subjectTwo] = setTwoDic



    for key, val in subjectClusterCompareDic.items():
        if val!={}:
            subjectClusterCompareDicNonEmpty[key]={}
            for againKey, againVal in val.items():
                if againVal != {}:
                    subjectClusterCompareDicNonEmpty[key][againKey]=againVal
    return subjectClusterDic,subjectClusterCompareDicNonEmpty


def clusterMatchingInBetweenSubjectsWithoutLength(autclustersDic,controlclustersDic,scoreThreshold1,scoreThreshold2):
    subjectClusterDic={}
    subjectClusterCompareDic={}
    subjectClusterCompareDicNonEmpty={}

    for subjectOne in list(autclustersDic.keys()):
        subjectOnelabelRegionDistDic, subjectOnelabelRegionLengthDic = getClusterRegionDistributionFromOneCluster(
            autclustersDic[subjectOne].labels_)
        subjectClusterCompareDic[subjectOne]={}
        flag1 = True
        flag2=False
        for subjectTwo in list(controlclustersDic.keys()):
            subjectTwolabelRegionDistDic, subjectTwolabelRegionLengthDic = getClusterRegionDistributionFromOneCluster(
                controlclustersDic[subjectTwo].labels_)
            setOneDic,setTwoDic=compareTwoClustersBetweenTwoThresholdWithoutLength(subjectOnelabelRegionDistDic,subjectTwolabelRegionDistDic,scoreThreshold1,scoreThreshold2)
            if setOneDic!={} and flag1:
                subjectClusterDic[subjectOne] = {}
                flag1=False
                flag2=True
            if  flag2:
                subjectClusterDic[subjectOne].update(setOneDic)
                if setTwoDic!={}:
                    subjectClusterCompareDic[subjectOne][subjectTwo] = setTwoDic



    for key, val in subjectClusterCompareDic.items():
        if val!={}:
            subjectClusterCompareDicNonEmpty[key]={}
            for againKey, againVal in val.items():
                if againVal != {}:
                    subjectClusterCompareDicNonEmpty[key][againKey]=againVal
    return subjectClusterDic,subjectClusterCompareDicNonEmpty




def getNonEmtpyDictionary(dic):
    nonEmtpyDic={}
    for key, val in dic.items():
        if val!={}:
            nonEmtpyDic[key]={}
            for againKey, againVal in val.items():
                if againVal != {}:
                    nonEmtpyDic[key][againKey]=againVal
    return nonEmtpyDic



def compareTwoClustersWithin(setOfClustersOneDic,setOfClustersTwoDic,clusterlengthThreshold,scoreThreshold):
    setOneClusterDic={}
    setTwoClusterDic={}
    for keyOne,valOne in setOfClustersOneDic.items():
        flag=True
        for keyTwo,valTwo in setOfClustersTwoDic.items():
            if len(valOne)>=clusterlengthThreshold and len(valTwo)>=clusterlengthThreshold:
                score=getClusterRegionMatchScore({keyOne:valOne},{keyTwo:valTwo})
                if score>=scoreThreshold:
                    if flag:
                        setOneClusterDic[keyOne]=valOne
                        flag=False
                    setTwoClusterDic[keyTwo]=valTwo
    return setOneClusterDic,setTwoClusterDic


def compareTwoClustersBetween(setOfClustersOneDic,setOfClustersTwoDic,clusterlengthThreshold,scoreThreshold):
    setOneClusterDic={}
    setTwoClusterDic={}
    for keyOne,valOne in setOfClustersOneDic.items():
        flag=True
        for keyTwo,valTwo in setOfClustersTwoDic.items():
            if len(valOne)>=clusterlengthThreshold and len(valTwo)>=clusterlengthThreshold:
                score=getClusterRegionMatchScore({keyOne:valOne},{keyTwo:valTwo})
                if score<=scoreThreshold:
                    if flag:
                        setOneClusterDic[keyOne]=valOne
                        flag=False
                    setTwoClusterDic[keyTwo]=valTwo
    return setOneClusterDic,setTwoClusterDic


def compareTwoClustersBetweenTwoThresholdWithoutLength(setOfClustersOneDic,setOfClustersTwoDic,scoreThreshold1,scoreThreshold2):
    setOneClusterDic={}
    setTwoClusterDic={}
    for keyOne,valOne in setOfClustersOneDic.items():
        flag=True
        for keyTwo,valTwo in setOfClustersTwoDic.items():
            if len(valOne)>=2 and len(valTwo)>=2:
                score=getClusterRegionMatchScore({keyOne:valOne},{keyTwo:valTwo})
                if score>=scoreThreshold1 and score<=scoreThreshold2:
                    if flag:
                        setOneClusterDic[keyOne]=valOne
                        flag=False
                    setTwoClusterDic[keyTwo]=valTwo
    return setOneClusterDic,setTwoClusterDic


def compareTwoClustersWithinWithoutLengthThreshold(setOfClustersOneDic,setOfClustersTwoDic,scoreThreshold):
    setOneClusterDic={}
    setTwoClusterDic={}
    for keyOne,valOne in setOfClustersOneDic.items():
        flag=True
        for keyTwo,valTwo in setOfClustersTwoDic.items():
            if len(valOne)>=2 and len(valTwo)>=2:
                score=getClusterRegionMatchScore({keyOne:valOne},{keyTwo:valTwo})
                if score>=scoreThreshold:
                    if flag:
                        setOneClusterDic[keyOne]=valOne
                        flag=False
                    setTwoClusterDic[keyTwo]=valTwo
    return setOneClusterDic,setTwoClusterDic


def compareTwoClustersWithinWithoutLengthTwoThreshold(setOfClustersOneDic,setOfClustersTwoDic,scoreThreshold1,scoreThreshold2):
    setOneClusterDic={}
    setTwoClusterDic={}
    for keyOne,valOne in setOfClustersOneDic.items():
        flag=True
        for keyTwo,valTwo in setOfClustersTwoDic.items():
            if len(valOne)>=2 and len(valTwo)>=2:
                score=getClusterRegionMatchScore({keyOne:valOne},{keyTwo:valTwo})
                if score>=scoreThreshold1 and score<=scoreThreshold2:
                    if flag:
                        setOneClusterDic[keyOne]=valOne
                        flag=False
                    setTwoClusterDic[keyTwo]=valTwo
    return setOneClusterDic,setTwoClusterDic


def getClusterRegionMatchScore(clusterOneDic,clusterTwoDic):

    unionOfDics=list(set(list(clusterOneDic.values())[0]).union(set(list(clusterTwoDic.values())[0])))
    intersectionOfDics=list(set(list(clusterOneDic.values())[0]) & set(list(clusterTwoDic.values())[0]))

    if len(intersectionOfDics)==0:
        return 0
    elif len(unionOfDics)==0:
        return 0
    else:
        return len(intersectionOfDics)/len(unionOfDics)



def getClusterRegionMatchScoreList(clusterOneArray,clusterTwoArray):

    unionOfDics=list(set(clusterOneArray) | set(clusterTwoArray))
    intersectionOfDics=list(set(clusterOneArray) & set(clusterTwoArray))

    if len(intersectionOfDics)==0:
        return 0
    elif len(unionOfDics)==0:
        return 0
    else:
        return len(intersectionOfDics)/len(unionOfDics)



def getTimePointsBasedDict(autTimes,controlTimes):
    timePointsDic={}
    for autkey,autVal in autTimes.items():
        if autVal in list(timePointsDic.keys()):
            timePointsDic[autVal]+=1
        else:
            timePointsDic[autVal]=1

    for contkey,contVal in controlTimes.items():
        if contVal  in list(timePointsDic.keys()):
            timePointsDic[contVal]+=1
        else:
            timePointsDic[contVal]=1

    return timePointsDic

def getSiteDictionaryCountFromSiteSubjectDict(siteSubjectDic,SiteInfo):

    siteSubjectTotalDic={}
    for key,val in siteSubjectDic.items():
        siteInfoName=SiteInfo[key]
        siteSubjectTotalDic[siteInfoName]=len(val)
    return siteSubjectTotalDic


# This function creates subjectID wise map of the data so that one ID is associated with one absolute path
def getSubjectPathMapping(completePath,subjectAutismAssoDic):

    subjectPathMapDic={}
    subjectAutismStatus={}
    filePaths=getAllDataFilesPath(completePath)
    for path in filePaths:
        subjectID, siteInfo = getSubjectIDFromDataFilePath(path)
        subjectID=int(subjectID)
        subjectPathMapDic[subjectID]=path
        subjectAutismStatus[subjectID] = subjectAutismAssoDic[subjectID]
    return subjectPathMapDic , subjectAutismStatus





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











#
#THE START




#autClusters,controlClusters=getAutismAndHealthyClusteringResults(dataFilesPath)
_,_,autClustersDic,controlClustersDic=getAutismAndHealthyClusteringResults(dataFilesPath)

# FROM HERE

autClustersDist=[]
contClustersDist=[]
for clusters in autClustersDic.values():
    autClustersDist.append(len(np.unique(clusters.labels_)))
for clusters in controlClustersDic.values():
    contClustersDist.append(len(np.unique(clusters.labels_)))

plt.plot(autClustersDist,label='Autism (505 Subjects)')
plt.plot(contClustersDist,label='Controls (530 Subjects')
plt.xlabel("Subjects")
plt.ylabel("Number of Clusters")
plt.title("Clustering Using Affinity Propagation")
plt.legend()
plt.show()



autClustersSize=getClusterSizeDistribution(autClustersDic.values())
contClustersSize=getClusterSizeDistribution(controlClustersDic.values())

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



# Analyzing Distributions of Clusters







'''

'''

#autTimePoints,controlTimePoints,aut_min_max,cont_min_max=getSubjectTimePoints(dataFilesPath)
#siteWiseTimePointsDic=getSiteWiseTimePoints(completePath)
#subjectTimePointsDic=getSiteWiseTimePoints(completePath)
#trainSiteDic,testSiteDic=getSixteenSitesDataBasedOnOneSite(completePath,[14,15,16,17])


#siteSubjectData,siteSubjectLabel=getSubjectListUsingTimePointsFilteringLower(1000,completePath)



autSubjects,controlSubjects=getAutismOrHealthySubjects(completePath)
plt.plot(autSubjects[51456][:,2])
plt.plot(controlSubjects[51476][:,2])
plt.show()


















'Comparing Clusters of one type that is comparing every Autistic Clusters or Healthy Clusters'
#AutClusters MAXSIZE=21
#ContClusters MAXSIZE=16



subjectClustersDicResults,subjectClustersDicCompareResults=clusterMatchingInWithinSubjectsWithoutLengthThreshold(autClustersDic,0.8,1.0)
subjectClustersDicCompareResults=getNonEmtpyDictionary(subjectClustersDicCompareResults)




controlClustersDicResults,controlClustersDicCompareResults=clusterMatchingInWithinSubjectsWithoutLengthThreshold(controlClustersDic,0.8,1.0)
controlClustersDicCompareResults=getNonEmtpyDictionary(controlClustersDicCompareResults)
singleTypeClustersDicResults,twoTypeClustersComparedResults=clusterMatchingInBetweenSubjectsWithoutLength(autClustersDic,controlClustersDic,0.0,0.2)
twoTypeClustersComparedResults=getNonEmtpyDictionary(twoTypeClustersComparedResults)

autismLengthDic={}
controlLengthDic={}
betweenSubjectControlDic={}

with  open("D:\Paper_Results\\autistic.txt",'w') as f:
    for index,key  in enumerate(subjectClustersDicResults):
        subject=key
        length=len(subjectClustersDicCompareResults[key])
        autismLengthDic[key]=length
        f.write(str(length)+"\n")

with  open("D:\Paper_Results\\controls.txt", 'w') as f:
    for index, key in enumerate(controlClustersDicResults):
        subject = key
        length = len(controlClustersDicCompareResults[key])
        controlLengthDic[key]=length
        f.write(str(length) + "\n")

with  open("D:\Paper_Results\\betweenSubjectControl.txt", 'w') as f:
    for index, key in enumerate(singleTypeClustersDicResults):
        subject = key
        length = len(twoTypeClustersComparedResults[key])
        betweenSubjectControlDic[key] = length
        f.write(str(length) + "\n")



'Comparing Subject vs Compared Subjects'

autSubject,autLength=list(autismLengthDic.keys()),list(autismLengthDic.values())
contSubject,contLength=list(controlLengthDic.keys()),list(controlLengthDic.values())
plt.plot(list(autismLengthDic.values()),label="Autism 505 Subjects")
plt.plot(list(controlLengthDic.values()),label="Healthy 529 Subjects")
plt.xlabel("Subjects")
plt.ylabel("Compared Subjects")
plt.title("Subject Vs (Number of Compared Subjects)")
plt.legend()
plt.show()












analyzeClustersEvolution(subjectClustersDicResults,subjectClustersDicCompareResults,0.8,1.0,1)
analyzeClustersEvolution(controlClustersDicResults,controlClustersDicCompareResults,0.8,1.0,2)
analyzeClustersEvolution(singleTypeClustersDicResults,twoTypeClustersComparedResults,0.8,1.0,3)
getemptyfiles("D:\Paper_Results\clusterComparisons\Autism_Controls")









timePointsDic=getTimePointsBasedDict(autTimePoints,controlTimePoints)




allTimePoints,allTimeFrequency=list(timePointsDic.keys()),list(timePointsDic.values())
plt.bar(allTimePoints,allTimeFrequency)
plt.xlabel("Time Points")
plt.ylabel("Number of Subjects")
plt.title("Time Points Distribution for Subjects")
plt.xticks(rotation=90)
plt.show()




#Site wise data of subjects, following function returns subjects ID in a site

siteSubjectDataSpecifics,siteSubjectLabelsSpecifics=getSubjectListUsingTimePointsFilteringSpecifics([202,124,177,232,234],completePath)

siteSubjectLabelsSpecificsTotal=getSiteDictionaryCountFromSiteSubjectDict(siteSubjectLabelsSpecifics,Site_Info_dic)


# A one time total Dictionary of sites and subjects from the data folder
siteSubjectIDDic,siteSubjectTotalDic=getDataFileSiteDic(completePath,Site_Info_dic)




#SubjectID and Absolute Path mapping with labels also
subjectPathMapDic,subjectAutismDic=getSubjectPathMapping(completePath,subject_autism_asso)




#Function that reads all the data given in the site dictionary and returns the data also in a site dictionary with labels
def getSubjectsDataBasedOnSiteWiseSubjectDictionary(siteSubjectDic,subjectPathMapping,subject_autism_asso):

    siteSubjectDataDic={}
    for key in list(siteSubjectDic.keys()):
        siteSubjectDataDic[key]={}
        for subjectID in siteSubjectDic[key]:
            subjectPath=subjectPathMapping[subjectID]
            subjectData=readFileData(subjectPath)
            timeRowsRegionCols = np.vstack(subjectData)
            timeRowsRegionCols = timeRowsRegionCols.astype(np.float)
            autismCondition=subject_autism_asso[subjectID]
            siteSubjectDataDic[key][subjectID]=[]
            siteSubjectDataDic[key][subjectID].append(timeRowsRegionCols)
            siteSubjectDataDic[key][subjectID].append(autismCondition)
    return siteSubjectDataDic


siteSubjectRawDataDic=getSubjectsDataBasedOnSiteWiseSubjectDictionary(siteSubjectLabelsSpecifics,subjectPathMapDic,subject_autism_asso)
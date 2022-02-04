import pandas as pd
from pathlib import Path
import glob
import numpy as np
from collections import Counter
from sklearn.linear_model import RANSACRegressor

"""
WORK IN PROGRESS, NOT RELIABLE

Package of methods used in the vessel registration project
"""

def readDataframe(path,separator=';',decimal=','):
    """
    Loads the processed scan as Pandas dataframe from the CSV file

    Parameters
        ----------
        path : Path object
            Absolute path to the CSV source file
        separator : str, optional
            Separator used in the CSV file (default is ';')
        decimal : str, optional
            Decimal symbol used in the CSV file (default is ',')
    
    """
    dataframe = pd.read_csv(path,sep=separator,decimal=decimal)
    return dataframe

def findRadiusOfNeighbours(sourceTable,filename = None,csvSeparator = ';'):
    """
    Extracts relevant data from skeleton analysis via Pandas module.

    Parameters
        ----------
        sourceTable : pandas.DataFrame()
            Original data gained from skeleton analysis
        filename : str, optional
            Path of the file for data export (default is None)
        csvSeparator : str, optional
            Separator for export to CSV files (default is ';')
    """
    neighbours = 0
    for nei in sourceTable.keys():
        if('connectedEdgesB' in nei):
            neighbours +=1
            
    df = pd.DataFrame()
    df['id'] = sourceTable['id']
    df['rad0'] = sourceTable['radius_mm']
    df['nodeIdA'] = sourceTable['nodeIdA']
    df['nodeIdB'] = sourceTable['nodeIdB']
                
    #iteration over each possible column name
    for letter in ['A','B']:
        for number in range(0,neighbours):
            radius = [] #new list for each column
            key = f'connectedEdges{letter} {number}' #regex for key genereation
            column = sourceTable[key]
            for i in range(len(column)):
                if(column[i] > 0):
                    rad = sourceTable.at[int(column[i]-1),'radius_mm']
                    radius.append(rad)
                else:
                    radius.append(np.nan)
            df[f'radius{letter} {number}'] = pd.Series(radius)
    
    df['nodeA_ZYX_mm 0'] = sourceTable['nodeA_ZYX_mm 0']
    df['nodeA_ZYX_mm 1'] = sourceTable['nodeA_ZYX_mm 1']
    df['nodeA_ZYX_mm 2'] = sourceTable['nodeA_ZYX_mm 2']
    df['nodeB_ZYX_mm 0'] = sourceTable['nodeB_ZYX_mm 0']
    df['nodeB_ZYX_mm 1'] = sourceTable['nodeB_ZYX_mm 1']
    df['nodeB_ZYX_mm 2'] = sourceTable['nodeB_ZYX_mm 2']
    
    df['phiAa'] = sourceTable['phiAa']
    df['phiAb'] = sourceTable['phiAb']
    df['phiAc'] = sourceTable['phiAc']
    df['phiBa'] = sourceTable['phiBa']
    df['phiBb'] = sourceTable['phiBb']
    df['phiBc'] = sourceTable['phiBc']
    
    if(filename is not None):
        df.to_csv('output_significant_nodes_' + str(filename) +'.csv',sep=csvSeparator)
    return df

def findRadiusNode(dataframe,filename = None,csvSeparator = ';'):
    """
    Rearranges the dataframe for further use.

    Parameters
        ----------
        dataframe : pandas.DataFrame()
            Dataframe containing extracted data
        filename : str, optional
            Path of the file for data export, (default is None)
        csvSeparator : str, optional
            Separator for export to CSV files (default is ';')
    """
    df = pd.DataFrame()
    listA = list(dataframe['nodeIdA'])
    listB = list(dataframe['nodeIdB'])
    
    nodeIdList = np.array(listA+listB)
    nodeIdList = nodeIdList[nodeIdList<0]
    
    dfA = dataframe.loc[dataframe['nodeIdA'] == listA]
    dfB = dataframe.loc[dataframe['nodeIdB'] == listB]
    
    dfA[['radius 1','radius 2','radius 3']] = dataframe[['radiusA 0','radiusA 1','radiusA 2']]
    dfA[['node_ZYX_mm 1','node_ZYX_mm 2','node_ZYX_mm 3']] = dataframe[['nodeA_ZYX_mm 0','nodeA_ZYX_mm 1','nodeA_ZYX_mm 2']]
    dfA[['phi 1','phi 2','phi 3']] = dataframe[['phiAa','phiAb','phiAc']]
    
    dfB[['radius 1','radius 2','radius 3']] = dataframe[['radiusB 0','radiusB 1','radiusB 2']]
    dfB[['node_ZYX_mm 1','node_ZYX_mm 2','node_ZYX_mm 3']] = dataframe[['nodeB_ZYX_mm 0','nodeB_ZYX_mm 1','nodeB_ZYX_mm 2']]
    dfB[['phi 1','phi 2','phi 3']] = dataframe[['phiBa','phiBb','phiBc']]
    
    subDf = pd.concat([dfA,dfB],ignore_index=True)
    df['nodeID'] = nodeIdList
    radKeys = ['rad0']
    radKeys += [f'radius {x}' for x in range(1,4)]
    radKeys += [f'node_ZYX_mm {x}' for x in range(1,4)]
    radKeys += [f'phi {x}' for x in range(1,4)]
    df[radKeys] = subDf[radKeys]
    
    df = df.sort_values(by=['nodeID'])
    
    if(filename is not None):
        df.to_csv('output_significant_nodes_' + str(filename) +'.csv',sep=csvSeparator)
    return df

def removeInsignificantNodes(dataframe,filename = None,csvSeparator = ';'):
    """
    Removes nodes that occur only once, as they are just the supposed ends of given tube.
    Parameters
        ----------
        dataframe : pandas.DataFrame()
            Dataframe containing arranged data
        filename : str, optional
            Path of the file for data export, (default is None)
        csvSeparator : str, optional
            Separator for export to CSV files (default is ';')
    """
    df = dataframe
    radKeys = [x for x in df.keys() if ('radius' in x)]
    for rowIndex in df.index:
        row = df.loc[rowIndex]
        radiusList = row[radKeys].tolist()
        if np.all(np.isnan(radiusList)):
            df = df.drop(rowIndex)
    df = df.drop_duplicates()
    
    if(filename is not None):
        df.to_csv('output_significant_nodes_' + str(filename) +'.csv',sep=csvSeparator)
    return df

def nLargestTubes(n,row):
    """
    Extracts n largest tubes in the given node.

    Parameters
        ----------
        n : int
            Number of desired tubes to extract from a node
        row : int
            Index of a row representing the given node in a dataframe
    Raises
        ------
        Exception
            If n exceeds the number of connected tubes.
    """
    if(n>len(row)):
        raise Exception('n should not exceed the number of connexted tubes')
        

    rowArray = row.copy()
    rowArray = rowArray.to_numpy()
    
    largest = []
    for i in range(n):
        if(np.all(pd.isna(rowArray))):
            return np.concatenate((largest,np.full((n-len(largest)),np.nan)))
        maxVal = np.nanmax(rowArray)
        maxIdx = np.nanargmax(rowArray)
        largest.append(maxVal)
        rowArray = np.delete(rowArray,maxIdx)
    return largest

def nLargestNodes(n,dataframe):
    """
    Extracts the largest nodes in the detailed scan according to the whole scan
    
    Parameters
        ----------
        n : int
            Number of desired nodes to extract from a dataframe
        dataframe : pandas.DataFrame()
            Dataframe containing arranged data
    """
    reducedDf = pd.DataFrame()
    reducedDf['nodeID'] = dataframe['nodeID']
    reducedDf['rad0'] = dataframe['rad0']

    columns = [x for x in dataframe.keys() if ('radius' in x)]
    columnsA= [x for x in columns if "A " in x]
    columnsB = np.setdiff1d(columns, columnsA)
    
    keysA = [f'radiusA {x}' for x in range(n)]
    keysB = [f'radiusB {x}' for x in range(n)]
    
    for i in range(len(dataframe)):
        largestA = nLargestTubes(n,dataframe.loc[i,columnsA])
        largestB = nLargestTubes(n,dataframe.loc[i,columnsB])

        reducedDf.loc[i,keysA] = largestA
        reducedDf.loc[i,keysB] = largestB

    #display(reducedDf)
    return reducedDf

def pairNodes(dataframe1,dataframe2):
    """
    Pairs nodes in both dataframes by their Euclidean distance
    
    Parameters
        ----------
        dataframe1 : pandas.DataFrame()
            Dataframe containing data from the lower resolution scan
        dataframe2 : pandas.DataFrame()
            Dataframe containing data from the higher resolution scan
    """
    df1 = dataframe1
    df2 = dataframe2
    for col1,col2 in zip(dataframe1,dataframe2):
        df1[col1] = df1[col1].fillna(5*df1[col1].max())
        df2[col2] = df2[col2].fillna(5*df2[col2].max())
    pairs = []
    radKeys = ['rad0'] 
    radKeys += [x for x in dataframe1.keys() if (('radius' in x) or ('phi' in x))]
    
    for i in range(len(df1)):
        distances = []
        for j in range(len(df2)):
            pt1 = df1[radKeys].iloc[i].values
            pt2 = df2[radKeys].iloc[j].values
            distances.append(np.linalg.norm(pt1-pt2))
        minDistances = np.argwhere(distances == np.nanmin(distances))
        
        for dist in minDistances:
            pairs.append((i,dist[0]))
    return pairs

    
def extractCoordinates(dataframe1,dataframe2,paired):
    """
    Extracts coordinates of paired nodes in both scans
    
    Parameters
        ----------
        dataframe1 : pandas.DataFrame()
            Dataframe containing data from the lower resolution scan
        dataframe2 : pandas.DataFrame()
            Dataframe containing data from the higher resolution scan
        paired : list of tuples
            List of each pair of the most similar nodes across both scans, represented by the id of the nodes
    """
    keys = ['node_ZYX_mm 1','node_ZYX_mm 2','node_ZYX_mm 3']
    coordsA = []
    coordsB = []
    for i in range(len(paired)):
        x = []
        y = []
        for key in keys:
            x.append(dataframe1.loc[paired[i][0],key])
            y.append(dataframe2.loc[paired[i][1],key])
        coordsA.append(x)
        coordsB.append(y)
    return coordsA,coordsB

def getTransformMatrix(coords1,coords2):
    """
    Returns transformation matrix using the RANSAC algorithm on paired nodes
    
    Parameters
        ----------
        coords1 : list of lists
            List containing coordinates of nodes in the lower resolution scan
        coords2 : list of lists
            List containing coordinates of nodes in the higher resolution scan
    """
    reg = RANSACRegressor(random_state=0).fit(coords1, coords2)
    transMat = np.column_stack([reg.estimator_.coef_, reg.estimator_.intercept_])
    transMat = np.row_stack([transMat, [0, 0, 0, 1]])
    inliers = reg.inlier_mask_
    return transMat,inliers

def register_vessels(df1,df2):
    """
    Returns both the transformation matrix and lists of point coordinates from both dataframes
    
    Parameters
        ----------
        df1 : pandas.DataFrame()
            Dataframe containing data from the lower resolution scan
        df2 : pandas.DataFrame()
            Dataframe containing data from the higher resolution scan
    """
    df1n = findRadiusOfNeighbours(df1)
    df2n = findRadiusOfNeighbours(df2)

    dfn1 = findRadiusNode(df1n)
    dfn2 = findRadiusNode(df2n)

    significantOnly1 = removeInsignificantNodes(dfn1)
    significantOnly2 = removeInsignificantNodes(dfn2)

    paired = pairNodes(significantOnly1,significantOnly2)

    coA,coB = extractCoordinates(dfn1,dfn1,paired)

    transMat,inliers = getTransformMatrix(coA,coB)

    validPairs = [paired[i] for i in range(len(paired)) if inliers[i]]
    coA,coB = extractCoordinates(dfn1,dfn1,validPairs)

    return coA,coB,transMat
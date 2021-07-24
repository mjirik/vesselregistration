import pandas as pd
from pathlib import Path
import glob
import numpy as np
from collections import Counter

"""
WORK IN PROGRESS, NOT RELIABLE

Package of methods used in the vessel registration project
"""

def findRadiusOfNeighbours(sourceTable,filename = None,csvSeparator = ';'):
    """
    Extracts relevant data from skeleton analysis via Pandas module.

    Parameters
        ----------
        sourceTable : pandas.DataFrame()
            Original data gained from skeleton analysis
        filename : str, optional
            Path of the file for data export, (default is None)
        csvSeparator : str, optional
            Separator for export to CSV files (default is ';')
    """
    index = 0
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
                    rad = sourceTable.at[column[i]-1,'radius_mm']
                    radius.append(rad)
                else:
                    radius.append(np.nan)
            df[f'radius{letter} {number}'] = pd.Series(radius)
            
    #display(df)
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
    listA = list(dataframe['nodeIdA'])#[dataframe['nodeIdA']<0])
    listB = list(dataframe['nodeIdB'])#[dataframe['nodeIdB']<0])
    
    nodeIdList = np.array(listA+listB)
    nodeIdList = nodeIdList[nodeIdList<0]
    
    dfA = dataframe.loc[dataframe['nodeIdA'] == listA]
    dfB = dataframe.loc[dataframe['nodeIdB'] == listB]
    subDf = pd.concat([dfA,dfB],ignore_index=True)
    
    df['nodeID'] = nodeIdList
    radKeys = [x for x in dataframe.keys() if ('radius' in x)]
    radKeys.insert(0,'rad0')
    df[radKeys] = subDf[radKeys]
    
    df = df.sort_values(by=['nodeID'])
    #display(df)
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
    nodes = dataframe['nodeID']
    counted = Counter(nodes)
    
    remove = [node for node in nodes if counted[node] < 2]
    removeIdx = [idx for idx in range(len(nodes)) if nodes[idx] in remove]
    df = dataframe.drop(removeIdx)
    df.index = range(len(df))
    #display(df)
    
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
        raise Exception('n should not exceed the number of connected tubes')
        #return

    x = np.array(row.copy())
    largest = []
    for i in range(n):
        if(np.all(np.isnan(x))):
            return np.concatenate((largest,np.full((n-len(largest)),np.nan)))
        maxVal = np.nanmax(x)
        maxIdx = np.nanargmax(x)
        largest.append(maxVal)
        x = np.delete(x,maxIdx)
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

def pairNodes(dataframe1,dataframe2,nanConstant):
    """
    Pairs nodes in both dataframes by their Euclidean distance
    
    Parameters
        ----------
        dataframe1 : pandas.DataFrame()
            Dataframe containing data from the lower resolution scan
        dataframe2 : pandas.DataFrame()
            Dataframe containing data from the higher resolution scan
        nanConstant : float
            Small number used for substitution of missing tubes
    """
    dataframe1[np.isnan(dataframe1)] = nanConstant
    dataframe2[np.isnan(dataframe2)] = nanConstant
    pairs = []
    radKeys1 = [x for x in dataframe1.keys() if ('radius' in x)]
    radKeys2 = [x for x in dataframe2.keys() if ('radius' in x)]

    for i in range(len(dataframe1)):
        distances = []
        for j in range(len(dataframe2)):
            pt1 = dataframe1[radKeys1].iloc[i].values
            pt2 = dataframe2[radKeys2].iloc[j].values
            distances.append(np.linalg.norm(pt1-pt2))
        #print(np.argmin(distances))
        
        pairs.append((i,np.nanargmin(distances)))
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
    keys = ['node_ZYX_mm 0','node_ZYX_mm 1','node_ZYX_mm 2']
    coordsA = []
    coordsB = []
    for i in range(len(paired)):
        x = []
        y = []
        for key in keys:
            x.append(dataframe1.loc[paired[i][0],key])
            y.append(dataframe2.loc[paired[i][0],key])
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
    temp_mat = reg.estimator_.coef_
    temp_column = reg.estimator_.intercept_
    transMat = np.column_stack([reg.estimator_.coef_, reg.estimator_.intercept_])
    transMat = np.row_stack([transMat, [0, 0, 0, 1]])
    
    return transMat
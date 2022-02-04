from vesreg3d import *
import pandas as pd
import glob
import numpy as np
import copy

def register_vessels(df1,df2):
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

def test(df1,df2, tolerance):
    coordinates1, coordinates2, transformMatrix = register_vessels(df1,df2)
    error = []
    for x,y in zip(copy.copy(coordinates1),copy.copy(coordinates2)):
        x = np.append(x,1)
        y = np.append(y,1)
        newY = transformMatrix@x
        error.append(np.abs(y-newY))
    assert (np.mean(error)<tolerance), print("Mean error should be in tolerance!")
    print("Test passed succesfully.")




if '__name__' == '__main__':
    pth = './artif/'
    pths = glob.glob(pth+"*.csv")
    path1 = pths[0]
    path2 = pths[1]
    path3 = pths[2]
    df1 = readDataframe(path1)
    df2 = readDataframe(path2)
    df3 = readDataframe(path3)

    test(df1,df1,5)
    test(df1,df2,5)
    test(df1,df3,5)
    
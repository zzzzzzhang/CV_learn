
# coding: utf-8

import numpy as np

def costFun(v_riverList, v_man, sList, T, angleList, w):
    '''
    v_riverList: list of different river's speed
    v_man: speed of man
    sList: list of different river's width
    T: time limited
    angleList: list of angles
    w: weight of punishment
    return: -h
    '''
    totalTimeNeeded = 0.
    totalH = 0.
#     calculate totalH and totalTimeNeeded
    for i, s in enumerate(sList):
        t = s/(v * np.cos(angleList[i]))
        totalTimeNeeded += t
        h = (v_riverList[i] + v * np.sin(angleList[i])) * t
        totalH += h
    
    score = totalH - w*(totalTimeNeeded - T)
    return -score


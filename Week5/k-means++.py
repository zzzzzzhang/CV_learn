# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def genSamples():
    '''
    generate 200 samples with 4 centers
    '''
    centroids_ori = [(50,50), (100,100), (50,100), (100,50)]
    xPoints = []
    yPoints = []
    for c in centroids_ori:
        x = c[0]
        y = c[1]
        for i in range(50):
            xPoints.append(x + np.round(np.random.normal(0,10)))
            yPoints.append(y + np.round(np.random.normal(0,10)))
            
    return np.abs(xPoints), np.abs(yPoints)

def assignment(df, centroids, colmap):
    for i in centroids.keys():
#         d = sqrt((x1 - x2)^2 - (y1 - y2)^2)
        df['distance_from_{0}'.format(i)] = ((df['x'] - centroids[i][0])**2 + (df['y'] - centroids[i][1])**2) **0.5
#     find the closest centroid & give it color
    df['closest'] = df.iloc[:, 2:6].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x[-1]))
    df['color'] = df['closest'].map(lambda x: colmap[x])
    
    return df

def updateCentroids_byFarther(df, centroids):
#     k-means++ way
    for i in centroids.keys():
        prob = df[df['closest'] == i]['distance_from_{0}'.format(i)] / df[df['closest'] == i]['distance_from_{0}'.format(i)].sum()
        prob = [prob[:j + 1].sum() for j in range(len(prob))]
        r = np.random.rand()
        for k in range(len(prob)):
            if prob[k] > r:
                centroids[i][0] = df[df['closest'] == i]['x'].values[k]
                centroids[i][1] = df[df['closest'] == i]['y'].values[k]
                break
    return centroids

def updateCentroids_byMean(df, centroids):
#     k-means way
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
    return centroids

def fig(df, centroids, colmap):
    plt.figure(0,(6,6))
    plt.scatter(df['x'], df['y'], color = df['color'], alpha= 0.5)
    for i in centroids.keys():
        plt.scatter(x = centroids[i][0], y = centroids[i][1], s= 100, color=colmap[i])
    plt.xlim(0, max(df['x']) + 10)
    plt.ylim(0, max(df['y']) + 10)
    plt.show()

def main():
    # step 1: generate source data
    x, y = genSamples()
    df = pd.DataFrame({'x': x, 'y': y})
    
    # step 2: generate center
    k = 4
    # centroids[i] = [x, y]
    centroids = {i: [np.random.randint(min(x), max(x) + 1), np.random.randint(min(y), max(y))] for i in range(k)}

    # step 3: assign centroid for each source data
    colmap = {0: 'r', 1: 'g', 2: 'b', 3: 'k'}
    df = assignment(df, centroids, colmap)
    #figure
    fig(df, centroids, colmap)
    #update centroids
    for epoch in range(10):
        plt.close()
        closest_centroids = df['closest'].copy(deep=True)
        centroids = updateCentroids_byFarther(df, centroids)
        df = assignment(df, centroids, colmap)
        
        if closest_centroids.equals(df['closest']):
            break
            
        fig(df, centroids, colmap)
        
    plt.close()

if __name__ == '__main__':
    main()


# coding: utf-8

def getIOU(points1, points2):
    '''
    points1/points2 : list like [x1,x2,y1,y2]
    '''
    if (min(points1[1],points2[1]) > max(points1[0],points2[0]) and (min(points1[3],points2[3]) > max(points1[2],points2[2]))):
        s1 = (points1[1] - points1[0])*(points1[3] - points1[2])
        s2 = (points2[1] - points2[0])*(points1[3] - points1[2])
        s3 = (min(points1[1],points2[1]) - max(points1[0],points2[0])) *  (min(points1[3],points2[3]) - max(points1[2],points2[2]))
#         print(points1,points2,s1,s2,s3,s3/(s1+s2-s3))
        return s3/(s1+s2-s3)
    else:
#         print(min(points1[1],points2[1]), max(points1[0],points2[0]), \
#               min(points1[3],points2[3]), max(points1[2],points2[2]))
        return 0
    
def NMS(lists,thre):
    '''
    lists: list of list contains left-top and right-bottom points of box x1,x2,y1,y2 and score
    threshold : NMS iou threshold
    return : boxes choosed
    '''
    M = []
    D = []
    maxScore = 0
    #排序求最大值
    lists_sorted = sorted(lists, key= lambda x: x[-1])
    maxScore = lists_sorted[-1][-1]
    M = lists_sorted[-1][0:4]
    D.append(M)
    lists_sorted.pop(-1)
    IOU = 0
    for box_score in lists_sorted:
        IOU = getIOU(box_score[:-1], M)
#         print(box_score, IOU)
        if IOU >= thre:
            continue
        else:
            D.append(box_score[:-1])
    return D

if __name__ == '__main__':
    # lists: list of list contains left-top and right-bottom points of box x1,x2,y1,y2 and score
    lists = [[0,4,0,4,0.5],[1,5,1,5,0.8],[3,6,3,6,0.6],[4,6,4,6,0.3]]
    # threshold : NMS iou threshold
    threshold = 0.3
    output = NMS(lists, threshold)
    print(output)


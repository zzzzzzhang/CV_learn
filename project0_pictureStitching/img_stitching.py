# coding: utf-8

import cv2
import numpy as np

def imShowWithWaitKey(img):
    cv2.imshow('img_test', img)
    if cv2.waitKey() == 27:
        cv2.destroyAllWindows()
        
def getPoints(img, hessianThreshold, savePath = None):
    '''
    return key points & descriptions
    '''
    # 创建SURF detector
    detector = cv2.xfeatures2d.SURF_create(hessianThreshold = hessianThreshold)

    # 获得SURF特征点和描述子
    kpts, decs = detector.detectAndCompute(img,None)

    #画出key_points
    img_kp = cv2.drawKeypoints(img, kpts, outImage = img.copy(), flags =  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #show keypoints
    if savePath:
        imShowWithWaitKey(img_kp)
        cv2.imwrite(savePath, img_kp)

    return kpts, decs

def ketPointsMatching(img_1, kpts_1, decs_1, img_2, kpts_2, decs_2):
    '''
    return img_3 with matched points drawn on 
    '''
    # define a point matcher
    mathcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck = True)
    # match 'nearest point' by Manhattan Distance
    matches = mathcher.match(decs_1, decs_2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)[:40]
    # Draw first 10 matches.
    img = cv2.drawMatches(img_1, kpts_1, img_2, kpts_2, matches, outImg = np.array([0]), flags= 2)
    #get points' loc
    kp_matched_ori = [kpts_1[i.queryIdx].pt for i in matches]
    kp_matched_taget = [kpts_2[i.trainIdx].pt for i in matches]
    
    return img, np.array(kp_matched_ori), np.array(kp_matched_taget)

def pictureStitch(img_1, kp_matched_ori, img_2, kp_matched_taget, methed = 1):
    '''
    return stitched picture
    '''
    # 求透视变化矩阵
    homo,_ = cv2.findHomography(kp_matched_ori,kp_matched_taget,cv2.RANSAC)

    #根据对应关系判断是homo/np.linalg.inv(homo)
    img_4 = cv2.warpPerspective(img_2, homo, (img_1.shape[1] + img_2.shape[1], img_1.shape[0]))
    
    sum_row = img_4[:,:,0].sum(axis = 0)
    endRight = np.where(sum_row > 0)[0][-1] #右结束边界
    
    if methed:
        # 权重叠加
        right = img_1.shape[1] #右边界
        left = np.where(sum_row > 0)[0][0] #左边界
        

        # 根据到两边距离赋予权重
        for col in range(right):
            if col <= left:
                img_4[:,col,:] = img_1[:,col,:]
            else:
                a = img_1[:,col,:]
                b = img_4[:,col,:]
                b = np.where(b == 0, a, b) #没有值的区域填充a的值
                w1 = (col - left)/(right - left)
                w2 = (right - col)/(right - left)
                img_4[:,col,:] = w2 * a + w1 * b
#                 print(w1 * a + w2 * b)

        img_4 = img_4[:,:endRight,:] #去掉右边黑边
    else:
        # 直接叠加
        img_4[0:img_1.shape[0], 0:img_1.shape[1]] = img_1
        
    img_4 = img_4[:,:endRight,:] #去掉右边黑边
    return img_4

if __name__ == '__main__':
    path_1 = r'source/1.jpg'
    path_2 = r'source/2.jpg'
    img_1 = cv2.imread(path_1)
    img_2 = cv2.imread(path_2)

    kpts_1, decs_1 = getPoints(img_1,hessianThreshold= 1000, savePath= r'out/1_kp.jpg')

    kpts_2, decs_2 = getPoints(img_2,hessianThreshold= 1000, savePath= r'out/1_kp.jpg')

    img_3,kp_matched_ori,kp_matched_taget = ketPointsMatching(img_2, kpts_2, decs_2, img_1, kpts_1, decs_1)

    img_4 = pictureStitch(img_1, kp_matched_ori, img_2, kp_matched_taget, methed= 1)

    imShowWithWaitKey(img_4)
    cv2.imwrite(r'out/test_stitched_2.jpg', img_4)

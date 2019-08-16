## 技术途径

---
(基于opencv库)

1.获取图像SURF特征点

2.挑选最佳匹配点20个

3.使用ransac方法挑选特征点建模，获得透视变换矩阵

4.重叠部分根据离边界距离远近赋予权重融合

5.保存结果

---
两张原始图片

![ori](https://github.com/zzzzzzhang/CV_learn/blob/master/project0_pictureStitching/source/seal1.jpg)
![ori](https://github.com/zzzzzzhang/CV_learn/blob/master/project0_pictureStitching/source/seal2.jpg)
---
特征点

![kp](https://github.com/zzzzzzhang/CV_learn/blob/master/project0_pictureStitching/out/seal1_kp.jpg)
![kp](https://github.com/zzzzzzhang/CV_learn/blob/master/project0_pictureStitching/out/seal2_kp.jpg)
---
匹配结果

![kplines](https://github.com/zzzzzzhang/CV_learn/blob/master/project0_pictureStitching/out/seal_kp_lines.jpg)
---
融合结果

![result](https://github.com/zzzzzzhang/CV_learn/blob/master/project0_pictureStitching/out/seal_stitched_2.jpg)

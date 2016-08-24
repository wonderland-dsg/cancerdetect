//
// Created by dang on 16-8-7.
//

#ifndef BREAST_CONCER_DETECTION_LBP_H
#define BREAST_CONCER_DETECTION_LBP_H

#include <opencv2/opencv.hpp>
static int g_lookTable[256]={
        56, 0, 7, 1, 14, 57, 8, 2, 21, 57, 57, 57, 15, 57, 9, 3, 28, 57, 57, 57, 57, 57, 57, 57, 22, 57, 57, 57,
        16, 57, 10, 4, 35, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 29, 57, 57, 57, 57, 57,
        57, 57, 23, 57, 57, 57, 17, 57, 11, 5, 42, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57,
        57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 36, 57, 57, 57, 57, 57, 57, 57, 57, 57,
        57, 57, 57, 57, 57, 57, 30, 57, 57, 57, 57, 57, 57, 57, 24, 57, 57, 57, 18, 57, 12, 6, 49, 50, 57, 51,
        57, 57, 57, 52, 57, 57, 57, 57, 57, 57, 57, 53, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57,
        57, 54, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57,
        57, 57, 57, 57, 57, 57, 57, 55, 43, 44, 57, 45, 57, 57, 57, 46, 57, 57, 57, 57, 57, 57, 57, 47, 57, 57,
        57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 48, 37, 38, 57, 39, 57, 57, 57, 40, 57, 57, 57, 57,
        57, 57, 57, 41, 31, 32, 57, 33, 57, 57, 57, 34, 25, 26, 57, 27, 19, 20, 13, 56
};
class LBP{
private:

    void getLBPImage(cv::Mat& src,cv::Mat& lbp);

    void histFill(cv::Mat& img, cv::Rect rect, float* feature, int len = 58)
    {
        cv::Mat tmpImg = img(rect);
        for (int r = 0; r < tmpImg.rows; ++r) {
            for (int c = 0; c < tmpImg.cols; ++c) {
                int v = (int)tmpImg.at<uchar>(r, c);
                feature[v] += 1.0;
            }
        }
        int s = tmpImg.rows * tmpImg.cols;
        for (int i = 0; i < len; ++i) feature[i] /= s;
    }
public:
    LBP(){
    }
    ~LBP(){
    }
    void getLBPVector(cv::Mat& img,std::vector<float>& lbp_vector);

};


#endif //BREAST_CONCER_DETECTION_LBP_H

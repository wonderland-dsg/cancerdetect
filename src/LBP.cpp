//
// Created by dang on 16-8-7.
//

#include "LBP.h"

void LBP::getLBPImage(cv::Mat &src, cv::Mat &lbp) {
    cv::Mat grayImg;
    if (src.channels() == 3) {
        cv::cvtColor(src, grayImg, CV_BGR2GRAY);
    }
    else {
        src.copyTo(grayImg);
    }
    cv::equalizeHist(grayImg,grayImg);
    lbp.create(src.size(), src.depth());
    lbp.setTo(0);
    for (int r = 1; r < grayImg.rows - 1; ++r) {
        const uchar* prev = grayImg.ptr(r - 1);
        const uchar* curr = grayImg.ptr(r);
        const uchar* next = grayImg.ptr(r + 1);
        uchar* pdst = lbp.ptr(r);
        for (int c = 1; c < grayImg.cols - 1; ++c) {
            uchar center = curr[c];
            int value = 0;
            if (curr[c + 1] > center)  value |= 0x01 << 0;
            if (next[c + 1] > center)  value |= 0x01 << 1;
            if (next[c] > center)      value |= 0x01 << 2;
            if (next[c - 1] > center)  value |= 0x01 << 3;
            if (curr[c - 1] > center)  value |= 0x01 << 4;
            if (prev[c - 1] > center)  value |= 0x01 << 5;
            if (prev[c] > center)      value |= 0x01 << 6;
            if (prev[c + 1] > center)  value |= 0x01 << 7;
            pdst[c] = (uchar)g_lookTable[value];
        }
    }
}

void LBP::getLBPVector(cv::Mat &img, std::vector<float> &lbp_vector) {
    cv::resize(img,img,cv::Size(480,360));
    cv::Mat lbp;
    getLBPImage(img,lbp);
    lbp_vector.resize(58*576,0);
    float* temp = &(lbp_vector[0]);
    // the boundary of the lbp is ignored, actually 78x58 rectangle is used.
    for (int r = 1; r <= 329; r += 14) {
        for (int c = 1; c <= 439; c += 19) {
            cv::Rect rect(c, r, 40, 30);
            histFill(lbp, rect, temp);
            temp += 58;
        }
    }
}
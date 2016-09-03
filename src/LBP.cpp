//
// Created by dang on 16-8-7.
//

#include "LBP.h"
#include <omp.h>
#include "Canny.h"

void LBP::getLBPImage(cv::Mat &src, cv::Mat &lbp) {
    cv::Mat grayImg;
    if (src.channels() == 3) {
        cv::cvtColor(src, grayImg, CV_BGR2GRAY);
    }
    else {
        src.copyTo(grayImg);
    }
    //cv::equalizeHist(grayImg,grayImg);
    lbp.create(src.size(), src.depth());
    lbp.setTo(0);
    omp_set_num_threads(8);
#pragma omp parallel for
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
    cv::Mat grayImg;
    if (img.channels() == 3) {
        cv::cvtColor(img, grayImg, CV_BGR2GRAY);
    }
    else {
        img.copyTo(grayImg);
    }
    //cv::equalizeHist(grayImg,grayImg);
    sigma=1;
    int winsize=(int)(sigma*4);
    if(winsize%2==0)winsize++;
    cv::resize(grayImg,grayImg,cv::Size(480+winsize-3,360+winsize-3));

    grayImg=getCanny(grayImg,sigma);

   // std::cout<<"grayImg size:"<<grayImg.rows<<std::endl;

    cv::Mat lbp;
    getLBPImage(grayImg,lbp);
    //std::cout<<"getLBPImage "<<std::endl;

    lbp_vector.resize(58*576,0);
    //std::cout<<"resizeLbp "<<std::endl;
    float* temp = &(lbp_vector[0]);
    // the boundary of the lbp is ignored, actually 78x58 rectangle is used.
    //std::cout<<"temp "<<std::endl;
    omp_set_num_threads(8);
#pragma omp parallel for
    for (int r = 1; r <= 329;r += 14) {//loop 24 times
        for (int c = 1; c <= 439; c += 19) {//loop 24 times
            cv::Rect rect(c, r, 40, 30);
            histFill(lbp, rect, temp+58*(((r-1)/14)*24+((c-1)/19)));
            //temp += 58;

        }
    }

    //std::cout<<"finish LBP "<<std::endl;
}
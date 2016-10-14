//
// Created by dang on 16-8-7.
//

#include "LBP.h"
#include <omp.h>
#include "Canny.h"

#define scala 7
double sigmas[scala]={0.5,1,1.5,2,3,3.5,4};

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

cv::Mat cannyProcess(cv::Mat grayImg,double sigma){
    int winsize=(int)(sigma*4);
    //std::cout<<"winsize"<<winsize<<std::endl;
    if(winsize%2==0)winsize++;
    //std::cout<<"winsize"<<winsize<<std::endl;
    cv::resize(grayImg,grayImg,cv::Size(480+winsize-3,360+winsize-3));
    grayImg=getCanny(grayImg,sigma);
    return grayImg;
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
    //sigma=0.5;



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

void LBP::getLBPVectorDebug(cv::Mat &img, std::vector<float> &lbp_vector,cv::Mat& lbpImg,cv::Mat& lbpImg2) {
    cv::Mat grayImg;
    if (img.channels() == 3) {
        cv::cvtColor(img, grayImg, CV_BGR2GRAY);
    }
    else {
        img.copyTo(grayImg);
    }
    //cv::equalizeHist(grayImg,grayImg);
    //sigma=0.5;



    // std::cout<<"grayImg size:"<<grayImg.rows<<std::endl;

    cv::Mat lbp;
    getLBPImage(grayImg,lbp);
    //std::cout<<"getLBPImage "<<std::endl;
    lbpImg=lbp;
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
    lbpImg2.create(58,576,CV_8UC1);
    for(int nr=0;nr<lbpImg2.rows;nr++){
        uchar* curr = lbpImg2.ptr(nr);
        for(int nl=0;nl<lbpImg2.cols;nl++){
            curr[nl]=uchar(lbp_vector[nr*58+nl]*255);
        }
    }
    //std::cout<<"finish LBP "<<std::endl;
}

void LBP::getLBPScalaVector(cv::Mat &img, std::vector<float> &lbp_vector) {
    cv::Mat grayImg;
    if (img.channels() == 3) {
        cv::cvtColor(img, grayImg, CV_BGR2GRAY);
    }
    else {
        img.copyTo(grayImg);
    }

    //cv::Mat grayImg1,grayImg2,grayImg3,grayImg4;
    //array sigmas is defined in the head of this cpp file
    std::vector<cv::Mat> immages;//,lbpImages1,lbpImages2;

    for(int i=0;i<scala;i++){
        immages.push_back(cannyProcess(grayImg,sigmas[i]));
    }
    std::vector<float> temp;
    for(int i=0;i<immages.size();i++){
        getLBPVector(immages[i],temp);
        //getLBPVectorDebug(immages[i],temp,lbpImages1[i],lbpImages2[i]);
        for(int j=0;j<temp.size();j++){
            lbp_vector.push_back(temp[j]);
        }
    }

}

void LBP::getLBPScalaVectorDebug(cv::Mat &img, std::vector<float> &lbp_vector,std::vector<cv::Mat>& immages,std::vector<cv::Mat>& lbpImgs1,std::vector<cv::Mat>& lbpImgs2) {
    /*
     * immages: different scalas images
     * lbpImgs1: the lbd features Mat for each scala images
     * lbpImgs2: the lbp features Mat mapping
     *
     * */
    cv::Mat grayImg;
    if (img.channels() == 3) {
        cv::cvtColor(img, grayImg, CV_BGR2GRAY);
    }
    else {
        img.copyTo(grayImg);
    }
    //double sigmas[scala]={0.5,1,1.5,2,3,3.5,4};
    //cv::Mat grayImg1,grayImg2,grayImg3,grayImg4;
    //std::vector<cv::Mat> immages;
    lbpImgs1.resize(scala);
    lbpImgs2.resize(scala);
    for(int i=0;i<scala;i++){
        immages.push_back(cannyProcess(grayImg,sigmas[i]));
    }
    std::vector<float> temp;
    for(int i=0;i<immages.size();i++){
        //getLBPVector(immages[i],temp);
        getLBPVectorDebug(immages[i],temp,lbpImgs1[i],lbpImgs2[i]);
        for(int j=0;j<temp.size();j++){
            lbp_vector.push_back(temp[j]);
        }
    }

}
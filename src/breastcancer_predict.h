//
// Created by dang on 16-8-7.
//

#ifndef BREAST_CONCER_DETECTION_BREASTCANCER_PREDICT_H
#define BREAST_CONCER_DETECTION_BREASTCANCER_PREDICT_H

#include <fstream>
#include "svm.h"
#include "LBP.h"
class CancerPredict{
private:
    LBP mLBP;
    void readImagePaths(std::string txtPathFile,std::vector<std::string>& imgPaths);
    void readTrainSample(const std::vector<std::string>& img_path,double *y,double flag,svm_node **pnodes,int cur);
public:
    CancerPredict(){
        mLBP=LBP();
    };
    ~CancerPredict(){};
    void trainModel(const char* pos_txt_file,const char* neg_txt_file,const char* model_file);
    double predictSample(cv::Mat img,const char* model_file);
    double predictSample(cv::Mat img,svm_model* model);
    void testModel(const char* txt_file,const char* model_file,double target=1);
};

#endif //BREAST_CONCER_DETECTION_BREASTCANCER_PREDICT_H

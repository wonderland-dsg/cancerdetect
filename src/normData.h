//
// Created by dang on 16-8-14.
//

#ifndef BREAST_CONCER_DETECTION_NORMDATA_H
#define BREAST_CONCER_DETECTION_NORMDATA_H
#include "svm.h"
#include "GLCM.h"

namespace nd{
    class normData{
    private:
        vi::GLCM* pGLCM=NULL;
        void getVector(cv::Mat source_imgage,std::vector<double>& feature);
        void readTrainSample(const std::vector<std::string>& img_path,double *y,double flag,svm_node **pnodes,int cur);

        inline void getDescripter(double*feature ,const vi::glcmParams gP){
            feature[0]=gP.clustershade;
            feature[1]=gP.contrast;
            feature[2]=gP.correlation;
            feature[3]=gP.dissimilarity;
            feature[4]=gP.energy;

            feature[5]=gP.entropy;
            feature[6]=gP.homogenity;
            feature[7]=gP.mean;
            feature[8]=gP.mean_actual;
            feature[9]=gP.stddev;

            feature[10]=gP.stdev_actual;
        }
    public:
        normData(){
            pGLCM=new vi::GLCM(8);
        };
        ~normData(){
            delete pGLCM;
        };
        void start(char *pos_txt_file,char*neg_txt_file,char *outfile);
    };
}
#endif //BREAST_CONCER_DETECTION_NORMDATA_H

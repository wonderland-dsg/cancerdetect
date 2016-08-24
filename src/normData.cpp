//
// Created by dang on 16-8-14.
//
#include "normData.h"
#include <fstream>
#include <iostream>
#include <vector>
#include "my_scale.h"
#include <opencv2/opencv.hpp>


#define SAMPLE_HOME "/home/dang/ClionProjects/breast_concer_detection/resource/samples/"

#define store_param "/home/dang/ClionProjects/breast_concer_detection/resource/samples_all_pos_neg.param"
#define restore_param "/home/dang/ClionProjects/breast_concer_detection/resource/myrestore.param"
#define scale_data "/home/dang/ClionProjects/breast_concer_detection/resource/scale_samples_all_pos_neg.dat"



#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

namespace nd{

    void readImagePaths(std::string txtPathFile, std::vector<std::string> &imgPaths) {
        imgPaths.resize(0);
        std::ifstream textFile(txtPathFile.c_str());
        if (!textFile.is_open()) {
            std::cerr << "can not read the image path text file." << std::endl;
            return;
        }
        std::string path;
        while (std::getline(textFile, path, '\n')) {
            path=SAMPLE_HOME+path;
            imgPaths.push_back(path);
        }
    }

    inline void copyFeatureToNode(std::vector<double>& feature, svm_node* node)
    {
        for (int i = 0; i < (int)feature.size(); ++i) {
            node[i].index = i + 1;
            node[i].value = feature[i];
            //std::cout<<i<<":node[i].index"<<node[i].index<<":node[i].value"<<node[i].value<<std::endl;
        }
        node[(int)feature.size()].index = -1;
    }
    void normData::getVector(cv::Mat source_imgage,std::vector<double>& feature){
        cv::resize(source_imgage,source_imgage,cv::Size(480,360));
        cv::Mat grayImg;
        if (source_imgage.channels() == 3) {
            cv::cvtColor(source_imgage, grayImg, CV_BGR2GRAY);
        }
        else {
            source_imgage.copyTo(grayImg);
        }
        pGLCM->setImage(grayImg);
        pGLCM->splitToWindows(40);
        pGLCM->calcGLCM();

        tbb::concurrent_vector<vi::glcmParams> myGlcmBlocks;
        myGlcmBlocks=pGLCM->_glcmBlocks;
        feature.resize(108*11,0);
        double* temp = &(feature[0]);
        for(int i=0;i<myGlcmBlocks.size();i++){
            getDescripter(temp,myGlcmBlocks[i]);
            temp+=11;
        }
    }
    void normData::readTrainSample(const std::vector<std::string>& img_path,double *y,double flag,svm_node **pnodes,int cur){
        for (int i = 0; i < (int)img_path.size(); ++i) {
            cv::Mat img = cv::imread(img_path[i]);
            std::vector<double> feature;
            getVector(img, feature);
            pnodes[cur+i] = new svm_node[(int)feature.size() + 1];
            copyFeatureToNode(feature, pnodes[cur+i]);
            y[i+cur] = flag;
        }
    }

    void normData::start(char *pos_txt_file,char*neg_txt_file, char *outfile) {
        //prepare the train data
        std::vector<std::string> pos_img_path,neg_img_path;
        readImagePaths(pos_txt_file, pos_img_path);
        readImagePaths(neg_txt_file, neg_img_path);
        int nsamples = (int)(pos_img_path.size() + neg_img_path.size());
        int nsamples_pos=(int)pos_img_path.size();
        svm_node** pnodes = new svm_node*[nsamples];
        double* y=new double[nsamples];
        std::cout<<"load sample..."<<std::endl;
        readTrainSample(pos_img_path,y,1.0,pnodes,0);
        readTrainSample(neg_img_path,y,-1.0,pnodes,nsamples_pos);
        std::cout<<"load sample success"<<std::endl;
        svm_problem mproblem;
        //init the svm problem
        mproblem.l=nsamples;
        mproblem.x=pnodes;
        mproblem.y=y;
        std::cout<<"train problem is done"<<std::endl;

        std::cout<<"begin file"<<std::endl;
        std::cout<<"open file"<<std::endl;
        std::ofstream outFile(outfile);
        if (!outFile.is_open()) {
            std::cerr << "can not read the image path text file." << std::endl;
            return;
        }
        //outFile<< std::fixed;
        std::cout<<"write file"<<std::endl;

        std::cout<<mproblem.l<<std::endl;
        for (int i=0;i<mproblem.l;i++){
            int j=0;
            outFile<<y[i];
            //outFile<<x[j]->value<<std::endl;
            while(mproblem.x[i][j].index!=-1){ //x[j]->index!=-1
                outFile<<"\t";
                outFile<<mproblem.x[i][j].index;
                outFile<<":";
                outFile<<mproblem.x[i][j].value;
                j++;
            }
            outFile<<std::endl;
        }
        outFile.close();
        doscale(store_param,NULL,outfile,scale_data);
    }
}
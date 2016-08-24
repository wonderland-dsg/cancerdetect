//
// Created by dang on 16-8-13.
//

#define SAMPLE_HOME "/home/dang/ClionProjects/breast_concer_detection/resource/samples/"

#include "predict_glcm.h"


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

void CancerPredictGlcm::getFeatureVector(cv::Mat source_imgage, std::vector<double> &feature) {
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
    for(int i=0;i<(int)feature.size();i++){
        if(feature_max[i]<DBL_MAX&&feature_min[i]>-DBL_MAX){
            double value=feature[i];
            value = lower + (upper-lower) *
                            (value-feature_min[i])/
                            (feature_max[i]-feature_min[i]);
            feature[i]=value;
            std::cout<<value<<std::endl;
        }
    }
}

void CancerPredictGlcm::readImagePaths(std::string txtPathFile, std::vector<std::string> &imgPaths) {
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
    }
    node[(int)feature.size()].index = -1;
}

void CancerPredictGlcm::readTrainSample(const std::vector<std::string> &img_path, double *y, double flag,
                                        svm_node **pnodes, int cur) {
    for (int i = 0; i < (int)img_path.size(); ++i) {
        cv::Mat img = cv::imread(img_path[i]);
        std::vector<double> feature;
        this->getFeatureVector(img,feature);
        pnodes[cur+i] = new svm_node[(int)feature.size() + 1];
        copyFeatureToNode(feature, pnodes[cur+i]);
        y[i+cur] = flag;
    }
}

void CancerPredictGlcm::trainModel(const char *pos_txt_file, const char *neg_txt_file, const char *model_file) {
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

    svm_parameter param;
    // default values
    param.svm_type =C_SVC ;//;ONE_CLASS(so bad)  NU_SVC(maybe)C_SVC  NU_SVC
    param.kernel_type = LINEAR; //RBF(validation accuracy<0.7 )  LINEAR(<0.65) SIGMOID(<0.7) //SIGMOID(2016-8-10 first run) (second is RBF) third is LINEAR
    param.degree = 3;
    param.gamma = 1.0/1188;	// 1/num_features
    param.coef0 = 0;
    param.nu = 0.5;
    param.cache_size = 100;
    param.C = 0.53; //this parameter always need change,
    param.eps = 1e-3;
    param.p = 0.1;
    param.shrinking = 1;
    param.probability = 0;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;
    std::cout << "param is done." << std::endl;

    //do_cross_validation(mproblem,param,nsamples);

    const char* errmsg = svm_check_parameter(&mproblem, &param);
    if (errmsg) {
        std::cout << "something error: " << errmsg << std::endl;
        return;
    }
    svm_model* model = svm_train(&mproblem, &param);
    std::cout << "training is done" << std::endl;
    if(svm_save_model(model_file,model))
    {
        fprintf(stderr, "can't save model to file %s\n", model_file);
        exit(1);
    }
    int success_count=0;
    for(int i=0;i<nsamples;i++){
        svm_node* x=pnodes[i];
        double predict_label=svm_predict(model,x);
        std::cout<<predict_label;//<<std::endl;
        if(predict_label==y[i]){
            success_count++;
        }
    }
    std::cout<<"accuracy is: "<<(double)success_count/nsamples<<std::endl;
    std::cout << "model has saved" << std::endl;
    svm_free_and_destroy_model(&model);
    svm_destroy_param(&param);
    for (int i = 0; i < nsamples; ++i) delete[] pnodes[i];
    delete[] pnodes;
    delete[] y;
}

void CancerPredictGlcm::testModel(const char *txt_file, const char *model_file, double target) {
    std::cout<<"begin test..."<<std::endl;
    svm_model* model=svm_load_model(model_file);
    int success_count=0;
    std::vector<std::string> img_path;
    std::cout<<"begin read img_path..."<<std::endl;
    readImagePaths(txt_file, img_path);
    std::cout<<"success read img_path..."<<std::endl;
    for(int i=0;i<img_path.size();i++){
        std::cout<<"read img"<<img_path[i]<<std::endl;
        cv::Mat img=cv::imread(img_path[i]);
        double predict_label=predictSample(img,model);
        std::cout<<predict_label<<std::endl;
        if(predict_label==target)
            success_count++;
    }
    std::cout<<"accuracy is: "<<(double)success_count/img_path.size()<<std::endl;
}

double CancerPredictGlcm::predictSample(cv::Mat img, svm_model *model) {
    std::vector<double> feature;
    this->getFeatureVector(img, feature);
    svm_node *x;
    x=new svm_node[feature.size()+1];
    x[feature.size()].index=-1;
    copyFeatureToNode(feature,x);
    std::cout<<"get feature success!"<<feature.size()<<std::endl;
    double predict_label=svm_predict(model,x);
    std::cout<<"predict success!"<<predict_label<<std::endl;
    return predict_label;
}

double CancerPredictGlcm::predictSample(cv::Mat img, const char *model_file) {
    svm_model* model=svm_load_model(model_file);
    std::cout<<"load model success!"<<std::endl;
    return predictSample(img,model);
}
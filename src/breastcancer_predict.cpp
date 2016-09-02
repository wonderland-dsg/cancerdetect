//
// Created by dang on 16-8-7.
//


#include "breastcancer_predict.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

void CancerPredict::readImagePaths(std::string txtPathFile, std::vector<std::string> &imgPaths) {
        imgPaths.resize(0);
        std::ifstream textFile(txtPathFile.c_str());
        if (!textFile.is_open()) {
            std::cerr << "can not read the image path text file." << std::endl;
            return;
        }
        std::string path;
        while (std::getline(textFile, path, '\n')) {
            path="/home/dang/ClionProjects/breast_concer_detection/resource/samples/"+path;
            imgPaths.push_back(path);
        }
}

inline void copyFeatureToNode(std::vector<float>& feature, svm_node* node)
{
    for (int i = 0; i < (int)feature.size(); ++i) {
        node[i].index = i + 1;
        node[i].value = feature[i];
    }
    node[(int)feature.size()].index = -1;
}


void do_cross_validation(svm_problem prob,svm_parameter param,int nr_fold)
{
    int i;
    int total_correct = 0;
    double total_error = 0;
    double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
    double *target = Malloc(double,prob.l);

    svm_cross_validation(&prob,&param,nr_fold,target);
    if(param.svm_type == EPSILON_SVR ||
       param.svm_type == NU_SVR)
    {
        for(i=0;i<prob.l;i++)
        {
            double y = prob.y[i];
            double v = target[i];
            total_error += (v-y)*(v-y);
            sumv += v;
            sumy += y;
            sumvv += v*v;
            sumyy += y*y;
            sumvy += v*y;
        }
        printf("Cross Validation Mean squared error = %g\n",total_error/prob.l);
        printf("Cross Validation Squared correlation coefficient = %g\n",
               ((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
               ((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))
        );
    }
    else
    {
        for(i=0;i<prob.l;i++)
            if(target[i] == prob.y[i])
                ++total_correct;
        printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
    }
    free(target);
}

void CancerPredict::readTrainSample(const std::vector<std::string>& img_path,double *y,double flag,svm_node **pnodes,int cur){
    for (int i = 0; i < (int)img_path.size(); ++i) {
        cv::Mat img = cv::imread(img_path[i]);
        std::vector<float> feature;
        mLBP.getLBPVector(img, feature);
        pnodes[cur+i] = new svm_node[(int)feature.size() + 1];
        copyFeatureToNode(feature, pnodes[cur+i]);
        y[i+cur] = flag;
    }
}
void CancerPredict::trainModel(const char *pos_txt_file, const char *neg_txt_file, const char *model_file) {
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
    param.gamma = 2.0/nsamples;	// 1/num_features
    param.coef0 = 0;
    param.nu = 0.5;
    param.cache_size = 100;
    param.C = 5; //this parameter always need change,
    param.eps = 1e-3;
    param.p = 0.1;
    param.shrinking = 1;
    param.probability = 1;
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
        //double predict_label=svm_predict(model,x);
        double prob_estimates[2];
        double predict_label=svm_predict_probability(model,x,prob_estimates);
        std::cout<<"probability is: "<<prob_estimates[0]<<"  "<<prob_estimates[1]<<std::endl;
        std::cout<<predict_label<<std::endl;
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

double CancerPredict::predictSample(cv::Mat img, const char *model_file) {
    svm_model* model=svm_load_model(model_file);
    double predict_label=predictSample(img,model);
    return predict_label;
}

double CancerPredict::predictSample(cv::Mat img, svm_model* model) {
    std::vector<float> feature;
    mLBP.getLBPVector(img, feature);
    svm_node *x;
    x=new svm_node[feature.size()+1];
    x[feature.size()].index=-1;
    copyFeatureToNode(feature,x);
    double predict_label=svm_predict(model,x);
    //double prob_estimates[2];
    //double predict_label=svm_predict_probability(model,x,prob_estimates);
    //std::cout<<"probability is: "<<prob_estimates[0]<<"  "<<prob_estimates[1]<<" model(label<0,1>):"<<model->label[0]<<" "<<model->label[1]<<std::endl;
    //svm_free_and_destroy_model(&model);
    delete[] x;
    feature.clear();
    return predict_label;
}
void CancerPredict::testModel(const char *txt_file, const char *model_file,double target) {
    std::cout<<"begin test..."<<std::endl;
    svm_model* model=svm_load_model(model_file);
    int success_count=0;
    std::vector<std::string> img_path;
    readImagePaths(txt_file, img_path);
    std::cout<<img_path.size()<<std::endl;
    for(int i=0;i<img_path.size();i++){
        cv::Mat img=cv::imread(img_path[i]);
        double predict_label=predictSample(img,model);
        std::cout<<predict_label<<std::endl;
        if(predict_label==target)
            success_count++;
    }
    std::cout<<"accuracy is: "<<(double)success_count/img_path.size()<<std::endl;
}

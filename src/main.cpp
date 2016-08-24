#define tpl "/home/dang/ClionProjects/breast_concer_detection/resource/train_norm_pos.lst"  //pos_all_train.lst" //
#define tnl "/home/dang/ClionProjects/breast_concer_detection/resource/train_norm_neg.lst"   //neg_all_train.lst" //
#define mp "/home/dang/ClionProjects/breast_concer_detection/resource/svm_for_glcm_all_2_linear.model" //svm_for_glcm_1.model"
//#define vpl "/home/dang/ClionProjects/breast_concer_detection/resource/validation_norm_pos.lst" //pos_all_test.lst" //
//#define vnl "/home/dang/ClionProjects/breast_concer_detection/resource/validation_norm_neg.lst"  //neg_all_test.lst" //
#define vpl "/home/dang/ClionProjects/breast_concer_detection/resource/samples/pos_all.lst"
#define vnl "/home/dang/ClionProjects/breast_concer_detection/resource/samples/neg_all.lst"

#define norm_file "/home/dang/ClionProjects/breast_concer_detection/resource/samples__all_neg_pos.waitForScale"

#define scale_train_file "/home/dang/ClionProjects/breast_concer_detection/resource/scale_data_train_all.dat"
#define scale_test_file "/home/dang/ClionProjects/breast_concer_detection/resource/scale_data_test_all.dat"

#define store_param "/home/dang/ClionProjects/breast_concer_detection/resource/samples_all_pos_neg.param"

//#define myGLCM
//#define myNORM
//#define myLBP
//#define myTEST
//#define QTdialog
#define myGUI

#ifdef myLBP
#include <iostream>
#include "breastcancer_predict.h"
#include "GLCM.h"
using namespace std;
//#define test_pos
#define test_neg
#define train
int main() {
    CancerPredict mcp;
    #ifdef train
    mcp.trainModel("/home/dang/ClionProjects/breast_concer_detection/resource/pos_all_train.lst",
                   "/home/dang/ClionProjects/breast_concer_detection/resource/neg_all_train.lst",
                   "/home/dang/ClionProjects/breast_concer_detection/resource/svm_for_all_2.model");
    #endif
    #ifdef test_pos
    mcp.testModel("/home/dang/ClionProjects/breast_concer_detection/pos_all_test.lst","/home/dang/ClionProjects/breast_concer_detection/svm_for_all_2.model",1.0);
    #endif
    #ifdef test_neg
    mcp.testModel("/home/dang/ClionProjects/breast_concer_detection/resource/neg_all_test.lst","/home/dang/ClionProjects/breast_concer_detection/respurce/svm_for_all_2.model",-1.0);
    #endif

    return 0;
}
#endif

#ifdef myGLCM
#include "predict_glcm.h"
//#define test_pos
#define test_neg
//#define train
int main(){
    CancerPredictGlcm* mcp;
    mcp=new CancerPredictGlcm(store_param);
#ifdef train
    mcp->trainModel(tpl,
                   tnl,
                   mp);
#endif
#ifdef test_pos
    mcp->testModel(vpl,mp,1.0);
#endif
#ifdef test_neg
    mcp->testModel(vnl,mp,-1.0);
#endif
    return 0;
}
#endif

#ifdef myNORM

#include <fstream>
#include "normData.h"
#include "mysvmtrain.h"
#include <string>
int main(){
    nd::normData* mynorm;
    mynorm=new nd::normData();
    //mynorm->start(vpl,vnl,norm_file);

    std::ifstream fp("/home/dang/ClionProjects/breast_concer_detection/resource/scale_samples_all_pos_neg.dat");
    std::ofstream fp_t(scale_train_file);
    std::ofstream fp_v(scale_test_file);
    if (!fp.is_open()|!fp_t.is_open()|!fp_v.is_open()) {
        std::cerr << "can not read the image path text file." << std::endl;
        return 1;
    }
    cv::string temp;
    int i=0;
    for(;i<0;i++){

        getline(fp,temp);
        fp_v<<temp<<std::endl;
    }
    for(;i<2700;i++){ //2450
        getline(fp,temp);
        fp_t<<temp<<std::endl;
    }
    for(;i<0;i++){//2753

        getline(fp,temp);
        fp_v<<temp<<std::endl;
    }
    fp.close();
    fp_t.close();
    fp_v.close();
    mytrain::mytrain(scale_train_file,mp,scale_train_file);//scale_test_file
    return 0;
}
#endif
#ifdef myTEST
#include "predict_glcm.h"
int main(int argc,char **argv){
    if(argc!=4){
        std::cout<<"param1: param file path\nparam2: image file path\nparam3:model file\n";
        return 0;
    }
    CancerPredictGlcm* mcp;
    mcp=new CancerPredictGlcm(argv[1]);
    cv::Mat img=cv::imread(argv[2]);
    std::cout<<"The result is:"<<mcp->predictSample(img,argv[3]);
    return 0;
}
#endif

#ifdef QTdialog

#include <QFileDialog>

#include <QApplication>
#include <iostream>

#include "predict_glcm.h"

int main(int argc,char** argv){

    QApplication a(argc,argv);
    QString file_name = QFileDialog::getOpenFileName(NULL, //parent moudle
                                                     QObject::tr("Open File"), //dialog title
                                                     ".", //the init directory
                                                     "JPEG Files(*.jpg);;PNG Files(*.png)",
                                                     0);

    if (!file_name.isNull())
    {
        std::cout<<file_name.toStdString()<<std::endl;
        CancerPredictGlcm* mcp;
        mcp=new CancerPredictGlcm(store_param);
        cv::Mat img=cv::imread(file_name.toStdString());
        cv::imshow("fdfdfd",img);
        cv::waitKey(0);
        std::cout<<"The result is:"<<mcp->predictSample(img,mp);

    }
    else{
        std::cout<<"you have choose cancel;"<<std::endl;

    }

    return  0;
}

#endif

#ifdef myGUI
#include "mainwindow.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();

    return a.exec();
}

#endif
















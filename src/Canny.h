//
// Created by dang on 16-9-3.
//

#ifndef BREAST_CONCER_DETECTION_CANNY_H
#define BREAST_CONCER_DETECTION_CANNY_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>

#define Melloc3X3Uchar(t) {t=new uchar*[3];for(int i=0;i<3;i++){t[i]=new uchar[3];}}
#define Free3X3Uchar(t) {for(int i=0;i<3;i++)delete[] t[i];}

#define MellocnXnDouble(t,n) {t=new double*[n];for(int i=0;i<n;i++){t[i]=new double[n];}}
#define FreenXnarray(t,n) {for(int i=0;i<n;i++)delete[] t[i];}

#define PI 3.141592653

#define CPU_COREs 8

using namespace std;

void array3X32Mat_freeArray(cv::Mat& temp,uchar** t){
    temp.create(3,3,CV_8UC1);
    for(int r=0;r<temp.rows;r++){
        uchar* curr = temp.ptr(r);
        for(int l=0;l<temp.cols;l++){
            curr[l]=t[r][l];
        }
    }
    Free3X3Uchar(t)
}

void arraynXn2Mat_freeArray(cv::Mat& temp,double** t,int n){
    temp.create(n,n,CV_64FC1);
    for(int r=0;r<temp.rows;r++){
        double* curr = temp.ptr<double>(r);
        for(int l=0;l<temp.cols;l++){
            curr[l]=t[r][l];
        }
    }
    FreenXnarray(t,n)
}

void arraynXn2Mat_freeArray_(cv::Mat& temp,double t[3][3],int n){
    temp.create(n,n,CV_64FC1);
    for(int r=0;r<temp.rows;r++){
        double* curr = temp.ptr<double>(r);
        for(int l=0;l<temp.cols;l++){
            curr[l]=t[r][l];
        }
    }
}

void Mat2array3X3(const cv::Mat& temp,uchar** t){
    Melloc3X3Uchar(t)
    for(int r=0;r<temp.rows;r++){
        const uchar* curr = temp.ptr(r);
        for(int l=0;l<temp.cols;l++){
            t[r][l]=curr[l];
        }
    }
}

void Mat2array3X3(const cv::Mat& temp,double** &t){
    MellocnXnDouble(t,3)
    for(int r=0;r<temp.rows;r++){
        const double* curr = temp.ptr<double>(r);
        for(int l=0;l<temp.cols;l++){
            t[r][l]=curr[l];
        }
    }
}

void Mat2arraynXn(const cv::Mat& temp,double** &t){
    MellocnXnDouble(t,temp.rows)
    for(int r=0;r<temp.rows;r++){
        const double* curr = temp.ptr<double>(r);
        for(int l=0;l<temp.cols;l++){
            t[r][l]=curr[l];
        }
    }
}

cv::Mat convolution3X3Kernel(const cv::Mat& source ,cv::Mat temp){
    cv::Mat result;
    cv::Mat grayImg;
    if (source.channels() == 3) {
        cv::cvtColor(source, grayImg, CV_BGR2GRAY);
    }
    else {
        source.copyTo(grayImg);
    }
    double** t;
    Mat2array3X3(temp,t);
    /*for(int i=0;i<3;i++){
        for(int j=0;j<3;j++)
            cout<<t[i][j]<<" ";
        cout<<endl;

    }*/


    std::cout<<"begin convolution Mat"<<std::endl;
    result.create(grayImg.rows-2,grayImg.cols-2,grayImg.type());
    cv::Mat re_d;
    re_d.create(grayImg.rows-2,grayImg.cols-2,CV_64FC1);
    double mx=0,mn=0;
    //omp_set_num_threads(CPU_COREs);
//#pragma omp parallel for
    for(int r=1;r<grayImg.rows-1;r++){
        //std::cout<<"r:"<<r<<std::endl;
        const uchar* prev = grayImg.ptr(r - 1);
        const uchar* curr = grayImg.ptr(r);
        const uchar* next = grayImg.ptr(r + 1);
        double* pdst = re_d.ptr<double>(r-1);
        for(int l=1;l<grayImg.cols-1;l++){
            //std::cout<<"l:"<<l<<std::endl;
            double p[3],c[3],n[3];
            for(int i=0;i<3;i++){
                p[i]=prev[l-1+i];
                c[i]=curr[l-1+i];
                n[i]=next[l-1+i];
                //std::cout<<"p c n:"<<p[i]<<" " <<c[i]<<" "<<n[i]<<std::endl;
            }
            double re=p[0]*t[0][0]+p[1]*t[0][1]+p[2]*t[0][2]+c[0]*t[1][0]+c[1]*t[1][1]+c[2]*t[1][2]+n[0]*t[2][0]+n[1]*t[2][1]+n[2]*t[2][2];
            /*double re=prev[l-1]*t[0][0]+prev[l]*t[0][1]+prev[l+1]*t[0][2]+\
                      curr[l-1]*t[1][0]+curr[l]*t[1][1]+curr[l+1]*t[1][2]+\
                      next[l-1]*t[2][0]+next[l]*t[2][1]+next[l+1]*t[2][2];*/
            //std::cout<<"res:"<<re<<std::endl;
            //if(re<0.0000001)
            //   re=0;
            pdst[l-1]=re;
        }
    }
    std::cout<<"finish!"<<std::endl;
    FreenXnarray(t,3)
    return re_d;
}



cv::Mat convolutionnXnKernel(const cv::Mat& source ,cv::Mat temp){
    int winsize=temp.rows;
    int center=winsize/2;
    if(winsize%2==0) {std::cout<<"kernel size error!"<<std::endl;exit(-1);}
    cv::Mat result;
    cv::Mat grayImg;
    if (source.channels() == 3) {
        cv::cvtColor(source, grayImg, CV_BGR2GRAY);
    }
    else {
        source.copyTo(grayImg);
    }
    double** t;
    Mat2arraynXn(temp,t);
    /*for(int i=0;i<winsize;i++){
        for(int j=0;j<winsize;j++)
            cout<<t[i][j]<<" ";
        cout<<endl;
    }*/


    //std::cout<<"begin convolution Mat"<<std::endl;
    result.create(grayImg.rows-2*center,grayImg.cols-2*center,grayImg.type());
    cv::Mat re_d;
    re_d.create(grayImg.rows-2*center,grayImg.cols-2*center,CV_64FC1);

    //omp_set_num_threads(CPU_COREs);
//#pragma omp parallel for
    for(int r=center;r<grayImg.rows-center;r++){
        //std::cout<<"r:"<<r<<std::endl;
        uchar** roi;
        roi=new uchar*[winsize];

        for(int i=0;i<winsize;i++){
            roi[i]=grayImg.ptr(r+i-center);
        }
        double* pdst = re_d.ptr<double>(r-center);
        for(int l=center;l<grayImg.cols-center;l++){
            double re=0;
            for(int i=0;i<winsize;i++){
                for(int j=0;j<winsize;j++){
                    re+=roi[i][l+j-center]*t[i][j];
                }
            }
            //std::cout<<"res:"<<re<<std::endl;
            //if(re<0.0000001)
            //   re=0;
            pdst[l-center]=re;
        }
    }
    //std::cout<<"finish!"<<std::endl;
    FreenXnarray(t,temp.rows)
    return re_d;
}




cv::Mat convolution(const cv::Mat& source ,double t[3][3]){
    cv::Mat result;
    cv::Mat grayImg;
    if (source.channels() == 3) {
        cv::cvtColor(source, grayImg, CV_BGR2GRAY);
    }
    else {
        source.copyTo(grayImg);
    }
    //std::cout<<"begin convolution double sizeof(double)"<<sizeof(double)<<std::endl;
    result.create(grayImg.rows-2,grayImg.cols-2,CV_64FC1);
    //omp_set_num_threads(CPU_COREs);
//#pragma omp parallel for
    for(int r=1;r<grayImg.rows-1;r++){
        //std::cout<<"r:"<<r<<std::endl;
        const double* prev = grayImg.ptr<double>(r - 1);
        const double* curr = grayImg.ptr<double>(r);
        const double* next = grayImg.ptr<double>(r + 1);
        double* pdst = result.ptr<double>(r-1);
        for(int l=1;l<grayImg.cols-1;l++){
            pdst[l-1]=prev[l-1]*t[0][0]+prev[l]*t[0][1]+prev[l+1]*t[0][2]+\
                      curr[l-1]*t[1][0]+curr[l]*t[1][1]+curr[l+1]*t[1][2]+\
                      next[l-1]*t[2][0]+next[l]*t[2][1]+next[l+1]*t[2][2];
            //std::cout<<pdst[l-1]<<"  ";
        }
    }
    //std::cout<<"finish double[][] convolution"<<std::endl;
    return result;
}


void getGuassCannyKernel(cv::Mat& Kx,cv::Mat& Ky,double sigma){
    double** gK;
    int winsize=(int)(sigma*4);
    if(winsize%2==0)winsize++;
    MellocnXnDouble(gK,winsize)
    double sigma_2 = sigma * sigma;
    double sum = 0;
    //std::cout<<"begin calculate Kernel"<<std::endl;
    omp_set_num_threads(CPU_COREs);
#pragma omp parallel for
    for(int i=0;i<winsize;i++){
        for(int j=0;j<winsize;j++){
            gK[i][j]=exp(-((i-2)*(i-2)+(j-2)*(j-2))/(2*sigma_2))/(2*PI*sigma_2);
            sum+=gK[i][j];
        }
    }
    omp_set_num_threads(CPU_COREs);
#pragma omp parallel for
    for(int i=0;i<winsize;i++){
        for(int j=0;j<winsize;j++){
            gK[i][j]/=sum;
        }
    }
    double Cx[3][3]={{-1,0,1},{-2,0,2},{-1,0,1}};
    double Cy[3][3]={{-1,-2,-1},{0,0,0},{1,2,1}};
    //std::cout<<"begin strans to Mat"<<std::endl;
    cv::Mat K;
    arraynXn2Mat_freeArray(K,gK,winsize);
    //std::cout<<"begin convolution Canny Kernel"<<std::endl;
    Ky=convolution(K,Cy);
    Kx=convolution(K,Cx);
}

void normMat(const cv::Mat& src,cv::Mat& dst){
    dst.create(src.rows,src.cols,CV_8UC1);
    double mx=0,mn=0;
    for(int r=1;r<src.rows-1;r++){
        const double* curr = src.ptr<double>(r);
        for(int l=1;l<src.cols-1;l++){
            if(curr[l]<mn) mn=curr[l];
            if(curr[l]>mx) mx=curr[l];
        }
    }
    for(int r=1;r<src.rows-1;r++){
        const double* curr = src.ptr<double>(r);
        uchar * pdst = dst.ptr(r);
        for(int l=1;l<src.cols-1;l++){
            pdst[l]=(uchar)(((curr[l]-mn)/(mx-mn))*255);
        }
    }
}

cv::Mat getCanny(const cv::Mat& src,double sigma){
    //cout<<"run getCanny!"<<endl;
    cv::Mat Kx,Ky;
    getGuassCannyKernel(Kx,Ky,sigma);
    cv::Mat res=convolutionnXnKernel(src,Kx);
    cv::Mat res2=convolutionnXnKernel(src,Ky);
    cv::Mat dst;
    dst.create(res.rows,res.cols,res.type());
    for(int r=0;r<res.rows;r++){
        const double* currx = res.ptr<double>(r);
        const double* curry = res2.ptr<double>(r);
        double * pdst = dst.ptr<double>(r);
        for(int l=0;l<res.cols;l++){
            pdst[l]=sqrt(currx[l]*currx[l]+curry[l]*curry[l]);
            //cout<< pdst[l]<<endl;
        }
    }
    cv::Mat d;
    normMat(dst,d);
    return d;
}


#endif //BREAST_CONCER_DETECTION_CANNY_H

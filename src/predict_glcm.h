//
// Created by dang on 16-8-13.
//

#ifndef BREAST_CONCER_DETECTION_PREDICT_GLCM_H
#define BREAST_CONCER_DETECTION_PREDICT_GLCM_H
#include <fstream>
#include <tbb/concurrent_vector.h>
#include "svm.h"
#include "GLCM.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define max(x,y) (((x)>(y))?(x):(y))
#define min(x,y) (((x)<(y))?(x):(y))

class CancerPredictGlcm{
private:
    vi::GLCM* pGLCM=NULL;
    double Range[1189][2];
    void readImagePaths(std::string txtPathFile,std::vector<std::string>& imgPaths);
    void readTrainSample(const std::vector<std::string>& img_path,double *y,double flag,svm_node **pnodes,int cur);
    void getFeatureVector(cv::Mat source_imgage,std::vector<double>& feature);


    char *line = NULL;
    int max_line_len = 1024;
    double lower=-1.0,upper=1.0,y_lower,y_upper;
    int y_scaling = 0;
    double *feature_max;
    double *feature_min;
    double y_max = -DBL_MAX;
    double y_min = DBL_MAX;
    int max_index=1188;
    int min_index;
    long int num_nonzeros = 0;
    long int new_num_nonzeros = 0;


    void clean_up(FILE *fp_restore, const char* msg)
    {
        //fprintf(stderr,	"%s", msg);
        std::cout<<"error:"<<msg<<std::endl;
        free(line);
        free(feature_max);
        free(feature_min);
        if (fp_restore)
            fclose(fp_restore);
        exit(-1);
    }


    char* readline(FILE *input)
    {
        int len;

        if(fgets(line,max_line_len,input) == NULL)
            return NULL;

        while(strrchr(line,'\n') == NULL)
        {
            max_line_len *= 2;
            line = (char *) realloc(line, max_line_len);
            len = (int) strlen(line);
            if(fgets(line+len,max_line_len-len,input) == NULL)
                break;
        }
        return line;
    }


public:
    CancerPredictGlcm(char* param_file){
        pGLCM=new vi::GLCM(8);

#define SKIP_TARGET\
	while(isspace(*p)) ++p;\
	while(!isspace(*p)) ++p;

#define SKIP_ELEMENT\
	while(*p!=':') ++p;\
	++p;\
	while(isspace(*p)) ++p;\
	while(*p && !isspace(*p)) ++p;

        max_index = 1188;
        min_index = 1;
        FILE* fp_restore;
        fp_restore = fopen(param_file,"r");
        if(!fp_restore){
            std::cout<<"open param file failed!"<<std::endl;
            exit(-1);
        }


        feature_max = (double *)malloc((max_index+1)* sizeof(double));
        feature_min = (double *)malloc((max_index+1)* sizeof(double));

        for(int i=0;i<=max_index;i++)
        {
            feature_max[i]=-DBL_MAX;
            feature_min[i]=DBL_MAX;
        }

        /* fp_restore rewinded in finding max_index */
        int idx, c;
        double fmin, fmax;
        int next_index = 1;
        int i=0;

        std::cout<<"open param file success!"<<std::endl;
        if (fgetc(fp_restore) == 'x')
        {
            //std::cout<<"begin ....!"<<std::endl;
            if(fscanf(fp_restore, "%lf %lf\n", &lower, &upper) != 2)
                clean_up(fp_restore,"ERROR: failed to read scaling parameters\n");
            while(fscanf(fp_restore,"%d %lf %lf\n",&idx,&fmin,&fmax)==3)
            {
                //std::cout<<idx<<":"<<fmin<<":"<<fmax<<std::endl;
                for(i = next_index;i<idx;i++)
                    if(feature_min[i] != feature_max[i]){
                        fprintf(stderr,"WARNING: feature index %d appeared in file -- was not seen in the scaling factor file %s.\n", i,  param_file);
                    }


                feature_min[idx] = fmin;
                feature_max[idx] = fmax;

                next_index = idx + 1;
            }
            //std::cout<<"out loop"<<std::endl;
            for(i=next_index;i<=max_index;i++)
                if(feature_min[i] != feature_max[i])
                    fprintf(stderr,
                            "WARNING: feature index %d appeared in file -- was not seen in the scaling factor file %s.\n",
                            i, param_file);
        }
        fclose(fp_restore);
        for(i=0;i<=max_index;i++){
            Range[i][0]=feature_min[i];
            Range[i][1]=feature_max[i];
        }
        //std::cout<<"dsfsdfds"<<std::endl;

    };
    ~CancerPredictGlcm(){
        delete pGLCM;
    };
    void trainModel(const char* pos_txt_file,const char* neg_txt_file,const char* model_file);
    double predictSample(cv::Mat img,const char* model_file);
    double predictSample(cv::Mat img,svm_model* model);
    void testModel(const char* txt_file,const char* model_file,double target=1);
};

#endif //BREAST_CONCER_DETECTION_PREDICT_GLCM_H

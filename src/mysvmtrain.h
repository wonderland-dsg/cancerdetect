//
// Created by dang on 16-8-14.
//

#ifndef BREAST_CONCER_DETECTION_MYSVMTRAIN_H
#define BREAST_CONCER_DETECTION_MYSVMTRAIN_H

#include <iostream>
#include <string.h>
#include "svm.h"

namespace mytrain{
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

    static char *line = NULL;
    static int max_line_len;
    struct svm_node *x_space;

    static char* readline(FILE *input)
    {
        int len;

        if(fgets(line,max_line_len,input) == NULL)
            return NULL;

        while(strrchr(line,'\n') == NULL)
        {
            max_line_len *= 2;
            line = (char *) realloc(line,max_line_len);
            len = (int) strlen(line);
            if(fgets(line+len,max_line_len-len,input) == NULL)
                break;
        }
        return line;
    }


    void exit_input_error(int line_num)
    {
        fprintf(stderr,"Wrong input format at line %d\n", line_num);
        exit(1);
    }

    // read in a problem (in svmlight format)

    void read_problem(const char *filename,svm_problem& prob,svm_parameter& param)
    {
        int max_index, inst_max_index, i;
        size_t elements, j;
        FILE *fp = fopen(filename,"r");
        char *endptr;
        char *idx, *val, *label;

        if(fp == NULL)
        {
            fprintf(stderr,"can't open input file %s\n",filename);
            exit(1);
        }

        prob.l = 0;
        elements = 0;

        max_line_len = 1024;
        line = Malloc(char,max_line_len);
        while(readline(fp)!=NULL)
        {
            char *p = strtok(line," \t"); // label

            // features
            while(1)
            {
                p = strtok(NULL," \t");
                if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
                    break;
                ++elements;
            }
            ++elements;
            ++prob.l;
        }
        rewind(fp);

        prob.y = Malloc(double,prob.l);
        prob.x = Malloc(struct svm_node *,prob.l);
        x_space = Malloc(struct svm_node,elements);

        max_index = 0;
        j=0;
        for(i=0;i<prob.l;i++)
        {
            inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
            readline(fp);
            prob.x[i] = &x_space[j];
            label = strtok(line," \t\n");
            if(label == NULL) // empty line
                exit_input_error(i+1);

            prob.y[i] = strtod(label,&endptr);
            if(endptr == label || *endptr != '\0')
                exit_input_error(i+1);

            while(1)
            {
                idx = strtok(NULL,":");
                val = strtok(NULL," \t");

                if(val == NULL)
                    break;

                errno = 0;
                x_space[j].index = (int) strtol(idx,&endptr,10);
                if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
                    exit_input_error(i+1);
                else
                    inst_max_index = x_space[j].index;

                errno = 0;
                x_space[j].value = strtod(val,&endptr);
                if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
                    exit_input_error(i+1);

                ++j;
            }

            if(inst_max_index > max_index)
                max_index = inst_max_index;
            x_space[j++].index = -1;
        }

        if(param.gamma == 0 && max_index > 0)
            param.gamma = 1.0/max_index;

        if(param.kernel_type == PRECOMPUTED)
            for(i=0;i<prob.l;i++)
            {
                if (prob.x[i][0].index != 0)
                {
                    fprintf(stderr,"Wrong input format: first column must be 0:sample_serial_number\n");
                    exit(1);
                }
                if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
                {
                    fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
                    exit(1);
                }
            }

        fclose(fp);
    }

    void free_svm_prob(svm_problem mproblem){
        svm_node** pnodes=mproblem.x;
        int nsamples=mproblem.l;
        double* y=mproblem.y;
        for (int i = 0; i < nsamples; ++i) delete[] pnodes[i];
        delete[] pnodes;
        delete[] y;
    }

    void mytrain(char* prob_file,char* model_file,char* test_file){
        svm_parameter param;
        // default values
        param.svm_type =C_SVC ;//;ONE_CLASS(so bad)  NU_SVC(maybe)C_SVC  NU_SVC
        param.kernel_type = LINEAR; //RBF(validation accuracy<0.7 )  LINEAR(<0.65) SIGMOID(<0.7) //SIGMOID(2016-8-10 first run) (second is RBF) third is LINEAR
        param.degree = 3;
        param.gamma = 0;	// 1/num_features
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

        svm_problem mproblem;
        read_problem(prob_file,mproblem,param);

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
        int nsamples=mproblem.l;
        for(int i=0;i<nsamples;i++){
            svm_node* x=mproblem.x[i];
            double predict_label=svm_predict(model,x);
            std::cout<<predict_label;//<<std::endl;
            if(predict_label==mproblem.y[i]){
                success_count++;
            }
        }
        std::cout<<" train accuracy is: "<<(double)success_count/nsamples<<std::endl;
        std::cout << "model has saved" << std::endl;
        svm_problem mproblem_test;
        read_problem(test_file,mproblem_test,param);
        std::cout << "read test data syccess!" << std::endl;
        success_count=0;
        int success_count_pos=0,success_count_neg=0,count_pos=0,count_neg=0;
        nsamples=mproblem_test.l;
        for(int i=0;i<nsamples;i++){
            svm_node* x=mproblem_test.x[i];
            double predict_label=svm_predict(model,x);
            std::cout<<predict_label;//<<std::endl;
            if(mproblem_test.y[i]==1){
                count_pos++;
            }else{
                count_neg++;
            }
            if(predict_label==mproblem_test.y[i]){
                success_count++;
                if(mproblem_test.y[i]==1){
                    success_count_pos++;
                }else{
                    success_count_neg++;
                }
            }
        }
        std::cout<<std::endl<<" test accuracy is: "<<(double)success_count/nsamples<<std::endl;
        std::cout<<" test pos accuracy is: "<<(double)success_count_pos/count_pos<<std::endl;
        std::cout<<" test neg accuracy is: "<<(double)success_count_neg/count_neg<<std::endl;
        svm_free_and_destroy_model(&model);
        svm_destroy_param(&param);
        free_svm_prob(mproblem);
        free_svm_prob(mproblem_test);
    }

    void mypredict(char* prob_file,char* model_file);
    //void mypredict(cv::Mat img,char* model_file){

    };

}

#endif //BREAST_CONCER_DETECTION_MYSVMTRAIN_H

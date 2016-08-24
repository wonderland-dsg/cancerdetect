/***************************************************************************
 *   Copyright (c) Anubhav Rohatgi by 2014
 *   anubhavroh@gmail.com   *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 ***************************************************************************/

#pragma once
#ifndef BREAST_CONCER_DETECTION_GLCM_H
#define BREAST_CONCER_DETECTION_GLCM_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>

namespace vi {

	#define myDEBUG 1

	struct glcmParams
	{
		cv::Mat imgPart;
		cv::Mat glcm_0,glcm_45,glcm_90,glcm_135,glcm_avg;
		cv::Rect rectLoc; // location rectangle reference in the original image -- acts as label
		double tp0,tp45,tp90,tp135; // total pairs
		double entropy;
		double contrast;
		double homogenity;
		double energy;
		double dissimilarity;
		double correlation;
		double clustershade;
		double mean;
		double stddev;
		double mean_actual;
		double stdev_actual;
	};


	//! Gray Level Co-Occurrence Matrix and Haralick's Texture Paramters
	/*!
	  * This class divides the grayscale image into n number of windows of square dimensions.
	  * Then it calculates the GLCM for each window in the directions - 0, 45,90 and135 degrees
	  * and then averages the all 4 directions and calculates the various parameters like
	  * Energy,Entropy,Dissimilarity,Homogeneity, Correlation, Contrast, Cluster Shade, mean and standard deviation of each window in 4 directions.
	  * Ref: http://www.iaeng.org/IJCS/issues_v36/issue_1/IJCS_36_1_09.pdf
	  */
	class GLCM
	{

	private:
		cv::Mat src;
		cv::Mat src_real;
		cv::Mat padded;
		int image_levels;
		int winSize;

		/*!
					 \fn    computeGLCMParams()
					 \brief It computes the Haralick's Parameters for Texture. All the parameters as described in the glcmParams Structure.
							It normalizes and makes the matrices symmetric. It takes the average of the all directions before computing the parameters.

		*/
		void computeGLCMParams();

	public:

		/*!
					 \fn    GLCM(int numLevels)
					 \brief instantiates the object of the class and defines the number of quantization levels.
					 \param	numLevels number of quantization levels.
		*/
		GLCM(int numLevels);

		/*!
					 \fn    ~GLCM(void)
					 \brief Destroys the class objects.
		*/
		~GLCM(void);


		/*!
					 \fn    setImage(cv::Mat& input)
					 \brief Sets the grayscale image to be used for GLCM computation. It normalizes the image to the set number of levels.
					 \param input The input grayscale single channel CV_8UC1 image.
		*/
		void setImage(const cv::Mat input);


		/*!
					 \fn    splitToWindows(int win_size);
					 \brief splits the grayscale image into the entered window size.
							The window size should be a whole number and a multiple of 2. preferably powers of 2.
							The image size if not a power of 2 then it is padded with the reflecting values.
							This function also fills the _glcmBlocks concurrent vector with the initialized structure glcmParams.
					 \param win_size Size of the window
		*/
		void splitToWindows(int win_size); /*size should be a square size*/

		/*!
					 \fn    calcGLCM()
					 \brief Calculates the Gray Level Occurrence Matrix in directions 0 ,45, 90, 135 degrees.
							It also calls the private member function which computes the Haralick's Parameters for Texture.
		*/
		void calcGLCM();

		/*Variables*/
		tbb::concurrent_vector<glcmParams> _glcmBlocks;
		cv::Mat padded_real;
	};

}//end of namespace vi


#endif
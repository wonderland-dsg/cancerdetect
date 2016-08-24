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

#define myMatDiv(M,s) {cv::divide(s,M,M);cv::divide(1,M,M);}
#define myMatAdd(M,N) {cv::add(N,M,N);}

#include "GLCM.h"
namespace vi {

GLCM::GLCM(int numLevels)
{
	if(numLevels < 4)
			image_levels = 4;
	else
			image_levels = numLevels;
		winSize= 10;
}


GLCM::~GLCM(void)
{
	src.release();
	src_real.release();
	padded.release();
	padded_real.release();
	_glcmBlocks.clear();
}


void GLCM::setImage(const cv::Mat input)
{
	_glcmBlocks.clear();
	this->src_real = input.clone();
	cv::normalize(input,src,0.0,image_levels,CV_MINMAX);
	src.convertTo(src,CV_64FC1);
}


void GLCM::splitToWindows(int win_size) //size should be a square size
{
	winSize = win_size;

	//divide into 10 x 10 blocks>> implises the image dimensions should be perfect multiple of 10 or whatever the size is given by user;
	cv::copyMakeBorder(src,padded,0,win_size - (src.rows % win_size <= 0?win_size:(src.rows % win_size) ),0,win_size - ((src.cols % win_size <=0)?win_size:(src.cols % win_size) ),cv::BORDER_REFLECT101,cv::Scalar::all(0));
	cv::copyMakeBorder(src_real,padded_real,0,win_size - (src.rows % win_size <= 0?win_size:(src.rows % win_size) ),0,win_size - ((src.cols % win_size <=0)?win_size:(src.cols % win_size) ),cv::BORDER_REFLECT101,cv::Scalar::all(0));

	int numBlocks = (padded.rows * padded.cols) / (win_size * win_size);
		
	if(myDEBUG)
	{
		std::cout<< "\n\n padded = "<<padded.size();
		std::cout<< "\n\n winSize = "<<win_size;
		std::cout<< "\n\n Num of blocks = "<<numBlocks<<"\n";
		
	}
	cv::Mat glBlank;
	glBlank= cv::Mat::zeros(image_levels,image_levels,CV_64FC1);

	for(int i = 0; i < padded.cols; i+=win_size) //traversing from top to bottom and left to right
		for(int j = 0; j < padded.rows; j+=win_size)
		{
			cv::Rect area = cv::Rect(i,j,win_size,win_size);
				
			glcmParams x1;

			x1.contrast = 0.0;
			x1.energy = 0.0;
			x1.entropy = 0.0;
			x1.homogenity = 0.0;
			x1.dissimilarity = 0.0;
			x1.correlation = 0.0;
			x1.clustershade = 0.0;
			x1.mean = 0.0;
			x1.stddev = 0.0;
			x1.tp0 = x1.tp45 = x1.tp90 = x1.tp135 = 0.0;

			x1.glcm_0 = glBlank.clone();
			x1.glcm_45 = glBlank.clone();
			x1.glcm_90 = glBlank.clone();
			x1.glcm_135 = glBlank.clone();
			x1.rectLoc = area;
			x1.imgPart = padded(area);
			x1.glcm_avg = glBlank.clone();

			//mean and stdev of actual
			cv::Scalar mean,stdev;
			cv::meanStdDev(padded_real(area),mean,stdev);

			x1.mean_actual = mean[0];
			x1.stdev_actual = stdev[0];

			_glcmBlocks.push_back(x1);
			/*cv::rectangle(padded,area,cv::Scalar::all(125));
			cv::imshow("rectsw",padded);
			cv::waitKey(0);	*/
		}

		//free the memory
		glBlank.release();
}



void GLCM::calcGLCM()
{		
	//parallel_for for the number of blocks
	tbb::parallel_for(tbb::blocked_range<size_t>(0,_glcmBlocks.size()),[=](const tbb::blocked_range<size_t> &bl) {
	for(size_t k = bl.begin(); k!=bl.end();k++)
	{
		////parallel for for all 4 directions/angles 0,45,90 and 135
		for(int angle = 0; angle <4 ; angle++)
			{
				//Direction Setter --- 
				int ioffset = 0;
				int joffset = 1;

				for(int i = 0; i < image_levels; i++)//glcm size
					{
						for(int j =0 ; j < image_levels; j++)
						{
							for(int p = 0; p < winSize; p++)
							{
								for(int q = 0; q < winSize; q++)
								{
									int x = p;
									int y = q;

										if(angle == 0) // main glcm loop for 0 and 315 degrees
										{
												ioffset = 0;
												joffset = 1;
												x = p + ioffset;
												y = q + joffset;
											
											if(x < winSize && y < winSize)
												if(_glcmBlocks[k].imgPart.at<double>(p,q) == i && _glcmBlocks[k].imgPart.at<double>(x,y)==j)
															_glcmBlocks[k].glcm_0.at<double>(i,j) +=1.0;
													

										} // end if angles 0

										else if(angle == 1) // 45 degrees
										{
												int x = q - 1;
												int y = p + 1;
														
												if(x < 0 || y <0)
													continue;
												else
												{

													if(x < winSize && y < winSize)
																if(_glcmBlocks[k].imgPart.at<double>(q,p) == i && _glcmBlocks[k].imgPart.at<double>(x,y)==j)
																			_glcmBlocks[k].glcm_45.at<double>(i,j) +=1.0;										
												}

										}// end if angles 45 

										else if (angle == 2) // 90 degrees
										{

												int x = q - 1; //c
												int y = p ; // r
													
												if(x < 0 || y <0)
														continue;
												else
												{
													if(x < winSize && y < winSize)
															if(_glcmBlocks[k].imgPart.at<double>(q,p) == i && _glcmBlocks[k].imgPart.at<double>(x,y)==j)
																_glcmBlocks[k].glcm_90.at<double>(i,j) +=1.0;
													}

										}// end if angles 90


										else if (angle == 3) // 135 degrees
										{

												int x = q - 1; //c
												int y = p - 1; // r
													
												if(x < 0 || y <0)
														continue;
												else
												{
													if(x < winSize && y < winSize)
															if(_glcmBlocks[k].imgPart.at<double>(q,p) == i && _glcmBlocks[k].imgPart.at<double>(x,y)==j)
																_glcmBlocks[k].glcm_135.at<double>(i,j) +=1.0;
													}

										}// end if angles 135
									}
							}
						}
					}
			}//for loop angles

		}//functional for for tbb

	});//end for tbb for


	//Compute the parameters and populate the structure elements
	computeGLCMParams();
}


void GLCM::computeGLCMParams()
{
	//compute sum of elements and make the matrix symmetric and normalize them
	tbb::parallel_for(tbb::blocked_range<size_t>(0,_glcmBlocks.size()),[=](const tbb::blocked_range<size_t> &bl) {

	for(size_t k = bl.begin(); k!=bl.end();k++)
	{
		//make the matrices symmetric

		//_glcmBlocks[k].glcm_0 = _glcmBlocks[k].glcm_0 +_glcmBlocks[k].glcm_0.t();
		cv::add(_glcmBlocks[k].glcm_0,_glcmBlocks[k].glcm_0.t(),_glcmBlocks[k].glcm_0);
		cv::add(_glcmBlocks[k].glcm_0,_glcmBlocks[k].glcm_45.t(),_glcmBlocks[k].glcm_45);
		cv::add(_glcmBlocks[k].glcm_0,_glcmBlocks[k].glcm_90.t(),_glcmBlocks[k].glcm_90);
		cv::add(_glcmBlocks[k].glcm_0,_glcmBlocks[k].glcm_135.t(),_glcmBlocks[k].glcm_135);
		//_glcmBlocks[k].glcm_45 = _glcmBlocks[k].glcm_45 + _glcmBlocks[k].glcm_45.t();
		//_glcmBlocks[k].glcm_90 = _glcmBlocks[k].glcm_90 + _glcmBlocks[k].glcm_90.t();
		//_glcmBlocks[k].glcm_135 = _glcmBlocks[k].glcm_135 + _glcmBlocks[k].glcm_135.t();
		


		//sum of elements of all the 5 directions
			
		cv::Scalar s1 = cv::sum(_glcmBlocks[k].glcm_0);
		_glcmBlocks[k].tp0 = ceil(s1[0]);

		cv::Scalar s2 = cv::sum(_glcmBlocks[k].glcm_45);
		_glcmBlocks[k].tp45 = ceil(s2[0]);

		cv::Scalar s3 = cv::sum(_glcmBlocks[k].glcm_90);
		_glcmBlocks[k].tp90 = ceil(s3[0]);

		cv::Scalar s4 = cv::sum(_glcmBlocks[k].glcm_135);
		_glcmBlocks[k].tp135 = ceil(s4[0]);

		//normalize weights

		//cv::divide(_glcmBlocks[k].tp0,_glcmBlocks[k].glcm_0,_glcmBlocks[k].glcm_0);
		//cv::divide(1,_glcmBlocks[k].glcm_0,_glcmBlocks[k].glcm_0);
		myMatDiv(_glcmBlocks[k].glcm_0,_glcmBlocks[k].tp0);
		myMatDiv(_glcmBlocks[k].glcm_45,_glcmBlocks[k].tp45);
		myMatDiv(_glcmBlocks[k].glcm_90,_glcmBlocks[k].tp90);
		myMatDiv(_glcmBlocks[k].glcm_135,_glcmBlocks[k].tp135);
		//_glcmBlocks[k].glcm_0   /= _glcmBlocks[k].tp0;
		//_glcmBlocks[k].glcm_45  /= _glcmBlocks[k].tp45;
		//_glcmBlocks[k].glcm_90  /= _glcmBlocks[k].tp90;
		//_glcmBlocks[k].glcm_135  /= _glcmBlocks[k].tp135;


		//take avg of all 4 directions
		cv::Mat temp;
		temp=cv::Mat::zeros(_glcmBlocks[k].glcm_0.size(),_glcmBlocks[k].glcm_0.type());
		myMatAdd(_glcmBlocks[k].glcm_0,temp);
		myMatAdd(_glcmBlocks[k].glcm_45,temp);
		myMatAdd(_glcmBlocks[k].glcm_90,temp);
		myMatAdd(_glcmBlocks[k].glcm_135,temp);
		myMatDiv(temp,4.0)
		_glcmBlocks[k].glcm_avg=temp;
		//temp=_glcmBlocks[k].glcm_0 + _glcmBlocks[k].glcm_45 + _glcmBlocks[k].glcm_90 + _glcmBlocks[k].glcm_135;
		//_glcmBlocks[k].glcm_avg = (temp) / 4.0;


		//calculate mean and stdev for the average glcm Matrix
		cv::Scalar mean, stdev;
		cv::meanStdDev(_glcmBlocks[k].glcm_avg,mean,stdev);
		_glcmBlocks[k].mean = mean[0];
		_glcmBlocks[k].stddev = stdev[0];


		/*calculate all the parameters */
						
			for(int i =0; i < image_levels; i++)
			{
				for(int j=0; j < image_levels; j++)
					{
						 
						if (_glcmBlocks[k].glcm_avg.at<double>(i,j) >= 0.0)
						{
							//Homogeneity 
							_glcmBlocks[k].homogenity += _glcmBlocks[k].glcm_avg.at<double>(i,j) / (1.0 + (abs(i - j))) ;	

							//Angular Second Moment or Energy
							_glcmBlocks[k].energy += pow((double)_glcmBlocks[k].glcm_avg.at<double>(i,j),2);
							 
							//Entropy							 
							_glcmBlocks[k].entropy += (_glcmBlocks[k].glcm_avg.at<double>(i,j) * (-1.0 * (_glcmBlocks[k].glcm_avg.at<double>(i,j) == 0.0)?0.0:log(_glcmBlocks[k].glcm_avg.at<double>(i,j)) ) );

							//Dissimilarity
							_glcmBlocks[k].dissimilarity += (_glcmBlocks[k].glcm_avg.at<double>(i,j) * abs(i-j));

							//Contrast or Inertia
							_glcmBlocks[k].contrast += (_glcmBlocks[k].glcm_avg.at<double>(i,j) * (i-j) * (i-j));

							//Correlation
							_glcmBlocks[k].correlation += (i*j) * (_glcmBlocks[k].glcm_avg.at<double>(i,j) - (mean[0]*mean[0]));

							//Cluster Shade
							_glcmBlocks[k].clustershade += pow(((i - mean[0]) + (j - mean[0])),3) *  _glcmBlocks[k].glcm_avg.at<double>(i,j);

						}
					}
			}

			
			//correlation continued from reduced set due to symmetricity
			_glcmBlocks[k].correlation /= (stdev[0] * stdev[0]);
	}
	});
}
}//end of namespace vi
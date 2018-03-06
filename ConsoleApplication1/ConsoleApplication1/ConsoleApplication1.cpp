// ConsoleApplication1.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <cv.h>
#include <highgui.h>

using namespace std;
using namespace cv;

int _tmain(int argc, _TCHAR* argv[])
{
	CvCapture *capture = cvCreateFileCapture("D:\\XJL\\test\\003.mp4");
	Mat img;
	IplImage  *mframe, *current, *frg, *test;
	int *fg, *bg_bw, *rank_ind;
	double *w, *mean, *sd, *u_diff, *rank;
	int C, M, sd_init, i, j, k, m, rand_temp = 0, rank_ind_temp = 0, min_index = 0, x = 0, y = 0, counter_frame = 0;
	double D, alph, thresh, p, temp;
	CvRNG state;
	int match, height, width;
	mframe = cvQueryFrame(capture);
	if (mframe==NULL){
		return -1;
	}
	frg = cvCreateImage(cvSize(mframe->width, mframe->height), IPL_DEPTH_8U, 1);
	current = cvCreateImage(cvSize(mframe->width, mframe->height), IPL_DEPTH_8U, 1);
	test = cvCreateImage(cvSize(mframe->width, mframe->height), IPL_DEPTH_8U, 1);

	C = 4;                      //number of gaussian components (typically 3-5)  
	M = 4;                      //number of background components  
	sd_init = 6;                //initial standard deviation (for new components) var = 36 in paper  
	alph = 0.01;                //learning rate (between 0 and 1) (from paper 0.01)  
	D = 2.5;                    //positive deviation threshold  
	thresh = 0.25;              //foreground threshold (0.25 or 0.75 in paper)  
	p = alph / (1 / C);         //initial p variable (used to update mean and sd)  

	height = current->height; width = current->widthStep;

	fg = (int *)malloc(sizeof(int)*width*height);                   //foreground array  
	bg_bw = (int *)malloc(sizeof(int)*width*height);                //background array  
	rank = (double *)malloc(sizeof(double)* 1 * C);                    //rank of components (w/sd)  
	w = (double *)malloc(sizeof(double)*width*height*C);            //weights array  
	mean = (double *)malloc(sizeof(double)*width*height*C);         //pixel means  
	sd = (double *)malloc(sizeof(double)*width*height*C);           //pixel standard deviations  
	u_diff = (double *)malloc(sizeof(double)*width*height*C);       //difference of each pixel from mean  

	for (i = 0; i<height; i++)
	{
		for (j = 0; j<width; j++)
		{
			for (k = 0; k<C; k++)
			{
				mean[i*width*C + j*C + k] = cvRandReal(&state) * 255;
				w[i*width*C + j*C + k] = (double)1 / C;
				sd[i*width*C + j*C + k] = sd_init;
			}
		}
	}

	while (1){
		rank_ind = (int *)malloc(sizeof(int)*C);
		cvCvtColor(mframe, current, CV_BGR2GRAY);
		// calculate difference of pixel values from mean  
		for (i = 0; i<height; i++)
		{
			for (j = 0; j<width; j++)
			{
				for (m = 0; m<C; m++)
				{
					u_diff[i*width*C + j*C + m] = abs((uchar)current->imageData[i*width + j] - mean[i*width*C + j*C + m]);
				}
			}
		}
		//update gaussian components for each pixel  
		for (i = 0; i<height; i++)
		{
			for (j = 0; j<width; j++)
			{
				match = 0;
				temp = 0;
				for (k = 0; k<C; k++)
				{
					if (abs(u_diff[i*width*C + j*C + k]) <= D*sd[i*width*C + j*C + k])      //pixel matches component  
					{
						match = 1;                                                 // variable to signal component match  

						//update weights, mean, sd, p  
						w[i*width*C + j*C + k] = (1 - alph)*w[i*width*C + j*C + k] + alph;
						p = alph / w[i*width*C + j*C + k];
						mean[i*width*C + j*C + k] = (1 - p)*mean[i*width*C + j*C + k] + p*(uchar)current->imageData[i*width + j];
						sd[i*width*C + j*C + k] = sqrt((1 - p)*(sd[i*width*C + j*C + k] * sd[i*width*C + j*C + k]) + p*(pow((uchar)current->imageData[i*width + j] - mean[i*width*C + j*C + k], 2)));
					}
					else{
						w[i*width*C + j*C + k] = (1 - alph)*w[i*width*C + j*C + k];           // weight slighly decreases  
					}
					temp += w[i*width*C + j*C + k];
				}

				for (k = 0; k<C; k++)
				{
					w[i*width*C + j*C + k] = w[i*width*C + j*C + k] / temp;
				}

				temp = w[i*width*C + j*C];
				bg_bw[i*width + j] = 0;
				for (k = 0; k<C; k++)
				{
					bg_bw[i*width + j] = bg_bw[i*width + j] + mean[i*width*C + j*C + k] * w[i*width*C + j*C + k];
					if (w[i*width*C + j*C + k] <= temp)
					{
						min_index = k;
						temp = w[i*width*C + j*C + k];
					}
					rank_ind[k] = k;
				}

				test->imageData[i*width + j] = (uchar)bg_bw[i*width + j];

				//if no components match, create new component  
				if (match == 0)
				{
					mean[i*width*C + j*C + min_index] = (uchar)current->imageData[i*width + j];
					//printf("%d ",(uchar)bg->imageData[i*width+j]);  
					sd[i*width*C + j*C + min_index] = sd_init;
				}
				for (k = 0; k<C; k++)
				{
					rank[k] = w[i*width*C + j*C + k] / sd[i*width*C + j*C + k];
					//printf("%f ",w[i*width*C+j*C+k]);  
				}

				//sort rank values  
				for (k = 1; k<C; k++)
				{
					for (m = 0; m<k; m++)
					{
						if (rank[k] > rank[m])
						{
							//swap max values  
							rand_temp = rank[m];
							rank[m] = rank[k];
							rank[k] = rand_temp;

							//swap max index values  
							rank_ind_temp = rank_ind[m];
							rank_ind[m] = rank_ind[k];
							rank_ind[k] = rank_ind_temp;
						}
					}
				}

				//calculate foreground  
				match = 0; k = 0;
				//frg->imageData[i*width+j]=0;  
				while ((match == 0) && (k<M)){
					if (w[i*width*C + j*C + rank_ind[k]] >= thresh)
					if (abs(u_diff[i*width*C + j*C + rank_ind[k]]) <= D*sd[i*width*C + j*C + rank_ind[k]]){
						frg->imageData[i*width + j] = 0;
						match = 1;
					}
					else
						frg->imageData[i*width + j] = (uchar)current->imageData[i*width + j];
					k = k + 1;
				}
			}
		}

		mframe = cvQueryFrame(capture);
		cvShowImage("fore", frg);
		cvShowImage("back", test);
		char s = cvWaitKey(33);
		if (s == 27) break;
		free(rank_ind);
	}

	free(fg); free(w); free(mean); free(sd); free(u_diff); free(rank);
	cvNamedWindow("back", 0);
	cvNamedWindow("fore", 0);
	cvReleaseCapture(&capture);
	cvDestroyWindow("fore");
	cvDestroyWindow("back");
	return 0;
}


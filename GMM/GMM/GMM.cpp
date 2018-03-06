// GMM.cpp : 定义控制台应用程序的入口点。
//
#include<cv.h>
#include <highgui.h>
#include "stdafx.h"
using namespace std;
using namespace cv;

int _tmain(int argc, _TCHAR* argv[])
{
	IplImage *test;
	test = cvLoadImage("D:\XJL\test\001.png");
	cvNamedWindow("test_demo", 1);
	cvShowImage("test_demo", test);
	cvWaitKey(0);
	cvDestroyWindow("test_demo");
	cvReleaseImage(&test);  // 释放内存
	return 0;
}


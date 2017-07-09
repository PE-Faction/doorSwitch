/*
    ����һ��ͼƬ��������ɫ��
*/
/*#include<opencv2/opencv.hpp>
#include "cv.h"
#include "highgui.h"
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include "iostream"

using namespace std;
using namespace cv;

int main()
{
	int iLowH = 120;
	int iHighH = 179;

	int iLowS = 0;
	int iHighS = 255;

	int iLowV = 0;
	int iHighV = 255;

	namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"

												//Create trackbars in "Control" window
	cvCreateTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
	cvCreateTrackbar("HighH", "Control", &iHighH, 179);

	cvCreateTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
	cvCreateTrackbar("HighS", "Control", &iHighS, 255);

	cvCreateTrackbar("LowV", "Control", &iLowV, 255); //Value (0 - 255)
	cvCreateTrackbar("HighV", "Control", &iHighV, 255);


	Mat imgOriginal = imread("door4.jpg");
	double fScale = 0.2;//����ϵ��  
	Size dsize = Size(imgOriginal.cols*fScale, imgOriginal.rows*fScale);
	resize(imgOriginal, imgOriginal, dsize);


	Mat imgHSV;
	vector<Mat> hsvSplit;
	cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

												  //��Ϊ���Ƕ�ȡ���ǲ�ɫͼ��ֱ��ͼ���⻯��Ҫ��HSV�ռ���
	split(imgHSV, hsvSplit);
	equalizeHist(hsvSplit[2], hsvSplit[2]);
	merge(hsvSplit, imgHSV);
	Mat imgThresholded;

	while (1)
	{
		inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image

																									  //������ (ȥ��һЩ���)
		Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
		morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element);

		//�ղ��� (����һЩ��ͨ��)
		morphologyEx(imgThresholded, imgThresholded, MORPH_CLOSE, element);

		imshow("Thresholded Image", imgThresholded); //show the thresholded image
		imshow("Original", imgOriginal); //show the original image
		waitKey(27);
	}


	waitKey(0);
}*/
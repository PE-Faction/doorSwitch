 /*
         ����������¼����鿴ͼƬÿһ�����ص�HSVֵ
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

//��ȡÿһ�����ص�HSVֵ
IplImage* pImg, *imgRGB, *imgHSV;
int flags = 0;
CvPoint pt;
CvScalar s = { 0.0 }, ss = { 0.0 };

void on_mouse(int event, int x, int y, int flags, void* param)
{

if (!imgRGB)
return;

switch (event)
{
case CV_EVENT_LBUTTONDOWN:
{
s = cvGet2D(imgRGB, y, x);
printf("(%d,%d)����RGBֵ�ֱ��ǣ�B = %f,G = %f, R = %f \n", x, y, s.val[0], s.val[1], s.val[2]);
ss = cvGet2D(imgHSV, y, x);
printf("(%d,%d)����RGBֵ�ֱ��ǣ�H = %f,S = %f, V = %f \n\n", x, y, ss.val[0], ss.val[1], ss.val[2]);
}
break;
}
}

int main(int argc, char** argv)
{
imgRGB = cvLoadImage("12.jpg", 1);
imgHSV = cvCreateImage(cvGetSize(imgRGB), 8, 3);
cvNamedWindow("imgRGB", 2);
cvSetMouseCallback("imgRGB", on_mouse, 0);
cvShowImage("imgRGB", imgRGB); //��ʾͼ��
cvCvtColor(imgRGB, imgHSV, CV_RGB2HSV);


cvWaitKey(); //�ȴ�����
cvDestroyWindow("imgRGB");//���ٴ���
cvReleaseImage(&imgRGB); //�ͷ�ͼ��
return 0;
}
*/
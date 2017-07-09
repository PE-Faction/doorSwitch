
/*
   ������Ƶ�е�ɫ�飬�������ſ��ص��ж�
*/
#include<opencv2/opencv.hpp>
#include "cv.h"
#include "highgui.h"
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include "iostream"
#include <math.h>

using namespace std;
using namespace cv;


/*
���վ��Ȼ�
1����ȡԴͼI��ƽ���Ҷȣ�����¼rows��cols��
2������һ����С����ΪN*M�����飬���ÿ���ƽ��ֵ���õ��ӿ�����Ⱦ���D��
3���þ���D��ÿ��Ԫ�ؼ�ȥԴͼ��ƽ���Ҷȣ��õ��ӿ�����Ȳ�ֵ����E��
4����˫������ֵ����������E��ֵ����Դͼһ����С�����ȷֲ�����R��
5���õ��������ͼ��result=I-R��
*/

void unevenLightCompensate(Mat &image, int blockSize)
{
	if (image.channels() == 3) cvtColor(image, image, 7);
	double average = mean(image)[0];
	int rows_new = ceil(double(image.rows) / double(blockSize));
	int cols_new = ceil(double(image.cols) / double(blockSize));
	Mat blockImage;
	blockImage = Mat::zeros(rows_new, cols_new, CV_32FC1);
	for (int i = 0; i < rows_new; i++)
	{
		for (int j = 0; j < cols_new; j++)
		{
			int rowmin = i*blockSize;
			int rowmax = (i + 1)*blockSize;
			if (rowmax > image.rows) rowmax = image.rows;
			int colmin = j*blockSize;
			int colmax = (j + 1)*blockSize;
			if (colmax > image.cols) colmax = image.cols;
			Mat imageROI = image(Range(rowmin, rowmax), Range(colmin, colmax));
			double temaver = mean(imageROI)[0];
			blockImage.at<float>(i, j) = temaver;
		}
	}
	blockImage = blockImage - average;
	Mat blockImage2;
	resize(blockImage, blockImage2, image.size(), (0, 0), (0, 0), INTER_CUBIC);
	Mat image2;
	image.convertTo(image2, CV_32FC1);
	Mat dst = image2 - blockImage2;
	dst.convertTo(image, CV_8UC1);
}


//��˹������ģ
/*BackgroundSubtractorMOG  mog;
void computForeground(const cv::Mat & in, cv::Mat & out)
{
	// ���±�����������ǰ��
	mog(in, out, 0.01);
	// ��ͼ��ȡ��
	//cv::threshold(out, out, 128, 255, cv::THRESH_BINARY_INV);
}
*/

//��ȡͼ��Ĵ�ֱֱ��ͼ
void   changeHistogramImage(Mat &blackFrame)
{
   //���㴹ֱͶӰ
  int *colheight = new int[blackFrame.cols];
	//������븳��ֵΪ0����������޷���������
	memset(colheight, 0, blackFrame.cols * 4);
	int value;
	for (int i = 0; i < blackFrame.rows; i++)
	{
		for (int j = 0; j < blackFrame.cols; j++)
		{
			value = blackFrame.at<uchar>(i, j);
			if (value == 255)
				colheight[j]++;

		}
	}
	Mat  histogramImage(blackFrame.rows, blackFrame.cols, CV_8UC1);
	for (int i = 0; i < blackFrame.rows; i++)
	{
		for (int j = 0; j < blackFrame.cols; j++)
		{
			value = 0;
			histogramImage.at<uchar>(i, j) = value;

		}
	}
	for (int i = 0; i < blackFrame.cols; i++)
	{
		for (int j = 0; j < colheight[i]; j++)
		{
			value = 255;
			histogramImage.at<uchar>(j, i) = value;
		}
	}

	imshow("��ֱͶӰ", histogramImage);
}



//Ѱ��ͼ�������,���õ���������
int  cache[7];   //����7֡ͼ�����洢ÿһ֡��������Ϣ
Point2f  cacheLeft[7];//����Ϊ2ʱ��һ������������
Point2f  cacheRight[7];//����Ϊ2ʱ����һ������������
Point2f  cacheCenter[7];//����Ϊ1ʱ����������
int time=1;//������������֤�����ͼƬ֡��Ϊ7֡
int time2 = 0;//������¼���������
int temp1 = 0;
int temp2 = 0;
int temp3 = 0;
int status[1000];//��¼ÿ�����ڵ��жϽ��  0Ϊ�ر� 1Ϊ�� 2Ϊ���ڴ�  3Ϊ���ڹر�
void centroid(Mat& img)
{
	int thresh = 30;
	Mat cannyOutput;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;


	//canny���ӱ�Ե���
	Canny(img, cannyOutput, thresh, thresh * 3, 3);
	//imshow("canny", cannyOutput);

	//��������
	findContours(cannyOutput,contours,hierarchy, RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	
	//����������  ������
	vector<Moments> mu(contours.size());
	for (int i = 0; i < contours.size(); i++)
		mu[i] = moments(contours[i], false);

	//��������������
	vector<Point2f> mc(contours.size());
	for (int i = 0; i < contours.size(); i++)
		mc[i] = Point2d(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);

	//���������������Ĳ���ʾ����
	Mat  drawing = Mat::zeros(cannyOutput.size(),CV_8UC3);
	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(255,0,0);
		drawContours(drawing,contours,i,color,2,8, hierarchy, 0, Point());
		circle(drawing, mc[i], 5, Scalar(0, 0, 255), -1, 8, 0);
		rectangle(drawing, boundingRect(contours.at(i)), cvScalar(0, 255, 0));
		char tam[100];
		sprintf(tam, "(%0.0f,%0.0f)", mc[i].x, mc[i].y);
		putText(drawing, tam, Point(mc[i].x, mc[i].y), FONT_HERSHEY_SIMPLEX, 0.4, cvScalar(255, 0, 255), 1);
		
	}
	imshow("Contours", drawing);





	//======================================================�������״̬����
	// printf("%d\n", contours.size());   //��ʾ�����ĸ���
	cache[time] = contours.size();   //����ͼƬ������
	
	//��¼����Ϊ1��Ϊ2ʱ����������
	if (cache[time] == 2) 
	{
		cacheLeft[temp1].x = mc[0].x;
		cacheLeft[temp1].y = mc[0].y;
		cacheRight[temp1].x = mc[1].x;
		cacheRight[temp1].y = mc[1].y;
		temp1++;
	}
	else if (cache[time] == 1)
	{
		cacheCenter[temp2].x = mc[0].x;
		cacheCenter[temp2].y = mc[0].y;
		temp2++;
	}


	//�������ͼƬ��7��ʱ���������״̬�ж�����
	int a = 0,b = 0,c = 0;
	if (time==7)
	{
		if (time2 == 0)//�Թ���һ�����ڣ��������
		{
			time2++;
			time = 1;
			return;
		}

		for (int i = 1; i <= time; i++)//ͳ��ÿ��������7֡ͼ���� ����Ϊ0,1,2�ĸ��Ե�ͼ��֡��
		{
			if (cache[i] == 0)
				a++;
			else if (cache[i] == 1)
				b++;
			else if (cache[i] == 2)
				c++;
		}
	
		if (a > b&&a > c)     //��������Ϊ0��ͼ��ռ���Ա���ʱ���϶�����Ϊ��״̬���������������еĵ���״̬���浽status������
		{
			system("cls");
			printf("����ĿǰΪ��״̬\n");
			status[temp3] = 1;
			temp3++;
		}
		else if (b > a&&b > c)//��������Ϊ1��ͼ��ռ���Ա�����һ���϶�Ϊ���ݴ��ڹر�״̬��������Ƶ�������Ϣ�����⴦��
		{
			/*
			������Ƶ��������Ӱ�ڵ������ܻ����ֻ��һ��������������Դ˸������һ������
			�����һ֡��x������һ��ʼ�������������������趨����ֵ���ó����ڵ���״̬���ǹر�
			���������ڴ򿪻��߹رգ����������֮�У���Ƕ����һ��������Ƶ����ʱ����֣��������Ļ�
			��Ϊ�������ƶ��������෴�ķ���仯���������ڴ�ʹ��״̬���飬���֮ǰ�ĵ���״̬�����ж�Ŀǰ�ĵ���״̬
			ͬʱ��������������Ը����жϵ��ݵĿ����߹أ�һ������
			*/
			if (abs(cacheCenter[temp2 - 1].x - cacheCenter[0].x) > (float)5)//�ж��˵��������ڱ仯״̬
			{
				if (status[temp3 - 1] == 2 && status[temp3 - 2] == 2)
				{
					system("cls");
					printf("�������ڹر�");
					status[temp3] = 2;
					temp3++;

				}
				else
				{
				status[temp3] = 3;
				temp3++;
				system("cls");
				printf("�������ڴ�\n");
				}

			}
			
			else
			{
				status[temp3] = 0;
				temp3++;
				system("cls");
				printf("����ĿǰΪ�ر�״̬\n");
			}
			
			temp2 = 0;
		}
		else if (c > a&&c > b)//��������Ϊ2��ͼ��ռ���Ա��������ǿ�֪�����������ڱ仯״̬��������Ŀ��ػ�Ҫ�ֱ��ж�
		{
			if (temp1 < 3)   //����������Ϊ2��ͼ������3֡����ֻ��ǰһ֡����һ֡������֮֡�䣬����������ʲô�仯�����ݴ��жϵ��ݿ���
			{

				if (abs(cacheRight[temp1 - 1].x - cacheLeft[temp1 - 1].x) > abs(cacheRight[0].x - cacheLeft[0].x))
				{
					//printf("���һ֡�ľ����%0.0f\n", abs(cacheRight[temp1 - 1].x - cacheLeft[temp1 - 1].x));
					//printf("��һ֡�ľ����%0.0f\n", abs(cacheRight[0].x - cacheLeft[0].x));
					status[temp3] = 3;
					temp3++;
					system("cls");
					printf("�������ڴ�\n");
				}
				else if (abs(cacheRight[temp1 - 1].x - cacheLeft[temp1 - 1].x) < abs(cacheRight[0].x - cacheLeft[0].x))
				{
					//printf("���һ֡�ľ����%0.0f\n", abs(cacheRight[temp1 - 1].x - cacheLeft[temp1 - 1].x));
					//printf("��һ֡�ľ����%0.0f\n", abs(cacheRight[0].x - cacheLeft[0].x));
					status[temp3] = 2;
					temp3++;
					system("cls");
					printf("�������ڹر�\n");
			
				}
			}
			else   //����֡������3֡�ģ���ȡ�����֡���һ֡�ľ���ıȽϡ�ͬʱ����Ų����жϣ������ʼ�����ݷ��أ��м�ͬ�����ܳ��ֹرյ���
				   //�ڴ����������ͬ�����ж�
			{
		      if ((abs(cacheRight[temp1 - 1].x - cacheLeft[temp1 - 1].x) < abs(cacheRight[0].x - cacheLeft[0].x)) &&
			      (abs(cacheRight[temp1 - 2].x - cacheLeft[temp1 - 2].x) < abs(cacheRight[0].x - cacheLeft[0].x)) )

		     {
			  //printf("���һ֡�ľ����%0.0f\n", abs(cacheRight[temp1 - 1].x - cacheLeft[temp1 - 1].x));
			 // printf("����֡�ľ����%0.0f\n", abs(cacheRight[temp1 - 2].x - cacheLeft[temp1 - 2].x));
			  //printf("��һ֡�ľ����%0.0f\n", abs(cacheRight[0].x - cacheLeft[0].x));
			  status[temp3] = 2;
			  temp3++;
			  system("cls");
			  printf("�������ڹر�\n");
		     }

			  else	if ((abs(cacheRight[temp1 - 1].x - cacheLeft[temp1 - 1].x) > abs(cacheRight[0].x - cacheLeft[0].x))&&
					(abs(cacheRight[temp1 - 2].x - cacheLeft[temp1 - 2].x) > abs(cacheRight[0].x - cacheLeft[0].x)))
				{
					//printf("���һ֡�ľ����%0.0f\n", abs(cacheRight[temp1 - 1].x - cacheLeft[temp1 - 1].x));
					//printf("����֡�ľ����%0.0f\n", abs(cacheRight[temp1 - 2].x - cacheLeft[temp1 - 2].x));
					//printf("��һ֡�ľ����%0.0f\n", abs(cacheRight[0].x - cacheLeft[0].x));
					if(status[temp3-1]==2&&status[temp3-2]==2)
					{
						system("cls");
						printf("�������ڹر�");
						status[temp3] = 2;
						temp3++;

					}
					else
					{
						status[temp3] = 3;
						temp3++;
						system("cls");
						printf("�������ڴ�\n");
					}
				}
			  else
			  {
				  temp1 = 0;
				  time = 1;
				  return;
			  }
			
		    }
				
			temp1 = 0;
		}

		time = 0;//����ͼƬ���㣬Ϊ��һ����������ʼ��

	 }
	time++;
	if (temp3 == 100)//��temp3 100�����ᣬҲ���ǵ���״̬���棬100�����ڳ�ʼ��һ��
		temp3 =0;


}




int main()
{

	int temp = 0;
	//����HSV�Ľ�����
	int iLowH = 92;
	int iHighH = 100;
	int iLowS = 80;
	int iHighS = 180;
	int iLowV = 0;
	int iHighV = 255;
	namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control
	cvCreateTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
	cvCreateTrackbar("HighH", "Control", &iHighH, 179);
	cvCreateTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
	cvCreateTrackbar("HighS", "Control", &iHighS, 255);
	cvCreateTrackbar("LowV", "Control", &iLowV, 255); //Value (0 - 255)
	cvCreateTrackbar("HighV", "Control", &iHighV, 255);



	//������Ƶ�ĵ��룬��ͨ��HSV����ɫ�������Ȧ��
	VideoCapture cap("close.avi"); //capture the video from web cam

	if (!cap.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the web cam" << endl;
		return -1;
	}

	while (true)
	{
		
	
		if (temp % 5 != 0)
		{
			temp++;
			continue;
		}
		temp++;
		Mat imgOriginal;
		bool bSuccess = cap.read(imgOriginal); // read a new frame from video

		if (!bSuccess) //if not success, break loop
		{
			cout << "Cannot read a frame from video stream" << endl;
			break;
		}


		double fScale = 0.5;//����ϵ��  
		Size dsize = Size(imgOriginal.cols*fScale, imgOriginal.rows*fScale);
		resize(imgOriginal, imgOriginal, dsize);
		

		Mat imgHSV;
		
		cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

		vector<Mat> hsvSplit;										  //��Ϊ���Ƕ�ȡ���ǲ�ɫͼ��ֱ��ͼ���⻯��Ҫ��HSV�ռ���
		hsvSplit.resize(3);
		split(imgHSV, hsvSplit);
		equalizeHist(hsvSplit[2], hsvSplit[2]);
		merge(hsvSplit, imgHSV);
		Mat imgThresholded;

		inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image

																										  
		Mat element = getStructuringElement(MORPH_RECT, Size(5, 5)); //������ (ȥ��һЩ���)
		morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element);

		//�ղ��� (����һЩ��ͨ��)
		morphologyEx(imgThresholded, imgThresholded, MORPH_CLOSE, element);

		//computForeground(imgThresholded, imgThresholded);
		centroid(imgThresholded);
		imshow("Thresholded Image", imgThresholded); //show the thresholded image	
		imshow("Original", imgOriginal); //show the original image
		

		char key = (char)waitKey(300);
		if (key == 27)
			break;
   }
	waitKey(0);
	return 0;
}
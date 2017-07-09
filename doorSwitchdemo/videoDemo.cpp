
/*
   查找视频中的色块，并进行门开关的判断
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
光照均匀化
1、求取源图I的平均灰度，并记录rows和cols；
2、按照一定大小，分为N*M个方块，求出每块的平均值，得到子块的亮度矩阵D；
3、用矩阵D的每个元素减去源图的平均灰度，得到子块的亮度差值矩阵E；
4、用双立方差值法，将矩阵E差值成与源图一样大小的亮度分布矩阵R；
5、得到矫正后的图像result=I-R；
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


//高斯背景建模
/*BackgroundSubtractorMOG  mog;
void computForeground(const cv::Mat & in, cv::Mat & out)
{
	// 更新背景，并返回前景
	mog(in, out, 0.01);
	// 对图像取反
	//cv::threshold(out, out, 128, 255, cv::THRESH_BINARY_INV);
}
*/

//求取图像的垂直直方图
void   changeHistogramImage(Mat &blackFrame)
{
   //计算垂直投影
  int *colheight = new int[blackFrame.cols];
	//数组必须赋初值为0，否则出错，无法遍历数组
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

	imshow("垂直投影", histogramImage);
}



//寻找图像的质心,并得到质心坐标
int  cache[7];   //缓存7帧图像来存储每一帧的轮廓信息
Point2f  cacheLeft[7];//轮廓为2时的一个轮廓的坐标
Point2f  cacheRight[7];//轮廓为2时的另一个轮廓的坐标
Point2f  cacheCenter[7];//轮廓为1时的轮廓坐标
int time=1;//用来计数，保证缓存的图片帧数为7帧
int time2 = 0;//用来记录计算的周期
int temp1 = 0;
int temp2 = 0;
int temp3 = 0;
int status[1000];//记录每个周期的判断结果  0为关闭 1为打开 2为正在打开  3为正在关闭
void centroid(Mat& img)
{
	int thresh = 30;
	Mat cannyOutput;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;


	//canny算子边缘检测
	Canny(img, cannyOutput, thresh, thresh * 3, 3);
	//imshow("canny", cannyOutput);

	//查找轮廓
	findContours(cannyOutput,contours,hierarchy, RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	
	//计算轮廓距  ？？？
	vector<Moments> mu(contours.size());
	for (int i = 0; i < contours.size(); i++)
		mu[i] = moments(contours[i], false);

	//计算轮廓的质心
	vector<Point2f> mc(contours.size());
	for (int i = 0; i < contours.size(); i++)
		mc[i] = Point2d(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);

	//画出轮廓及其质心并显示出来
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





	//======================================================进项电梯状态分类
	// printf("%d\n", contours.size());   //显示轮廓的个数
	cache[time] = contours.size();   //缓存图片轮廓数
	
	//记录轮廓为1和为2时的轮廓坐标
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


	//当缓存的图片到7个时，进入电梯状态判断周期
	int a = 0,b = 0,c = 0;
	if (time==7)
	{
		if (time2 == 0)//略过第一个周期，减少误差
		{
			time2++;
			time = 1;
			return;
		}

		for (int i = 1; i <= time; i++)//统计每个周期中7帧图像中 轮廓为0,1,2的各自的图像帧数
		{
			if (cache[i] == 0)
				a++;
			else if (cache[i] == 1)
				b++;
			else if (cache[i] == 2)
				c++;
		}
	
		if (a > b&&a > c)     //对于轮廓为0的图像占绝对比例时，认定电梯为打开状态，并将整个周期中的电梯状态保存到status数组中
		{
			system("cls");
			printf("电梯目前为打开状态\n");
			status[temp3] = 1;
			temp3++;
		}
		else if (b > a&&b > c)//对于轮廓为1的图像占绝对比例，一般认定为电梯处于关闭状态，对于视频中误差信息做特殊处理
		{
			/*
			对于视频中由于阴影遮挡，可能会出现只有一个轮廓的情况，对此跟踪这个一个轮廓
			将最后一帧的x坐标与一开始的坐标相减，如果大于设定的阈值，得出现在电梯状态不是关闭
			而可能正在打开或者关闭，在这个误差处理之中，又嵌套了一个误差处理，视频中有时会出现，轮廓质心会
			因为轮廓的移动，而向相反的方向变化，产生误差，在此使用状态数组，结合之前的电梯状态分析判断目前的电梯状态
			同时这个方法，还可以辅助判断电梯的开或者关，一举两得
			*/
			if (abs(cacheCenter[temp2 - 1].x - cacheCenter[0].x) > (float)5)//判断了电梯正处于变化状态
			{
				if (status[temp3 - 1] == 2 && status[temp3 - 2] == 2)
				{
					system("cls");
					printf("电梯正在关闭");
					status[temp3] = 2;
					temp3++;

				}
				else
				{
				status[temp3] = 3;
				temp3++;
				system("cls");
				printf("电梯正在打开\n");
				}

			}
			
			else
			{
				status[temp3] = 0;
				temp3++;
				system("cls");
				printf("电梯目前为关闭状态\n");
			}
			
			temp2 = 0;
		}
		else if (c > a&&c > b)//对于轮廓为2的图像占绝对比例，我们可知两个轮廓处于变化状态，但具体的开关还要分别判断
		{
			if (temp1 < 3)   //如果获得轮廓为2的图像少于3帧，就只能前一帧减后一帧，看两帧之间，轮廓距离有什么变化，依据此判断电梯开关
			{

				if (abs(cacheRight[temp1 - 1].x - cacheLeft[temp1 - 1].x) > abs(cacheRight[0].x - cacheLeft[0].x))
				{
					//printf("最后一帧的距离差%0.0f\n", abs(cacheRight[temp1 - 1].x - cacheLeft[temp1 - 1].x));
					//printf("第一帧的距离差%0.0f\n", abs(cacheRight[0].x - cacheLeft[0].x));
					status[temp3] = 3;
					temp3++;
					system("cls");
					printf("电梯正在打开\n");
				}
				else if (abs(cacheRight[temp1 - 1].x - cacheLeft[temp1 - 1].x) < abs(cacheRight[0].x - cacheLeft[0].x))
				{
					//printf("最后一帧的距离差%0.0f\n", abs(cacheRight[temp1 - 1].x - cacheLeft[temp1 - 1].x));
					//printf("第一帧的距离差%0.0f\n", abs(cacheRight[0].x - cacheLeft[0].x));
					status[temp3] = 2;
					temp3++;
					system("cls");
					printf("电梯正在关闭\n");
			
				}
			}
			else   //对于帧数大于3帧的，就取最后两帧与第一帧的距离的比较。同时满足才采纳判断，否则初始化数据返回，中间同样可能出现关闭的误差，
				   //在此因此与上述同样的判断
			{
		      if ((abs(cacheRight[temp1 - 1].x - cacheLeft[temp1 - 1].x) < abs(cacheRight[0].x - cacheLeft[0].x)) &&
			      (abs(cacheRight[temp1 - 2].x - cacheLeft[temp1 - 2].x) < abs(cacheRight[0].x - cacheLeft[0].x)) )

		     {
			  //printf("最后一帧的距离差%0.0f\n", abs(cacheRight[temp1 - 1].x - cacheLeft[temp1 - 1].x));
			 // printf("最后二帧的距离差%0.0f\n", abs(cacheRight[temp1 - 2].x - cacheLeft[temp1 - 2].x));
			  //printf("第一帧的距离差%0.0f\n", abs(cacheRight[0].x - cacheLeft[0].x));
			  status[temp3] = 2;
			  temp3++;
			  system("cls");
			  printf("电梯正在关闭\n");
		     }

			  else	if ((abs(cacheRight[temp1 - 1].x - cacheLeft[temp1 - 1].x) > abs(cacheRight[0].x - cacheLeft[0].x))&&
					(abs(cacheRight[temp1 - 2].x - cacheLeft[temp1 - 2].x) > abs(cacheRight[0].x - cacheLeft[0].x)))
				{
					//printf("最后一帧的距离差%0.0f\n", abs(cacheRight[temp1 - 1].x - cacheLeft[temp1 - 1].x));
					//printf("最后二帧的距离差%0.0f\n", abs(cacheRight[temp1 - 2].x - cacheLeft[temp1 - 2].x));
					//printf("第一帧的距离差%0.0f\n", abs(cacheRight[0].x - cacheLeft[0].x));
					if(status[temp3-1]==2&&status[temp3-2]==2)
					{
						system("cls");
						printf("电梯正在关闭");
						status[temp3] = 2;
						temp3++;

					}
					else
					{
						status[temp3] = 3;
						temp3++;
						system("cls");
						printf("电梯正在打开\n");
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

		time = 0;//缓存图片清零，为下一个周期做初始化

	 }
	time++;
	if (temp3 == 100)//给temp3 100个机会，也就是电梯状态保存，100个周期初始化一次
		temp3 =0;


}




int main()
{

	int temp = 0;
	//创建HSV的进度条
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



	//进行视频的导入，并通过HSV进行色块区域的圈定
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


		double fScale = 0.5;//缩放系数  
		Size dsize = Size(imgOriginal.cols*fScale, imgOriginal.rows*fScale);
		resize(imgOriginal, imgOriginal, dsize);
		

		Mat imgHSV;
		
		cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

		vector<Mat> hsvSplit;										  //因为我们读取的是彩色图，直方图均衡化需要在HSV空间做
		hsvSplit.resize(3);
		split(imgHSV, hsvSplit);
		equalizeHist(hsvSplit[2], hsvSplit[2]);
		merge(hsvSplit, imgHSV);
		Mat imgThresholded;

		inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image

																										  
		Mat element = getStructuringElement(MORPH_RECT, Size(5, 5)); //开操作 (去除一些噪点)
		morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element);

		//闭操作 (连接一些连通域)
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
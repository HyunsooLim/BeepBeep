#pragma once
#include <sstream>
#include <string>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <stdlib.h> 
#include <vector> 
#include <cstdlib>

using namespace cv;
using namespace std;

int H_MIN = 0;
int H_MAX = 255;
int S_MIN = 0;
int S_MAX = 255;
int V_MIN = 0;
int V_MAX = 255;
//default capture width and height
const int FRAME_WIDTH = 672;
const int FRAME_HEIGHT = 376;
//max number of objects to be detected in frame
const int MAX_NUM_OBJECTS = 50;
//minimum and maximum object area
const int MIN_OBJECT_AREA = 20 * 20;
const int MAX_OBJECT_AREA = FRAME_HEIGHT*FRAME_WIDTH / 1.5;
//names that will appear at the top of each window
const string windowName = "Original Image";
const string windowName1 = "HSV Image";
const string windowName2 = "Thresholded Image";
const string windowName3 = "After Morphological Operations";
const string trackbarWindowName = "Trackbars";
void on_trackbar(int, void*)
{//This function gets called whenever a
 // trackbar position is changed
}

using namespace cv::xfeatures2d;
void quickSort(int* data, int start, int end);
void Sort(int* data, int n);
Mat colorchange(Mat camerafeed, int height, int width);

class xy {
public:
	int cnt;
	double sum;
	double mean;
};

Mat ROIcreate(Mat threshold, Mat camerafeed) 
{
	int x_histogram[672] = { 0, }, x_hist_max = 0, max_x = 0;
	int y_histogram[376] = { 0, }, y_hist_max = 0, max_y = 0;
	//width
	int min_left = 672;
	int max_right = 0;

	//height
	int min_up = 376;
	int max_down = 0;
	
	int width = 0, height = 0;

	//x축 경계 구하기
	for (width = 0; width < 672; width++) {
		for (height = 0; height < 376; height++) {
			if (threshold.at<uchar>(height, width) == 255) {
				x_histogram[width]++; //histogram에 각 픽셀마다 흰색인지 판단 후 흰색이면 카운트
				if (x_histogram[width] > x_hist_max) {
					x_hist_max = x_histogram[width];
					max_x = width;
				}
			}
		}
	}

	for (int i = max_x; x_histogram[i] != 0; i--) {
		if (i >= 0) {
			min_left = i;
		}
		else {
			break;
		}
	}
	for (int i = max_x; x_histogram[i] != 0; i++) {
		if (i < 672) {
			max_right = i;
		}
		else {
			break;
		}
	}

	//y축 경계 구하기
	for (height = 0; height < 376; height++) {
		for (width = 0; width < 672; width++) {
			if (threshold.at<uchar>(height, width) == 255) {
				y_histogram[height]++; //histogram에 각 픽셀마다 흰색인지 판단 후 흰색이면 카운트
				if (y_histogram[height] > y_hist_max) {
					y_hist_max = y_histogram[height];
					max_y = height;
				}
			}
		}
	}

	for (int i = max_y; y_histogram[i] != 0; i--) {
		if (i >= 0) {
			min_up = i;
		}
		else {
			break;
		}
	}
	for (int i = max_y; y_histogram[i] != 0; i++) {
		if (i < 376) {
			max_down = i;
		}
		else {
			break;
		}
	}

	printf("(%d,%d)", min_left, max_right);
	printf("(%d,%d)\n", min_up, max_down);


	for (height = 0; height < min_up; height++) {
		for (width = 0; width < 672; width++) {
			camerafeed = colorchange(camerafeed, height, width);
		}
	}

	for (height = max_down; height < 376; height++) {
		for (width = 0; width < 672; width++) {
			camerafeed = colorchange(camerafeed, height, width);
		}
	}

	for (width = 0; width < min_left; width++) {
		for (height = 0; height < 376; height++) {
			camerafeed = colorchange(camerafeed, height, width);
		}
	}

	for (width = max_right; width < 672; width++) {
		for (height = 0; height < 376; height++) {
			camerafeed = colorchange(camerafeed, height, width);
		}
	}

	return camerafeed;
}

Mat colorchange(Mat camerafeed, int height, int width)
{
	camerafeed.at<Vec3b>(height, width)[0] = 255;
	camerafeed.at<Vec3b>(height, width)[1] = 255;
	camerafeed.at<Vec3b>(height, width)[2] = 255;
	return camerafeed;
}

void quickSort(int* data, int start, int end) {
	if (start >= end) { // 원소가 1개인 경우 그대로 두기
		return;
	}
	int key = start; // 키는 첫 번째 원소 
	int i = start + 1, j = end, temp;
	while (i <= j) { // 엇갈릴 때까지 반복 
		while (data[i] <= data[key]) { // 키 값보다 큰 값을 만날 때까지
			i++;
		}
		while (data[j] >= data[key] && j > start) {
			// 키 값보다 작은 값을 만날 때까지
			j--;
		}
		if (i > j) { // 현재 엇갈린 상태면 키 값과 교체 
			temp = data[j];
			data[j] = data[key];
			data[key] = temp;
		}
		else { // 엇갈리지 않았다면 i와 j를 교체 
			temp = data[i];
			data[i] = data[j];
			data[j] = temp;
		}
	}
	quickSort(data, start, j - 1);
	quickSort(data, j + 1, end);
}
void Sort(int* data, int n) {
	int i, j;
	for (i = 0; i < n; i++) {
		for (j = n - 1; i < j; j--) {
			if (data[j - 1] > data[j]) {
				int temp = data[j - 1];
				data[j - 1] = data[j];
				data[j] = temp;
			}
		}
	}
}
void createTrackbars() {
	//create window for trackbars


	namedWindow(trackbarWindowName, 0);
	//create memory to store trackbar name on window
	char TrackbarName[50];
	sprintf_s(TrackbarName, "H_MIN", H_MIN);
	sprintf_s(TrackbarName, "H_MAX", H_MAX);
	sprintf_s(TrackbarName, "S_MIN", S_MIN);
	sprintf_s(TrackbarName, "S_MAX", S_MAX);
	sprintf_s(TrackbarName, "V_MIN", V_MIN);
	sprintf_s(TrackbarName, "V_MAX", V_MAX);
	//create trackbars and insert them into window
	//3 parameters are: the address of the variable that is changing when the trackbar is moved(eg.H_LOW),
	//the max value the trackbar can move (eg. H_HIGH), 
	//and the function that is called whenever the trackbar is moved(eg. on_trackbar)
	//                                  ---->    ---->     ---->      
	createTrackbar("H_MIN", trackbarWindowName, &H_MIN, H_MAX, on_trackbar);
	createTrackbar("H_MAX", trackbarWindowName, &H_MAX, H_MAX, on_trackbar);
	createTrackbar("S_MIN", trackbarWindowName, &S_MIN, S_MAX, on_trackbar);
	createTrackbar("S_MAX", trackbarWindowName, &S_MAX, S_MAX, on_trackbar);
	createTrackbar("V_MIN", trackbarWindowName, &V_MIN, V_MAX, on_trackbar);
	createTrackbar("V_MAX", trackbarWindowName, &V_MAX, V_MAX, on_trackbar);


}
string intToString(int number) {


	std::stringstream ss;
	ss << number;
	return ss.str();
}

void drawObject(int x, int y, Mat &frame) {

	circle(frame, Point(x, y), 20, Scalar(0, 255, 0), 2);
	if (y - 25>0)
		line(frame, Point(x, y), Point(x, y - 25), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(x, 0), Scalar(0, 255, 0), 2);
	if (y + 25<FRAME_HEIGHT)
		line(frame, Point(x, y), Point(x, y + 25), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(x, FRAME_HEIGHT), Scalar(0, 255, 0), 2);
	if (x - 25>0)
		line(frame, Point(x, y), Point(x - 25, y), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(0, y), Scalar(0, 255, 0), 2);
	if (x + 25<FRAME_WIDTH)
		line(frame, Point(x, y), Point(x + 25, y), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(FRAME_WIDTH, y), Scalar(0, 255, 0), 2);

	putText(frame, intToString(x) + "," + intToString(y), Point(x, y + 30), 1, 1, Scalar(0, 255, 0), 2);

}
void morphOps(Mat &thresh) {

	Mat erodeElement = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat dilateElement = getStructuringElement(MORPH_RECT, Size(8, 8));

	erode(thresh, thresh, erodeElement);
	erode(thresh, thresh, erodeElement);


	dilate(thresh, thresh, dilateElement);
	dilate(thresh, thresh, dilateElement);


}
void trackFilteredObject(int &x, int &y, Mat threshold, Mat &cameraFeed) {

	Mat temp;
	threshold.copyTo(temp);
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(temp, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
	double refArea = 0;
	bool objectFound = false;
	if (hierarchy.size() > 0) {
		int numObjects = hierarchy.size();
		if (numObjects<MAX_NUM_OBJECTS) {
			for (int index = 0; index >= 0; index = hierarchy[index][0]) {

				Moments moment = moments((cv::Mat)contours[index]);
				double area = moment.m00;

				if (area>MIN_OBJECT_AREA && area<MAX_OBJECT_AREA && area>refArea) {
					x = moment.m10 / area;
					y = moment.m01 / area;
					objectFound = true;
					refArea = area;
				}
				else objectFound = false;


			}
			//let user know you found an object
			if (objectFound == true) {
				putText(cameraFeed, "Tracking Object", Point(0, 50), 2, 1, Scalar(0, 255, 0), 2);
				//draw object location on screen
				drawObject(x, y, cameraFeed);
			}

		}
		else putText(cameraFeed, "TOO MUCH NOISE! ADJUST FILTER", Point(0, 50), 1, 2, Scalar(0, 0, 255), 2);
	}
}
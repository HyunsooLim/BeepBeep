#include "rover.h"

int main(int argc, char* argv[])
{
	bool trackObjects = false;
	bool useMorphOps = false;
	Mat cameraFeed, feed_left, feed_right;
	Mat HSV;
	Mat threshold, threshold_left, threshold_right;
	//Mat filtered_frame;
	int x = 0, y = 0;
	Mat img;
	int num = 0;
	int row = 0, column = 0;

	createTrackbars();

	VideoCapture capture;
	capture.open(1);
	if (!capture.isOpened())
	{
		cout << "not connected" << endl;
		return 0; // 연결실패
	}
	capture.set(CAP_PROP_FRAME_WIDTH, 1344);
	capture.set(CAP_PROP_FRAME_HEIGHT, 376);

	while (1) {
		capture.read(cameraFeed);
		resize(cameraFeed, cameraFeed, Size(1344, 376), 0, 0, INTER_CUBIC);
		cvtColor(cameraFeed, HSV, COLOR_BGR2HSV);
		inRange(HSV, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), threshold);
		if (useMorphOps)
			morphOps(threshold);
		if (trackObjects)
			trackFilteredObject(x, y, threshold, cameraFeed);

		imshow(windowName2, threshold);

		threshold_left = threshold(Range(0, 376), Range(0, 672));
		threshold_right = threshold(Range(0, 376), Range(672, 1344));

		feed_left = cameraFeed(Range(0, 376), Range(0, 672));
		feed_right = cameraFeed(Range(0, 376), Range(672, 1344));
		
		//ROI 생성
		feed_left = ROIcreate(threshold_left, feed_left);
		feed_right = ROIcreate(threshold_right, feed_right);


		//==============================SIFT algorithm==================================
		vector<KeyPoint> keypoints1, keypoints2;
		Mat descriptors1, descriptors2;
		Ptr<Feature2D> ptrFeature2D = SIFT::create(200);

		ptrFeature2D->detectAndCompute(feed_left, noArray(), keypoints1, descriptors1);
		ptrFeature2D->detectAndCompute(feed_right, noArray(), keypoints2, descriptors2);

		BFMatcher matcher(NORM_L1, false);
		//FlannBasedMatcher matcher;
		vector<DMatch> matches, goodMatches;

		matcher.match(descriptors1, descriptors2, matches);

		if (matches.size() == 0)
			continue;

		double mindis, maxdis;
		mindis = matches[0].distance;
		maxdis = matches[0].distance;
		for (int i = 1; i < matches.size(); i++) {
			if (maxdis < matches[i].distance)
				maxdis = matches[i].distance;
			if (mindis > matches[i].distance)
				mindis = matches[i].distance;
		}
		cout << "minDist = " << mindis << endl;
		cout << "maxDist = " << maxdis << endl;
		double fTh = 4 * mindis;
		for (int i = 0; i < matches.size(); i++)
		{
			if (matches[i].distance <= max(fTh, 0.005))
				goodMatches.push_back(matches[i]);
		}
		//
		if (goodMatches.size() == 0)
			continue;

		cout << "goodMatches.size() = " << goodMatches.size() << endl;
		const int b = goodMatches.size();
		int *A = new int[b];

		for (int i = 0; i < goodMatches.size(); i++) {
			A[i] = abs(keypoints1[goodMatches[i].queryIdx].pt.x - keypoints2[goodMatches[i].trainIdx].pt.x);
		}
		quickSort(A, 0, b - 1);
		//Sort(A, b);
		int disparity;   //그냥 중간값
		if (b % 2 == 0)
			disparity = A[b / 2];
		else
			disparity = A[(b + 1) / 2];

		int gcnt = 0;
		int count = 0;
		double sum = 0;
		xy s[10];


		for (int i = 0; i < goodMatches.size(); i++) {   //갯수
			if (abs(A[i] - A[i + 1])< 420) {
				gcnt++;
				sum += A[i];
			}
			else {
				s[count].cnt = gcnt;
				s[count].sum = sum;
				s[count].mean = sum / (double)gcnt;
				gcnt = 0;
				sum = 0;
				count++;//
			}
		}

		int maxcnt = s[0].cnt;
		int maxindex = 0;
		for (int i = 0; i < count; i++) {
			if (maxcnt < s[i].cnt) {
				maxindex = i;
				maxcnt = s[i].cnt;
			}
		}

		int disparity1 = s[maxindex].mean;
		cout << "group cnt : " << count << endl;
		cout << "maxgroup cnt : " << s[maxindex].cnt << endl;

		double focal = 350;
		double depth = (120 * focal / abs(disparity)) / 10.0; //cm
		double depth1 = (120 * focal / abs(disparity1)) / 10.0;
		//double depth = (120 * focal / abs(Total_x)) / 10.0; //cm 
		cout << "[RESULT VALUE] Depth = " << depth << "cm" << endl << endl;
		//cout << "[RESULT VALUE] Depth1 = " << depth1 << "cm" << endl << endl;
		//Total_x = Total_x / goodMatches.size();

		delete[] A;

		Mat imageMatches;
		drawMatches(feed_left, keypoints1, feed_right, keypoints2, goodMatches, imageMatches);
		namedWindow("Matches");
		imshow("Matches", imageMatches);

		if (waitKey(10) == 27) break;
	}
	return 0;
}
#include<opencv2\opencv.hpp>  
#include <iostream>  
#include <stdio.h>  

using namespace std;
using namespace cv;

String face_cascade_name = "haarcascade_frontalface_alt.xml";
String smile_cascade_name = "haarcascade_smile.xml";
CascadeClassifier face_cascade;
CascadeClassifier smile_cascade;
String window_name = "TEST";
String window_name2 = "TEST2";


Mat mask, output;


//
//Mat putMask(Mat src, Point2d center, Size face_size)
//{
//	Mat mask1, src1;
//	resize(mask, mask1, face_size);
//
//	// ROI selection
//	Rect2d roi(center.x - face_size.width / 2, center.y - face_size.width / 2, face_size.width, face_size.width);
//	src(roi).copyTo(src1);
//
//	// to make the white region transparent
//	Mat mask2, m, m1;
//	cvtColor(mask1, mask2, CV_BGR2GRAY);
//	threshold(mask2, mask2, 230, 255, CV_THRESH_BINARY_INV);
//
//	vector<Mat> maskChannels(3), result_mask(3);
//	split(mask1, maskChannels);
//	bitwise_and(maskChannels[0], mask2, result_mask[0]);
//	bitwise_and(maskChannels[1], mask2, result_mask[1]);
//	bitwise_and(maskChannels[2], mask2, result_mask[2]);
//	merge(result_mask, m);         //    imshow("m",m);
//
//	mask2 = 255 - mask2;
//	vector<Mat> srcChannels(3);
//	split(src1, srcChannels);
//	bitwise_and(srcChannels[0], mask2, result_mask[0]);
//	bitwise_and(srcChannels[1], mask2, result_mask[1]);
//	bitwise_and(srcChannels[2], mask2, result_mask[2]);
//	merge(result_mask, m1);        //    imshow("m1",m1);
//
//	addWeighted(m, 1, m1, 1, 0, m1);    //    imshow("m2",m1);
//
//	m1.copyTo(src(roi));
//
//	return src;
//}


int main()
{
	VideoCapture capture;
	Mat frame;
	//Mat frame2;
	Mat mask =imread("5.jpg");
	Mat outputimage;
	Mat imageROI;

	if (!face_cascade.load(face_cascade_name))
	{
		printf("--(!)Error loading face cascade\n");
		return -1;
	};
	if (!smile_cascade.load(smile_cascade_name))
	{
		printf("--(!)Error loading eyes cascade\n");
		return -1;
	};


	capture.open(1);


	if (!capture.isOpened())
	{
		printf("--(!)Error opening video capture\n");
		return -1;
	}

	while (capture.read(frame))
	{
		if (frame.empty())
		{
			printf(" --(!) No captured frame -- Break!");
			break;
		}


		capture >> frame;
		output = frame.clone();
		outputimage = frame.clone();
		std::vector<Rect> faces;
		Mat frame_gray;

		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
		equalizeHist(frame_gray, frame_gray);
		face_cascade.detectMultiScale(frame_gray, faces, 1.05, 8, CASCADE_SCALE_IMAGE);
		Mat faceROI;
		for (size_t i = 0; i < faces.size(); i++)
		{
			rectangle(frame, faces[i], Scalar(255, 0, 0), 2, 8, 0);
			
			faceROI = frame_gray(faces[i]);
			std::vector<Rect> smile;

			//-- In each face, detect smile
			smile_cascade.detectMultiScale(faceROI, smile, 1.1, 55, CASCADE_SCALE_IMAGE);

			for (size_t j = 0; j < smile.size(); j++)
			{
				Rect rect(faces[i].x + smile[j].x, faces[i].y + smile[j].y, smile[j].width, smile[j].height);
				rectangle(frame, rect, Scalar(0, 0, 255), 2, 8, 0);
				putText(frame, string("you are happy!"), Point(20, 20), 0, 1, Scalar(0, 0, 0), 3);
				Point2d center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
				//frame = putMask(frame, center, Size(faces[i].width, faces[i].height));

				//*************************************************************

				Mat mask1, src1;
				resize(mask, mask1, faces[i].size());

				// ROI selection
				//Rect roi(center.x - faces[i].width / 2, center.y - faces[i].width / 2, faces[i].width, faces[i].width);

				//frame2.copyTo(frame, mask1);

				//roi.copyTo(fame(Rect(left, top, src.cols, src.rows)));
				//src(roi).copyTo(src1);

				//// to make the white region transparent
				//Mat mask2, m, m1;
				//cvtColor(mask1, mask2, CV_BGR2GRAY);
				//threshold(mask2, mask2, 230, 255, CV_THRESH_BINARY_INV);

				//vector<Mat> maskChannels(3), result_mask(3);
				//split(mask1, maskChannels);
				//bitwise_and(maskChannels[0], mask2, result_mask[0]);
				//bitwise_and(maskChannels[1], mask2, result_mask[1]);
				//bitwise_and(maskChannels[2], mask2, result_mask[2]);
				//merge(result_mask, m);         //    imshow("m",m);

				//mask2 = 255 - mask2;
				//vector<Mat> srcChannels(3);
				//split(src1, srcChannels);
				//bitwise_and(srcChannels[0], mask2, result_mask[0]);
				//bitwise_and(srcChannels[1], mask2, result_mask[1]);
				//bitwise_and(srcChannels[2], mask2, result_mask[2]);
				//merge(result_mask, m1);        //    imshow("m1",m1);

				//addWeighted(m, 1, m1, 1, 0, m1);    //    imshow("m2",m1);

				//m1.copyTo(src(roi));


				//*************************************************************
				//for (int n = faces[i].x; n <faces[i].x + faces[i].width; n++)
				//{
				//	for (int m = faces[i].y; m < faces[i].y + faces[i].height; m++)
				//	{


				//		for (int l = 0; l < faces[i].width; l++)
				//		{
				//			for (int k = 0; k <  faces[i].height; k++)
				//			{
				//				//frame.at<Vec3b>(Point(i, j)).s

				//				// get pixel
				//				

				//				// ... do something to the color ....

				//				// set pixel
				//				//frame.at<Vec3b>(Point(n, m)) = color;
				//			

				//		/*int intensity = mask1.at<uchar>(k, l);
				//		frame.
				//		frame.at<Vec3b>(Point(n, m))= mask1.*/
				//				Vec3b color = mask1.at<Vec3b>(Point(l, k));
				//		frame.at<Vec3b>(Point(n, m)).val[0] = color.val[0];
				//		frame.at<Vec3b>(Point(n, m)).val[1] = color.val[1];
				//		frame.at<Vec3b>(Point(n, m)).val[2] = color.val[2];

				//			}
				//		}

				//	}
				//}
				
			
				Rect subimage = cv::Rect(0,0, 150,148);
				outputimage = output(subimage);


				//cv::Mat image = cv::imread("./image.png", -1);

				//cv::Rect rect(startColumn, startRow, image.cols, image.rows);
				//cv::Mat subdst = dst(rect);
				//mask1.copyTo(subdst);
				
				
				mask.copyTo(outputimage);

			

				//imageROI = frame(cv::Rect(20,20, 50, 50));

				//addWeighted(imageROI, 0.5, mask, 0.5, 0., imageROI);






			}
		}
		//-- Show what you got  
		namedWindow(window_name, 1);
		imshow(window_name, frame);

		namedWindow(window_name2, 1);
		imshow(window_name2, outputimage);
		waitKey(30);
	}
	int c = waitKey(0);
	if ((char)c == 27) { return 0; }

	return 0;
}


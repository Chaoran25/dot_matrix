#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/features2d/features2d.hpp" 
#include <opencv2/imgproc/imgproc.hpp>
#include "main.h"
#include <fstream>

using namespace std;
using namespace cv;

string haarcascade = "D:\\Master\\dot\\cascade_dot\\cascade.xml";
string LBPcascade = "D:\\Master\\dot\\LBP\\cascade.xml";
string smallhaarcascade = "D:\\Master\\dot\\HaarLike\\cascade.xml";
CascadeClassifier Mycascade;
string window_name = "Detection";
Size minsize(55,12);
Size maxsize(55 * 100, 12 * 100);
// Function
void Border_extention(Mat input, Mat & output,int border) {
	for (int r = 0; r < input.rows; r++) {
		for (int c = 0; c < input.cols; c++) {
			output.at<uchar>(border + r, border + c) = input.at<uchar>(r, c);
		}
	}
	// Up & Down edage
	for (int r = 0; r < border; r++) 
	{
		for (int c = border; c < output.cols - border; c++)
		{
			output.at<uchar>(r, c) = input.at<uchar>(0, c - border);
			output.at<uchar>(r + input.rows + border, c) = input.at<uchar>(input.rows - 1, c - border);
		}
	}
	// left & right edages
	for (int r = 0; r < output.rows; r++)
	{
		for (int c = 0; c < border; c++)
		{
			output.at<uchar>(r, c) = output.at<uchar>(r, border);
			output.at<uchar>(r, c+input.cols+border) = output.at<uchar>(r, border+input.cols-1);
		}
	}

}

void FastPoints(Mat Input,Mat &output,int neighborhood,int n,int t){
	for (int r = 0; r < Input.rows; r++) {
		for (int c = 0; c < Input.cols; c++)
		{
			int bright_point = 0;
			int darker_point = 0;
			for (int k = -neighborhood; k < neighborhood; k++) {
				for (int m = -neighborhood; m < neighborhood; m++) {
					if (Input.at<uchar>(r + k, c + m) - Input.at<uchar>(r, c) > t)
						bright_point++;
					if (Input.at<uchar>(r + k, c + m) - Input.at<uchar>(r, c) < -t)
						darker_point++;
				}
			}
		}
	}
}

//按照X坐标排序
bool rect_rank_x(vector<Rect> &vec_rects) {
	Rect vec_temp;
	for (int l = 1; l < vec_rects.size(); l++) {
		for (int m = vec_rects.size() - 1; m >= l; m--) {
			if (vec_rects[m].x < vec_rects[m - 1].x) {
				vec_temp = vec_rects[m - 1];
				vec_rects[m - 1] = vec_rects[m];
				vec_rects[m] = vec_temp;
			}
		}
	}
	return true;
}
//按照X坐标排序
bool rect_rank_y(vector<Rect> &vec_rects) {
	Rect vec_temp;
	for (int l = 1; l < vec_rects.size(); l++) {
		for (int m = vec_rects.size() - 1; m >= l; m--) {
			if (vec_rects[m].y < vec_rects[m - 1].y) {
				vec_temp = vec_rects[m - 1];
				vec_rects[m - 1] = vec_rects[m];
				vec_rects[m] = vec_temp;
			}
		}
	}
	return true;
}


/*将rect上下合并
* 参数：vec_rects：输入的所有的rect集合；
*      vec_rects_out:输出的上下合并后的所有的rect集合；
*      x_dif：进行上下合并的x差值；y_dif：进行上下合并的y差值；
*      width:进行上下合并的width最大值；height:进行上下合并的height最大值；
width_rect:合并后的rect的width的值大于width_rect为满足条件
*/
bool rect_combine_uplow(vector<Rect> &vec_rects,
	vector<Rect>&vec_rects_out, int x_dif, int y_dif, int width, int height,
	int width_rect) {
	rect_rank_y(vec_rects);
	//将上下部分分裂的，合并
	int num_rect = vec_rects.size();
	for (int j = 0; j < num_rect; j++) {
		if (vec_rects[j].width > 0) {
			Rect r;
			for (int p = 0; p < num_rect; p++) {
				if ((vec_rects[p].width > 0) && (p > j || p < j)) {
					if ((((abs(vec_rects[p].x - vec_rects[j].x) < x_dif)
						|| (abs(
							vec_rects[p].x + vec_rects[p].width
							- vec_rects[j].x
							- vec_rects[j].width) < x_dif))
						&& ((abs(
							vec_rects[p].y
							- (vec_rects[j].y
								+ vec_rects[j].height))
							< y_dif)
							|| (abs(
								vec_rects[j].y
								- (vec_rects[p].y
									+ vec_rects[p].height))
								< y_dif))
						&& (vec_rects[p].height < height)
						&& (vec_rects[j].height < height)
						&& (vec_rects[p].width < width)
						&& (vec_rects[j].width < width))) {

						r.x = min(vec_rects[j].x, vec_rects[p].x);
						r.y = min(vec_rects[j].y, vec_rects[p].y);
						r.width = max(
							vec_rects[p].x + vec_rects[p].width
							- vec_rects[j].x,
							vec_rects[j].x + vec_rects[j].width
							- vec_rects[p].x);
						r.height = max(
							vec_rects[j].y + vec_rects[j].height
							- vec_rects[p].y,
							vec_rects[p].y + vec_rects[p].height
							- vec_rects[j].y);
						if (vec_rects[p].y < vec_rects[j].y) {
							vec_rects[p].width = 0;
							vec_rects[p].x = 0;
							vec_rects[p].height = 0;
							vec_rects[p].y = 0;
							vec_rects[j] = r;
						}
						else {
							vec_rects[j].width = 0;
							vec_rects[j].x = 0;
							vec_rects[j].height = 0;
							vec_rects[j].y = 0;
							vec_rects[p] = r;
						}

					}
				}
			}
		}
	}

	for (int j = 0; j < num_rect; j++) {
		if (vec_rects[j].width > width_rect) {
			vec_rects_out.push_back(vec_rects[j]);
		}
	}
	return true;
}

/*将rect左右合并
* 参数：
* show:输入图像；
* vec_rects：输入的所有的rect集合；
* vec_rects_out:输出的左右合并后的所有的rect集合；
* x_dif：进行左右合并的x差值；y_dif：进行左右合并的y差值；
* width:进行左右合并的width最大值；height:进行左右合并的height最大值；
* rate1：rect的长宽比最小值1;rate2:rect的长宽比最小值2;
* width_rect:合并后的rect的width的值大于width_rect为满足条件
*/
bool rect_combine_leftright(Mat & show, vector<Rect> &vec_rects,
	vector<Rect>&vec_rects_out, int x_dif, int y_dif, int width, int height,
	double rate1, double rate2, int width_rect) {
	int num = vec_rects.size();
	for (int j = 0; j < num - 1; j++) {
		if (vec_rects[j].width > 0) {
			for (int q = j + 1; q < num; q++) {
				if (vec_rects[q].width > 0) {
					Rect r;
					if ((max(vec_rects[q].x - x_dif, 0)
						< min(vec_rects[j].x + vec_rects[j].width,
							show.cols))
						&& ((abs(vec_rects[q].y - vec_rects[j].y) < y_dif)
							|| (abs(
								min(
									vec_rects[q].y
									+ vec_rects[q].height,
									show.rows)
								- min(
									vec_rects[j].y
									+ vec_rects[j].height,
									show.rows)) < y_dif))
						&& (vec_rects[q].width < width)
						&& (vec_rects[j].width < width)
						&& (((vec_rects[q].height
							/ (double)vec_rects[q].width > rate1)
							&& (vec_rects[j].height
								/ (double)vec_rects[j].width
						> rate2))
							|| ((vec_rects[j].height
								/ (double)vec_rects[j].width
									> rate1)
								&& (vec_rects[q].height
									/ (double)vec_rects[q].width
						> rate2)))) {
						if ((vec_rects[j].x + vec_rects[j].width
						> show.cols / 10 * 8.5)
							&& (vec_rects[q].x > show.cols / 10 * 8.5)
							&& abs(vec_rects[j].width - vec_rects[q].width)
							< 4
							&& abs(
								vec_rects[j].height
								- vec_rects[q].height) < 3) {
							;
						}
						else {
							r.x = vec_rects[j].x;
							r.y = min(vec_rects[j].y, vec_rects[q].y);
							r.width = vec_rects[q].x + vec_rects[q].width
								- vec_rects[j].x;
							r.height = max(vec_rects[j].y + vec_rects[j].height,
								vec_rects[q].y + vec_rects[q].height) - r.y;
							vec_rects[q].width = 0;
							vec_rects[q].x = 0;
							vec_rects[j] = r;
						}
					}
				}
			}
		}
	}
	for (int j = 0; j < num; j++) {
		if (vec_rects[j].width > width_rect) {
			vec_rects_out.push_back(vec_rects[j]);
		}
	}
	return true;
}
Mat detectAndDisplay(Mat frame)
{
	std::vector<Rect> dot_matrix;
	std::vector<Rect> dot_matrix2;
	Mat frame_gray(frame.size(), CV_8U);
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	Mycascade.detectMultiScale(frame_gray, dot_matrix, 1.01, 1, 2,minsize,maxsize);
	//groupRectangles(dot_matrix, 1,5);
	for (int i = 0; i < dot_matrix2.size(); i++) {
		rectangle(frame,                   //图像.  
			dot_matrix[i],
			Scalar(255, 0, 0),     //线条颜色 (RGB) 或亮度（灰度图像 ）(grayscale image）  
			1);                   //组成矩形的线条的粗细程度。取负值时（如 CV_FILLED）函数绘制填充了色彩的矩形 
	}
	
	imshow(window_name, frame);
	return frame;
}
// Main
int main()
{
	if (!Mycascade.load(haarcascade)) { printf("[error] 无法加载级联分类器文件！\n");   return -1; }
	Mat Src,Src_resize, Src_detection,Src_detected;
	string Img_Name = "D:\\Master\\dot\\pos\\0002.bmp";
	Src = imread(Img_Name);//图片路径    
	if (!Src.data)
	{
		cout << "无法读取图片 !!!!" << endl;
		system("pause");
		return -1;
	}
	//缩放
	if (Src.cols >= 1000 || Src.rows >= 1000) {
		resize(Src, Src_resize, Size(Src.cols / 2, Src.rows / 2), 0, 0, INTER_LINEAR);
	}
	else {
		Src.copyTo(Src_resize);
	}
	//namedWindow("result", WINDOW_AUTOSIZE);
	imshow("original",Src_resize);
	cvWaitKey(0);
	Src_resize.copyTo(Src_detection);
	Src_detected = detectAndDisplay(Src_detection);
	waitKey(0);

	return 0;
}
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void rotate_arbitrarily_angle(const cv::Mat& src, cv::Mat& dst, float angle)
{
	float radian = (float)(angle / 180.0 * CV_PI);

	//填充图像
	int maxBorder = (int)(std::max(src.cols, src.rows) * 1.414); //即为sqrt(2)*max
	int dx = (maxBorder - src.cols) / 2;
	int dy = (maxBorder - src.rows) / 2;
	cv::copyMakeBorder(src, dst, dy, dy, dx, dx, cv::BORDER_CONSTANT);

	//旋转
	cv::Point2f center((float)(dst.cols / 2), (float)(dst.rows / 2));
	cv::Mat affine_matrix = cv::getRotationMatrix2D(center, angle, 1.0);//求得旋转矩阵
	cv::warpAffine(dst, dst, affine_matrix, dst.size());

	//计算图像旋转之后包含图像的最大的矩形
	float sinVal = abs(sin(radian));
	float cosVal = abs(cos(radian));
	cv::Size targetSize((int)(src.cols * cosVal + src.rows * sinVal),
		(int)(src.cols * sinVal + src.rows * cosVal));

	//剪掉多余边框
	int x = (dst.cols - targetSize.width) / 2;
	int y = (dst.rows - targetSize.height) / 2;
	cv::Rect rect(x, y, targetSize.width, targetSize.height);
	dst = cv::Mat(dst, rect);
}
int main(int argc, char** argv)
{
	//读取图像
	cv::Mat img_1 = cv::imread("Images/L10.png");
	cv::Mat img_2_ = cv::imread("Images/R10.png");
	cv::Mat img_2;

	rotate_arbitrarily_angle(img_2_, img_2, 30);

	//初始化
	std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
	cv::Mat descriptors_1, descriptors_2;
	cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
	cv::Ptr<cv::FeatureDetector> descriptor = cv::ORB::create();
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

	//第一步，检测Oriented FAST 角点位置

	detector->detect(img_1, keypoints_1);
	detector->detect(img_2, keypoints_2);

	//第二部，根据角点位置计算BRIEF描述子
	descriptor->compute(img_1, keypoints_1, descriptors_1);
	descriptor->compute(img_2, keypoints_2, descriptors_2);

	cv::Mat outimg1;
	drawKeypoints(img_1, keypoints_1, outimg1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
	cv::imshow("ORB features", outimg1);

	//第三步：对两幅图像中打BRIEF描述子进行匹配，使用Hamming距离
	std::vector<cv::DMatch> matches;

	matcher->match(descriptors_1, descriptors_2, matches);

	//第四步：匹配点对筛选   计算最小距离和最大距离
	auto min_max = minmax_element(matches.begin(), matches.end(), [](const cv::DMatch& m1, const cv::DMatch& m2) { return m1.distance < m2.distance; });
	double min_dist = min_max.first->distance;
	double max_dist = min_max.second->distance;

	printf("-- Max dist : %lf \n", max_dist);
	printf("-- Min dist : %lf \n", min_dist);

	//当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
	std::vector<cv::DMatch> good_matches;
	for (int i = 0; i < descriptors_1.rows; ++i) {
		if (matches[i].distance <= std::max(2 * min_dist, 30.0)) {
			good_matches.push_back(matches[i]);
		}
	}

	//第五步，绘制匹配结果
	cv::Mat img_match;
	cv::Mat img_goodmatch;
	drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
	drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
	cv::imshow("all matches", img_match);
	cv::imwrite("Images/img_match_1.jpg", img_match);
	cv::imshow("good matches", img_goodmatch);
	cv::imwrite("Images/img_goodmatch_1.jpg", img_goodmatch);
	cv::waitKey(0);

	return 0;
}

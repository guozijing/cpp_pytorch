#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>
#include <time.h>

using namespace std;
using namespace cv;
using namespace torch;

// int main(int argc, const char* argv[]) {
int main () {
	/*if (argc != 4) {
		std::cerr << "usage: super_resolution <path-to-exported-script-module>  <path-to-image> <using-GPU-0:false-1:true>\n";
		return -1;
	}
	const string using_gpu = argv[3];*/
	string model_path = "C:/Users/guojingf/Desktop/py_pro/super_resolution/model_epoch_30.pt";
	// string model_path = "C:/Users/guojingf/Desktop/py_pro/model_epoch_30.pth";
	string img_path = "C:/Users/guojingf/Desktop/py_pro/examples-master/super_resolution/dataset/BSDS300/images/test/16077.jpg";
	Mat image = imread(img_path, 1);
	// Mat image = imread(argv[2], 1);
	Mat channels[3];
	cvtColor(image, image, COLOR_RGB2YCrCb);
	split(image, channels);
	int row = channels[0].rows;
	int col = channels[0].cols;
	Mat output_image(row*3, col*3, CV_8UC1);
	Mat out_Cb, out_Cr;
	Mat images_out;

	// torch::jit::script::Module module = torch::jit::load(argv[1]);
	torch::jit::script::Module module = torch::jit::load(model_path, torch::kCPU);
	/*if (using_gpu == "1")
	{
		module.to(at::kCUDA);	
	}*/
	
	Tensor tensor_image = torch::from_blob(channels[0].data, {1, row, col, 1}, torch::kByte); // [[],[],[]]
	tensor_image = tensor_image.permute({ 0, 3, 1, 2 }); //[[[[],[],[],[]]]]
	tensor_image = tensor_image.to(at::kFloat);
	tensor_image = tensor_image.div(255);
	/*if (using_gpu == "1")
	{
		tensor_image.to(at::kCUDA);
	}*/

	at::Tensor output = module.forward({ tensor_image }).toTensor();
	output = output[0].squeeze().detach();
	output = output.mul(255).clamp(0, 255).to(torch::kU8);
	output = output.to(kCPU);
	
	std::memcpy((void*)output_image.data, output[0].data_ptr(), sizeof(torch::kU8) * output.numel());

	Size ResImgSiz = Size(col*3, row*3);
	cv::resize(channels[1], out_Cb, ResImgSiz, INTER_CUBIC);
	cv::resize(channels[2], out_Cr, ResImgSiz, INTER_CUBIC);
	vector<Mat> images(3);
	images[0] = output_image;
	images[1] = out_Cb;
	images[2] = out_Cr;
	cv::merge(images, images_out);
	cvtColor(images_out, images_out, COLOR_YCrCb2RGB);
	Mat bicubic_out;
	cv::resize(image, bicubic_out, ResImgSiz, INTER_CUBIC);
	cvtColor(bicubic_out, bicubic_out, COLOR_YCrCb2RGB);
	// imwrite("C:/Users/guojingf/Desktop/py_pro/examples-master/super_resolution/dataset/BSDS300/images/test/out_bicubic.png", bicubic_out);
	imwrite("C:/Users/guojingf/Desktop/py_pro/examples-master/super_resolution/dataset/BSDS300/images/test/out_sim.png", images_out);
	cv::imshow("img", images_out);
	cv::waitKey(); // show image 
}

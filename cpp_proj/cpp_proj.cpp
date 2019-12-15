#include <memory>
#include <vector>
#include <string>

//Some magic spells for torch v1.3, see https://github.com/pytorch/pytorch/issues/27568
#ifdef UNICODE
#define UNICODE_TMP UNICODE
#undef UNICODE
#endif
#pragma warning( push )
#pragma warning( disable : 4146 )
#include <torch/script.h>
#pragma warning( pop )
#ifdef UNICODE_TMP
#define UNICODE UNICODE_TMP
#undef UNICODE_TMP
#endif
#include <torch/cuda.h> // need to use torch::cuda::is_available()

#include <opencv2/opencv.hpp>

int main(int argc, const char* argv[])
{
	// Load test image. Don't forget to convert OpenCV's BGR to RGB
	cv::Mat img = cv::imread("../pics/cat.jpg");
	cv::Size target_size(224, 224);
	cv::resize(img, img, target_size);
	switch (img.channels())
	{
	case 4:
		cv::cvtColor(img, img, cv::COLOR_BGRA2RGB);
		break;
	case 3:
		cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
		break;
	default:
		throw new std::runtime_error("incorrect image depth!");
	}

	torch::NoGradGuard no_grad; // same as `with torch.no_grad()` in Python. Don't forget it or don't be surprized that more than expected GPU resources are eaten!
	const bool cuda_is_available = torch::cuda::is_available();

	////////////
	// ResNet34 Classification
	///////////

	// CPU
	{
		torch::jit::script::Module module = torch::jit::load("../resnet34_infer.pth");

		torch::Tensor tensor_img = torch::from_blob(img.data, { img.rows, img.cols, 3 }, torch::kByte);
		at::Tensor output = module.forward( { tensor_img } ).toTensor(); // remember {...} or conscruct a vector of arguments to pass in module.forward() before

		int class_index = output.argmax().item().to<int>();
		auto output_a = output.accessor<float, 1>();
		auto response = output_a[class_index];
		std::cout << "ResNet34 on CPU results for cat.jpg: class_index=" << class_index << " (282 - cat), response=" << response << '\n';
	}

	// GPU
	if (cuda_is_available)
	{
		torch::jit::script::Module module = torch::jit::load("../resnet34_infer_cuda.pth", torch::kCUDA);

		torch::Tensor tensor_img = torch::from_blob(img.data, { img.rows, img.cols, 3 }, torch::kByte).cuda();
		at::Tensor output = module.forward({ tensor_img }).toTensor().cpu();

		int class_index = output.argmax().item().to<int>();
		auto output_a = output.accessor<float, 1>();
		auto response = output_a[class_index];
		std::cout << "ResNet34 on CUDA results for cat.jpg: class_index=" << class_index << " (282 - cat), response=" << response << '\n';
	}
	else
	{
		std::cout << "CUDA is not available" << '\n';
	}

	////////////
	// DeepLabV3 segmentation
	///////////

	// CPU or CPU is selected automatically
	{
		auto device = cuda_is_available ? torch::kCUDA : torch::kCPU;
		std::string device_name = cuda_is_available ? "CUDA" : "CPU";
		torch::jit::script::Module module = torch::jit::load("../deep_lab_v3_infer.pth", device);

		torch::Tensor tensor_img = torch::from_blob(img.data, { img.rows, img.cols, 3 }, torch::kByte).to(device);
		torch::List<torch::Tensor> input_list( { tensor_img } ); // this model accepts not a tensor but a list of tensors
		if (cuda_is_available) // if you have GPU, use it hard! Recognize 20 cats for the price of 1!
			for (int i = 0; i < 19; ++i)
				input_list.push_back( tensor_img );
		auto outputs = module.forward({ input_list }).toTuple(); // This model returns tuple (logits, labels, demo_imgs), not tensor 
		torch::Tensor demo_img = outputs->elements()[2].toTensor().cpu()[0]; // take demo_imgs from tuple, remove batch dim

		cv::Mat prediction(cv::Size(img.cols, img.rows), CV_8UC3, demo_img.data_ptr());
		cv::cvtColor(prediction, prediction, cv::COLOR_BGR2RGB);
		cv::imshow("DeepLabV3 results", prediction);
		std::cout << "DeepLabV3 finished on "<< device_name << ". Select image window and press any key." << '\n';
		cv::waitKey();
	}
}

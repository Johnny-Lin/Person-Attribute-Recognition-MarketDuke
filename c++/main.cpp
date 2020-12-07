#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>


void test_pt(){
    
    //std::shared_ptr<torch::jit::script::Module> module = torch::jit::load("./out.pt");
    torch::jit::Module module = torch::jit::load("../out.pt");

    cv::Mat image = cv::imread("../test_market.jpg");
    cv::resize(image,image, cv::Size(144,288));
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    std::cout << "image shape:" << image.rows <<" " << image.cols <<" " << image.channels() << std::endl;
    torch::Tensor img_tensor = torch::from_blob(image.data, {1, image.rows, image.cols, 3}, torch::kByte);
    img_tensor = img_tensor.permute({0, 3, 1, 2}); // 调整 opencv 矩阵的维度使其和 torch 维度一致
    img_tensor = img_tensor.toType(torch::kFloat);
    img_tensor = img_tensor.div(255);
    img_tensor = img_tensor.to(at::kCUDA);
	//均值归一化
	img_tensor[0][0] = img_tensor[0][0].sub_(0.485).div_(0.229);
	img_tensor[0][1] = img_tensor[0][1].sub_(0.456).div_(0.224);
	img_tensor[0][2] = img_tensor[0][2].sub_(0.406).div_(0.225);

    int i = 1000;
    while (i--)
    {
        torch::Tensor output = module.forward({img_tensor}).toTensor();
    }
    
    torch::Tensor output = module.forward({img_tensor}).toTensor();
    std::cout << "forward done " << std::endl;
    //output = output.to(at::kCPU);
    std::cout << output << std::endl;
    //auto out_select = torch::index_select(output.to(at::kCPU),0,3);
    std::cout << output[0][0].item().toFloat() << std::endl;
    std::cout << output[0][0].item().toFloat() << std::endl;
    std::cout << output[0][4] << std::endl;

    //tensor切片
    

    // 转换成float
    std::vector<float> out_f(30);
    for (int i = 0;i<30;i++)
    {
        // 转化成Float
        out_f[i] = output[0][i].item().toFloat();
        printf("\n   out_f[i]:%d  %f",i,out_f[i]);
    }
    for (int i = 0;i<30;i++)
    {
        printf("\nout float: %d %f",i, out_f[i] );
    }

    printf("\nout float:\n %f", out_f[1] );
    std::cout << "\ncout:\n" << out_f << std::endl;
    //printf( output[0][4] );
    //auto max_result = output.max(1, true);
    //auto max_index = std::get<1>(max_result).item<float>();
    //std::cout << max_index << std::endl;
}


int main() {
//    TorchTest();
    std::cout << "torch::cuda::is_available():"<< torch::cuda::is_available() << std::endl;

    test_pt();

    return 0;
}


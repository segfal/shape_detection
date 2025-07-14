#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

class ShapeRecognizer {
private:
    torch::jit::script::Module model;
    
public:
    ShapeRecognizer(const std::string& model_path) {
        try {
            // Load the TorchScript model
            model = torch::jit::load(model_path);
            model.eval();
            std::cout << "✅ Model loaded successfully from: " << model_path << std::endl;
        } catch (const c10::Error& e) {
            std::cerr << "❌ Error loading model: " << e.msg() << std::endl;
            throw;
        }
    }
    
    std::string predictShape(const cv::Mat& image) {
        try {
            // Preprocess image
            cv::Mat processed = preprocessImage(image);
            
            // Convert to tensor
            torch::Tensor tensor_image = cvMatToTensor(processed);
            
            // Add batch dimension
            tensor_image = tensor_image.unsqueeze(0);
            
            // Run inference
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(tensor_image);
            
            torch::NoGradGuard no_grad;
            auto output = model.forward(inputs).toTensor();
            
            // Get prediction
            auto prediction = output.argmax(1);
            int predicted_class = prediction.item<int>();
            
            return getShapeName(predicted_class);
            
        } catch (const std::exception& e) {
            std::cerr << "❌ Error during prediction: " << e.what() << std::endl;
            return "Unknown";
        }
    }
    
private:
    cv::Mat preprocessImage(const cv::Mat& image) {
        cv::Mat gray, resized, normalized;
        
        // Convert to grayscale if needed
        if (image.channels() == 3) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image.clone();
        }
        
        // Resize to 28x28 (assuming your model expects this size)
        cv::resize(gray, resized, cv::Size(28, 28));
        
        // Normalize to [0, 1]
        resized.convertTo(normalized, CV_32F, 1.0/255.0);
        
        return normalized;
    }
    
    torch::Tensor cvMatToTensor(const cv::Mat& image) {
        // Convert CV_32F Mat to tensor
        torch::Tensor tensor = torch::from_blob(
            image.data, 
            {image.rows, image.cols}, 
            torch::kFloat32
        );
        
        // Add channel dimension (1 for grayscale)
        tensor = tensor.unsqueeze(0);
        
        return tensor;
    }
    
    std::string getShapeName(int class_id) {
        std::vector<std::string> shapes = {"circle", "square", "triangle", "rectangle"};
        if (class_id >= 0 && class_id < shapes.size()) {
            return shapes[class_id];
        }
        return "unknown";
    }
};

int main(int argc, char* argv[]) {
    std::cout << "🔍 Shape Recognizer with PyTorch (LibTorch)" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    // Check command line arguments
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <image_path> [model_path]" << std::endl;
        std::cout << "Example: " << argv[0] << " test_images/circle.png shape_model.pt" << std::endl;
        return 1;
    }
    
    std::string image_path = argv[1];
    std::string model_path = (argc > 2) ? argv[2] : "shape_model.pt";
    
    try {
        // Load image
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "❌ Error: Could not load image from " << image_path << std::endl;
            return 1;
        }
        
        std::cout << "📸 Loaded image: " << image_path << std::endl;
        std::cout << "   Size: " << image.cols << "x" << image.rows << std::endl;
        
        // Initialize shape recognizer
        ShapeRecognizer recognizer(model_path);
        
        // Perform prediction
        std::string predicted_shape = recognizer.predictShape(image);
        
        std::cout << "🎯 Prediction: " << predicted_shape << std::endl;
        
        // Display image with prediction
        cv::putText(image, "Shape: " + predicted_shape, 
                   cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 
                   1.0, cv::Scalar(0, 255, 0), 2);
        
        cv::imshow("Shape Recognition Result", image);
        cv::waitKey(0);
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 
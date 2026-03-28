#include <iostream>
#include <memory>
#include <string>
#include "insider_detector.hpp"

using namespace insider;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <model_path> [threshold]" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    double threshold = 0.5;
    
    if (argc >= 3) {
        threshold = std::stod(argv[2]);
    }
    
    InsiderDetector detector(threshold);
    
    try {
        detector.load_model(model_path);
        std::cout << "Model loaded from " << model_path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return 1;
    }
    
    TradeFeatures features;
    features.return_7d = 0.05;
    features.return_30d = 0.12;
    features.volatility_30d = 0.02;
    features.rsi_14d = 65.0;
    features.macd = 0.5;
    features.macd_signal = 0.3;
    features.volume_ratio_30d = 2.5;
    features.insider_net_activity = 10;
    features.abnormal_return_30d = 0.08;
    features.beta = 1.2;
    
    detector.add_trade_features(features);
    
    auto result = detector.analyze_trade();
    
    std::cout << "\n=== Detection Result ===" << std::endl;
    std::cout << "Insider Trade: " << (result.is_insider ? "YES" : "NO") << std::endl;
    std::cout << "Probability: " << (result.probability * 100) << "%" << std::endl;
    std::cout << "Confidence: " << (result.confidence * 100) << "%" << std::endl;
    
    if (!result.flags.empty()) {
        std::cout << "\nFlags:" << std::endl;
        for (const auto& flag : result.flags) {
            std::cout << "  - " << flag << std::endl;
        }
    }
    
    return 0;
}

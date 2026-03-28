#ifndef INSIDER_DETECTOR_H
#define INSIDER_DETECTOR_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <string>
#include <fstream>
#include <sstream>
#include <memory>

namespace insider {

struct TradeFeatures {
    double return_7d = 0;
    double return_14d = 0;
    double return_30d = 0;
    double return_60d = 0;
    double volatility_7d = 0;
    double volatility_14d = 0;
    double volatility_30d = 0;
    double volatility_60d = 0;
    double sma_ratio_50_200 = 0;
    double rsi_14d = 50;
    double macd = 0;
    double macd_signal = 0;
    double volume_ratio_7d = 1;
    double volume_ratio_14d = 1;
    double volume_ratio_30d = 1;
    double insider_buys_30d = 0;
    double insider_sells_30d = 0;
    double insider_net_activity = 0;
    double market_cap = 0;
    double pe_ratio = 0;
    double insider_percent = 0;
    double institutional_percent = 0;
    double beta = 1.0;
    double abnormal_return_7d = 0;
    double abnormal_return_14d = 0;
    double abnormal_return_30d = 0;
    double abnormal_return_60d = 0;
    
    std::vector<double> to_vector() const {
        return {
            return_7d, return_14d, return_30d, return_60d,
            volatility_7d, volatility_14d, volatility_30d, volatility_60d,
            sma_ratio_50_200, rsi_14d, macd, macd_signal,
            volume_ratio_7d, volume_ratio_14d, volume_ratio_30d,
            insider_buys_30d, insider_sells_30d, insider_net_activity,
            market_cap, pe_ratio, insider_percent, institutional_percent, beta,
            abnormal_return_7d, abnormal_return_14d, abnormal_return_30d, abnormal_return_60d
        };
    }
};

struct ScalerParams {
    std::vector<double> mean;
    std::vector<double> scale;
    
    std::vector<double> transform(const std::vector<double>& x) const {
        std::vector<double> result(x.size());
        for (size_t i = 0; i < x.size() && i < scale.size(); ++i) {
            if (scale[i] != 0) {
                result[i] = (x[i] - mean[i]) / scale[i];
            } else {
                result[i] = 0;
            }
        }
        return result;
    }
};

struct TreeNode {
    int feature_index = -1;
    double threshold = 0;
    int left = -1;
    int right = -1;
    double value = 0;
    bool is_leaf = true;
};

class RandomForestDetector {
private:
    std::vector<TreeNode> trees_;
    ScalerParams scaler_;
    int n_classes_ = 2;
    int n_features_ = 28;
    
public:
    void load(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open model file: " + path);
        }
        
        int n_trees;
        file.read(reinterpret_cast<char*>(&n_trees), sizeof(int));
        file.read(reinterpret_cast<char*>(&n_classes_), sizeof(int));
        file.read(reinterpret_cast<char*>(&n_features_), sizeof(int));
        
        trees_.resize(n_trees);
        
        for (int i = 0; i < n_trees; ++i) {
            file.read(reinterpret_cast<char*>(&trees_[i].feature_index), sizeof(int));
            file.read(reinterpret_cast<char*>(&trees_[i].threshold), sizeof(double));
            file.read(reinterpret_cast<char*>(&trees_[i].left), sizeof(int));
            file.read(reinterpret_cast<char*>(&trees_[i].right), sizeof(int));
            file.read(reinterpret_cast<char*>(&trees_[i].value), sizeof(double));
            file.read(reinterpret_cast<char*>(&trees_[i].is_leaf), sizeof(bool));
        }
        
        int n_scaler;
        file.read(reinterpret_cast<char*>(&n_scaler), sizeof(int));
        scaler_.mean.resize(n_scaler);
        scaler_.scale.resize(n_scaler);
        file.read(reinterpret_cast<char*>(scaler_.mean.data()), n_scaler * sizeof(double));
        file.read(reinterpret_cast<char*>(scaler_.scale.data()), n_scaler * sizeof(double));
        
        file.close();
    }
    
    int predict_single(const std::vector<double>& features) const {
        std::vector<double> scaled = scaler_.transform(features);
        double sum = 0;
        
        for (const auto& node : trees_) {
            sum += predict_tree(scaled, node);
        }
        
        return (sum / trees_.size() > 0.5) ? 1 : 0;
    }
    
    double predict_proba(const std::vector<double>& features) const {
        std::vector<double> scaled = scaler_.transform(features);
        double sum = 0;
        
        for (const auto& node : trees_) {
            sum += predict_tree(scaled, node);
        }
        
        return sum / trees_.size();
    }
    
    std::vector<int> predict_batch(const std::vector<std::vector<double>>& features) const {
        std::vector<int> results;
        results.reserve(features.size());
        
        for (const auto& f : features) {
            results.push_back(predict_single(f));
        }
        
        return results;
    }
    
private:
    double predict_tree(const std::vector<double>& scaled, const TreeNode& root) const {
        const TreeNode* current = &root;
        
        while (!current->is_leaf) {
            if (current->feature_index < 0 || current->feature_index >= (int)scaled.size()) {
                break;
            }
            
            double feature_value = scaled[current->feature_index];
            
            if (feature_value <= current->threshold) {
                if (current->left >= 0 && current->left < (int)trees_.size()) {
                    current = &trees_[current->left];
                } else {
                    break;
                }
            } else {
                if (current->right >= 0 && current->right < (int)trees_.size()) {
                    current = &trees_[current->right];
                } else {
                    break;
                }
            }
        }
        
        return current->value;
    }
};

class InsiderDetector {
private:
    std::unique_ptr<RandomForestDetector> rf_model_;
    std::vector<TradeFeatures> recent_features_;
    double threshold_;
    
public:
    InsiderDetector(double threshold = 0.5) : threshold_(threshold) {
        rf_model_ = std::make_unique<RandomForestDetector>();
    }
    
    void load_model(const std::string& path) {
        rf_model_->load(path);
    }
    
    void add_trade_features(const TradeFeatures& features) {
        recent_features_.push_back(features);
    }
    
    bool is_insider_trade() const {
        if (recent_features_.empty() || !rf_model_) {
            return false;
        }
        
        const auto& latest = recent_features_.back();
        auto features_vec = latest.to_vector();
        
        double probability = rf_model_->predict_proba(features_vec);
        
        return probability > threshold_;
    }
    
    double get_insider_probability() const {
        if (recent_features_.empty() || !rf_model_) {
            return 0.0;
        }
        
        const auto& latest = recent_features_.back();
        return rf_model_->predict_proba(latest.to_vector());
    }
    
    struct DetectionResult {
        bool is_insider = false;
        double probability = 0.0;
        double confidence = 0.0;
        std::vector<std::string> flags;
    };
    
    DetectionResult analyze_trade() {
        DetectionResult result;
        
        if (recent_features_.empty() || !rf_model_) {
            return result;
        }
        
        const auto& latest = recent_features_.back();
        auto features_vec = latest.to_vector();
        
        result.probability = rf_model_->predict_proba(features_vec);
        result.is_insider = result.probability > threshold_;
        result.confidence = std::abs(result.probability - 0.5) * 2.0;
        
        if (std::abs(latest.abnormal_return_30d) > 0.10) {
            result.flags.push_back("High abnormal return");
        }
        if (latest.insider_net_activity > 5) {
            result.flags.push_back("High insider buying activity");
        }
        if (latest.rsi_14d > 70) {
            result.flags.push_back("Overbought (RSI > 70)");
        }
        if (latest.rsi_14d < 30) {
            result.flags.push_back("Oversold (RSI < 30)");
        }
        if (latest.volume_ratio_30d > 3.0) {
            result.flags.push_back("Unusual volume");
        }
        if (latest.return_30d > 0.20) {
            result.flags.push_back("High 30-day return");
        }
        
        return result;
    }
    
    void clear_history() {
        recent_features_.clear();
    }
};

}

#endif

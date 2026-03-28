#ifndef FEATURE_PROCESSOR_H
#define FEATURE_PROCESSOR_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace insider {

struct PriceData {
    std::vector<double> close;
    std::vector<double> volume;
    std::vector<double> open;
    std::vector<double> high;
    std::vector<double> low;
};

struct FeatureResult {
    std::vector<double> returns_7d, returns_14d, returns_30d, returns_60d;
    std::vector<double> volatility_7d, volatility_14d, volatility_30d, volatility_60d;
    std::vector<double> sma_7d, sma_14d, sma_30d, sma_60d;
    std::vector<double> ema_7d, ema_14d, ema_30d, ema_60d;
    std::vector<double> volume_sma_7d, volume_sma_14d, volume_sma_30d, volume_sma_60d;
    std::vector<double> rsi_14d;
    std::vector<double> macd, macd_signal;
};

class FeatureProcessor {
public:
    static std::vector<double> calculate_sma(const std::vector<double>& data, int window) {
        std::vector<double> result(data.size(), std::nan(""));
        
        if (data.size() < window) return result;
        
        double sum = 0;
        for (int i = 0; i < window; ++i) {
            sum += data[data.size() - 1 - i];
        }
        result[data.size() - 1] = sum / window;
        
        for (int i = static_cast<int>(data.size()) - window - 1; i >= 0; --i) {
            sum = sum - data[i + window] + data[i];
            result[i] = sum / window;
        }
        
        return result;
    }
    
    static std::vector<double> calculate_ema(const std::vector<double>& data, int window) {
        std::vector<double> result(data.size(), std::nan(""));
        
        if (data.size() < window) return result;
        
        double multiplier = 2.0 / (window + 1);
        double sum = 0;
        
        for (int i = 0; i < window; ++i) {
            sum += data[data.size() - 1 - i];
        }
        double ema = sum / window;
        result[data.size() - 1] = ema;
        
        for (int i = static_cast<int>(data.size()) - window - 1; i >= 0; --i) {
            ema = (data[i] - ema) * multiplier + ema;
            result[i] = ema;
        }
        
        return result;
    }
    
    static std::vector<double> calculate_returns(const std::vector<double>& prices, int window) {
        std::vector<double> result(prices.size(), std::nan(""));
        
        for (size_t i = window; i < prices.size(); ++i) {
            if (prices[i - window] != 0) {
                result[i] = (prices[i] - prices[i - window]) / prices[i - window];
            }
        }
        
        return result;
    }
    
    static std::vector<double> calculate_volatility(const std::vector<double>& prices, int window) {
        std::vector<double> result(prices.size(), std::nan(""));
        
        std::vector<double> daily_returns(prices.size() - 1);
        for (size_t i = 1; i < prices.size(); ++i) {
            daily_returns[i - 1] = (prices[i] - prices[i - 1]) / prices[i - 1];
        }
        
        for (size_t i = window; i < prices.size(); ++i) {
            double sum = 0;
            for (size_t j = i - window; j < i; ++j) {
                sum += daily_returns[j] * daily_returns[j];
            }
            result[i] = std::sqrt(sum / window);
        }
        
        return result;
    }
    
    static std::vector<double> calculate_rsi(const std::vector<double>& prices, int period = 14) {
        std::vector<double> result(prices.size(), std::nan(""));
        
        if (prices.size() < period + 1) return result;
        
        std::vector<double> gains, losses;
        gains.reserve(prices.size() - 1);
        losses.reserve(prices.size() - 1);
        
        for (size_t i = 1; i < prices.size(); ++i) {
            double diff = prices[i] - prices[i - 1];
            gains.push_back(std::max(diff, 0.0));
            losses.push_back(std::max(-diff, 0.0));
        }
        
        double avg_gain = 0, avg_loss = 0;
        for (int i = 0; i < period; ++i) {
            avg_gain += gains[i];
            avg_loss += losses[i];
        }
        avg_gain /= period;
        avg_loss /= period;
        
        result[period] = 100.0 - (100.0 / (1.0 + avg_gain / (avg_loss + 1e-10)));
        
        for (size_t i = period + 1; i < prices.size(); ++i) {
            avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period;
            avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period;
            
            double rs = avg_gain / (avg_loss + 1e-10);
            result[i] = 100.0 - (100.0 / (1.0 + rs));
        }
        
        return result;
    }
    
    static std::pair<std::vector<double>, std::vector<double>> calculate_macd(
        const std::vector<double>& prices, 
        int fast = 12, 
        int slow = 26, 
        int signal = 9
    ) {
        std::vector<double> ema_fast = calculate_ema(prices, fast);
        std::vector<double> ema_slow = calculate_ema(prices, slow);
        
        std::vector<double> macd_line(prices.size(), std::nan(""));
        for (size_t i = 0; i < prices.size(); ++i) {
            if (!std::isnan(ema_fast[i]) && !std::isnan(ema_slow[i])) {
                macd_line[i] = ema_fast[i] - ema_slow[i];
            }
        }
        
        std::vector<double> valid_macd;
        for (double v : macd_line) {
            if (!std::isnan(v)) valid_macd.push_back(v);
        }
        
        std::vector<double> signal_line = calculate_ema(valid_macd, signal);
        
        std::vector<double> macd_result(prices.size(), std::nan(""));
        std::vector<double> signal_result(prices.size(), std::nan(""));
        
        size_t valid_idx = 0;
        for (size_t i = 0; i < prices.size(); ++i) {
            if (!std::isnan(macd_line[i])) {
                if (valid_idx < signal_line.size()) {
                    macd_result[i] = macd_line[i];
                    signal_result[i] = signal_line[valid_idx];
                }
                valid_idx++;
            }
        }
        
        return {macd_result, signal_result};
    }
    
    static FeatureResult process_all(const PriceData& data) {
        FeatureResult result;
        
        result.returns_7d = calculate_returns(data.close, 7);
        result.returns_14d = calculate_returns(data.close, 14);
        result.returns_30d = calculate_returns(data.close, 30);
        result.returns_60d = calculate_returns(data.close, 60);
        
        result.volatility_7d = calculate_volatility(data.close, 7);
        result.volatility_14d = calculate_volatility(data.close, 14);
        result.volatility_30d = calculate_volatility(data.close, 30);
        result.volatility_60d = calculate_volatility(data.close, 60);
        
        result.sma_7d = calculate_sma(data.close, 7);
        result.sma_14d = calculate_sma(data.close, 14);
        result.sma_30d = calculate_sma(data.close, 30);
        result.sma_60d = calculate_sma(data.close, 60);
        
        result.ema_7d = calculate_ema(data.close, 7);
        result.ema_14d = calculate_ema(data.close, 14);
        result.ema_30d = calculate_ema(data.close, 30);
        result.ema_60d = calculate_ema(data.close, 60);
        
        result.volume_sma_7d = calculate_sma(data.volume, 7);
        result.volume_sma_14d = calculate_sma(data.volume, 14);
        result.volume_sma_30d = calculate_sma(data.volume, 30);
        result.volume_sma_60d = calculate_sma(data.volume, 60);
        
        result.rsi_14d = calculate_rsi(data.close, 14);
        
        auto macd_result = calculate_macd(data.close);
        result.macd = macd_result.first;
        result.macd_signal = macd_result.second;
        
        return result;
    }
};

}

#endif

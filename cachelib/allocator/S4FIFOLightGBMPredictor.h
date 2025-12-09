/*
 * S4FIFO LightGBM Ensemble Predictor
 * 
 * LightGBM ensemble model (5 models, 50 trees each) for predicting optimal
 * S4FIFO configuration parameters based on cache workload features.
 * 
 * This is generated from train_xgb_18class_lite.py and exported via m2cgen.
 * Model accuracy: ~47% top-1, Mean improvement over FIFO: 15.34%
 * 
 * NOTE: This header should be included AFTER S4FIFOFeatureVector and
 * S4FIFOPredictedParams are defined (e.g., by MMS4FIFO.h).
 */

#pragma once

#include <cmath>
#include <cstddef>  // for size_t

// Include the ensemble model (C code, compiled as extern "C")
extern "C" {
#include "s4fifo_model/S4FIFOEnsemble.c"
}

namespace facebook::cachelib {

// The 18 selected S3FIFO/S4FIFO configurations
// Format: (s_param, m_param, t_param, g_param, k_param)
// s_param -> tinySizePercent: small queue size as fraction (0.05-0.9) * 100 = 5-90%
// m_param -> moveToMainThreshold: frequency threshold for small->main (1-2)
// t_param -> ghostToMainThreshold: ghost->main threshold (0-1)
// g_param -> ghostSizePercent: ghost size as fraction (0.9-6.0) * 100 = 90-600%
// k_param -> smallSkipRatio: skip ratio for frequency increment (always 0.25)

struct S4FIFOConfigEntry {
    double s;  // tinySizePercent (fraction, multiply by 100)
    int m;     // moveToMainThreshold  
    int t;     // ghostToMainThreshold
    double g;  // ghostSizePercent (fraction, multiply by 100)
    double k;  // smallSkipRatio
};

constexpr S4FIFOConfigEntry kS4FIFOConfigs[18] = {
    {0.20, 1, 0, 3.0, 0.25},  // Class 0
    {0.05, 1, 0, 0.9, 0.25},  // Class 1
    {0.50, 1, 0, 0.9, 0.25},  // Class 2
    {0.20, 1, 0, 0.9, 0.25},  // Class 3
    {0.05, 2, 0, 6.0, 0.25},  // Class 4
    {0.10, 2, 1, 3.0, 0.25},  // Class 5
    {0.30, 2, 0, 3.0, 0.25},  // Class 6
    {0.05, 2, 0, 3.0, 0.25},  // Class 7
    {0.10, 2, 0, 0.9, 0.25},  // Class 8
    {0.70, 1, 1, 0.9, 0.25},  // Class 9
    {0.20, 1, 1, 0.9, 0.25},  // Class 10
    {0.05, 1, 1, 0.9, 0.25},  // Class 11
    {0.30, 1, 0, 6.0, 0.25},  // Class 12
    {0.20, 2, 0, 0.9, 0.25},  // Class 13
    {0.90, 2, 0, 3.0, 0.25},  // Class 14
    {0.10, 2, 0, 6.0, 0.25},  // Class 15
    {0.30, 2, 1, 3.0, 0.25},  // Class 16
    {0.05, 2, 0, 0.9, 0.25},  // Class 17
};

// Prepare model input features from S4FIFOFeatureVector
// Maps the CacheLib feature vector to the 73-feature input expected by the model
// Features are in SORTED order (alphabetical) as per feature_columns.json
inline void prepareModelInput(const S4FIFOFeatureVector& fv, double* input) {
    // Initialize all to 0
    for (int i = 0; i < 73; i++) input[i] = 0.0;
    
    // Compute derived features
    double probation_efficiency = static_cast<double>(fv.hitsSmall) / (fv.hitsMain + 1e-6);
    double total_hits_with_ghost = static_cast<double>(fv.totalHits + fv.hitsGhost);
    double ghost_pressure = static_cast<double>(fv.hitsGhost) / (total_hits_with_ghost + 1e-6);
    double entropy_gap = fv.hitRatioMain - fv.hitRatioSmall;  // H_m - H_s
    double decay_rate_small = fv.histSmall[0] - fv.histSmall[1];
    
    // tail_heaviness: sum of hist_main[10..19]
    double tail_heaviness = 0.0;
    for (int i = 10; i < 20; i++) tail_heaviness += fv.histMain[i];
    
    // Ratio placeholder - in real usage, this should be passed in
    double ratio = 0.01;
    double thrashing_risk = fv.uniqueRatio / (ratio * 100.0 + 1e-6);
    double scan_intensity = fv.oneHitRatio * (1.0 - ratio);
    
    // Map features to model input indices (SORTED alphabetical order)
    input[0] = fv.hitRatioGhost;  // H_g
    input[1] = fv.hitRatioMain;  // H_m
    input[2] = fv.hitRatioSmall;  // H_s
    input[3] = decay_rate_small;
    input[4] = entropy_gap;
    input[5] = ghost_pressure;
    input[6] = fv.histGhost[0];
    input[7] = fv.histGhost[1];
    input[8] = fv.histGhost[10];
    input[9] = fv.histGhost[11];
    input[10] = fv.histGhost[12];
    input[11] = fv.histGhost[13];
    input[12] = fv.histGhost[14];
    input[13] = fv.histGhost[15];
    input[14] = fv.histGhost[16];
    input[15] = fv.histGhost[17];
    input[16] = fv.histGhost[18];
    input[17] = fv.histGhost[19];
    input[18] = fv.histGhost[2];
    input[19] = fv.histGhost[3];
    input[20] = fv.histGhost[4];
    input[21] = fv.histGhost[5];
    input[22] = fv.histGhost[6];
    input[23] = fv.histGhost[7];
    input[24] = fv.histGhost[8];
    input[25] = fv.histGhost[9];
    input[26] = fv.histMain[0];
    input[27] = fv.histMain[1];
    input[28] = fv.histMain[10];
    input[29] = fv.histMain[11];
    input[30] = fv.histMain[12];
    input[31] = fv.histMain[13];
    input[32] = fv.histMain[14];
    input[33] = fv.histMain[15];
    input[34] = fv.histMain[16];
    input[35] = fv.histMain[17];
    input[36] = fv.histMain[18];
    input[37] = fv.histMain[19];
    input[38] = fv.histMain[2];
    input[39] = fv.histMain[3];
    input[40] = fv.histMain[4];
    input[41] = fv.histMain[5];
    input[42] = fv.histMain[6];
    input[43] = fv.histMain[7];
    input[44] = fv.histMain[8];
    input[45] = fv.histMain[9];
    input[46] = fv.histSmall[0];
    input[47] = fv.histSmall[1];
    input[48] = fv.histSmall[10];
    input[49] = fv.histSmall[11];
    input[50] = fv.histSmall[12];
    input[51] = fv.histSmall[13];
    input[52] = fv.histSmall[14];
    input[53] = fv.histSmall[15];
    input[54] = fv.histSmall[16];
    input[55] = fv.histSmall[17];
    input[56] = fv.histSmall[18];
    input[57] = fv.histSmall[19];
    input[58] = fv.histSmall[2];
    input[59] = fv.histSmall[3];
    input[60] = fv.histSmall[4];
    input[61] = fv.histSmall[5];
    input[62] = fv.histSmall[6];
    input[63] = fv.histSmall[7];
    input[64] = fv.histSmall[8];
    input[65] = fv.histSmall[9];
    input[66] = probation_efficiency;
    input[67] = fv.oneHitRatio;
    input[68] = fv.uniqueRatio;
    input[69] = scan_intensity;
    input[70] = tail_heaviness;
    input[71] = thrashing_risk;
    input[72] = static_cast<double>(fv.totalRequests);
}

// LightGBM prediction function to be used as S4FIFOPredictionCallback
inline S4FIFOPredictedParams lightGBMPredict(const S4FIFOFeatureVector& features) {
    S4FIFOPredictedParams params;
    
    // Skip prediction if not enough data (need some hits to be meaningful)
    if (features.totalRequests < 10000 || features.totalHits < 100) {
        // Return "no change" params
        return params;
    }
    
    // Prepare input features
    double input[73];
    prepareModelInput(features, input);
    
    // Run ensemble prediction
    int classId = ensemble_predict(input);
    
    // Validate class ID
    if (classId < 0 || classId >= 18) {
        // Return "no change" params on error
        return params;
    }
    
    // Get the predicted config
    const S4FIFOConfigEntry& config = kS4FIFOConfigs[classId];
    
    // Map config to CacheLib parameters
    // s_param (0.05-0.9) -> tinySizePercent (5-90%)
    params.tinySizePercent = static_cast<size_t>(config.s * 100);
    
    // m_param (1-2) -> moveToMainThreshold
    params.moveToMainThreshold = config.m;
    
    // t_param (0-1) -> ghostToMainThreshold
    params.ghostToMainThreshold = config.t;
    
    // g_param (0.9-6.0) -> ghostSizePercent (90-600%)
    size_t ghostPercent = static_cast<size_t>(config.g * 100);
    if (ghostPercent > 600) ghostPercent = 600;  // Cap at 600%
    if (ghostPercent < 10) ghostPercent = 10;    // Min 10%
    params.ghostSizePercent = ghostPercent;
    
    // k_param (always 0.25) -> smallSkipRatio
    params.smallSkipRatio = config.k;
    
    return params;
}

} // namespace facebook::cachelib

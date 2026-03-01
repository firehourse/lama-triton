#include <algorithm>
#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

/**
 * Image Processing Logic for LaMa Inpainting Preprocessing (C++ Version)
 */

namespace lama_preproc {

struct PreprocResult {
  cv::Mat img_tensor;  // FP32, NCHW, shape [3, H, W] flattened
  cv::Mat mask_tensor; // FP32, NCHW, shape [1, H, W] flattened
  int original_h;
  int original_w;
  int padded_h;
  int padded_w;
};

/**
 * Perform preprocessing on raw image and mask bytes.
 *
 * 1. Decode bytes using OpenCV
 * 2. Calculate padding to make dimensions multiples of 8 (Mod 8)
 * 3. Normalize image to [0, 1] float32
 * 4. Binarize mask to {0, 1} float32
 * 5. Convert HWC to NCHW
 */
PreprocResult preprocess(const std::vector<uint8_t> &img_raw,
                         const std::vector<uint8_t> &mask_raw) {
  PreprocResult res;

  // 1. Decode
  cv::Mat img = cv::imdecode(img_raw, cv::IMREAD_COLOR); // BGR
  cv::Mat mask = cv::imdecode(mask_raw, cv::IMREAD_GRAYSCALE);

  if (img.empty() || mask.empty()) {
    throw std::runtime_error("Failed to decode image or mask bytes");
  }

  res.original_h = img.rows;
  res.original_w = img.cols;

  // 2. Padding (Mod 8)
  int pad_h = (8 - (img.rows % 8)) % 8;
  int pad_w = (8 - (img.cols % 8)) % 8;

  cv::Mat img_padded, mask_padded;
  if (pad_h > 0 || pad_w > 0) {
    cv::copyMakeBorder(img, img_padded, 0, pad_h, 0, pad_w, cv::BORDER_CONSTANT,
                       cv::Scalar(0, 0, 0));
    cv::copyMakeBorder(mask, mask_padded, 0, pad_h, 0, pad_w,
                       cv::BORDER_CONSTANT, cv::Scalar(0));
  } else {
    img_padded = img;
    mask_padded = mask;
  }

  // 3. Normalization (To FP32)
  cv::Mat img_fp32;
  img_padded.convertTo(img_fp32, CV_32FC3, 1.0 / 255.0);

  cv::Mat mask_fp32;
  mask_padded.convertTo(mask_fp32, CV_32FC1, 1.0 / 255.0);
  cv::threshold(mask_fp32, mask_fp32, 0.5, 1.0, cv::THRESH_BINARY);

  // 4. HWC to NCHW Transformation
  // Image: [H, W, 3] -> [3, H, W]
  int H = img_fp32.rows;
  int W = img_fp32.cols;

  // Create CHW planar layout for Image
  std::vector<cv::Mat> channels(3);
  cv::split(img_fp32, channels);

  res.img_tensor = cv::Mat(3, H * W, CV_32FC1);
  for (int i = 0; i < 3; ++i) {
    channels[i].reshape(1, 1).copyTo(res.img_tensor.row(i));
  }
  // Result shape is [3, H, W] when flattened

  // Mask already [H, W, 1] -> [1, H, W]
  res.mask_tensor = mask_fp32.reshape(1, 1);
  res.padded_h = H;
  res.padded_w = W;

  return res;
}

} // namespace lama_preproc

/**
* This file is part of DynaProp.
* Copyright (C) 2022 XZ Yuan (Huazhong University of Science and Technology in China)
* For more information see <https://github.com/yxianzhe/DynaProp>.
*/

#ifndef MASKPROP_H
#define MASKPROP_H

#include <iostream>
#include <boost/math/distributions.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "Frame.h"
#include "ORBextractor.h"
#include "ORBmatcher.h"
#include "ORBVocabulary.h"
#include "Converter.h"

namespace DynaProp
{
#define GROW_SIZE 3

class MaskPropagation
{
public:
    MaskPropagation(const string &strVocFile, const string &strSettingsFile);

    cv::Mat DilaThisMask(const cv::Mat &mask, int dilation_size = 15);

    cv::Mat ErodeThisMask(const cv::Mat &mask, int erode_size = 15);

    cv::Mat RegionGrowing(const cv::Mat &im, int x, int y, float reg_maxdist, const cv::Mat &MaskLimit, const cv::Mat &outside_mask, const int &growing_size, float &reg_mean);

    cv::Mat DepthRegionGrowing(const std::vector<cv::KeyPoint> &vKeyPoints, const cv::Mat &outside_mask, const cv::Mat &imDepth, const cv::Mat &MaskLimit, const int &growing_size);

    void ComputeBoW(cv::Mat &Descriptors, ORB_SLAM2::ORBVocabulary* Vocabulary, DBoW2::BowVector &BowVec, DBoW2::FeatureVector &FeatVec);

    void ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);

    int SearchByBoW(const DBoW2::FeatureVector &vFeatVec1, const DBoW2::FeatureVector &vFeatVec2, 
        const std::vector<cv::KeyPoint> &vKeyPoint1, const std::vector<cv::KeyPoint> &vKeyPoint2,
        const cv::Mat Descriptors1, const cv::Mat Descriptors2, 
        std::vector<cv::DMatch> &matches, bool mbCheckOrientation, float mfNNratio = 0.6);

    void ConvertDepth(cv::Mat &depthimg);

    void UndistortKeyPoints();

    void FindNewKeypoints();

    cv::Mat GetMaskbyPropagation(const cv::Mat &newimg, const cv::Mat &newdepth, std::string dir="no_save", std::string rgb_name="no_file"); 

    void GetMaskbySegmentation(const cv::Mat &newimg, const cv::Mat &newdepth, const cv::Mat &newmask);

    void UpdateImg(const cv::Mat &img);

    void UpdateDepth(const cv::Mat &depth);

    void UpdateMask(const cv::Mat &mask);

    const std::vector<cv::KeyPoint>& GetNewImgKeyPoints();

    const cv::Mat& GetNewImgDescriptors();

    const std::vector<float>& GetNewImgDynamicProbablity();

private:

    ORB_SLAM2::ORBextractor* mextractor;

    ORB_SLAM2::ORBVocabulary* mVocabulary;

    cv::Mat mK, mDistCoef;

    string mstrVocFile, mstrSettingsFile;

    cv::Mat mmask_origin;

    cv::Mat mlastimg, mnewimg, mlastdepth, mnewdepth, mlastmask, mnewmask;

    cv::Mat mimgwithmask, mimgwithpoints;

    std::vector<cv::KeyPoint> mtarget_points, moutside_points;

    std::vector<std::pair<cv::KeyPoint, float>> regions; 

    std::vector<cv::KeyPoint> mnewimg_keypoints, mnewun_keypoints;
    cv::Mat F;
    std::vector<float> P_s_d; //segmentation dynamic
    std::vector<float> P_g_d; //geometry dynamic
    std::vector<float> P_o_d; //observation dynamic
    std::vector<float> P_d;
    cv::Mat mnewimg_descriptors;
    DBoW2::BowVector mnewimg_BowVec;
    DBoW2::FeatureVector mnewimg_FeatVec; 
    
    float mDepthMapFactor;
};


}

#endif
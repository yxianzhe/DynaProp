/**
* This file is a modified version of ORB-SLAM2.<https://github.com/raulmur/ORB_SLAM2>
*
* This file is part of DynaProp.
* Copyright (C) 2022 XZ Yuan (Huazhong University of Science and Technology in China)
* For more information see <https://github.com/yxianzhe/DynaProp>.
*/
#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <thread>
#include <mutex>

#include <ros/ros.h>
#include <cv_bridge_1/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <opencv2/core/core.hpp>

#include "../../../include/System.h"
#include "../../../include/MaskNet.h"
#include "../../../include/MaskPropogation.h"

using namespace std;

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM2::System* pSLAM, DynaSLAM::SegmentDynObject* pMaskNet, DynaProp::MaskPropogation* pMaskProp):mpSLAM(pSLAM),mpMaskNet(pMaskNet),mpMaskProp(pMaskProp){}

    void GrabRGBD(const sensor_msgs::ImageConstPtr& msgRGB,const sensor_msgs::ImageConstPtr& msgD);

    void segmask();
public:
    ORB_SLAM2::System* mpSLAM;
    DynaSLAM::SegmentDynObject *mpMaskNet;
    DynaProp::MaskPropogation *mpMaskProp;
    vector<pair<int,pair<cv::Mat,cv::Mat>>> imgVec;
    vector<pair<int,cv::Mat>> maskVec;
    vector<pair<int,cv::Mat>> segVec;
    mutex img_mtx;
    mutex mask_mtx;
    mutex seg_mtx;
    cv::Mat maskinuse; //上一帧使用的mask
    cv::Mat latest_mask, latest_RGB, latest_D;//存放最新一次分割的结果
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "RGBD");
    ros::start();

    if(argc != 3)
    {
        cerr << endl << "Usage: rosrun ORB_SLAM2 RGBD path_to_vocabulary path_to_settings" << endl;        
        ros::shutdown();
        return 1;
    }    

    DynaSLAM::SegmentDynObject *MaskNet;
    cout << "Loading Mask R-CNN. This could take a while..." << endl;
    MaskNet = new DynaSLAM::SegmentDynObject();
    cout << "Mask R-CNN loaded!" << endl;

    DynaProp::MaskPropogation MaskProp(argv[1],argv[2]);

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::RGBD,true);

    ImageGrabber igb(&SLAM,MaskNet,&MaskProp);

    ros::NodeHandle nh;

    message_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh, "/camera/rgb/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, "camera/depth_registered/image_raw", 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), rgb_sub,depth_sub);
    sync.registerCallback(boost::bind(&ImageGrabber::GrabRGBD,&igb,_1,_2));

    thread mask_seger{&ImageGrabber::segmask,&igb};
    mask_seger.detach();
    ros::spin();

    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    ros::shutdown();

    return 0;
}

void ImageGrabber::GrabRGBD(const sensor_msgs::ImageConstPtr& msgRGB,const sensor_msgs::ImageConstPtr& msgD)
{
    static int ni = 0;
    ni++;


    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptrRGB;
    try
    {
        cv_ptrRGB = cv_bridge::toCvShare(msgRGB);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv_bridge::CvImageConstPtr cv_ptrD;
    try
    {
        cv_ptrD = cv_bridge::toCvShare(msgD);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv::Mat imRGB = cv_ptrRGB->image;
    cv::Mat imgD = cv_ptrD->image;
    img_mtx.lock();
    imgVec.emplace_back(make_pair(ni,make_pair(imRGB,imgD)));
    img_mtx.unlock();

    // cout<<"ni = "<<ni<<endl;

    cv::Mat maskfore;
    cv::Mat mask = cv::Mat::ones(480,640,CV_8U);

    while(ni==1 && segVec.empty()){
        usleep(3000);
        continue;
    }

    if(!segVec.empty())
    {
        seg_mtx.lock();
        segVec.clear();
        seg_mtx.unlock();
        mpMaskProp->GetMaskbySegmentation(latest_RGB,latest_D,latest_mask);//用新分割的mask更新参考帧
    }
    if(ni==1) maskfore = latest_mask;
    else maskfore = mpMaskProp->GetMaskbyPropogation(imRGB,imgD,"no_save","no_file");
    mask = mask - maskfore;

    mpSLAM->TrackRGBD(imRGB,imgD,mask,cv_ptrRGB->header.stamp.toSec(),mpMaskProp->GetNewImgKeyPoints(),mpMaskProp->GetNewImgDescriptors(),true);
}


void ImageGrabber::segmask()
{
    int ni;
    cv::Mat imgRGB,imgD;
    cv::Mat maskfore;
    while(true)
    {
        if(imgVec.empty()){
            usleep(3000);
            continue;
        }
        // cout<<imgVec.size()<<endl;
        //取最新帧的值
        img_mtx.lock();
        ni = (imgVec.end()-1)->first;
        // cout<<"ni is "<<ni<<endl;
        imgRGB = (imgVec.end()-1)->second.first;
        imgD = (imgVec.end()-1)->second.second;
        // cout<<"..."<<endl;
        imgVec.clear();
        img_mtx.unlock();

        maskfore = mpMaskNet->GetSegmentation(imgRGB);
        latest_mask = maskfore;
        latest_RGB = imgRGB;
        latest_D = imgD;
        seg_mtx.lock();
        segVec.emplace_back(make_pair(ni,maskfore));
        seg_mtx.unlock();
    }
}
/**
* This file is a modified version of ORB-SLAM2.<https://github.com/raulmur/ORB_SLAM2>
*
* This file is a modified version of DynaSLAM.
* Copyright (C) 2018 Berta Bescos <bbescos at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/bertabescos/DynaSLAM>.
*
* This file is part of DynaProp.
* Copyright (C) 2022 XZ Yuan (Huazhong University of Science and Technology in China)
* For more information see <https://github.com/yxianzhe/DynaProp>.
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <unistd.h>
#include<opencv2/core/core.hpp>

#include "Geometry.h"
#include "MaskNet.h"
#include "MaskPropogation.h"
#include "System.h"

using namespace std;

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    if(argc != 5 && argc != 6 && argc != 7)
    {
        cerr << endl << "Usage: ./rgbd_tum path_to_vocabulary path_to_settings path_to_sequence path_to_association (path_to_masks) (path_to_output)" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    string strAssociationFilename = string(argv[4]);
    LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);

    // Check consistency in the number of images and depthmaps
    int nImages = vstrImageFilenamesRGB.size();
    // std::cout << "nImages: " << nImages << std::endl;

    if(vstrImageFilenamesRGB.empty())
    {
        cerr << endl << "No images found in provided path." << endl;
        return 1;
    }
    else if(vstrImageFilenamesD.size()!=vstrImageFilenamesRGB.size())
    {
        cerr << endl << "Different number of images for rgb and depth." << endl;
        return 1;
    }

    // Initialize Mask R-CNN
    DynaSLAM::SegmentDynObject *MaskNet;
    if (argc==6 || argc==7)
    {
        cout << "Loading Mask R-CNN. This could take a while..." << endl;
        MaskNet = new DynaSLAM::SegmentDynObject();
        cout << "Mask R-CNN loaded!" << endl;
    }

    DynaProp::MaskPropogation MaskProp(argv[1],argv[2]);

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::RGBD,true); // true

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Dilation settings
    int dilation_size = 15;
    cv::Mat kernel = getStructuringElement(cv::MORPH_ELLIPSE,
                                           cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                           cv::Point( dilation_size, dilation_size ) );

    if (argc==7) //创建三个输出文件夹
    {
        std::string dir = string(argv[6]);
        mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        dir = string(argv[6]) + "/rgb/";
        mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        dir = string(argv[6]) + "/depth/";
        mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        dir = string(argv[6]) + "/mask/";
        mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }

    // Main loop
        cv::Mat imRGB, imD;
        cv::Mat imRGBOut, imDOut,maskOut;//输出：rgb，depth，mask

    for(int ni=0; ni<nImages; ni++)
    {
        // Read image and depthmap from file
        imRGB = cv::imread(string(argv[3])+"/"+vstrImageFilenamesRGB[ni],CV_LOAD_IMAGE_UNCHANGED);//原图RGB
        imD = cv::imread(string(argv[3])+"/"+vstrImageFilenamesD[ni],CV_LOAD_IMAGE_UNCHANGED);//原图D

        double tframe = vTimestamps[ni];

        if(imRGB.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenamesRGB[ni] << endl;
            return 1;
        }

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        // Segment out the images
        cv::Mat mask = cv::Mat::ones(480,640,CV_8U);
        cv::Mat imgwithmask = cv::Mat::zeros(480,640,imRGB.type());
        bool bypropagation = false;
        if (argc == 6 || argc == 7)
        {
            cv::Mat maskfore;
            // std::chrono::steady_clock::time_point tseg1 = std::chrono::steady_clock::now();
            if(ni%7==0)
            {
                maskfore = MaskNet->GetSegmentation(imRGB,string(argv[5]),vstrImageFilenamesRGB[ni].replace(0,4,""));//把“rgb/”四个字换掉。。。
                MaskProp.GetMaskbySegmentation(imRGB,imD,maskfore);
            }
            else
            {
                maskfore = MaskProp.GetMaskbyPropogation(imRGB,imD,string(argv[5]),vstrImageFilenamesRGB[ni].replace(0,4,""));
                bypropagation = true;
            }
            // maskfore = MaskNet->GetSegmentation(imRGB,string(argv[5]),vstrImageFilenamesRGB[ni].replace(0,4,""));

            // std::chrono::steady_clock::time_point tseg2 = std::chrono::steady_clock::now();
            // double t_seg_total= std::chrono::duration_cast<std::chrono::duration<double> >(tseg2 - tseg1).count();
            // cout<<"segmentation for image "<<ni<<" takes "<<t_seg_total<<" seconds."<<endl;
            // cv::Mat maskforedil = maskfore.clone();
            // cv::dilate(maskfore,maskforedil, kernel);
            mask = mask - maskfore;//全1 - 前景1 = 把前景扣掉
            // cv::Mat imgwithmask = cv::Mat::zeros(480,640,imRGB.type());
            imRGB.copyTo(imgwithmask,mask);
            // cv::imshow("imgwithmask",imgwithmask);
            // cv::waitKey(0);

        }

        // Pass the image to the SLAM system
        //std::chrono::steady_clock::time_point tSLAM1 = std::chrono::steady_clock::now();
        if (argc == 7){
            SLAM.TrackRGBD(imRGB,imD,mask,tframe,imRGBOut,imDOut,maskOut,MaskProp.GetNewImgKeyPoints(),MaskProp.GetNewImgDescriptors(),bypropagation);
        }//RGB原图，D原图，扣掉前景的mask，时间戳，输出RGB，输出D，输出mask（这个应该是结合了几何之后的结果）
        else {
            SLAM.TrackRGBD(imRGB,imD,mask,tframe,MaskProp.GetNewImgKeyPoints(),MaskProp.GetNewImgDescriptors(),bypropagation);
        }
        //std::chrono::steady_clock::time_point tSLAM2 = std::chrono::steady_clock::now();
        //double t_SLAM_total= std::chrono::duration_cast<std::chrono::duration<double> >(tSLAM2 - tSLAM1).count();
        //cout<<"SLAM for image "<<ni<<" takes "<<t_SLAM_total<<" seconds."<<endl;

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        if (argc == 7)
        {
            cv::imwrite(string(argv[6]) + "/rgb/" + vstrImageFilenamesRGB[ni],imgwithmask);
            vstrImageFilenamesD[ni].replace(0,6,"");
            // cv::imwrite(string(argv[6]) + "/depth/" + vstrImageFilenamesD[ni],imDOut);
            // cv::imwrite(string(argv[6]) + "/mask/" + vstrImageFilenamesRGB[ni],maskOut);
        }

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        // cout<<"Tracking for image "<<ni<<" takes "<<ttrack<<" seconds in total."<<endl;

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
        {
            // cout<<"sleep....."<<endl;
            usleep((T-ttrack)*1e6);
        }
            
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    return 0;
}

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while(!fAssociation.eof())
    {
        string s;
        getline(fAssociation,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);

        }
    }
}

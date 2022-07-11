/**
* This file is part of DynaSLAM.
* Copyright (C) 2018 Berta Bescos <bbescos at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/bertabescos/DynaSLAM>.
*
*/

#include "Geometry.h"
#include <algorithm>
#include "Frame.h"
#include "Tracking.h"

namespace DynaSLAM
{

Geometry::Geometry()
{
    vAllPixels = cv::Mat(640*480,2,CV_32F);
    int m(0);
    for (int i(0); i < 640; i++){
        for (int j(0); j < 480; j++){
            vAllPixels.at<float>(m,0) = i;
            vAllPixels.at<float>(m,1) = j;
            m++;
        }
    }
}

void Geometry::GeometricModelCorrection(const ORB_SLAM2::Frame &currentFrame,
                                        cv::Mat &imDepth, cv::Mat &mask){
    if(currentFrame.mTcw.empty()){
        std::cout << "Geometry not working." << std::endl;
    }
    else if (mDB.mNumElem >= ELEM_INITIAL_MAP){
        vector<ORB_SLAM2::Frame> vRefFrames = GetRefFrames(currentFrame);//找最近的5个关键帧
        vector<DynKeyPoint> vDynPoints = ExtractDynPoints(vRefFrames,currentFrame);//用几何方法找动态点
        mask = DepthRegionGrowing(vDynPoints,imDepth);
        CombineMasks(currentFrame,mask);
    }
}

void Geometry::InpaintFrames(const ORB_SLAM2::Frame &currentFrame,
                             cv::Mat &imGray, cv::Mat &imDepth,
                             cv::Mat &imRGB, cv::Mat &mask){
    FillRGBD(currentFrame,mask,imGray,imDepth,imRGB);
}

void Geometry::GeometricModelUpdateDB(const ORB_SLAM2::Frame &currentFrame){
    if (currentFrame.mIsKeyFrame)
    {
        mDB.InsertFrame2DB(currentFrame);
    }
}

vector<ORB_SLAM2::Frame> Geometry::GetRefFrames(const ORB_SLAM2::Frame &currentFrame){

    cv::Mat rot1 = currentFrame.mTcw.rowRange(0,3).colRange(0,3);
    cv::Mat eul1 = rotm2euler(rot1);//旋转，欧拉角表示
    cv::Mat trans1 = currentFrame.mTcw.rowRange(0,3).col(3); // 平移
    cv::Mat vDist;
    cv::Mat vRot;

    //遍历关键帧数据库
    for (int i(0); i < mDB.mNumElem; i++){
        cv::Mat rot2 = mDB.mvDataBase[i].mTcw.rowRange(0,3).colRange(0,3);
        cv::Mat eul2 = rotm2euler(rot2);
        double distRot = cv::norm(eul2,eul1,cv::NORM_L2);
        vRot.push_back(distRot);//旋转距离

        cv::Mat trans2 = mDB.mvDataBase[i].mTcw.rowRange(0,3).col(3);
        double dist = cv::norm(trans2,trans1,cv::NORM_L2);
        vDist.push_back(dist);//平移距离
    }

    double minvDist, maxvDist;
    cv::minMaxLoc(vDist, &minvDist, &maxvDist);
    vDist /= maxvDist;//归一化

    double minvRot, maxvRot;
    cv::minMaxLoc(vRot, &minvRot, &maxvRot);
    vRot /= maxvRot;

    vDist = 0.7*vDist + 0.3*vRot;//取归一化后的加权，是一个n*1的矩阵
    cv::Mat vIndex;
    // cv::sortIdx(vDist,vIndex,CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);vIndex保存的是降序排列的索引，是否应该是升序排列啊.....
    cv::sortIdx(vDist,vIndex,CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);

    mnRefFrames = std::min(MAX_REF_FRAMES,vDist.rows);//5,n 关键帧数据库有大于5帧那就取5,否则有多少帧取多少帧

    vector<ORB_SLAM2::Frame> vRefFrames;

    for (int i(0); i < mnRefFrames; i++)//这里是取了距离最大的5帧吗？？？？？？？
    {
        int ind = vIndex.at<int>(0,i);
        vRefFrames.push_back(mDB.mvDataBase[ind]);
    }

    return(vRefFrames);
}

//用几何方法找动态点 入参：最近的5个参考关键帧，经过轻量化跟踪处理的本帧
vector<Geometry::DynKeyPoint> Geometry::ExtractDynPoints(const vector<ORB_SLAM2::Frame> &vRefFrames,
                                                         const ORB_SLAM2::Frame &currentFrame){
    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = currentFrame.fx;
    K.at<float>(1,1) = currentFrame.fy;
    K.at<float>(0,2) = currentFrame.cx;
    K.at<float>(1,2) = currentFrame.cy;

    cv::Mat vAllMPw;//存放所有的世界坐标系下地图点
    cv::Mat vAllMatRefFrame;//参考帧中关键点的u，v，1
    cv::Mat vAllLabels;
    cv::Mat vAllDepthRefFrame;

    //遍历参考关键帧
    for (int i(0); i < mnRefFrames; i++)
    {
        ORB_SLAM2::Frame refFrame = vRefFrames[i];

        // Fill matrix with points
        cv::Mat matRefFrame(refFrame.N,3,CV_32F);//n*3 参考帧的特征点
        cv::Mat matDepthRefFrame(refFrame.N,1,CV_32F);//n*1 参考帧特征点的深度
        cv::Mat matInvDepthRefFrame(refFrame.N,1,CV_32F);//n*1 参考帧特征点的逆深度
        cv::Mat vLabels(refFrame.N,1,CV_32F);
        int k(0);
        //遍历特征点
        for(int j(0); j < refFrame.N; j++){
            const cv::KeyPoint &kp = refFrame.mvKeys[j];
            const float &v = kp.pt.y;
            const float &u = kp.pt.x;
            const float d = refFrame.mImDepth.at<float>(v,u); //深度图上这一点的深度值
            if (d > 0 && d < 6){
                matRefFrame.at<float>(k,0) = refFrame.mvKeysUn[j].pt.x;
                matRefFrame.at<float>(k,1) = refFrame.mvKeysUn[j].pt.y;
                matRefFrame.at<float>(k,2) = 1.;
                matInvDepthRefFrame.at<float>(k,0) = 1./d;
                matDepthRefFrame.at<float>(k,0) = d;
                vLabels.at<float>(k,0) = i;//这个特征点是属于第i个参考关键帧的
                k++;
            }
        }

        //只有满足d > 0 && d < 6才能入选
        matRefFrame = matRefFrame.rowRange(0,k);
        matInvDepthRefFrame = matInvDepthRefFrame.rowRange(0,k);
        matDepthRefFrame = matDepthRefFrame.rowRange(0,k);
        vLabels = vLabels.rowRange(0,k);
        cv::Mat vMPRefFrame = K.inv()*matRefFrame.t();//归一化相机坐标系 3*k
        cv::vconcat(vMPRefFrame,matInvDepthRefFrame.t(),vMPRefFrame);//4*k 多一维逆深度 X/Z, Y/Z, 1, 1/Z
        cv::Mat vMPw = refFrame.mTcw.inv() * vMPRefFrame;//转到世界坐标系 4*k
        cv::Mat _vMPw = cv::Mat(4,vMPw.cols,CV_32F);//4*k
        cv::Mat _vLabels = cv::Mat(vLabels.rows,1,CV_32F); //k*1
        cv::Mat _matRefFrame = cv::Mat(matRefFrame.rows,3,CV_32F);//k*3
        cv::Mat _matDepthRefFrame = cv::Mat(matDepthRefFrame.rows,1,CV_32F);//k*1

        int h(0);
        mParallaxThreshold = 30; //视差角 我怀疑这个视差角的阈值是根据数据集调的。。。。可否换成自适应的？？？
        for (int j(0); j < k; j++)//筛除视差角大的物体
        {
            cv::Mat mp = cv::Mat(3,1,CV_32F);//世界坐标
            mp.at<float>(0,0) = vMPw.at<float>(0,j)/matInvDepthRefFrame.at<float>(0,j);
            mp.at<float>(1,0) = vMPw.at<float>(1,j)/matInvDepthRefFrame.at<float>(0,j);
            mp.at<float>(2,0) = vMPw.at<float>(2,j)/matInvDepthRefFrame.at<float>(0,j);
            cv::Mat tRefFrame = refFrame.mTcw.rowRange(0,3).col(3);//参考帧在世界坐标系下的位置

            cv::Mat tCurrentFrame = currentFrame.mTcw.rowRange(0,3).col(3);//本帧在世界坐标系下的位置
            cv::Mat nMPRefFrame = mp - tRefFrame;//kf到x的向量
            cv::Mat nMPCurrentFrame = mp - tCurrentFrame;//cf到x的向量

            double dotProduct = nMPRefFrame.dot(nMPCurrentFrame);//点乘
            double normMPRefFrame = cv::norm(nMPRefFrame,cv::NORM_L2);
            double normMPCurrentFrame = cv::norm(nMPCurrentFrame,cv::NORM_L2);
            double angle = acos(dotProduct/(normMPRefFrame*normMPCurrentFrame))*180/M_PI;//余弦公式计算夹角
            if (angle < mParallaxThreshold)//小于最大视差角才保存地图点，大于30度的点在论文中被认为是“有遮挡情况的静态点”，如果不筛除，后续就会被错误归为动态点
            {
                _vMPw.at<float>(0,h) = vMPw.at<float>(0,j);
                _vMPw.at<float>(1,h) = vMPw.at<float>(1,j);
                _vMPw.at<float>(2,h) = vMPw.at<float>(2,j);
                _vMPw.at<float>(3,h) = vMPw.at<float>(3,j);
                _vLabels.at<float>(h,0) = vLabels.at<float>(j,0);
                _matRefFrame.at<float>(h,0) = matRefFrame.at<float>(j,0);
                _matRefFrame.at<float>(h,1) = matRefFrame.at<float>(j,1);
                _matRefFrame.at<float>(h,2) = matRefFrame.at<float>(j,2);
                _matDepthRefFrame.at<float>(h,0) = matDepthRefFrame.at<float>(j,0);
                h++;
            }
        }

        // resize的操作
        vMPw = _vMPw.colRange(0,h);
        vLabels = _vLabels.rowRange(0,h);
        matRefFrame = _matRefFrame.rowRange(0,h);
        matDepthRefFrame = _matDepthRefFrame.rowRange(0,h);

        if (vAllMPw.empty())
        {
            vAllMPw = vMPw;
            vAllMatRefFrame = matRefFrame;
            vAllLabels = vLabels;
            vAllDepthRefFrame = matDepthRefFrame;
        }
        else
        {
            if (!vMPw.empty())
            {
                hconcat(vAllMPw,vMPw,vAllMPw);//4*（h1+...+h5）
                vconcat(vAllMatRefFrame,matRefFrame,vAllMatRefFrame);
                vconcat(vAllLabels,vLabels,vAllLabels);
                vconcat(vAllDepthRefFrame,matDepthRefFrame,vAllDepthRefFrame);
            }
        }
    }

    cv::Mat vLabels = vAllLabels;

    if (!vAllMPw.empty())
    {
        cv::Mat vMPCurrentFrame = currentFrame.mTcw * vAllMPw;//转到cf系下

        // Divide by last column
        for (int i(0); i < vMPCurrentFrame.cols; i++)// 把所有cf下的地图点转换为X Y Z 1，即cf系下的坐标
        {
            vMPCurrentFrame.at<float>(0,i) /= vMPCurrentFrame.at<float>(3,i);
            vMPCurrentFrame.at<float>(1,i) /= vMPCurrentFrame.at<float>(3,i);
            vMPCurrentFrame.at<float>(2,i) /= vMPCurrentFrame.at<float>(3,i);
            vMPCurrentFrame.at<float>(3,i) /= vMPCurrentFrame.at<float>(3,i);
        }
        cv::Mat matProjDepth = vMPCurrentFrame.row(2);//投影到当前帧相机坐标系的深度

        cv::Mat _vMPCurrentFrame = cv::Mat(vMPCurrentFrame.size(),CV_32F);
        cv::Mat _vAllMatRefFrame = cv::Mat(vAllMatRefFrame.size(),CV_32F);
        cv::Mat _vLabels = cv::Mat(vLabels.size(),CV_32F);
        cv::Mat __vAllDepthRefFrame = cv::Mat(vAllDepthRefFrame.size(),CV_32F);
        int h(0);
        cv::Mat __matProjDepth = cv::Mat(matProjDepth.size(),CV_32F);
        //遍历所有地图点
        for (int i(0); i < matProjDepth.cols; i++)
        {
            if (matProjDepth.at<float>(0,i) < 7)//深度小于7
            {
                __matProjDepth.at<float>(0,h) = matProjDepth.at<float>(0,i);

                _vMPCurrentFrame.at<float>(0,h) = vMPCurrentFrame.at<float>(0,i);
                _vMPCurrentFrame.at<float>(1,h) = vMPCurrentFrame.at<float>(1,i);
                _vMPCurrentFrame.at<float>(2,h) = vMPCurrentFrame.at<float>(2,i);
                _vMPCurrentFrame.at<float>(3,h) = vMPCurrentFrame.at<float>(3,i);

                _vAllMatRefFrame.at<float>(h,0) = vAllMatRefFrame.at<float>(i,0);
                _vAllMatRefFrame.at<float>(h,1) = vAllMatRefFrame.at<float>(i,1);
                _vAllMatRefFrame.at<float>(h,2) = vAllMatRefFrame.at<float>(i,2);

                _vLabels.at<float>(h,0) = vLabels.at<float>(i,0);

                __vAllDepthRefFrame.at<float>(h,0) = vAllDepthRefFrame.at<float>(i,0);

                h++;
            }
        }

        matProjDepth = __matProjDepth.colRange(0,h);
        vMPCurrentFrame = _vMPCurrentFrame.colRange(0,h);//4*h
        vAllMatRefFrame = _vAllMatRefFrame.rowRange(0,h);
        vLabels = _vLabels.rowRange(0,h);
        vAllDepthRefFrame = __vAllDepthRefFrame.rowRange(0,h);

        cv::Mat aux;
        cv::hconcat(cv::Mat::eye(3,3,CV_32F),cv::Mat::zeros(3,1,CV_32F),aux);//3*4
        cv::Mat matCurrentFrame = K*aux*vMPCurrentFrame;//vMPCurrentFrame存的是X Y Z 1 每个点的坐标是一个4*1，转到像素坐标z（u v 1）

        cv::Mat mat2CurrentFrame(matCurrentFrame.cols,2,CV_32F);//存投影到当前帧的像素坐标
        cv::Mat v2AllMatRefFrame(matCurrentFrame.cols,3,CV_32F);//存在参考帧中的像素坐标
        cv::Mat mat2ProjDepth(matCurrentFrame.cols,1,CV_32F);//存投影到当前帧的深度
        cv::Mat v2Labels(matCurrentFrame.cols,1,CV_32F);//存该点属于哪个关键帧
        cv::Mat _vAllDepthRefFrame(matCurrentFrame.cols,1,CV_32F);//存投影点的深度观测值

        int j = 0;
        for (int i(0); i < matCurrentFrame.cols; i++)
        {
            float x = ceil(matCurrentFrame.at<float>(0,i)/matCurrentFrame.at<float>(2,i));//像素是整数
            float y = ceil(matCurrentFrame.at<float>(1,i)/matCurrentFrame.at<float>(2,i));
            if (IsInFrame(x,y,currentFrame))// 这个像素点判定在图像的特定范围内
            {
                const float d = currentFrame.mImDepth.at<float>(y,x);// 取出这一点在深度图里的深度值
                if (d > 0)
                {
                    mat2CurrentFrame.at<float>(j,0) = x;//像素坐标
                    mat2CurrentFrame.at<float>(j,1) = y;
                    v2AllMatRefFrame.at<float>(j,0) = vAllMatRefFrame.at<float>(i,0);
                    v2AllMatRefFrame.at<float>(j,1) = vAllMatRefFrame.at<float>(i,1);
                    v2AllMatRefFrame.at<float>(j,2) = vAllMatRefFrame.at<float>(i,2);
                    _vAllDepthRefFrame.at<float>(j,0) = vAllDepthRefFrame.at<float>(i,0);
                    float d1 = matProjDepth.at<float>(0,i);
                    mat2ProjDepth.at<float>(j,0) = d1;
                    v2Labels.at<float>(j,0) = vLabels.at<float>(i,0);
                    j++;
                }
            }
        }
        vAllDepthRefFrame = _vAllDepthRefFrame.rowRange(0,j);//存投影点的深度观测值
        vAllMatRefFrame = v2AllMatRefFrame.rowRange(0,j);//存在参考帧中的像素坐标
        matProjDepth = mat2ProjDepth.rowRange(0,j);//存投影到当前帧的深度
        matCurrentFrame = mat2CurrentFrame.rowRange(0,j);//存投影到当前帧的像素坐标
        vLabels = v2Labels.rowRange(0,j);//存该点属于哪个关键帧

        // -20 -20
        // -20 -19
        // ... ...
        //  20  19
        //  20  20
        cv::Mat u1((2*mDmax+1)*(2*mDmax+1),2,CV_32F);//[41*41, 2] 干嘛用的....
        int m(0);
        for (int i(-mDmax); i <= mDmax; i++){
            for (int j(-mDmax); j <= mDmax; j++){
                u1.at<float>(m,0) = i;
                u1.at<float>(m,1) = j;
                m++;
            }
        }

        cv::Mat matDepthCurrentFrame(matCurrentFrame.rows,1,CV_32F);
        cv::Mat _matProjDepth(matCurrentFrame.rows,1,CV_32F);
        cv::Mat _matCurrentFrame(matCurrentFrame.rows,2,CV_32F);

        int _s(0);
        //遍历所有点
        for (int i(0); i < matCurrentFrame.rows; i++)
        {
            int s(0);
            cv::Mat _matDiffDepth(u1.rows,1,CV_32F);//记录投影深度-测量深度
            cv::Mat _matDepth(u1.rows,1,CV_32F);
            for (int j(0); j < u1.rows; j++)//这个像素点周围的41*41区域
            {
                int x = (int)matCurrentFrame.at<float>(i,0) + (int)u1.at<float>(j,0);
                int y = (int)matCurrentFrame.at<float>(i,1) + (int)u1.at<float>(j,1);
                float _d = currentFrame.mImDepth.at<float>(y,x);
                //这一点的实际测量值大于0 且小于投影获得的深度
                //注：实际测量值是这些点各自的测量值，投影深度都是同一个
                if ((_d > 0) && (_d < matProjDepth.at<float>(i,0)))
                {
                    _matDepth.at<float>(s,0) = _d;//记录测量值
                    _matDiffDepth.at<float>(s,0) = matProjDepth.at<float>(i,0) - _d;//记录投影深度-测量深度
                    s++;
                }
            }
            if (s > 0)
            {
                _matDepth = _matDepth.rowRange(0,s);
                _matDiffDepth = _matDiffDepth.rowRange(0,s);
                double minVal, maxVal;
                cv::Point minIdx, maxIdx;
                cv::minMaxLoc(_matDiffDepth,&minVal,&maxVal,&minIdx,&maxIdx);
                int xIndex = minIdx.x;// ==0 因为只有一行
                int yIndex = minIdx.y;// 最小深度差的index
                matDepthCurrentFrame.at<float>(_s,0) = _matDepth.at<float>(yIndex,0);//取最小深度差对应的深度测量值
                _matProjDepth.at<float>(_s,0) = matProjDepth.at<float>(i,0);//深度投影值
                _matCurrentFrame.at<float>(_s,0) = matCurrentFrame.at<float>(i,0);//投影的像素坐标
                _matCurrentFrame.at<float>(_s,1) = matCurrentFrame.at<float>(i,1);
                _s++;
            }
        }

        matDepthCurrentFrame = matDepthCurrentFrame.rowRange(0,_s);
        matProjDepth = _matProjDepth.rowRange(0,_s);
        matCurrentFrame = _matCurrentFrame.rowRange(0,_s);

        mDepthThreshold = 0.6;

        cv::Mat matDepthDifference = matProjDepth - matDepthCurrentFrame;

        mVarThreshold = 0.001; //0.040;

        vector<Geometry::DynKeyPoint> vDynPoints;

        for (int i(0); i < matCurrentFrame.rows; i++)
        {
            if (matDepthDifference.at<float>(i,0) > mDepthThreshold)//对于深度差大于0.6的点
            {
                int xIni = (int)matCurrentFrame.at<float>(i,0) - mDmax;
                int yIni = (int)matCurrentFrame.at<float>(i,1) - mDmax;
                int xEnd = (int)matCurrentFrame.at<float>(i,0) + mDmax + 1;
                int yEnd = (int)matCurrentFrame.at<float>(i,1) + mDmax + 1;
                cv::Mat patch = currentFrame.mImDepth.rowRange(yIni,yEnd).colRange(xIni,xEnd);//这一整个41*41的区域的深度测量值
                cv::Mat mean, stddev;
                cv::meanStdDev(patch,mean,stddev);//均值和标准差
                double _stddev = stddev.at<double>(0,0);
                double var = _stddev*_stddev;//方差
                if (var < mVarThreshold)//方差小于这个阈值，说明这一片是连在一起的，可以判定为前景
                {
                    DynKeyPoint dynPoint;//记录这一片的中心点，中心点的参考关键帧
                    dynPoint.mPoint.x = matCurrentFrame.at<float>(i,0);
                    dynPoint.mPoint.y = matCurrentFrame.at<float>(i,1);
                    dynPoint.mRefFrameLabel = vLabels.at<float>(i,0);
                    vDynPoints.push_back(dynPoint);
                }
            }
        }

        return vDynPoints;
    }
    else//找不到参考帧和点，就算了
    {
        vector<Geometry::DynKeyPoint> vDynPoints;
        return vDynPoints;
    }


    // vector<Geometry::DynKeyPoint> vDynPoints;
    // return vDynPoints;
}


cv::Mat Geometry::DepthRegionGrowing(const vector<DynKeyPoint> &vDynPoints,const cv::Mat &imDepth){

    cv::Mat maskG = cv::Mat::zeros(480,640,CV_32F);

    if (!vDynPoints.empty())
    {
        mSegThreshold = 0.20;

        //遍历每个动态中心点
        for (size_t i(0); i < vDynPoints.size(); i++){
            int xSeed = vDynPoints[i].mPoint.x;
            int ySeed = vDynPoints[i].mPoint.y;
            const float d = imDepth.at<float>(ySeed,xSeed);
            if (maskG.at<float>(ySeed,xSeed)!=1. && d > 0)
            {
                cv::Mat J = RegionGrowing(imDepth,xSeed,ySeed,mSegThreshold);//取中心点进行区域生长，阈值是0.2，生成mask
                maskG = maskG | J;
            }
        }

        int dilation_size = 15;
        cv::Mat kernel = getStructuringElement(cv::MORPH_ELLIPSE,
                                               cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                               cv::Point( dilation_size, dilation_size ) );
        maskG.cv::Mat::convertTo(maskG,CV_8U);
        cv::dilate(maskG, maskG, kernel);//进行膨胀操作
    }
    else
    {
        maskG.cv::Mat::convertTo(maskG,CV_8U);
    }

    cv::Mat _maskG = cv::Mat::ones(480,640,CV_8U);
    maskG = _maskG - maskG;

    return maskG;
}



void Geometry::CombineMasks(const ORB_SLAM2::Frame &currentFrame, cv::Mat &mask)
{
    cv::Mat _maskL = cv::Mat::ones(currentFrame.mImMask.size(),currentFrame.mImMask.type());
    _maskL = _maskL - currentFrame.mImMask;

    cv::Mat _maskG = cv::Mat::ones(mask.size(),mask.type());
    _maskG = _maskG - mask;

    cv::Mat _mask = _maskL | _maskG;

    cv::Mat __mask = cv::Mat::ones(_mask.size(),_mask.type());
    __mask = __mask - _mask;
    mask = __mask;

}

float Area(float x1, float x2, float y1, float y2){
    float xc1 = max(x1-0.5,x2-0.5);
    float xc2 = min(x1+0.5,x2+0.5);
    float yc1 = max(y1-0.5,y2-0.5);
    float yc2 = min(y1+0.5,y2+0.5);
    return (xc2-xc1)*(yc2-yc1);
}

void Geometry::FillRGBD(const ORB_SLAM2::Frame &currentFrame,cv::Mat &mask,cv::Mat &imGray,cv::Mat &imDepth){

    cv::Mat imGrayAccumulator = imGray.mul(mask);
    imGrayAccumulator.convertTo(imGrayAccumulator,CV_32F);
    cv::Mat imCounter;
    mask.convertTo(imCounter,CV_32F);
    cv::Mat imDepthAccumulator = imDepth.mul(imCounter);
    imDepthAccumulator.convertTo(imDepthAccumulator,CV_32F);
    cv::Mat imMinDepth = cv::Mat::zeros(imDepth.size(),CV_32F)+100.0;

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = currentFrame.fx;
    K.at<float>(1,1) = currentFrame.fy;
    K.at<float>(0,2) = currentFrame.cx;
    K.at<float>(1,2) = currentFrame.cy;

    for (int i(0); i < mDB.mNumElem; i++){

        ORB_SLAM2::Frame refFrame = mDB.mvDataBase[i];

        cv::Mat vPixels(640*480,2,CV_32F);
        cv::Mat mDepth(640*480,1,CV_32F);

        int n(0);
        for (int j(0); j < 640*480; j++){
            int x = (int)vAllPixels.at<float>(j,0);
            int y = (int)vAllPixels.at<float>(j,1);
            if ((int)refFrame.mImMask.at<uchar>(y,x) == 1){
                const float d = refFrame.mImDepth.at<float>(y,x);
                if (d > 0 && d < 7){
                    vPixels.at<float>(n,0) = vAllPixels.at<float>(j,0);
                    vPixels.at<float>(n,1) = vAllPixels.at<float>(j,1);
                    mDepth.at<float>(n,0) = 1./d;
                    n++;
                }
            }
        }

        vPixels = vPixels.rowRange(0,n);
        mDepth = mDepth.rowRange(0,n);
        hconcat(vPixels,cv::Mat::ones(n,1,CV_32F),vPixels);
        cv::Mat vMPRefFrame = K.inv() * vPixels.t();
        vconcat(vMPRefFrame,mDepth.t(),vMPRefFrame);

        cv::Mat vMPw = refFrame.mTcw.inv() * vMPRefFrame;
        cv::Mat vMPCurrentFrame = currentFrame.mTcw * vMPw;

        // Divide by last column
        for (int j(0); j < vMPCurrentFrame.cols; j++)
        {
            vMPCurrentFrame.at<float>(0,j) /= vMPCurrentFrame.at<float>(3,j);
            vMPCurrentFrame.at<float>(1,j) /= vMPCurrentFrame.at<float>(3,j);
            vMPCurrentFrame.at<float>(2,j) /= vMPCurrentFrame.at<float>(3,j);
            vMPCurrentFrame.at<float>(3,j) /= vMPCurrentFrame.at<float>(3,j);
        }

        cv::Mat matProjDepth = vMPCurrentFrame.row(2);
        cv::Mat aux;
        cv::hconcat(cv::Mat::eye(3,3,CV_32F),cv::Mat::zeros(3,1,CV_32F),aux);
        cv::Mat matCurrentFrame = K*aux*vMPCurrentFrame;

        cv::Mat vProjPixels(matCurrentFrame.cols,2,CV_32F);
        cv::Mat _matProjDepth(matCurrentFrame.cols,1,CV_32F);
        cv::Mat _vPixels(matCurrentFrame.cols,2,CV_32F);

        int p(0);
        for (int j(0); j < matCurrentFrame.cols; j++)
        {
            float x = matCurrentFrame.at<float>(0,j)/matCurrentFrame.at<float>(2,j);
            float y = matCurrentFrame.at<float>(1,j)/matCurrentFrame.at<float>(2,j);
            bool inFrame = (x > 1 && x < (currentFrame.mImDepth.cols - 1) && y > 1 && y < (currentFrame.mImDepth.rows - 1));
            if (inFrame && (mask.at<uchar>(y,x) == 0)){
                vProjPixels.at<float>(p,0) = x;
                vProjPixels.at<float>(p,1) = y;
                _matProjDepth.at<float>(p,0) = matProjDepth.at<float>(0,j);
                _vPixels.at<float>(p,0) = vPixels.at<float>(j,0);
                _vPixels.at<float>(p,1) = vPixels.at<float>(j,1);
                p++;
            }
        }
        vProjPixels = vProjPixels.rowRange(0,p);
        matProjDepth = _matProjDepth.rowRange(0,p);
        vPixels = _vPixels.rowRange(0,p);

        for (int j(0); j< p; j++)
        {


            int _x = (int)vPixels.at<float>(j,0);
            int _y = (int)vPixels.at<float>(j,1);
            float x = vProjPixels.at<float>(j,0);//x of *
            float y = vProjPixels.at<float>(j,1);//y of *
            /*
                -----------
                | A  | B  |
                ----*------ y
                | C  | D  |
                -----------
                     x
            */
            float x_a = floor(x);
            float y_a = floor(y);
            float x_b = ceil(x);
            float y_b = floor(y);
            float x_c = floor(x);
            float y_c = ceil(y);
            float x_d = ceil(x);
            float y_d = ceil(y);

            float weight = 0;

            if( IsInImage(x_a,y_a,imGrayAccumulator)){
                if(abs(imMinDepth.at<float>(y_a,x_a)-matProjDepth.at<float>(j,0)) < MIN_DEPTH_THRESHOLD )
                {
                    weight = Area(x,x_a,y,y_a);
                    imCounter.at<float>(int(y_a),int(x_a))+=weight;
                    imGrayAccumulator.at<float>(int(y_a),int(x_a))+=weight*(float)refFrame.mImGray.at<uchar>(_y,_x);
                    imDepthAccumulator.at<float>(int(y_a),int(x_a))+=weight*matProjDepth.at<float>(j,0);
                    mask.at<uchar>(int(y_a),int(x_a)) = 1;
                }
                else if ((imMinDepth.at<float>(y_a,x_a)-matProjDepth.at<float>(j,0)) > 0)
                {
                    weight = Area(x,x_a,y,y_a);
                    imCounter.at<float>(int(y_a),int(x_a))=weight;
                    imGrayAccumulator.at<float>(int(y_a),int(x_a))=weight*(float)refFrame.mImGray.at<uchar>(_y,_x);
                    imDepthAccumulator.at<float>(int(y_a),int(x_a))=weight*matProjDepth.at<float>(j,0);
                    mask.at<uchar>(int(y_a),int(x_a)) = 1;
                }
                imMinDepth.at<float>(y_a,x_a) = min(imMinDepth.at<float>(y_a,x_a),matProjDepth.at<float>(j,0));
            }
            if( IsInImage(x_b,y_b,imGrayAccumulator) && (x_a != x_b))
            {
                if(abs(imMinDepth.at<float>(y_b,x_b)-matProjDepth.at<float>(j,0)) < MIN_DEPTH_THRESHOLD )
                {
                    weight = Area(x,x_b,y,y_b);
                    imCounter.at<float>(int(y_b),int(x_b))+=weight;
                    imGrayAccumulator.at<float>(int(y_b),int(x_b))+=weight*(float)refFrame.mImGray.at<uchar>(_y,_x);
                    imDepthAccumulator.at<float>(int(y_b),int(x_b))+=weight*matProjDepth.at<float>(j,0);
                    mask.at<uchar>(int(y_b),int(x_b)) = 1;
                }
                else if ((imMinDepth.at<float>(y_b,x_b)-matProjDepth.at<float>(j,0)) > 0)
                {
                    weight = Area(x,x_b,y,y_b);
                    imCounter.at<float>(int(y_b),int(x_b)) = weight;
                    imGrayAccumulator.at<float>(int(y_b),int(x_b)) = weight*(float)refFrame.mImGray.at<uchar>(_y,_x);
                    imDepthAccumulator.at<float>(int(y_b),int(x_b)) = weight*matProjDepth.at<float>(j,0);
                    mask.at<uchar>(int(y_b),int(x_b)) = 1;
                }
                imMinDepth.at<float>(y_b,x_b) = min(imMinDepth.at<float>(y_b,x_b),matProjDepth.at<float>(j,0));
            }
            if( IsInImage(x_c,y_c,imGrayAccumulator) && (y_a != y_c) && (x_b != x_c && y_b != y_c))
            {
                if(abs(imMinDepth.at<float>(y_c,x_c)-matProjDepth.at<float>(j,0)) < MIN_DEPTH_THRESHOLD )
                {
                    weight = Area(x,x_c,y,y_c);
                    imCounter.at<float>(int(y_c),int(x_c))+=weight;
                    imGrayAccumulator.at<float>(int(y_c),int(x_c))+=weight*(float)refFrame.mImGray.at<uchar>(_y,_x);
                    imDepthAccumulator.at<float>(int(y_c),int(x_c))+=weight*matProjDepth.at<float>(j,0);
                    mask.at<uchar>(int(y_c),int(x_c)) = 1;
                }
                else if ((imMinDepth.at<float>(y_c,x_c)-matProjDepth.at<float>(j,0)) > 0)
                {
                    weight = Area(x,x_c,y,y_c);
                    imCounter.at<float>(int(y_c),int(x_c)) = weight;
                    imGrayAccumulator.at<float>(int(y_c),int(x_c)) = weight*(float)refFrame.mImGray.at<uchar>(_y,_x);
                    imDepthAccumulator.at<float>(int(y_c),int(x_c)) = weight*matProjDepth.at<float>(j,0);
                    mask.at<uchar>(int(y_c),int(x_c)) = 1;
                }
                imMinDepth.at<float>(y_c,x_c) = min(imMinDepth.at<float>(y_c,x_c),matProjDepth.at<float>(j,0));

            }
            if( IsInImage(x_d,y_d,imGrayAccumulator) && (x_a != x_d && y_a != y_d) && (y_b != y_d) && (x_d != x_c))
            {
                if (abs(imMinDepth.at<float>(y_d,x_d)-matProjDepth.at<float>(j,0)) < MIN_DEPTH_THRESHOLD )
                {
                    weight = Area(x,x_d,y,y_d);
                    imCounter.at<float>(int(y_d),int(x_d))+=weight;
                    imGrayAccumulator.at<float>(int(y_d),int(x_d))+=weight*(float)refFrame.mImGray.at<uchar>(_y,_x);
                    imDepthAccumulator.at<float>(int(y_d),int(x_d))+=weight*matProjDepth.at<float>(j,0);
                    mask.at<uchar>(int(y_d),int(x_d)) = 1;
                }
                else if ((imMinDepth.at<float>(y_d,x_d)-matProjDepth.at<float>(j,0)) > 0)
                {
                    weight = Area(x,x_d,y,y_d);
                    imCounter.at<float>(int(y_d),int(x_d)) = weight;
                    imGrayAccumulator.at<float>(int(y_d),int(x_d)) = weight*(float)refFrame.mImGray.at<uchar>(_y,_x);
                    imDepthAccumulator.at<float>(int(y_d),int(x_d)) = weight*matProjDepth.at<float>(j,0);
                    mask.at<uchar>(int(y_d),int(x_d)) = 1;
                }
                imMinDepth.at<float>(y_d,x_d) = min(imMinDepth.at<float>(y_d,x_d),matProjDepth.at<float>(j,0));
            }
        }
    }

    imGrayAccumulator = imGrayAccumulator.mul(1/imCounter);
    imDepthAccumulator = imDepthAccumulator.mul(1/imCounter);

    imGrayAccumulator.convertTo(imGrayAccumulator,CV_8U);
    imGray = imGray*0;
    imGrayAccumulator.copyTo(imGray,mask);
    imDepth = imDepth*0;
    imDepthAccumulator.copyTo(imDepth,mask);

}

void Geometry::FillRGBD(const ORB_SLAM2::Frame &currentFrame,cv::Mat &mask,cv::Mat &imGray,cv::Mat &imDepth,cv::Mat &imRGB){

    cv::Mat imGrayAccumulator = imGray.mul(mask);
    imGrayAccumulator.convertTo(imGrayAccumulator,CV_32F);
    cv::Mat bgr[3];
    cv::split(imRGB,bgr);
    cv::Mat imRAccumulator = bgr[2].mul(mask);
    imRAccumulator.convertTo(imRAccumulator,CV_32F);
    cv::Mat imGAccumulator = bgr[1].mul(mask);
    imGAccumulator.convertTo(imGAccumulator,CV_32F);
    cv::Mat imBAccumulator = bgr[0].mul(mask);
    imBAccumulator.convertTo(imBAccumulator,CV_32F);
    cv::Mat imCounter;
    mask.convertTo(imCounter,CV_32F);
    cv::Mat imDepthAccumulator = imDepth.mul(imCounter);
    imDepthAccumulator.convertTo(imDepthAccumulator,CV_32F);
    cv::Mat imMinDepth = cv::Mat::zeros(imDepth.size(),CV_32F)+100.0;

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = currentFrame.fx;
    K.at<float>(1,1) = currentFrame.fy;
    K.at<float>(0,2) = currentFrame.cx;
    K.at<float>(1,2) = currentFrame.cy;

    for (int i(0); i < mDB.mNumElem; i++){

        ORB_SLAM2::Frame refFrame = mDB.mvDataBase[i];
        cv::Mat bgr[3];
        cv::split(refFrame.mImRGB,bgr);
        cv::Mat imR = bgr[2];
        cv::Mat imG = bgr[1];
        cv::Mat imB = bgr[0];

        cv::Mat vPixels(640*480,2,CV_32F);
        cv::Mat mDepth(640*480,1,CV_32F);

        int n(0);
        for (int j(0); j < 640*480; j++){
            int x = (int)vAllPixels.at<float>(j,0);
            int y = (int)vAllPixels.at<float>(j,1);
            if ((int)refFrame.mImMask.at<uchar>(y,x) == 1){
                const float d = refFrame.mImDepth.at<float>(y,x);
                if (d > 0){
                    vPixels.at<float>(n,0) = vAllPixels.at<float>(j,0);
                    vPixels.at<float>(n,1) = vAllPixels.at<float>(j,1);
                    mDepth.at<float>(n,0) = 1./d;
                    n++;
                }
            }
        }

        vPixels = vPixels.rowRange(0,n);
        mDepth = mDepth.rowRange(0,n);
        hconcat(vPixels,cv::Mat::ones(n,1,CV_32F),vPixels);
        cv::Mat vMPRefFrame = K.inv() * vPixels.t();
        vconcat(vMPRefFrame,mDepth.t(),vMPRefFrame);

        cv::Mat vMPw = refFrame.mTcw.inv() * vMPRefFrame;
        cv::Mat vMPCurrentFrame = currentFrame.mTcw * vMPw;

        // Divide by last column
        for (int j(0); j < vMPCurrentFrame.cols; j++)
        {
            vMPCurrentFrame.at<float>(0,j) /= vMPCurrentFrame.at<float>(3,j);
            vMPCurrentFrame.at<float>(1,j) /= vMPCurrentFrame.at<float>(3,j);
            vMPCurrentFrame.at<float>(2,j) /= vMPCurrentFrame.at<float>(3,j);
            vMPCurrentFrame.at<float>(3,j) /= vMPCurrentFrame.at<float>(3,j);
        }

        cv::Mat matProjDepth = vMPCurrentFrame.row(2);
        cv::Mat aux;
        cv::hconcat(cv::Mat::eye(3,3,CV_32F),cv::Mat::zeros(3,1,CV_32F),aux);
        cv::Mat matCurrentFrame = K*aux*vMPCurrentFrame;

        cv::Mat vProjPixels(matCurrentFrame.cols,2,CV_32F);
        cv::Mat _matProjDepth(matCurrentFrame.cols,1,CV_32F);
        cv::Mat _vPixels(matCurrentFrame.cols,2,CV_32F);

        int p(0);
        for (int j(0); j < matCurrentFrame.cols; j++)
        {
            float x = matCurrentFrame.at<float>(0,j)/matCurrentFrame.at<float>(2,j);
            float y = matCurrentFrame.at<float>(1,j)/matCurrentFrame.at<float>(2,j);
            bool inFrame = (x > 1 && x < (currentFrame.mImDepth.cols - 1) && y > 1 && y < (currentFrame.mImDepth.rows - 1));
            if (inFrame && (mask.at<uchar>(y,x) == 0))
            {
                vProjPixels.at<float>(p,0) = x;
                vProjPixels.at<float>(p,1) = y;
                _matProjDepth.at<float>(p,0) = matProjDepth.at<float>(0,j);
                _vPixels.at<float>(p,0) = vPixels.at<float>(j,0);
                _vPixels.at<float>(p,1) = vPixels.at<float>(j,1);
                p++;
            }
        }
        vProjPixels = vProjPixels.rowRange(0,p);
        matProjDepth = _matProjDepth.rowRange(0,p);
        vPixels = _vPixels.rowRange(0,p);

        for (int j(0); j< p; j++)
        {


            int _x = (int)vPixels.at<float>(j,0);
            int _y = (int)vPixels.at<float>(j,1);
            float x = vProjPixels.at<float>(j,0);//x of *
            float y = vProjPixels.at<float>(j,1);//y of *
            /*
                -----------
                | A  | B  |
                ----*------ y
                | C  | D  |
                -----------
                     x
            */
            float x_a = floor(x);
            float y_a = floor(y);
            float x_b = ceil(x);
            float y_b = floor(y);
            float x_c = floor(x);
            float y_c = ceil(y);
            float x_d = ceil(x);
            float y_d = ceil(y);

            float weight = 0;

            if( IsInImage(x_a,y_a,imGrayAccumulator)){
                if(abs(imMinDepth.at<float>(y_a,x_a)-matProjDepth.at<float>(j,0)) < MIN_DEPTH_THRESHOLD )
                {
                    weight = Area(x,x_a,y,y_a);
                    imCounter.at<float>(int(y_a),int(x_a)) += weight;
                    imGrayAccumulator.at<float>(int(y_a),int(x_a)) += weight*(float)refFrame.mImGray.at<uchar>(_y,_x);
                    imRAccumulator.at<float>(int(y_a),int(x_a)) += weight*(float)imR.at<uchar>(_y,_x);
                    imGAccumulator.at<float>(int(y_a),int(x_a)) += weight*(float)imG.at<uchar>(_y,_x);
                    imBAccumulator.at<float>(int(y_a),int(x_a)) += weight*(float)imB.at<uchar>(_y,_x);
                    imDepthAccumulator.at<float>(int(y_a),int(x_a)) += weight*matProjDepth.at<float>(j,0);
                    mask.at<uchar>(int(y_a),int(x_a)) = 1;
                }
                else if ((imMinDepth.at<float>(y_a,x_a)-matProjDepth.at<float>(j,0)) > 0)
                {
                    weight = Area(x,x_a,y,y_a);
                    imCounter.at<float>(int(y_a),int(x_a)) = weight;
                    imGrayAccumulator.at<float>(int(y_a),int(x_a)) = weight*(float)refFrame.mImGray.at<uchar>(_y,_x);
                    imRAccumulator.at<float>(int(y_a),int(x_a)) = weight*(float)imR.at<uchar>(_y,_x);
                    imGAccumulator.at<float>(int(y_a),int(x_a)) = weight*(float)imG.at<uchar>(_y,_x);
                    imBAccumulator.at<float>(int(y_a),int(x_a)) = weight*(float)imB.at<uchar>(_y,_x);
                    imDepthAccumulator.at<float>(int(y_a),int(x_a)) = weight*matProjDepth.at<float>(j,0);
                    mask.at<uchar>(int(y_a),int(x_a)) = 1;
                }
                imMinDepth.at<float>(y_a,x_a) = min(imMinDepth.at<float>(y_a,x_a),matProjDepth.at<float>(j,0));
            }

            if( IsInImage(x_b,y_b,imGrayAccumulator) && (x_a != x_b))
            {
                if(abs(imMinDepth.at<float>(y_b,x_b)-matProjDepth.at<float>(j,0)) < MIN_DEPTH_THRESHOLD )
                {
                    weight = Area(x,x_b,y,y_b);
                    imCounter.at<float>(int(y_b),int(x_b)) += weight;
                    imGrayAccumulator.at<float>(int(y_b),int(x_b)) += weight*(float)refFrame.mImGray.at<uchar>(_y,_x);
                    imRAccumulator.at<float>(int(y_b),int(x_b)) += weight*(float)imR.at<uchar>(_y,_x);
                    imGAccumulator.at<float>(int(y_b),int(x_b)) += weight*(float)imG.at<uchar>(_y,_x);
                    imBAccumulator.at<float>(int(y_b),int(x_b)) += weight*(float)imB.at<uchar>(_y,_x);
                    imDepthAccumulator.at<float>(int(y_b),int(x_b)) += weight*matProjDepth.at<float>(j,0);
                    mask.at<uchar>(int(y_b),int(x_b)) = 1;
                }
                else if ((imMinDepth.at<float>(y_b,x_b)-matProjDepth.at<float>(j,0)) > 0)
                {
                    weight = Area(x,x_b,y,y_b);
                    imCounter.at<float>(int(y_b),int(x_b)) = weight;
                    imGrayAccumulator.at<float>(int(y_b),int(x_b)) = weight*(float)refFrame.mImGray.at<uchar>(_y,_x);
                    imRAccumulator.at<float>(int(y_b),int(x_b)) = weight*(float)imR.at<uchar>(_y,_x);
                    imGAccumulator.at<float>(int(y_b),int(x_b)) = weight*(float)imG.at<uchar>(_y,_x);
                    imBAccumulator.at<float>(int(y_b),int(x_b)) = weight*(float)imB.at<uchar>(_y,_x);
                    imDepthAccumulator.at<float>(int(y_b),int(x_b)) = weight*matProjDepth.at<float>(j,0);
                    mask.at<uchar>(int(y_b),int(x_b)) = 1;
                }
                imMinDepth.at<float>(y_b,x_b) = min(imMinDepth.at<float>(y_b,x_b),matProjDepth.at<float>(j,0));
            }
            if( IsInImage(x_c,y_c,imGrayAccumulator) && (y_a != y_c) && (x_b != x_c && y_b != y_c))
            {
                if(abs(imMinDepth.at<float>(y_c,x_c)-matProjDepth.at<float>(j,0)) < MIN_DEPTH_THRESHOLD )
                {
                    weight = Area(x,x_c,y,y_c);
                    imCounter.at<float>(int(y_c),int(x_c)) += weight;
                    imGrayAccumulator.at<float>(int(y_c),int(x_c)) += weight*(float)refFrame.mImGray.at<uchar>(_y,_x);
                    imRAccumulator.at<float>(int(y_c),int(x_c)) += weight*(float)imR.at<uchar>(_y,_x);
                    imGAccumulator.at<float>(int(y_c),int(x_c)) += weight*(float)imG.at<uchar>(_y,_x);
                    imBAccumulator.at<float>(int(y_c),int(x_c)) += weight*(float)imB.at<uchar>(_y,_x);
                    imDepthAccumulator.at<float>(int(y_c),int(x_c)) += weight*matProjDepth.at<float>(j,0);
                    mask.at<uchar>(int(y_c),int(x_c)) = 1;
                }
                else if ((imMinDepth.at<float>(y_c,x_c)-matProjDepth.at<float>(j,0)) > 0)
                {
                    weight = Area(x,x_c,y,y_c);
                    imCounter.at<float>(int(y_c),int(x_c)) = weight;
                    imGrayAccumulator.at<float>(int(y_c),int(x_c)) = weight*(float)refFrame.mImGray.at<uchar>(_y,_x);
                    imRAccumulator.at<float>(int(y_c),int(x_c)) = weight*(float)imR.at<uchar>(_y,_x);
                    imGAccumulator.at<float>(int(y_c),int(x_c)) = weight*(float)imG.at<uchar>(_y,_x);
                    imBAccumulator.at<float>(int(y_c),int(x_c)) = weight*(float)imB.at<uchar>(_y,_x);
                    imDepthAccumulator.at<float>(int(y_c),int(x_c)) = weight*matProjDepth.at<float>(j,0);
                    mask.at<uchar>(int(y_c),int(x_c)) = 1;
                }
                imMinDepth.at<float>(y_c,x_c) = min(imMinDepth.at<float>(y_c,x_c),matProjDepth.at<float>(j,0));

            }
            if( IsInImage(x_d,y_d,imGrayAccumulator) && (x_a != x_d && y_a != y_d) && (y_b != y_d) && (x_d != x_c))
            {
                if (abs(imMinDepth.at<float>(y_d,x_d)-matProjDepth.at<float>(j,0)) < MIN_DEPTH_THRESHOLD )
                {
                    weight = Area(x,x_d,y,y_d);
                    imCounter.at<float>(int(y_d),int(x_d)) += weight;
                    imGrayAccumulator.at<float>(int(y_d),int(x_d)) += weight*(float)refFrame.mImGray.at<uchar>(_y,_x);
                    imRAccumulator.at<float>(int(y_d),int(x_d)) += weight*(float)imR.at<uchar>(_y,_x);
                    imGAccumulator.at<float>(int(y_d),int(x_d)) += weight*(float)imG.at<uchar>(_y,_x);
                    imBAccumulator.at<float>(int(y_d),int(x_d)) += weight*(float)imB.at<uchar>(_y,_x);
                    imDepthAccumulator.at<float>(int(y_d),int(x_d)) += weight*matProjDepth.at<float>(j,0);
                    mask.at<uchar>(int(y_d),int(x_d)) = 1;
                }
                else if ((imMinDepth.at<float>(y_d,x_d)-matProjDepth.at<float>(j,0)) > 0)
                {
                    weight = Area(x,x_d,y,y_d);
                    imCounter.at<float>(int(y_d),int(x_d)) = weight;
                    imGrayAccumulator.at<float>(int(y_d),int(x_d)) = weight*(float)refFrame.mImGray.at<uchar>(_y,_x);
                    imRAccumulator.at<float>(int(y_d),int(x_d)) = weight*(float)imR.at<uchar>(_y,_x);
                    imGAccumulator.at<float>(int(y_d),int(x_d)) = weight*(float)imG.at<uchar>(_y,_x);
                    imBAccumulator.at<float>(int(y_d),int(x_d)) = weight*(float)imB.at<uchar>(_y,_x);
                    imDepthAccumulator.at<float>(int(y_d),int(x_d)) = weight*matProjDepth.at<float>(j,0);
                    mask.at<uchar>(int(y_d),int(x_d)) = 1;
                }
                imMinDepth.at<float>(y_d,x_d) = min(imMinDepth.at<float>(y_d,x_d),matProjDepth.at<float>(j,0));
            }
        }
    }

    imGrayAccumulator = imGrayAccumulator.mul(1/imCounter);
    imRAccumulator = imRAccumulator.mul(1/imCounter);
    imRAccumulator.convertTo(imRAccumulator,CV_8U);
    cv::Mat imR = cv::Mat::zeros(imRAccumulator.size(),imRAccumulator.type());
    imRAccumulator.copyTo(imR,mask);
    imGAccumulator = imGAccumulator.mul(1/imCounter);
    imGAccumulator.convertTo(imGAccumulator,CV_8U);
    cv::Mat imG = cv::Mat::zeros(imGAccumulator.size(),imGAccumulator.type());
    imGAccumulator.copyTo(imG,mask);
    imBAccumulator = imBAccumulator.mul(1/imCounter);
    imBAccumulator.convertTo(imBAccumulator,CV_8U);
    cv::Mat imB = cv::Mat::zeros(imBAccumulator.size(),imBAccumulator.type());
    imBAccumulator.copyTo(imB,mask);
    imDepthAccumulator = imDepthAccumulator.mul(1/imCounter);

    std::vector<cv::Mat> arrayToMerge;
    arrayToMerge.push_back(imB);
    arrayToMerge.push_back(imG);
    arrayToMerge.push_back(imR);
    cv::merge(arrayToMerge, imRGB);

    imGrayAccumulator.convertTo(imGrayAccumulator,CV_8U);
    imGray = imGray*0;
    imGrayAccumulator.copyTo(imGray,mask);
    imDepth = imDepth*0;
    imDepthAccumulator.copyTo(imDepth,mask);

}

void Geometry::GetClosestNonEmptyCoordinates(const cv::Mat &mask, const int &x, const int &y, int &_x, int &_y)
{
    cv::Mat neigbIni(4,2,CV_32F);
    neigbIni.at<float>(0,0) = -1;
    neigbIni.at<float>(0,1) = 0;
    neigbIni.at<float>(1,0) = 1;
    neigbIni.at<float>(1,1) = 0;
    neigbIni.at<float>(2,0) = 0;
    neigbIni.at<float>(2,1) = -1;
    neigbIni.at<float>(3,0) = 0;
    neigbIni.at<float>(3,1) = 1;

    cv::Mat neigb = neigbIni;

    bool found = false;
    int f(2);

    while (!found)
    {
        for (int j(0); j< 4; j++)
        {
            int xn = x + neigb.at<float>(j,0);
            int yn = y + neigb.at<float>(j,1);
            bool ins = ((xn >= 0) && (yn >= 0) && (xn <= mask.cols) && (yn <= mask.rows));
            if (ins && ((int)mask.at<uchar>(yn,xn) == 1))
            {
                found = true;
                _x = xn;
                _y = yn;
            }
        }
        neigb = f*neigbIni;
        f++;
    }

}


void Geometry::DataBase::InsertFrame2DB(const ORB_SLAM2::Frame &currentFrame){

    if (!IsFull()){
        mvDataBase[mFin] = currentFrame;
        mFin = (mFin + 1) % MAX_DB_SIZE;
        mNumElem += 1;
    }
    else {
        mvDataBase[mIni] = currentFrame;
        mFin = mIni;
        mIni = (mIni + 1) % MAX_DB_SIZE;
    }
}

bool Geometry::DataBase::IsFull(){
    return (mIni == (mFin+1) % MAX_DB_SIZE);
}

cv::Mat Geometry::rotm2euler(const cv::Mat &R){
    assert(isRotationMatrix(R));
    float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );
    bool singular = sy < 1e-6;
    float x, y, z;
    if (!singular)
    {
        x = atan2(R.at<double>(2,1) , R.at<double>(2,2));
        y = atan2(-R.at<double>(2,0), sy);
        z = atan2(R.at<double>(1,0), R.at<double>(0,0));
    }
    else
    {
        x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        y = atan2(-R.at<double>(2,0), sy);
        z = 0;
    }
    cv::Mat res = (cv::Mat_<double>(1,3) << x, y, z);
    return res;
}


bool Geometry::isRotationMatrix(const cv::Mat &R){
    cv::Mat Rt;
    transpose(R,Rt);
    cv::Mat shouldBeIdentity = Rt*R;
    cv::Mat I = cv::Mat::eye(3,3,shouldBeIdentity.type());
    return norm(I,shouldBeIdentity) < 1e-6;
}


bool Geometry::IsInFrame(const float &x, const float &y, const ORB_SLAM2::Frame &Frame)
{
    mDmax = 20;
    return (x > (mDmax + 1) && x < (Frame.mImDepth.cols - mDmax - 1) && y > (mDmax + 1) && y < (Frame.mImDepth.rows - mDmax - 1));
}
bool Geometry::IsInImage(const float &x, const float &y, const cv::Mat image)
{
    return (x >= 0 && x < (image.cols) && y >= 0 && y < image.rows);
}

cv::Mat Geometry::RegionGrowing(const cv::Mat &im,int &x,int &y,const float &reg_maxdist){

    cv::Mat J = cv::Mat::zeros(im.size(),CV_32F);

    float reg_mean = im.at<float>(y,x);
    int reg_size = 1;

    int _neg_free = 10000;
    int neg_free = 10000;
    int neg_pos = -1;
    cv::Mat neg_list = cv::Mat::zeros(neg_free,3,CV_32F);

    double pixdist=0;

    //Neighbor locations (footprint)
    cv::Mat neigb(4,2,CV_32F);
    neigb.at<float>(0,0) = -1;
    neigb.at<float>(0,1) = 0;
    neigb.at<float>(1,0) = 1;
    neigb.at<float>(1,1) = 0;
    neigb.at<float>(2,0) = 0;
    neigb.at<float>(2,1) = -1;
    neigb.at<float>(3,0) = 0;
    neigb.at<float>(3,1) = 1;

    while(pixdist < reg_maxdist && reg_size < im.total())
    {
        for (int j(0); j< 4; j++)
        {
            //Calculate the neighbour coordinate
            int xn = x + neigb.at<float>(j,0);
            int yn = y + neigb.at<float>(j,1);

            bool ins = ((xn >= 0) && (yn >= 0) && (xn < im.cols) && (yn < im.rows));
            if (ins && (J.at<float>(yn,xn) == 0.))
            {
                neg_pos ++;
                neg_list.at<float>(neg_pos,0) = xn;
                neg_list.at<float>(neg_pos,1) = yn;
                neg_list.at<float>(neg_pos,2) = im.at<float>(yn,xn);
                J.at<float>(yn,xn) = 1.;
            }
        }

        // Add a new block of free memory
        if((neg_pos + 10) > neg_free){
            cv::Mat _neg_list = cv::Mat::zeros(_neg_free,3,CV_32F);
            neg_free += 10000;
            vconcat(neg_list,_neg_list,neg_list);
        }

        // Add pixel with intensity nearest to the mean of the region, to the region
        cv::Mat dist;
        for (int i(0); i < neg_pos; i++){
            double d = abs(neg_list.at<float>(i,2) - reg_mean);
            dist.push_back(d);
        }
        double max;
        cv::Point ind, maxpos;
        cv::minMaxLoc(dist, &pixdist, &max, &ind, &maxpos);
        int index = ind.y;

        if (index != -1)
        {
            J.at<float>(y,x) = -1.;
            reg_size += 1;

            // Calculate the new mean of the region
            reg_mean = (reg_mean*reg_size + neg_list.at<float>(index,2))/(reg_size+1);

            // Save the x and y coordinates of the pixel (for the neighbour add proccess)
            x = neg_list.at<float>(index,0);
            y = neg_list.at<float>(index,1);

            // Remove the pixel from the neighbour (check) list
            neg_list.at<float>(index,0) = neg_list.at<float>(neg_pos,0);
            neg_list.at<float>(index,1) = neg_list.at<float>(neg_pos,1);
            neg_list.at<float>(index,2) = neg_list.at<float>(neg_pos,2);
            neg_pos -= 1;
        }
        else
        {
            pixdist = reg_maxdist;
        }

    }

    J = cv::abs(J);
    return(J);
}

}

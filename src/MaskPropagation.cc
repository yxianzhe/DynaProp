/**
* This file is part of DynaProp.
* Copyright (C) 2022 XZ Yuan (Huazhong University of Science and Technology in China)
* For more information see <https://github.com/yxianzhe/DynaProp>.
*/

#include <sys/stat.h>
#include <dirent.h>
#include "MaskPropagation.h"
#include "ORBmatcher.h"
#include "ORBVocabulary.h"
#include "Converter.h"

namespace DynaProp
{

MaskPropagation::MaskPropagation(const string &strVocFile, const string &strSettingsFile):mstrVocFile(strVocFile),mstrSettingsFile(strSettingsFile)
{
    mVocabulary = new ORB_SLAM2::ORBVocabulary();
    bool bVocLoad = mVocabulary->loadFromTextFile(mstrVocFile);
    if(!bVocLoad)
    {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Falied to open at: " << mstrVocFile << endl;
        exit(-1);
    }
    cout << "Vocabulary loaded for MaskPropagation!" << endl << endl;

    mextractor = new ORB_SLAM2::ORBextractor(1000, 1.2, 8, 20, 8);

    cv::FileStorage fSettings(mstrSettingsFile.c_str(), cv::FileStorage::READ);
    if(!fSettings.isOpened())
    {
       cerr << "Failed to open settings file at: " << mstrSettingsFile << endl;
       exit(-1);
    }
    float DepthMapFactor = fSettings["DepthMapFactor"];
    if(fabs(DepthMapFactor)<1e-5)
        mDepthMapFactor=1;
    else
        mDepthMapFactor = 1.0f/DepthMapFactor;

    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    //     |fx  0   cx|
    // K = |0   fy  cy|
    //     |0   0   1 |
    //构造相机内参矩阵
    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    // 图像矫正系数
    // [k1 k2 p1 p2 k3]
    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    //有些相机的畸变系数中会没有k3项
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

}

cv::Mat MaskPropagation::DilaThisMask(const cv::Mat &mask, int dilation_size)
{
    cv::Mat dilamask = mask.clone();
    cv::Mat kernel_dila = getStructuringElement(cv::MORPH_ELLIPSE,
                                           cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                           cv::Point( dilation_size, dilation_size ) );
    cv::dilate(mask,dilamask,kernel_dila);
    return dilamask;
}

cv::Mat MaskPropagation::ErodeThisMask(const cv::Mat &mask, int erode_size)
{
    cv::Mat erodemask = mask.clone();
    cv::Mat kernel_erode = getStructuringElement(cv::MORPH_ELLIPSE,
                                           cv::Size( 2*erode_size + 1, 2*erode_size+1 ),
                                           cv::Point( erode_size, erode_size ) );
    cv::erode(mask,erodemask,erode_size);
    return erodemask;
}

cv::Mat MaskPropagation::RegionGrowing(const cv::Mat &im, int x, int y, float reg_maxdist, const cv::Mat &MaskLimit, const cv::Mat &outside_mask, const int &growing_size, float &reg_mean)
{
    cv::Mat J = cv::Mat::zeros(im.size(),CV_32F);
    cv::Mat dilamask = DilaThisMask(MaskLimit);

    int reg_size = 1;

    int _neg_free = 15000;
    int neg_free = 15000;
    int neg_pos = -1;
    cv::Mat neg_list = cv::Mat::zeros(neg_free,3,CV_32F);

    double pixdist=0;

    //Neighbor locations (footprint)
    cv::Mat neigb(4,2,CV_32F);
    // -1,0  1,0  0,-1  0,1
    neigb.at<float>(0,0) = -growing_size;
    neigb.at<float>(0,1) = 0;
    neigb.at<float>(1,0) = growing_size;
    neigb.at<float>(1,1) = 0;
    neigb.at<float>(2,0) = 0;
    neigb.at<float>(2,1) = -growing_size;
    neigb.at<float>(3,0) = 0;
    neigb.at<float>(3,1) = growing_size;

    while(pixdist < reg_maxdist && reg_size < im.total())
    {
        for (int j(0); j< 4; j++)
        {
            //Calculate the neighbour coordinate
            int xn = x + neigb.at<float>(j,0);
            int yn = y + neigb.at<float>(j,1);

            bool ins = ((xn > 0) && (yn > 0) && (xn < im.cols-1) && (yn < im.rows-1));
            if(ins)
            {
                int outside_limit = (int)outside_mask.at<uchar>(yn,xn);
                int mask_limit = (int)MaskLimit.at<uchar>(yn,xn);
                int dila_limit = (int)dilamask.at<uchar>(yn,xn);
                if(outside_limit==1 && mask_limit!=1 )
                {
                    reg_maxdist = reg_maxdist*0.8;
                    if (reg_maxdist<1.0) reg_maxdist=1.0;
                }
                else if(mask_limit==1)
                {
                    reg_maxdist = reg_maxdist*1.25;
                    if (reg_maxdist>2.0) reg_maxdist=2.0;
                }
                if (dila_limit == 1 && (J.at<float>(yn,xn) == 0.))
                {
                    neg_pos ++;
                    neg_list.at<float>(neg_pos,0) = xn;
                    neg_list.at<float>(neg_pos,1) = yn;
                    neg_list.at<float>(neg_pos,2) = im.at<float>(yn,xn);//坐标，深度值
                    int bound = growing_size/2;
                    for (size_t i = 0; i < growing_size; i++)
                    {
                        for (size_t k = 0; k < growing_size; k++)
                        {
                            if(xn-bound+k>=growing_size/2 && xn-bound+k<640-growing_size/2 && yn-bound+i>=growing_size/2 && yn-bound+i<480-growing_size/2)
                                J.at<float>(yn-bound+i,xn-bound+k) = 1.;//这一块点mask置1
                        }
                    }
                }
            }
        }

        // Add a new block of free memory
        if((neg_pos + 30) > neg_free){
            cv::Mat _neg_list = cv::Mat::zeros(_neg_free,3,CV_32F);
            neg_free += 15000;
            vconcat(neg_list,_neg_list,neg_list);
        }

        // Add pixel with intensity nearest to the mean of the region, to the region
        cv::Mat dist;
        for (int i(0); i < neg_pos; i++){
            double d = abs(neg_list.at<float>(i,2) - reg_mean);
            dist.push_back(d);//深度值之差
        }
        double max;
        cv::Point ind, maxpos;
        cv::minMaxLoc(dist, &max, &pixdist, &ind, &maxpos);
        int index = ind.y;//最小深度值之差的索引


        if (index != -1 && neg_pos >= 0)
        {
            J.at<float>(y,x) = -1.;//中心置成-1
            reg_size += 1;

            // Calculate the new mean of the region
            reg_mean = (reg_mean*reg_size + neg_list.at<float>(index,2))/(reg_size+1);
            // reg_mean = neg_list.at<float>(index,2);

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

    cv::absdiff(J,cv::Scalar::all(0),J);

    return(J);
}

cv::Mat MaskPropagation::DepthRegionGrowing(const std::vector<cv::KeyPoint> &vKeyPoints, const cv::Mat &outside_mask, const cv::Mat &imDepth, const cv::Mat &MaskLimit, const int &growing_size)
{
    cv::Mat maskG = cv::Mat::zeros(480,640,CV_32F);
    regions.clear();

    if (!vKeyPoints.empty())
    {
        float mSegThreshold = 2.0;

        //遍历每个KeyPoint
        for (size_t i(0); i < vKeyPoints.size(); i++){
            int xSeed = vKeyPoints[i].pt.x;
            int ySeed = vKeyPoints[i].pt.y;
            const float d = imDepth.at<float>(ySeed,xSeed);
            // Mat dilamask = DilaThisMask(MaskLimit);
            int limit = (int)MaskLimit.at<uchar>(ySeed,xSeed); //开始生长的点条件苛刻一点，一定在上一帧中这个像素的点也是在mask内，从而确保开始生长点不在mask边缘
            bool bound = (xSeed>=growing_size/2 && xSeed<640-growing_size/2 && ySeed>=growing_size/2 && ySeed<480-growing_size/2); //没有越界
            if (maskG.at<float>(ySeed,xSeed)!=1. && d > 0 && limit==1 && bound)
            {
                float reg_mean = imDepth.at<float>(ySeed,xSeed);
                cv::Mat J = RegionGrowing(imDepth,xSeed,ySeed,mSegThreshold,MaskLimit,outside_mask,growing_size,reg_mean);//取中心点进行区域生长，阈值是2.0，生成mask，前景1
                // maskG = maskG | J;
                cv::bitwise_or(maskG,J,maskG);
                regions.emplace_back(make_pair(vKeyPoints[i], reg_mean));
            }
        }

        int dilation_size = 15;
        cv::Mat kernel = getStructuringElement(cv::MORPH_ELLIPSE,
                                               cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                               cv::Point( dilation_size, dilation_size ) );
        maskG.cv::Mat::convertTo(maskG,CV_8U);
        // cv::dilate(maskG, maskG, kernel);//进行膨胀操作
    }
    else
    {
        maskG.cv::Mat::convertTo(maskG,CV_8U);
    }

    // cv::Mat _maskG = cv::Mat::ones(480,640,CV_8U);
    // maskG = _maskG - maskG;

    cv::Mat _maskG;
    cv::Mat kernel_close = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(15,15));
    cv::morphologyEx(maskG,_maskG,cv::MORPH_CLOSE,kernel_close);
    // maskG = cv::Mat::ones(480,640,CV_8U) - _maskG;
    // cv::subtract(cv::Mat::ones(480,640,CV_8U),_maskG,maskG);

    return _maskG;
}

void MaskPropagation::ComputeBoW(cv::Mat &Descriptors, ORB_SLAM2::ORBVocabulary* Vocabulary, DBoW2::BowVector &BowVec, DBoW2::FeatureVector &FeatVec)
{
    // 判断是否以前已经计算过了，计算过了就跳过
    if(BowVec.empty())
    {
		// 将描述子Descriptors转换为DBOW要求的输入格式
        std::vector<cv::Mat> vCurrentDesc = ORB_SLAM2::Converter::toDescriptorVector(Descriptors);
		// 将特征点的描述子转换成词袋向量mBowVec以及特征向量mFeatVec
        Vocabulary->transform(vCurrentDesc,	//当前的描述子vector
								   BowVec,			//输出，词袋向量，记录的是单词的id及其对应权重TF-IDF值
								   FeatVec,		//输出，记录node id及其对应的图像 feature对应的索引
								   4);				//4表示从叶节点向前数的层数
    }
}

void MaskPropagation::ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1=0;
    int max2=0;
    int max3=0;

    for(int i=0; i<L; i++)
    {
        const int s = histo[i].size();
        if(s>max1)
        {
            max3=max2;
            max2=max1;
            max1=s;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        }
        else if(s>max2)
        {
            max3=max2;
            max2=s;
            ind3=ind2;
            ind2=i;
        }
        else if(s>max3)
        {
            max3=s;
            ind3=i;
        }
    }

    // 如果差距太大了,说明次优的非常不好,这里就索性放弃了,都置为-1
    if(max2<0.1f*(float)max1)
    {
        ind2=-1;
        ind3=-1;
    }
    else if(max3<0.1f*(float)max1)
    {
        ind3=-1;
    }
}

int MaskPropagation::SearchByBoW(const DBoW2::FeatureVector &vFeatVec1, const DBoW2::FeatureVector &vFeatVec2, 
        const std::vector<cv::KeyPoint> &vKeyPoint1, const std::vector<cv::KeyPoint> &vKeyPoint2,
        const cv::Mat Descriptors1, const cv::Mat Descriptors2, 
        std::vector<cv::DMatch> &matches, bool mbCheckOrientation, float mfNNratio)
{
    int nmatches=0;
    int HISTO_LENGTH = 30;
    int TH_LOW = 50;

    // 特征点角度旋转差统计用的直方图
    std::vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    // 将0~360的数转换到0~HISTO_LENGTH的系数
    // !原作者代码是 const float factor = 1.0f/HISTO_LENGTH; 是错误的，更改为下面代码  
    const float factor = HISTO_LENGTH/360.0f;

    // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
    // 将属于同一节点(特定层)的ORB特征进行匹配
    DBoW2::FeatureVector::const_iterator KFit = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator Fit = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator KFend = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator Fend = vFeatVec2.end();

    while(KFit != KFend && Fit != Fend)
    {
        // Step 1：分别取出属于同一node的ORB特征点(只有属于同一node，才有可能是匹配点)
        // first 元素就是node id，遍历
        if(KFit->first == Fit->first) 
        {
            // second 是该node内存储的feature index
            const std::vector<unsigned int> vIndicesKF = KFit->second;
            const std::vector<unsigned int> vIndicesF = Fit->second;

            // Step 2：遍历KF中属于该node的特征点
            for(size_t iKF=0; iKF<vIndicesKF.size(); iKF++)
            {
                // 关键帧该节点中特征点的索引
                const unsigned int realIdxKF = vIndicesKF[iKF];

                // 取出该特征点
                cv::KeyPoint kp1 = vKeyPoint1[realIdxKF]; 

                const cv::Mat &dKF= Descriptors1.row(realIdxKF); // 取出该特征对应的描述子

                int bestDist1=256; // 最好的距离（最小距离）
                int bestIdxF =-1 ;
                int bestDist2=256; // 次好距离（倒数第二小距离）

                // Step 3：遍历F中属于该node的特征点，寻找最佳匹配点
                for(size_t iF=0; iF<vIndicesF.size(); iF++)
                {
                    // 和上面for循环重名了,这里的realIdxF是指普通帧该节点中特征点的索引
                    const unsigned int realIdxF = vIndicesF[iF];

                    const cv::Mat &dF = Descriptors2.row(realIdxF); // 取出F中该特征对应的描述子
                    // 计算描述子的距离
                    const int dist =  ORB_SLAM2::ORBmatcher::DescriptorDistance(dKF,dF); 

                    // 遍历，记录最佳距离、最佳距离对应的索引、次佳距离等
                    // 如果 dist < bestDist1 < bestDist2，更新bestDist1 bestDist2
                    if(dist<bestDist1)
                    {
                        bestDist2=bestDist1;
                        bestDist1=dist;
                        bestIdxF=realIdxF;
                    }
                    // 如果bestDist1 < dist < bestDist2，更新bestDist2
                    else if(dist<bestDist2) 
                    {
                        bestDist2=dist;
                    }
                }

                // Step 4：根据阈值 和 角度投票剔除误匹配
                // Step 4.1：第一关筛选：匹配距离必须小于设定阈值
                if(bestDist1<=TH_LOW) 
                {
                    // Step 4.2：第二关筛选：最佳匹配比次佳匹配明显要好，那么最佳匹配才真正靠谱
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                    {
                        cv::DMatch match;
                        match.queryIdx =  realIdxKF;
                        match.trainIdx =  bestIdxF;
                        match.distance =  bestDist1;

                        // Step 4.4：计算匹配点旋转角度差所在的直方图
                        if(mbCheckOrientation)
                        {
                            // angle：每个特征点在提取描述子时的旋转主方向角度，如果图像旋转了，这个角度将发生改变
                            // 所有的特征点的角度变化应该是一致的，通过直方图统计得到最准确的角度变化值
                            float rot = kp1.angle-vKeyPoint2[bestIdxF].angle;// 该特征点的角度变化值
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);// 将rot分配到bin组, 四舍五入, 其实就是离散到对应的直方图组中
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdxF);       // 直方图统计
                        }
                        nmatches++;
                        matches.push_back(match);
                    }
                }

            }
            KFit++;
            Fit++;
        }
        else if(KFit->first < Fit->first)
        {
            // 对齐
            KFit = vFeatVec1.lower_bound(Fit->first);
        }
        else
        {
            // 对齐
            Fit = vFeatVec2.lower_bound(KFit->first);
        }
    }

    // Step 5 根据方向剔除误匹配的点
    if(mbCheckOrientation)
    {
        // index
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        // 筛选出在旋转角度差落在在直方图区间内数量最多的前三个bin的索引
        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            // 如果特征点的旋转角度变化量属于这三个组，则保留
            if(i==ind1 || i==ind2 || i==ind3)
                continue;

            // 剔除掉不在前三的匹配对，因为他们不符合“主流旋转方向”  
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                matches.erase(matches.begin()+rotHist[i][j]);
                // matches[rotHist[i][j]]=static_cast<DMatch>(NULL);
                nmatches--;
            }
        }
    }

    return nmatches;
}

void MaskPropagation::ConvertDepth(cv::Mat &depthimg)
{
    if((fabs(mDepthMapFactor-1.0f)>1e-5) || depthimg.type()!=CV_32F)
    depthimg.convertTo(  //将图像转换成为另外一种数据类型,具有可选的数据大小缩放系数
        depthimg,            //输出图像
        CV_32F,             //输出图像的数据类型
        mDepthMapFactor);   //缩放系数
}

void MaskPropagation::UndistortKeyPoints()
{
    // Step 1 如果第一个畸变参数为0，不需要矫正。第一个畸变参数k1是最重要的，一般不为0，为0的话，说明畸变参数都是0
	//变量mDistCoef中存储了opencv指定格式的去畸变参数，格式为：(k1,k2,p1,p2,k3)
    if(mDistCoef.at<float>(0)==0.0)
    {
        mnewun_keypoints=mnewimg_keypoints;
        return;
    }

    int N = mnewimg_keypoints.size();
    // Step 2 如果畸变参数不为0，用OpenCV函数进行畸变矫正
    // Fill matrix with points
    // N为提取的特征点数量，为满足OpenCV函数输入要求，将N个特征点保存在N*2的矩阵中
    cv::Mat mat(N,2,CV_32F);
	//遍历每个特征点，并将它们的坐标保存到矩阵中
    for(int i=0; i<N; i++)
    {
		//然后将这个特征点的横纵坐标分别保存
        mat.at<float>(i,0)=mnewimg_keypoints[i].pt.x;
        mat.at<float>(i,1)=mnewimg_keypoints[i].pt.y;
    }

    // Undistort points
    // 函数reshape(int cn,int rows=0) 其中cn为更改后的通道数，rows=0表示这个行将保持原来的参数不变
    //为了能够直接调用opencv的函数来去畸变，需要先将矩阵调整为2通道（对应坐标x,y） 
    mat=mat.reshape(2);
    cv::undistortPoints(	
		mat,				//输入的特征点坐标
		mat,				//输出的校正后的特征点坐标覆盖原矩阵
		mK,					//相机的内参数矩阵
		mDistCoef,			//相机畸变参数矩阵
		cv::Mat(),			//一个空矩阵，对应为函数原型中的R
		mK); 				//新内参数矩阵，对应为函数原型中的P
	
	//调整回只有一个通道，回归我们正常的处理方式
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector
    // Step 存储校正后的特征点
    mnewun_keypoints.resize(N);
	//遍历每一个特征点
    for(int i=0; i<N; i++)
    {
		//根据索引获取这个特征点
		//注意之所以这样做而不是直接重新声明一个特征点对象的目的是，能够得到源特征点对象的其他属性
        cv::KeyPoint kp = mnewimg_keypoints[i];
		//读取校正后的坐标并覆盖老坐标
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mnewun_keypoints[i]=kp;
    }
}

void MaskPropagation::FindNewKeypoints()
{
    mtarget_points.clear();
    moutside_points.clear();
    std::vector<cv::KeyPoint> keypoint_1, keypoint_2, undistored_1, undistored_2;
    cv::Mat descriptors_1, descriptors_2;

    // 仿函数，提取特征点和计算描述子
    keypoint_1 = mnewimg_keypoints;
    undistored_1 = mnewun_keypoints;
    descriptors_1 = mnewimg_descriptors;
    (*mextractor)(mnewimg,cv::Mat(),keypoint_2,descriptors_2);

    mnewimg_keypoints = keypoint_2;
    mnewimg_descriptors = descriptors_2;
    
    // 去畸变，得到去畸变后的像素坐标
    UndistortKeyPoints();
    undistored_2 = mnewun_keypoints;
    // for(size_t i(0); i<keypoint_2.size(); ++i)
    // {
    //     cout << "keypoint_2[" << i << "]: " << keypoint_2[i].pt.x << "," << keypoint_2[i].pt.y << "   "
    //          << "undistored_2[" << i << "]: " << undistored_2[i].pt.x << "," << undistored_2[i].pt.y << endl;
    // }

    //把描述子转换为BoW形式
    DBoW2::BowVector BowVec1, BowVec2;
    DBoW2::FeatureVector FeatVec1, FeatVec2;
    BowVec1 = mnewimg_BowVec;
    FeatVec1 = mnewimg_FeatVec;
    ComputeBoW(descriptors_2, mVocabulary, BowVec2, FeatVec2);

    //进行特征点匹配
    std::vector<cv::DMatch> matches;
    int num_matches = SearchByBoW(FeatVec1,FeatVec2,keypoint_1,keypoint_2,descriptors_1,descriptors_2,matches,true,0.6);

    std::vector<cv::Point2f> un_match_1, un_match_2; // 所有匹配上的去畸变点，用来计算点到极线的距离
    std::vector<cv::Point2f> un_static_1, un_static_2; // 用来计算F矩阵的点
    std::vector<float> P_old_d = P_d;
    P_d.clear();
    P_d.resize(mnewimg_keypoints.size(), 0.5);
    for( size_t i(0); i<matches.size(); ++i)
    {
        int val = (int)mlastmask.at<uchar>(keypoint_1[matches[i].queryIdx].pt.y, keypoint_1[matches[i].queryIdx].pt.x); // at行列的索引
        if(val==1)
        {
            mtarget_points.emplace_back(keypoint_2[matches[i].trainIdx]);
            // mtarget_descriptors.push_back(descriptors_2.row(matches[i].trainIdx));
        }
        else
        {
            moutside_points.emplace_back(keypoint_2[matches[i].trainIdx]);
            un_static_1.emplace_back(undistored_1[matches[i].queryIdx].pt);
            un_static_2.emplace_back(undistored_2[matches[i].trainIdx].pt);
        }
        P_d[matches[i].trainIdx] = P_old_d[matches[i].queryIdx];
        un_match_1.emplace_back(undistored_1[matches[i].queryIdx].pt);
        un_match_2.emplace_back(undistored_2[matches[i].trainIdx].pt);
    }

    //计算Fundamental Matrix
    // cout<< "size of matches: "<<undistored_1.size()<<","<<undistored_2.size()<<","<<un_static_1.size()<<","<<un_static_2.size()<<endl;
    if(matches.size()>8){
        F = cv::findFundamentalMat(un_static_1,un_static_2,cv::FM_RANSAC,3,0.99);
    }
    // cout << "Fundamental Matrix is: " << F << endl;
    P_g_d.clear();
    P_g_d.resize(mnewimg_keypoints.size(), -1);
    
    boost::math::normal_distribution<> norm(0,1);

    for( size_t i(0); i<matches.size(); ++i)
    {
        // cout<< "Check F for match " << i << ":" <<endl;
        cv::Mat y1 = ( cv::Mat_<double> (3,1) << un_match_1[i].x, un_match_1[i].y, 1 );
        cv::Mat y2 = ( cv::Mat_<double> (3,1) << un_match_2[i].x, un_match_2[i].y, 1 );
        // cout<<"y1, y2 = "<< y1 << "," << y2 <<endl;
        cv::Mat l = F * y1;
        cv::Mat d = y2.t() * l; //对极约束表达式，应该等于0
        double dist = fabs(d.at<double>(0,0)) / sqrt(l.at<double>(0,0) * l.at<double>(0,0) + l.at<double>(1,0) * l.at<double>(1,0));
        dist = dist<2? dist:2;
        double pgd = 2 * boost::math::cdf(norm,dist) - 1;
        // cout<< " epipolar constraint = " << d <<endl;
        // cout<< " dist = " << dist <<endl;
        // cout<< " probability of geometry dynamic: " << pgd << endl;
        P_g_d[matches[i].trainIdx] = pgd;
    }

    mnewimg_BowVec = BowVec2;
    mnewimg_FeatVec = FeatVec2;
}

cv::Mat MaskPropagation::GetMaskbyPropagation(const cv::Mat &newimg, const cv::Mat &newdepth, std::string dir, std::string name)
{
    mnewimg = newimg;
    mnewdepth = newdepth;

    if(mnewimg.channels()==3)
        cv::cvtColor(mnewimg,mnewimg,CV_RGB2GRAY);
    else if(mnewimg.channels()==4)
        cv::cvtColor(mnewimg,mnewimg,CV_RGBA2GRAY);

    ConvertDepth(mnewdepth);

    cv::Mat maskthisframe = cv::imread(dir+"/"+name,CV_LOAD_IMAGE_UNCHANGED);
    // if(maskthisframe.empty())
    if(1)
    {
        if(!mlastdepth.empty() && !mlastimg.empty() && !mlastmask.empty())
        {
            FindNewKeypoints();//找到一些新图像的特征点

            cv::Mat outside_mask = cv::Mat::zeros(480,640,CV_8U); //区域生长需要停止的地方
            for (size_t i = 0; i < moutside_points.size(); i++)
            {
                int tempy = moutside_points[i].pt.y;
                int tempx = moutside_points[i].pt.x;
                int bound = GROW_SIZE/2;
                if( tempx>=bound && tempx<640-bound && tempy>=bound && tempy<480-bound )
                {
                    for (size_t j = 0; j < GROW_SIZE; j++)
                    {
                        for (size_t k = 0; k < GROW_SIZE; k++)
                        {
                            outside_mask.at<uchar>(tempy-bound+j, tempx-bound+k)=1;
                        }
                    }
                }
            }

            maskthisframe = DepthRegionGrowing(mtarget_points, outside_mask, mnewdepth, mlastmask, GROW_SIZE);

            P_s_d.clear();
            P_s_d.resize(mnewimg_keypoints.size()); 
            cv::Mat imgp = mnewimg;
            int Ns = 0;
            int Ng = 0;
            for (size_t i(0); i < mnewimg_keypoints.size(); i++)
            {
                float min_dist = 640000;
                float min_ddepth = 10.0;
                if(mnewdepth.at<float>(mnewimg_keypoints[i].pt.y, mnewimg_keypoints[i].pt.x) < 0.01) {
                    P_s_d[i] = 0.5;
                } else {
                    for(auto &reg:regions)
                    {
                        float dist = (reg.first.pt.y - mnewimg_keypoints[i].pt.y) * (reg.first.pt.y - mnewimg_keypoints[i].pt.y)
                                    +(reg.first.pt.x - mnewimg_keypoints[i].pt.x) * (reg.first.pt.x - mnewimg_keypoints[i].pt.x);
                        if(dist < min_dist) {
                            min_dist = dist;
                            min_ddepth = fabs(mnewdepth.at<float>(mnewimg_keypoints[i].pt.y, mnewimg_keypoints[i].pt.x) - mnewdepth.at<float>(reg.second));
                        }
                    }
                    if(min_ddepth < 2.5) {
                        P_s_d[i] = 1.0;
                    }else {
                        // P_s_d[i] = 1.0 / (exp(-0.5 / min_ddepth) + 1);
                        P_s_d[i] = 2 * (1.0 / (exp(-0.5 / min_ddepth) + 1) - 0.5);
                    }
                }
                if(P_g_d[i] !=-1) {
                    if(P_g_d[i] > 0.8) {
                        Ng++;
                    }
                    if(P_s_d[i] > 0.8) {
                        Ns++;
                    }
                }
                // if(P_g_d[i] > 0.75) {
                //     cv::circle(imgp, mnewimg_keypoints[i].pt, 3, 0, -1);
                // }
                // else if (P_g_d[i] == -1) {
                //     cv::circle(imgp, mnewimg_keypoints[i].pt, 3, 125, -1);
                // }
                // else {
                //     cv::circle(imgp, mnewimg_keypoints[i].pt, 3, 255, -1);
                // }
                // std::cout << "keypoint " << i << " minddepth: " << min_ddepth << std::endl;
                // std::cout << "keypoint " << i << " (" << mnewimg_keypoints[i].pt.y << "," << mnewimg_keypoints[i].pt.x << ") " << "P_s_d is: " << P_s_d.back() << std::endl;
            }
            float w = (Ng+Ns==0)? 0 : Ng / (Ng+Ns);
            P_o_d.clear();
            P_o_d.resize(mnewimg_keypoints.size());
            for (size_t i(0); i < mnewimg_keypoints.size(); i++) {
                if(P_g_d[i] !=-1){
                    P_o_d[i] = w * P_g_d[i] + (1-w) * P_s_d[i];
                }else {
                    P_o_d[i] = P_s_d[i];
                }
                float pd = P_o_d[i] * P_d[i];
                float ps = (1 - P_o_d[i]) * (1 - P_d[i]);
                P_d[i] = pd / (pd + ps);
                // if(P_d[i] > 0.75) {
                //     cv::circle(imgp, mnewimg_keypoints[i].pt, 3, 0, -1);
                // }
                // else {
                //     cv::circle(imgp, mnewimg_keypoints[i].pt, 3, 255, -1);
                // }
            }
            // cv::imshow("imgp:" , imgp);
            // cv::waitKey(0);
            if(dir.compare("no_save")!=0)
            {
                DIR* _dir = opendir(dir.c_str());
                if (_dir) {closedir(_dir);}
                else if (ENOENT == errno)
                {
                    const int check = mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
                    if (check == -1) {
                        std::string str = dir;
                        str.replace(str.end() - 6, str.end(), "");
                        mkdir(str.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
                    }
                }
                cv::imwrite(dir+"/"+name,maskthisframe);
            }
        }
    }

    mnewmask = maskthisframe;
    UpdateImg(mnewimg);
    UpdateDepth(mnewdepth);
    UpdateMask(mnewmask);

    return mlastmask;

}

void MaskPropagation::GetMaskbySegmentation(const cv::Mat &newimg, const cv::Mat &newdepth, const cv::Mat &newmask)
{
    mnewimg = newimg;
    mnewdepth = newdepth;
    mnewmask = newmask;

    if(mnewimg.channels()==3)
        cv::cvtColor(mnewimg,mnewimg,CV_RGB2GRAY);
    else if(mnewimg.channels()==4)
        cv::cvtColor(mnewimg,mnewimg,CV_RGBA2GRAY);

    ConvertDepth(mnewdepth);
    std::vector<cv::KeyPoint> keypoint_1, undistored_1, keypoint_2, undistored_2;
    cv::Mat descriptors_1, descriptors_2;
    DBoW2::BowVector BowVec1, BowVec2;
    DBoW2::FeatureVector FeatVec1, FeatVec2;

    if(!mnewimg_keypoints.empty()) {
        keypoint_1 = mnewimg_keypoints;
        undistored_1 = mnewun_keypoints;
        descriptors_1 = mnewimg_descriptors;
        BowVec1 = mnewimg_BowVec;
        FeatVec1 = mnewimg_FeatVec;
    }
    
    // 仿函数，提取特征点和计算描述子
    (*mextractor)(mnewimg,cv::Mat(),keypoint_2,descriptors_2);

    //把描述子转换为BoW形式
    ComputeBoW(descriptors_2, mVocabulary, BowVec2, FeatVec2);

    mnewimg_keypoints = keypoint_2;
    mnewimg_descriptors = descriptors_2;

    UndistortKeyPoints();
    undistored_2 = mnewun_keypoints;
    mnewimg_BowVec = BowVec2;
    mnewimg_FeatVec = FeatVec2;

    P_g_d.clear();
    P_g_d.resize(mnewimg_keypoints.size(), -1);
    if(!keypoint_1.empty()) {
        //进行特征点匹配
        std::vector<cv::DMatch> matches;
        int num_matches = SearchByBoW(FeatVec1,FeatVec2,keypoint_1,keypoint_2,descriptors_1,descriptors_2,matches,true,0.6);

        std::vector<cv::Point2f> un_match_1, un_match_2; // 所有匹配上的去畸变点，用来计算点到极线的距离
        std::vector<cv::Point2f> un_static_1, un_static_2; // 用来计算F矩阵的点
        std::vector<float> P_old_d = P_d;
        P_d.clear();
        P_d.resize(mnewimg_keypoints.size(), 0.5);
        for( size_t i(0); i<matches.size(); ++i)
        {
            int val = (int)mlastmask.at<uchar>(keypoint_1[matches[i].queryIdx].pt.y, keypoint_1[matches[i].queryIdx].pt.x);
            if(val==0)
            {
                un_static_1.emplace_back(undistored_1[matches[i].queryIdx].pt);
                un_static_2.emplace_back(undistored_2[matches[i].trainIdx].pt);
            }
            P_d[matches[i].trainIdx] = P_old_d[matches[i].queryIdx];
            un_match_1.emplace_back(undistored_1[matches[i].queryIdx].pt);
            un_match_2.emplace_back(undistored_2[matches[i].trainIdx].pt);
        }

        //计算Fundamental Matrix
        if(matches.size()>8){
            F = cv::findFundamentalMat(un_static_1,un_static_2,cv::FM_RANSAC,3,0.99);
        }
        // cout << "Fundamental Matrix is: " << F << endl;

        boost::math::normal_distribution<> norm(0,1);

        for( size_t i(0); i<matches.size(); ++i)
        {
            // cout<< "Check F for match " << i << ":" <<endl;
            cv::Mat y1 = ( cv::Mat_<double> (3,1) << un_match_1[i].x, un_match_1[i].y, 1 );
            cv::Mat y2 = ( cv::Mat_<double> (3,1) << un_match_2[i].x, un_match_2[i].y, 1 );
            // cout<<"y1, y2 = "<< y1 << "," << y2 <<endl;
            cv::Mat l = F * y1;
            cv::Mat d = y2.t() * l; //对极约束表达式，应该等于0
            double dist = fabs(d.at<double>(0,0)) / sqrt(l.at<double>(0,0) * l.at<double>(0,0) + l.at<double>(1,0) * l.at<double>(1,0));
            dist = dist<2? dist:2;
            double pgd = 2 * boost::math::cdf(norm,dist) - 1;
            // cout<< " epipolar constraint = " << d <<endl;
            // cout<< " dist = " << dist <<endl;
            // cout<< " probability of geometry dynamic: " << pgd << endl;
            P_g_d[matches[i].trainIdx] = pgd;
        }
    } else {
        P_d.resize(mnewimg_keypoints.size(), 0.5);
    }

    UpdateImg(mnewimg);
    UpdateDepth(mnewdepth);
    UpdateMask(mnewmask);

    P_s_d.clear();
    P_s_d.resize(mnewimg_keypoints.size());
    cv::Mat imgp = mnewimg;
    int Ns = 0;
    int Ng = 0;
    for (size_t i(0); i < mnewimg_keypoints.size(); i++) 
    {
        if(mnewmask.at<uchar>(mnewimg_keypoints[i].pt.y, mnewimg_keypoints[i].pt.x)==1) {
            P_s_d[i] = 1.0;
        } else {
            P_s_d[i] = 0.0;
        }
        if(P_g_d[i] !=-1) {
            if(P_g_d[i] > 0.8) {
                Ng++;
            }
            if(P_s_d[i] > 0.8) {
                Ns++;
            }
        }
        // if(P_g_d[i] > 0.75) {
        //     cv::circle(imgp, mnewimg_keypoints[i].pt, 3, 0, -1);
        // }
        // else if(P_g_d[i] == -1) {
        //     cv::circle(imgp, mnewimg_keypoints[i].pt, 3, 125, -1);
        // }
        // else {
        //     cv::circle(imgp, mnewimg_keypoints[i].pt, 3, 255, -1);
        // }
    }
    float w = (Ng+Ns==0)? 0 : Ng / (Ng+Ns);
    P_o_d.clear();
    P_o_d.resize(mnewimg_keypoints.size());
    for (size_t i(0); i < mnewimg_keypoints.size(); i++) {
        if(P_g_d[i] !=-1){
            P_o_d[i] = w * P_g_d[i] + (1-w) * P_s_d[i];
        }else {
            P_o_d[i] = P_s_d[i];
        }
        float pd = P_o_d[i] * P_d[i];
        float ps = (1 - P_o_d[i]) * (1 - P_d[i]);
        P_d[i] = pd / (pd + ps);
        // if(P_d[i] > 0.75) {
        //     cv::circle(imgp, mnewimg_keypoints[i].pt, 3, 0, -1);
        // }
        // else {
        //     cv::circle(imgp, mnewimg_keypoints[i].pt, 3, 255, -1);
        // }
    }
    // cv::imshow("imgp:" , imgp);
    // cv::waitKey(0);
}

void MaskPropagation::SetRefMask(const cv::Mat &newimg, const cv::Mat &newdepth, const cv::Mat &newmask)
{
    mnewimg = newimg;
    mnewdepth = newdepth;
    mnewmask = newmask;

    if(mnewimg.channels()==3)
        cv::cvtColor(mnewimg,mnewimg,CV_RGB2GRAY);
    else if(mnewimg.channels()==4)
        cv::cvtColor(mnewimg,mnewimg,CV_RGBA2GRAY);

    ConvertDepth(mnewdepth);
    std::vector<cv::KeyPoint> keypoint_2;
    cv::Mat descriptors_2;
    DBoW2::BowVector BowVec2;
    DBoW2::FeatureVector FeatVec2;
    
    // 仿函数，提取特征点和计算描述子
    (*mextractor)(mnewimg,cv::Mat(),keypoint_2,descriptors_2);

    //把描述子转换为BoW形式
    ComputeBoW(descriptors_2, mVocabulary, BowVec2, FeatVec2);

    mnewimg_keypoints = keypoint_2;
    mnewimg_descriptors = descriptors_2;

    UndistortKeyPoints();
    mnewimg_BowVec = BowVec2;
    mnewimg_FeatVec = FeatVec2;

    UpdateImg(mnewimg);
    UpdateDepth(mnewdepth);
    UpdateMask(mnewmask);

    P_d.clear();
    P_d.resize(mnewimg_keypoints.size(), 0.5);
    P_o_d.clear();
    P_o_d.resize(mnewimg_keypoints.size());
    for (size_t i(0); i < mnewimg_keypoints.size(); i++) 
    {
        if(mnewmask.at<uchar>(mnewimg_keypoints[i].pt.y, mnewimg_keypoints[i].pt.x)==1){
            P_o_d[i] = 1.0;
        }else{
            P_o_d[i] = 0.0;
        }
        float pd = P_o_d[i] * P_d[i];
        float ps = (1 - P_o_d[i]) * (1 - P_d[i]);
        P_d[i] = pd / (pd + ps);
    }
}

void MaskPropagation::UpdateImg(const cv::Mat &img)
{
    mlastimg = img;
}

void MaskPropagation::UpdateDepth(const cv::Mat &depth)
{
    mlastdepth = depth;
}

void MaskPropagation::UpdateMask(const cv::Mat &mask)
{
    mlastmask = mask;
}

const std::vector<cv::KeyPoint>& MaskPropagation::GetNewImgKeyPoints() 
{
    return mnewimg_keypoints;
}

const cv::Mat& MaskPropagation::GetNewImgDescriptors()
{
    return mnewimg_descriptors;
}

const std::vector<float>& MaskPropagation::GetNewImgDynamicProbablity()
{
    return P_d;
}

}
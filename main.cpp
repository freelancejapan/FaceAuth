#include <iostream>
#include <librealsense2/rs.hpp>
#include <chrono>

#include "cv-helpers.hpp"
#include "filter.hpp"

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include <dlib/dnn.h>

#include <random>
#include <numeric>

#include "svm.hpp"

template<template<int, template<typename> class, int, typename> class block, int N,
        template<typename> class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N, BN, 1, dlib::tag1<SUBNET>>>;
template<template<int, template<typename> class, int, typename> class block, int N,
        template<typename> class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2, 2, 2, 2, dlib::skip1<dlib::tag2<block<N, BN, 2, dlib::tag1<SUBNET>>>>>>;
template<int N, template<typename> class BN, int stride, typename SUBNET>
using block  = BN<dlib::con<N, 3, 3, 1, 1, dlib::relu<BN<dlib::con<N, 3, 3, stride, stride, SUBNET>>>>>;
template<int N, typename SUBNET> using ares      = dlib::relu<residual<block, N, dlib::affine, SUBNET>>;
template<int N, typename SUBNET> using ares_down = dlib::relu<residual_down<block, N, dlib::affine, SUBNET>>;
template<typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template<typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template<typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template<typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template<typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;
using anet_type = dlib::loss_metric<dlib::fc_no_bias<128, dlib::avg_pool_everything<
        alevel0<
                alevel1<
                        alevel2<
                                alevel3<
                                        alevel4<
                                                dlib::max_pool<3, 3, 2, 2, dlib::relu<dlib::affine<dlib::con<32, 7, 7, 2, 2,
                                                        dlib::input_rgb_image_sized<150>
                                                >>>>>>>>>>>>;

std::vector<std::string> split(const std::string &s, char delim) {
    std::stringstream ss(s);
    std::string item;
    std::vector<std::string> elems;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

#define Malloc(type, n) (type *)malloc((n)*sizeof(type))


const int rt_succeed = 0;
const int rt_user128dfilenotfoundorbroken = 100;
const int rt_mlmodelfilenotfoundorbroken = 101;
const int rt_hardwarefail = 110;
const int rt_authfailtoomuch = 120;
const int rt_timeout = 121;
const int rt_parameterwrong = 130;

// code definitions
//   -> 0 means succeed
//   -> 100 means user's 128d config file not found or broken ...
//   -> 101 means machine learning model file not found or broken ...
//   -> 110 camera hardware not found / hardware can not use ...
//   -> 120 means face auth continuous fail count > 3 (continuous succeed 3 times can login)
//   -> 121 means timeout (5 seconds no face detected)
//   -> 130 parameter wrong
int main(int argc, char *argv[]) {
    std::string userId = "";
    if (argc == 2) {
        userId = std::string(argv[1]);

        //check user config exist
        if (fileExists(userConfigPath + userId)) {
            //user 128D data exist
        } else {
            return rt_user128dfilenotfoundorbroken;
        }

        //check machine learning model file exist
        if (fileExists(tensorflowConfigFile)) {
            //ml model file exist
        } else {
            return rt_mlmodelfilenotfoundorbroken;
        }
        if (fileExists(tensorflowModelFile)) {
            //ml model file exist
        } else {
            return rt_mlmodelfilenotfoundorbroken;
        }
        if (fileExists(torchStereoDetectFile)) {
            //ml model file exist
        } else {
            return rt_mlmodelfilenotfoundorbroken;
        }
        if (fileExists(dlibShapeConfigFile)) {
            //ml model file exist
        } else {
            return rt_mlmodelfilenotfoundorbroken;
        }
        if (fileExists(dlibResNetFace128dFile)) {
            //ml model file exist
        } else {
            return rt_mlmodelfilenotfoundorbroken;
        }
        if (fileExists(otherConfigPath)) {
            //ml model file exist
        } else {
            return rt_mlmodelfilenotfoundorbroken;
        }
        if (fileExists(svmConfig)) {
            //ml model file exist
        } else {
            return rt_mlmodelfilenotfoundorbroken;
        }
        if (fileExists(userSVMMapConfig)) {
            //ml model file exist
        } else {
            return rt_mlmodelfilenotfoundorbroken;
        }
    } else {
        std::cout << "Usage:" << std::endl;
        std::cout << "      FaceAuth userid" << std::endl;
        return rt_parameterwrong;
    }

    bool debugFlag = true;

    const int CamWidth = 1280;
    const int CamHeight = 720;
    const int ViewWidth = 720;
    const int ViewHeight = 720;
    const int TimeoutSeconds = 5;

    const int Margin = 20;
    const float AreanRatioMin = 1.0 / 9.0;

    cv::Size cropSize(ViewWidth, ViewHeight);
    cv::Rect crop(cv::Point((ViewWidth - cropSize.width) / 2,
                            (ViewHeight - cropSize.height) / 2),
                  cropSize);

    cv::dnn::Net tfFaceNet;
    cv::dnn::Net torchDepthNet;

    dlib::frontal_face_detector dlibHogDetector = dlib::get_frontal_face_detector();
    dlib::shape_predictor dlibShapePredictor;
    anet_type dlibFace128dNet;

    try {
        tfFaceNet = cv::dnn::readNetFromTensorflow(tensorflowModelFile, tensorflowConfigFile);
        torchDepthNet = cv::dnn::readNetFromONNX(torchStereoDetectFile);
        dlib::deserialize(dlibShapeConfigFile) >> dlibShapePredictor;
        dlib::deserialize(dlibResNetFace128dFile) >> dlibFace128dNet;
    } catch (...) {
        return rt_mlmodelfilenotfoundorbroken;
    }

    if (tfFaceNet.empty() || torchDepthNet.empty()) {
        return rt_mlmodelfilenotfoundorbroken;
    }

    int resCode = rt_timeout;

    auto start = std::chrono::system_clock::now();

    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_DEPTH, CamWidth, CamHeight);
    cfg.enable_stream(RS2_STREAM_COLOR, CamWidth, CamHeight);
    pipe.start(cfg);

    rs2::align align_to_depth(RS2_STREAM_DEPTH);
    rs2::align align_to_color(RS2_STREAM_COLOR);

    cv::String window_name = "Linux Face Authentication";
    if (debugFlag == true) {
        namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    }

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(1.0, 10.0);

    //opencv svm predict face label
    //to get name we use map : face label -> name
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(svmConfig);
    std::map<std::string, std::string> mp;
    std::ifstream file(userSVMMapConfig);
    std::string tmpstr;
    while (std::getline(file, tmpstr)) {
        auto tmpres = split(tmpstr, '\t');
        mp.insert(std::make_pair(tmpres[0], tmpres[1]));
    }
    file.close();

//    //test
//    //here I will create a small artificial problem just for illustration
//    std::vector<std::vector<double>> data;
//    std::vector<int> labels;
//    std::map<std::string, std::string> mp2;
//    LoadDataForSVM(data, labels, mp2);
//    int sizeOfProblem = data.size(); //number of lines with labels
//    int elements = 128; //number of features for each data vector
//    //read data
//    //initialize the size of the problem with just an int
//    struct svm_parameter param;        // set by parse_command_line
//    struct svm_problem prob;        // set by read_problem
//    struct svm_model *model;
//    struct svm_node *x_space;
//    prob.l = sizeOfProblem;
//    //here we need to give some memory to our structures
//    // @param prob.l = number of labels
//    // @param elements = number of features for each label
//    prob.y = Malloc(double, prob.l); //space for prob.l doubles
//    prob.x = Malloc(struct svm_node *, prob.l); //space for prob.l pointers to struct svm_node
//    x_space = Malloc(struct svm_node, (elements + 1) * prob.l); //memory for pairs of index/value
//    //initialize the different lables with an array of labels
//    for (int i = 0; i < prob.l; ++i) {
//        prob.y[i] = labels[i];
//    }
//    //initialize the svm_node vector with input data array as follows:
//    int trainj = 0; //counter to traverse x_space[i];
//    for (int i = 0; i < prob.l; ++i) {
//        //set i-th element of prob.x to the address of x_space[j].
//        //elements from x_space[j] to x_space[j+data[i].size] get filled right after next line
//        prob.x[i] = &x_space[trainj];
//        for (int k = 0; k < data[i].size(); ++k, ++trainj) {
//            x_space[trainj].index = k + 1; //index of value
//            x_space[trainj].value = data[i][k]; //value
//        }
//        x_space[trainj].index = -1;//state the end of data vector
//        x_space[trainj].value = 0;
//        trainj++;
//    }
//    //set all default parameters for param struct
//    param.svm_type = C_SVC;
//    param.kernel_type = RBF;
//    param.degree = 3;
//    param.gamma = 1.0 / 256;    // 1/num_features
//    param.coef0 = 0;
//    param.nu = 0.5;
//    param.cache_size = 200;
//    param.C = 100000000;
//    param.eps = 1e-3;
//    param.p = 0.1;
//    param.shrinking = 1;
//    param.probability = 1;
//    param.nr_weight = 0;
//    param.weight_label = NULL;
//    param.weight = NULL;
//    model = svm_train(&prob, &param);
//    //end test

    cv::Mat userMat;
    cv::FileStorage tmpload(userConfigPath + userId, cv::FileStorage::READ);
    tmpload["matdata"] >> userMat;
    tmpload.release();

    while (true) {
        //check execution time > 5 seconds
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        if (elapsed_seconds.count() > TimeoutSeconds) {
            resCode = rt_timeout;
            break;
        }

        auto idx = dist(mt);

        rs2::frameset frames = pipe.wait_for_frames();
        frames = align_to_color.process(frames);
        auto depthFrame = frames.get_depth_frame();
        auto colorFrame = frames.get_color_frame();

        //copy from wrappers/opencv/dnn/rs-dnn.cpp
        static int last_frame_number = 0;
        if (colorFrame.get_frame_number() == last_frame_number) {
            std::cout << "@@@@ get_frame_number of colorFrame not equal" << std::endl;
            continue;
        }
        last_frame_number = colorFrame.get_frame_number();

        cv::Mat color_mat = frame_to_mat(colorFrame);
        cv::Mat depth_mat = frame_to_mat(depthFrame);
        color_mat = color_mat(crop).clone();
        depth_mat = depth_mat(crop).clone();

        int opencvDetectedCount = 0;
        bool isFacePositionNormal = false;
        bool isDepthNormal = false;
        cv::Mat inputBlob = cv::dnn::blobFromImage(color_mat,
                                                   1.0,
                                                   cv::Size(300, 300),
                                                   cv::Scalar(104.0, 177.0, 123.0),
                                                   false); //Convert Mat to batch of images
        tfFaceNet.setInput(inputBlob, "data");
        cv::Mat detection = tfFaceNet.forward("detection_out");
        cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
        int x1 = 0;
        int y1 = 0;
        int x2 = 0;
        int y2 = 0;
        for (int i = 0; i < detectionMat.rows; i++) {
            float confidence = detectionMat.at<float>(i, 2);
            if (confidence > 0.9) {
                opencvDetectedCount++;
                x1 = static_cast<int>(detectionMat.at<float>(i, 3) * color_mat.cols);
                y1 = static_cast<int>(detectionMat.at<float>(i, 4) * color_mat.rows);
                x2 = static_cast<int>(detectionMat.at<float>(i, 5) * color_mat.cols);
                y2 = static_cast<int>(detectionMat.at<float>(i, 6) * color_mat.rows);

                //ignore faces too much close to border
                if (x1 >= Margin &&
                    y1 >= Margin &&
                    x2 <= (ViewWidth - Margin) &&
                    y2 <= (ViewHeight - Margin) &&
                    1.0 * (x2 - x1) * (y2 - y1) / (ViewHeight * ViewWidth) > AreanRatioMin) {
                    isFacePositionNormal = true;
                }
            }
        }

        //use onnx check depth info
        //if real face detected
        //set isDepthNormal = true
        if (opencvDetectedCount == 1 && isFacePositionNormal == true) {
            cv::Rect roi;
            roi.x = x1;
            roi.y = y1;
            roi.width = x2 - x1;
            roi.height = y2 - y1;
            cv::Mat depthMatToCheck = depth_mat(roi).clone();

            cv::Mat padded = ResizeAndPaddingMat(depthMatToCheck, 64, 64);

            cv::Mat resMat = FilterMatWithHist(padded);

            cv::Mat testInput = cv::dnn::blobFromImage(resMat,
                                                       1.0,
                                                       cv::Size(64, 64),
                                                       0,
                                                       false,
                                                       false);

            torchDepthNet.setInput(testInput);
            cv::Mat resultMat = torchDepthNet.forward();

            float tmpa = resultMat.at<float>(0, 0);
            float tmpb = resultMat.at<float>(0, 1);

            if (tmpa > tmpb) {
                isDepthNormal = true;
                cv::rectangle(color_mat, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
            } else if (tmpa < tmpb) {
                isDepthNormal = false;
                cv::rectangle(color_mat, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 2);
            } else {
                isDepthNormal = false;
                cv::rectangle(color_mat, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 255, 255), 2);
            }
        }

        //debug
        isDepthNormal = true;

        if (opencvDetectedCount == 1 && isFacePositionNormal == true && isDepthNormal == true) {
            cv::Mat resized;
            cv::resize(color_mat, resized, cv::Size(360, 360));
            dlib::cv_image<dlib::bgr_pixel> dlibImg(resized);
            std::vector<dlib::rectangle> faceRectList = dlibHogDetector(dlibImg);
            std::vector<dlib::matrix<dlib::rgb_pixel>> faces;

            if (faceRectList.size() == 1) {
                for (auto face : faceRectList) {
                    x1 = face.left() * 2;
                    y1 = face.top() * 2;
                    x2 = face.right() * 2;
                    y2 = face.bottom() * 2;

                    //ignore faces too much close to border
                    if (x1 >= Margin &&
                        y1 >= Margin &&
                        x2 <= (ViewWidth - Margin) &&
                        y2 <= (ViewHeight - Margin) &&
                        1.0 * (x2 - x1) * (y2 - y1) / (ViewHeight * ViewWidth) > AreanRatioMin) {
                        auto shape = dlibShapePredictor(dlibImg, face);
                        dlib::matrix<dlib::rgb_pixel> face_chip;

                        //use dlib window display face_chip will find that
                        //extract_image_chip allready did face alignment
                        extract_image_chip(dlibImg, get_face_chip_details(shape, 150, 0.25), face_chip);

                        faces.push_back(std::move(face_chip));
                    }
                }

                std::vector<dlib::matrix<float, 0, 1>> face_descriptors = dlibFace128dNet(faces);
                if (face_descriptors.size() == 1) {
                    //fresh time when face detected
                    start = std::chrono::system_clock::now();

                    auto face128d = dlib::toMat(face_descriptors[0]).reshape(0, 1);

                    //std::cout << face128d.type() << std::endl;

                    //distance vector's data type is double
                    //Opencv calculate histgram need integer
                    std::vector<double> disVec;
                    for (int tmpi = 0; tmpi < userMat.rows; tmpi++) {
                        double tmpres = 0.0;
                        for (int tmpj = 0; tmpj < userMat.cols; tmpj++) {
                            tmpres +=
                                    (face128d.at<float>(0, tmpj) - userMat.at<float>(tmpi, tmpj)) *
                                    (face128d.at<float>(0, tmpj) - userMat.at<float>(tmpi, tmpj));
                        }
                        tmpres = std::sqrt(tmpres);
                        disVec.push_back(tmpres);
                    }

                    //SVC is not reliable comparing caclulate distance
                    double sum = std::accumulate(disVec.begin(), disVec.end(), 0.0);
                    double mean = sum / disVec.size();
                    double sq_sum = std::inner_product(disVec.begin(), disVec.end(), disVec.begin(), 0.0);
                    double stdev = std::sqrt(sq_sum / disVec.size() - mean * mean);
                    std::cout << disVec.size() << " mean is " << mean <<
                              " stdev is " << stdev << std::endl;

                    //authentication succeed
                    //break
                    if (mean < 0.4 && stdev < 0.2) {
                        resCode = rt_succeed;
                        break;
                    }

//svc do not work properly as intent
//                    double *resprob = new double[3];
//                    struct svm_node *svmVec;
//                    svmVec = Malloc(struct svm_node, (elements + 1));
//                    int c = 0;
//                    for (; c < face128d.cols; c++) {
//                        svmVec[c].index = c + 1; // Index starts from 1; Pre-computed kernel starts from 0
//                        svmVec[c].value = face128d.at<float>(0, c);
//                    }
//                    svmVec[c].index = -1;
//                    auto predictedlabel = svm_predict_probability(model, svmVec, resprob);
//                    std::cout << "predicted result is " << predictedlabel << ":" << resprob[0] << "," << resprob[1]
//                              << "," << resprob[2] << std::endl;
//                    delete resprob;
//                    free(svmVec);

//opencv's svm work better than top ,but it is still not reliable
//                    Mat predictOutput;
//                    auto predictRes = svm->predict(face128d);
//                    if (mp.count(std::to_string(static_cast<int>(predictRes))) == 1) {
//                        std::cout << "predicted user is " << mp.at(std::to_string(static_cast<int>(predictRes)))
//                                  << std::endl;
//                        if (userId == mp.at(std::to_string(static_cast<int>(predictRes)))) {
//                            //std::cout << "authentication for [" << userId << "] succeed " << std::endl;
//                            resCode = rt_succeed;
//                            //break;
//                        }
//                    } else {
//                        std::cout << "predicted result is " << predictRes << std::endl;
//                    }

                }
            }
        }

        if (debugFlag == true) {
            imshow(window_name, color_mat);
            if (cv::waitKey(1) >= 0) break;
        }
    }

    pipe.stop();

    std::cout << "@@@@ Test Execution by Pam, Return " << resCode << std::endl;
    return resCode;
}

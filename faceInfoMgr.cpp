//
// Created by hjd on 3/22/20.
//

#include <iostream>
#include <stdio.h>

#include <functional>

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




int main(int argc, char *argv[]) {
    bool shouldExtract128dFlag = false;
    std::vector<cv::Mat> face128dList;
    bool addUserFlag = false;
    std::string filename = "";
    std::string userId = "";

    if (argc == 3) {
        if (strcmp(argv[1], "test") == 0
            && strcmp(argv[2], "camera") == 0) {
            //test camera
        } else if (strcmp(argv[1], "list") == 0
                   && strcmp(argv[2], "all") == 0) {
            //list all user
        } else if (strcmp(argv[1], "add") == 0) {
            //add user
            //remove exist and create new
            //recalculate svc
            addUserFlag = true;
            shouldExtract128dFlag = true;
            userId = std::string(argv[2]);
            filename = userConfigPath + userId;

        } else if (strcmp(argv[1], "del") == 0) {
            //delete user
            cv::String userId(argv[2]);
        } else {
            std::cout << "Usage:" << std::endl;
            std::cout << "      faceInfoMgr add userid" << std::endl;
            std::cout << "      faceInfoMgr del userid" << std::endl;
            std::cout << "      faceInfoMgr list all" << std::endl;
            std::cout << "      faceInfoMgr test camera" << std::endl;
            return 0;
        }
    } else {
        std::cout << "Usage:" << std::endl;
        std::cout << "      faceInfoMgr add userid" << std::endl;
        std::cout << "      faceInfoMgr del userid" << std::endl;
        std::cout << "      faceInfoMgr list all" << std::endl;
        std::cout << "      faceInfoMgr test camera" << std::endl;
        return 0;
    }

    const int CamWidth = 1280;
    const int CamHeight = 720;
    const int ViewWidth = 720;
    const int ViewHeight = 720;
    const int TimeoutSeconds = 50;
    const int MinFace128dCount = 64;

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
        return 101;
    }

    if (tfFaceNet.empty() || torchDepthNet.empty()) {
        return 101;
    }

    int resCode = 0;

    auto start = std::chrono::system_clock::now();

    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_DEPTH, CamWidth, CamHeight);
    cfg.enable_stream(RS2_STREAM_COLOR, CamWidth, CamHeight);
    pipe.start(cfg);

    rs2::align align_to_depth(RS2_STREAM_DEPTH);
    rs2::align align_to_color(RS2_STREAM_COLOR);

    cv::String window_name = "Linux Face Authentication";
    namedWindow(window_name, cv::WINDOW_AUTOSIZE);

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(1.0, 10.0);

    while (true) {
        //check execution time > 5 seconds
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        if (elapsed_seconds.count() > TimeoutSeconds) {
            resCode = 121;
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
        tfFaceNet.setInput(inputBlob);
        cv::Mat detection = tfFaceNet.forward();
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

        if (opencvDetectedCount == 1
            && isFacePositionNormal == true
            && isDepthNormal == true
            && shouldExtract128dFlag == true) {
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

                    auto face128d = dlib::toMat(face_descriptors[0]);
                    //std::cout << "After " << face128d.reshape(0, 1) << std::endl;
                    if (addUserFlag == true) {
                        //reset time when succeed add face
                        start = std::chrono::system_clock::now();
                        face128dList.push_back(face128d.reshape(0, 1));
                        if (face128dList.size() > MinFace128dCount) {
                            break;
                        }
                    }
                }
            }
        }

        imshow(window_name, color_mat);
        if (cv::waitKey(1) >= 0) break;
    }

    pipe.stop();

    if (face128dList.size() > MinFace128dCount) {
        cv::Mat toSave = face128dList[0];
        for (int i = 1; i < face128dList.size(); i++) {
            toSave.push_back(face128dList[i]);
        }
        if (fileExists(filename)) {
            //delete
            remove(filename.c_str());
        }
        if (fileExists(filename)) {
            //check delete succeed
            std::cout << "Delete Old Face info for [" << userId << "] failed." << std::endl;
            return 2;
        }
        cv::FileStorage saveFaceInfo(filename, cv::FileStorage::WRITE);
        saveFaceInfo << "matdata" << toSave;
        saveFaceInfo.release();
        if (fileExists(filename)) {
            std::cout << "Face info for [" << userId << "] added succeed." << std::endl;
            updateSVC();
            return 0;
        } else {
            std::cout << "Face info for [" << userId << "] added failed." << std::endl;
            return 1;
        }
    }

    std::cout << "@@@@ Test Execution by Pam, Return " << resCode << std::endl;
    return resCode;
}

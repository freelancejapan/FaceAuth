//
// Created by hjd on 3/20/20.
//
// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#pragma once

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/opencv.hpp>   // Include OpenCV API
#include <exception>
#include <opencv2/ml.hpp>
#include <dirent.h>
#include <fstream>

const std::string tensorflowConfigFile = "/usr/lib64/security/pam_camera/opencv_face_detector.pbtxt";
const std::string tensorflowModelFile = "/usr/lib64/security/pam_camera/opencv_face_detector_uint8.pb";

const std::string torchStereoDetectFile = "/usr/lib64/security/pam_camera/torch_depth_cls.onnx";

const std::string dlibShapeConfigFile = "/usr/lib64/security/pam_camera/shape_predictor_5_face_landmarks.dat";
const std::string dlibResNetFace128dFile = "/usr/lib64/security/pam_camera/dlib_face_recognition_resnet_model_v1.dat";

const std::string otherConfigPath = "/usr/lib64/security/pam_camera/OtherConfig.xml";
const std::string userConfigPath = "/usr/lib64/security/pam_camera/userlist/";

const std::string svmConfig = "/usr/lib64/security/pam_camera/svmConfig";
const std::string userSVMMapConfig = "/usr/lib64/security/pam_camera/mapConfig";

bool fileExists(const std::string &name) {
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }
}

bool updateSVC() {
    std::vector<std::string> fileList;
    if (auto dir = opendir(userConfigPath.c_str())) {
        while (auto f = readdir(dir)) {
            if (!f->d_name || f->d_name[0] == '.') {
                continue; // Skip everything that starts with a dot
            }
            if (f->d_type == DT_REG) {
                std::string tmpfile(f->d_name);
                fileList.push_back(tmpfile);
            }
        }
        closedir(dir);
    }

    cv::FileStorage savefaceconfig(otherConfigPath, cv::FileStorage::READ);
    cv::Mat otherMat;
    savefaceconfig["matdata"] >> otherMat;
    savefaceconfig.release();
    //std::cout << otherMat.type() << " == " << CV_32F << std::endl;
    std::map<std::string, std::string> mp;

    if (otherMat.rows > 0 && otherMat.cols == 128) {
        std::string str = "..";
        std::hash<std::string> hasher;
        auto hashed = hasher(str); //returns std::size_t

        int rows = otherMat.rows;
        int labels[rows];
        //std::fill_n(labels, rows, static_cast<short>(hashed));
        std::fill_n(labels, rows, -1);
        cv::Mat labelsMat(rows, 1, CV_32SC1, labels);

        int processedCount = 0;
        for (auto tmpfile : fileList) {

            cv::Mat tmpMat;
            try {
                cv::FileStorage tmpload(userConfigPath + tmpfile, cv::FileStorage::READ);
                tmpload["matdata"] >> tmpMat;
                tmpload.release();
            } catch (...) {
                std::cout << "processing file [" << tmpfile << "] failed" << std::endl;
                continue;
            }

            if (tmpMat.rows > 0 && tmpMat.cols == 128) {
                processedCount++;

                auto tmphashed = hasher(tmpfile); //returns std::size_t
                int tmprows = tmpMat.rows;
                int tmplabels[tmprows];
                //std::fill_n(tmplabels, tmprows, static_cast<short>(tmphashed));
                std::fill_n(tmplabels, tmprows, processedCount);

                mp.insert(std::make_pair(tmpfile, std::to_string(processedCount)));

                cv::Mat tmpLabelsMat(tmprows, 1, CV_32SC1, tmplabels);
                otherMat.push_back(tmpMat);
                labelsMat.push_back(tmpLabelsMat);
            }
        }

        if (processedCount < 1) {
            std::cout << "no face config found" << std::endl;
            return false;
        }

        cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
        svm->setType(cv::ml::SVM::C_SVC);
        svm->setKernel(cv::ml::SVM::RBF);
        svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, std::numeric_limits<int>::max(), 1e-6));
        svm->train(otherMat, cv::ml::ROW_SAMPLE, labelsMat);

        if (fileExists(svmConfig)) {
            remove(svmConfig.c_str());
        }
        if (fileExists(svmConfig)) {
            std::cout << "Delete old svm config file failed" << std::endl;
            return false;
        }
        if (fileExists(userSVMMapConfig)) {
            remove(userSVMMapConfig.c_str());
        }
        if (fileExists(userSVMMapConfig)) {
            std::cout << "Delete old svm config file failed" << std::endl;
            return false;
        }

        svm->save(svmConfig);
        if (fileExists(svmConfig)) {
            std::ofstream out(userSVMMapConfig);
            for (auto& [key, value]: mp) {
                out << value << "\t" << key << std::endl;
            }
            out.close();
            std::cout << "svm config update succeed" << std::endl;
        }

    } else {
        std::cout << "OtherConfig.xml not exist or broken" << std::endl;
        return false;
    }
    return true;
}

// Convert rs2::frame to cv::Mat
cv::Mat frame_to_mat(const rs2::frame &f) {
    using namespace cv;
    using namespace rs2;

    auto vf = f.as<video_frame>();
    const int w = vf.get_width();
    const int h = vf.get_height();

    if (f.get_profile().format() == RS2_FORMAT_BGR8) {
        return Mat(Size(w, h), CV_8UC3, (void *) f.get_data(), Mat::AUTO_STEP);
    } else if (f.get_profile().format() == RS2_FORMAT_RGB8) {
        auto r_rgb = Mat(Size(w, h), CV_8UC3, (void *) f.get_data(), Mat::AUTO_STEP);
        Mat r_bgr;
        cvtColor(r_rgb, r_bgr, COLOR_RGB2BGR);
        return r_bgr;
    } else if (f.get_profile().format() == RS2_FORMAT_Z16) {
        return Mat(Size(w, h), CV_16UC1, (void *) f.get_data(), Mat::AUTO_STEP);
    } else if (f.get_profile().format() == RS2_FORMAT_Y8) {
        return Mat(Size(w, h), CV_8UC1, (void *) f.get_data(), Mat::AUTO_STEP);
    } else if (f.get_profile().format() == RS2_FORMAT_DISPARITY32) {
        return Mat(Size(w, h), CV_32FC1, (void *) f.get_data(), Mat::AUTO_STEP);
    }

    throw std::runtime_error("Frame format is not supported yet!");
}

// Converts depth frame to a matrix of doubles with distances in meters
cv::Mat depth_frame_to_meters(const rs2::pipeline &pipe, const rs2::depth_frame &f) {
    using namespace cv;
    using namespace rs2;

    Mat dm = frame_to_mat(f);
    dm.convertTo(dm, CV_64F);
    auto depth_scale = pipe.get_active_profile()
            .get_device()
            .first<depth_sensor>()
            .get_depth_scale();
    dm = dm * depth_scale;
    return dm;
}
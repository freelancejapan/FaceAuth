//
// Created by hjd on 3/21/20.
//

#pragma once

#include <opencv2/opencv.hpp>   // Include OpenCV API
#include <stdexcept>
#include <iostream>

using namespace cv;

Mat ResizeAndPaddingMat(Mat &pSrcMat, int expectRowCount, int expectColCount) {
    Mat srcMat;
    pSrcMat.convertTo(srcMat, CV_32F);
    int currentRowCount = srcMat.rows;
    int currentColCount = srcMat.cols;
    float expectRatio = 1.0 * expectRowCount / expectColCount;
    float currentRatio = 1.0 * currentRowCount / currentColCount;
    int resizeRowCount = expectRowCount;
    int resizeColCount = expectColCount;

    if (currentRatio >= expectRatio) {
        resizeColCount = int(resizeRowCount / currentRatio);
    } else {
        resizeRowCount = int(resizeColCount * currentRatio);
    }

    int paddingTop = (expectRowCount - resizeRowCount) / 2;
    int paddingBot = expectRowCount - resizeRowCount - paddingTop;
    int paddingLft = (expectColCount - resizeColCount) / 2;
    int paddingRgt = expectColCount - resizeColCount - paddingLft;

    if (paddingTop + paddingBot + resizeRowCount > expectRowCount) {
        std::cout << paddingTop << "+" << paddingBot << "+" << resizeRowCount << ">" << expectRowCount << std::endl;
        throw std::runtime_error("Padding Top + Padding Bottom + Resize Row Count > Expected Row Count");
    }
    if (paddingLft + paddingRgt + resizeColCount > expectColCount) {
        std::cout << paddingLft << "+" << paddingRgt << "+" << resizeColCount << ">" << expectColCount << std::endl;
        throw std::runtime_error("Padding Left + Padding Right + Resize Column Count > Expected Column Count");
    }
    Mat output;
    resize(srcMat, output, Size(resizeColCount, resizeRowCount));

    //multiply depth data when resize
    output = output * 1.0 * resizeColCount / currentColCount;

    Mat outputPadding;
    copyMakeBorder(output, outputPadding,
                   paddingTop, paddingBot, paddingLft, paddingRgt,
                   BORDER_CONSTANT,
                   (float) 0.0);

    return outputPadding;
}

Mat FilterMatWithHist(Mat &pSrcMat) {
    double minVal;
    double maxVal;
    cv::Point minLoc;
    cv::Point maxLoc;
    minMaxLoc(pSrcMat, &minVal, &maxVal, &minLoc, &maxLoc);
    maxVal = maxVal + 1;
    int histSize = 10;
    float range[] = {(float) minVal, (float) maxVal}; //the upper boundary is exclusive
    const float *histRange = {range};
    cv::Mat histMat;
    calcHist(&pSrcMat,
             1,
             0,
             cv::Mat(),
             histMat,
             1,
             &histSize,
             &histRange,
             true,
             false);

    int binSum[10] = {};
    for (int i = 0; i < 10; i++) {
        binSum[i] = histMat.at<int>(i, 0);
    }
    float binAnchor[10 + 1] = {};
    binAnchor[0] = minVal;
    for (int i = 1; i < 10 + 1; i++) {
        binAnchor[i] = (maxVal - minVal) / 10.0 * i + binAnchor[0];
    }

    float minAnchor = binAnchor[0];
    float maxAnchor = binAnchor[10];
    int noneZeroCount = countNonZero(pSrcMat);

    for (int tmp = 0; tmp < 10; tmp++) {
        if (tmp < 10 - 1) {
            if (1.0 * (binSum[tmp] + binSum[tmp + 1]) / noneZeroCount > 0.9) {
                minAnchor = binAnchor[tmp];
                maxAnchor = binAnchor[tmp + 1 + 1];
                if (tmp + 1 + 1 < 10 and binSum[tmp + 1 + 1] > 0) {
                    maxAnchor = binAnchor[tmp + 1 + 1 + 1];
                }
            }
        }
    }

    float tmpTop = -1;
    float tmpBot = -1;
    float tmpLeft = -1;
    float tmpRight = -1;

    Mat res = pSrcMat.clone();

    for (int j = 0; j < res.rows; j++) {
        for (int i = 0; i < res.cols; i++) {
            if (j - 1 >= 0) {
                tmpTop = pSrcMat.at<float>(j - 1, i);
            } else {
                tmpTop = -1;
            }
            if (j + 1 < res.rows) {
                tmpBot = pSrcMat.at<float>(j + 1, i);
            } else {
                tmpBot = -1;
            }

            if (i - 1 >= 0) {
                tmpLeft = pSrcMat.at<float>(j, i - 1);
            } else {
                tmpLeft = -1;
            }
            if (i + 1 < res.cols) {
                tmpRight = pSrcMat.at<float>(j, i + 1);
            } else {
                tmpRight = -1;
            }

            float tmpVal = res.at<float>(j, i);

            if (tmpVal >= minAnchor && tmpVal <= maxAnchor) {
                // do nothing
            } else {
                std::vector<float> vec;
                if (tmpTop >= minAnchor && tmpTop <= maxAnchor) {
                    vec.push_back(tmpTop);
                }
                if (tmpBot >= minAnchor && tmpBot <= maxAnchor) {
                    vec.push_back(tmpBot);
                }
                if (tmpLeft >= minAnchor && tmpLeft <= maxAnchor) {
                    vec.push_back(tmpLeft);
                }
                if (tmpRight >= minAnchor && tmpRight <= maxAnchor) {
                    vec.push_back(tmpRight);
                }

                if (vec.size() >= 3) {
                    float sum_of_elems = 0;
                    for (auto& n : vec)
                        sum_of_elems += n;
                    res.at<float>(j, i) = sum_of_elems;
                } else {
                    res.at<float>(j, i) = (float)0.0;
                }
            }
        }
    }

    double minValAjust;
    double maxValAjust;
    cv::Point minLocAjust;
    cv::Point maxLocAjust;
    minMaxLoc(pSrcMat, &minValAjust, &maxValAjust, &minLocAjust, &maxLocAjust);

    res = (float)maxValAjust - res;

    for (int j = 0; j < res.rows; j++) {
        for (int i = 0; i < res.cols; i++) {
            float tmp = res.at<float>(j, i);
            if (tmp == (float)maxValAjust) {
                res.at<float>(j, i) = (float)0.0;
            }
        }
    }

    return res;
}

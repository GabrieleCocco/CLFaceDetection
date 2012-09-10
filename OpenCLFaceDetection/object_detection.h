//
//  object_detection.h
//  OpenCLFaceDetection
//
//  Created by Gabriele Cocco on 9/8/12.
//  Copyright (c) 2012 Gabriele Cocco. All rights reserved.
//

#ifndef OpenCLFaceDetection_object_detection_h
#define OpenCLFaceDetection_object_detection_h

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cvaux.hpp>
#include <sys/time.h>
#include <limits.h>
#include "image_computations.h"

typedef struct ElapseTime {
    double s;
    double e;
    struct timeval time;
    void start() {
        gettimeofday(&time, NULL);
        s = (double)(time.tv_sec * 1000) + ((double)time.tv_usec / 1000.0);    }
    double get() {
        gettimeofday(&time, NULL);
        e = (double)(time.tv_sec * 1000) + ((double)time.tv_usec / 1000.0);
        return e - s;
    }
} ElapseTime;

typedef struct CLRect {
    cl_int x;
    cl_int y;
    cl_int width;
    cl_int height;
} CLRect;

typedef struct CLWeightedRect {
    cl_int x;
    cl_int y;
    cl_int width;
    cl_int height;
    cl_float weight;
} CLWeightedRect;

CLWeightedRect*
detectObjects(IplImage* image,
              CvHaarClassifierCascade* cascade,
              CLEnvironmentData* data,
              cl_uint min_window_width,
              cl_uint min_window_height,
              cl_uint max_window_width,
              cl_uint max_window_height,
              cl_uint min_neighbors,
              cl_uint* final_match_count);

CLWeightedRect*
detectObjectsOptimized(IplImage* image,
                       CvHaarClassifierCascade* cascade,
                       CLEnvironmentData* data,
                       cl_uint min_window_width,
                       cl_uint min_window_height,
                       cl_uint max_window_width,
                       cl_uint max_window_height,
                       cl_uint min_neighbors,
                       cl_uint* final_match_count);


CLWeightedRect*
detectObjectsGPU(IplImage* image,
                 CvHaarClassifierCascade* cascade,
                 CLEnvironmentData* data,
                 cl_uint min_window_width,
                 cl_uint min_window_height,
                 cl_uint max_window_width,
                 cl_uint max_window_height,
                 cl_uint min_neighbors,
                 cl_uint* final_match_count);

#endif

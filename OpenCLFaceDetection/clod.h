//
//  object_detection.h
//  OpenCLFaceDetection
//
//  Created by Gabriele Cocco on 9/8/12.
//  Copyright (c) 2012 Gabriele Cocco. All rights reserved.
//

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cvaux.hpp>
#include <sys/time.h>
#include <limits.h>
#include <params.h>
#include "clif.h"

#define CLOD_PRECOMPUTE_FEATURES  (2 << 0)
#define CLOD_BLOCK_IMPLEMENTATION (2 << 1)
#define CLOD_PER_STAGE_ITERATIONS (2 << 2)

typedef cl_uint clod_flags;

typedef struct ElapseTime {
    double s;
    double e;
    struct timeval time;
    void start() {
        gettimeofday(&time, NULL);
        s = (double)(time.tv_sec * 1000) + ((double)time.tv_usec / 1000.0);
    }
    double get() {
        gettimeofday(&time, NULL);
        e = (double)(time.tv_sec * 1000) + ((double)time.tv_usec / 1000.0);
        return e - s;
    }
} ElapseTime;


typedef struct CLODWeightedRect {
    CvRect rect;
    cl_float weight;
} CLODWeightedRect;

typedef struct CLODDetectObjectsResult {
    CLODWeightedRect* matches;
    cl_uint match_count;
} CLODDetectObjectsResult;

typedef struct CLODDetectsObjectsData {
    cl_mem buffers[5];
    size_t global_size[1];
    size_t local_size[1];
} CLODDetectObjectsData;

typedef struct CLODFEnvironmentData {
    CLIFEnvironmentData* clif;
    CLDeviceEnvironment environment;
    CLODDetectObjectsData detect_objects_data;
} CLODEnvironmentData;

CLODEnvironmentData*
clodInitEnvironment(const cl_uint device_index);

void
clodReleaseEnvironment(CLODFEnvironmentData* data);

void
clodInitBuffers(CLODEnvironmentData* data,
                const CvSize* integral_image_size);
void
clodReleaseBuffers(CLODEnvironmentData* data);

CLODDetectObjectsResult
clodDetectObjects(const IplImage* image,
                  const CvHaarClassifierCascade* cascade,
                  const CLODEnvironmentData* data,
                  const CvSize min_window_size,
                  const CvSize max_window_size,
                  const cl_uint min_neighbors,
                  const clod_flags flags,
                  const cl_bool use_opencl);

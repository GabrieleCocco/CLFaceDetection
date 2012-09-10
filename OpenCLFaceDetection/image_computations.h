//
//  integral_image.h
//  OpenCLFaceDetection
//
//  Created by Gabriele Cocco on 9/4/12.
//  Copyright (c) 2012 Gabriele Cocco. All rights reserved.
//

#ifndef OpenCLFaceDetection_integral_image_h
#define OpenCLFaceDetection_integral_image_h

extern "C" {
#include "CLEnvironment.h"
}
//#include "tempcv.hpp"

typedef struct CLOriginalImageData {
    cl_uint width;
    cl_uint height;
    cl_uint stride;
    cl_uint channels;
} CLOriginalImageData;

typedef struct CLBgrToGrayData {
    cl_mem buffers[2];
    void* ptr;
    size_t global_size[2];
    size_t local_size[2];
} CLBgrToGayData;

typedef struct CLIntegralImageData {
    cl_mem buffers[5];
    void* ptr;
    void* square_ptr;
    size_t global_size[2];
    size_t local_size[2];
} CLIntegralImageData;

typedef struct CLEnvironmentData {
    CLDeviceEnvironment environment;
    CLOriginalImageData original_data;
    CLBgrToGayData bgr_to_gray_data;
    CLIntegralImageData integral_image_data;
} CLEnvironmentData;

// Init and release OpenCL environment
CLEnvironmentData
initCLEnvironment(cl_uint width,
                  cl_uint height,
                  cl_uint row_stride,
                  cl_uint channels,
                  cl_uint device_index);
void
releaseCLEnvironment(CLEnvironmentData data);

// OpenCL computations
cl_uchar*
clBgrToGrayscale(cl_uchar* source,
                 CLEnvironmentData);
void
clIntegralImage(cl_uchar* source,
                CLEnvironmentData data,
                cl_uint** result,
                cl_ulong** square_result);
void
clIntegralImage(CLEnvironmentData data,
                cl_uint** result,
                cl_ulong** square_result);

// Host sequential computations
cl_uint*
integralImage(cl_uchar* source,
              cl_uint width,
              cl_uint height);
cl_uchar*
bgrToGrayScale(cl_uchar* src,
               cl_uint width,
               cl_uint height,
               cl_uint row_stride,
               cl_uint channels);
#endif

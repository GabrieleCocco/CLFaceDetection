#ifndef OpenCLIFFaceDetection_integral_image_h
#define OpenCLIFFaceDetection_integral_image_h

extern "C" {
#include "CLEnvironment.h"
#include "CLDevice.h"
}
#include <stdio.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/cvaux.hpp>

typedef struct CLIFBgrToGrayData {
    cl_mem buffers[2];
    void* ptr;
    size_t global_size[2];
    size_t local_size[2];
} CLIFBgrToGayData;

typedef struct CLIFIntegralImageData {
    cl_mem buffers[5];
    void* ptr;
    void* square_ptr;
    size_t global_size[2];
    size_t local_size[2];
} CLIFIntegralImageData;

typedef struct CLIFEnvironmentData {
    CLDeviceEnvironment environment;
    CLIFBgrToGayData bgr_to_gray_data;
    CLIFIntegralImageData integral_image_data;
} CLIFEnvironmentData;

typedef struct CLIFIntegralResult {
    CvMat* image;
    CvMat* square_image;
} CLIFIntegralResult;

typedef struct CLIFGrayscaleResult {
    IplImage* image;
} CLIFGrayscaleResult;

// Init and release OpenCLIF environment
CLIFEnvironmentData*
clifInitEnvironment(const cl_uint device_index);

void
clifReleaseEnvironment(CLIFEnvironmentData* data);

void
clifInitBuffers(CLIFEnvironmentData* data,
                const cl_uint image_width,
                const cl_uint image_height,
                const cl_uint image_stride,
                const cl_uint image_channels);

void
clifReleaseBuffers(CLIFEnvironmentData* data);

// OpenCLIF computations
CLIFGrayscaleResult
clifGrayscale(const IplImage* source,
              CLIFEnvironmentData* data,
              const cl_bool use_opencl);

CLIFIntegralResult
clifIntegral(const IplImage* source,
             CLIFEnvironmentData data,
             const cl_bool use_opencl);

CLIFIntegralResult
clifGrayscaleIntegral(const IplImage* source,
                      CLIFEnvironmentData* data,
                      const cl_bool use_opencl);
#endif

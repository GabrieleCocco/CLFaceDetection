//
//  integral_image.c
//  OpenCLFaceDetection
//
//  Created by Gabriele Cocco on 8/22/12.
//  Copyright (c) 2012 Gabriele Cocco. All rights reserved.
//

#include "image_computations.h"
#include <stdio.h>

static float rgb_to_grayscale_coeff[3] = {
    0.2989,
    0.5870,
    0.1140 };
enum
{
    yuv_shift = 14,
    xyz_shift = 12,
    R2Y = 4899,
    G2Y = 9617,
    B2Y = 1868,
    BLOCK_SIZE = 256
};

// Init and release OpenCL environment
CLEnvironmentData
initCLEnvironment(cl_uint width,
                  cl_uint height,
                  cl_uint row_stride,
                  cl_uint channels,
                  cl_uint device_index)
{
    cl_int error = CL_SUCCESS;
    CLEnvironmentData data;
    
    // Setup original image data
    data.original_data.width = width;
    data.original_data.height = height;
    data.original_data.stride = row_stride;
    data.original_data.channels = channels;
    
    // Get available devices
    cl_uint platform_device_count;
    CLDeviceInfo* platform_device_list = clGetDeviceList(&platform_device_count);
    
    // Get selected device
    CLDeviceInfo device = platform_device_list[device_index];
    
    // Set up kernel file path and functions
    const char* kernel_path = "/Users/Gabriele/Desktop/OpenCLFaceDetection/OpenCLFaceDetection/image_computations.cl";
    const char* kernel_functions[] = { "bgrToGrayscale", "integralImageSumRows", "integralImageSumCols" };
   
    // Create device environment
    char build_options[1024] = { 0 };
    sprintf(build_options, "-D CHANNELS=%d", channels);
    clCreateDeviceEnvironment(&device, 1, kernel_path, kernel_functions, 3, build_options, 0, 0, &data.environment);
    
    /*
     // Create source image
     cl_mem source_image = NULL;
     if(use_host_ptr) {
     source_image = clCreateImage(environment.context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, &image_format, &image_description, source, &error);
     clCheckOrExit(error);
     }
     
     // Create destination image
     cl_mem dest_image = NULL;
     dest_image = clCreateImage(environment.context, CL_MEM_WRITE_ONLY, &image_format, &image_description, NULL, &error);
     clCheckOrExit(error);
     */
    
    // Setup bgr to gray buffers
    data.bgr_to_gray_data.buffers[0] =
        clCreateBuffer(data.environment.context,
                       CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY,
                       data.original_data.stride * data.original_data.height,
                       NULL, &error);
    clCheckOrExit(error);
    data.bgr_to_gray_data.buffers[1] =
        clCreateBuffer(data.environment.context,
                       CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE,
                       data.original_data.width * data.original_data.height,
                       NULL, &error);
    clCheckOrExit(error);
    
    // Setup integral image buffers
    data.integral_image_data.buffers[0] =
        clCreateBuffer(data.environment.context,
                       CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY,
                       data.original_data.width * data.original_data.height,
                       NULL, &error);
    clCheckOrExit(error);
    data.integral_image_data.buffers[1] =
        clCreateBuffer(data.environment.context,
                       CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE,
                       (data.original_data.width + 1) * (data.original_data.height + 1) * sizeof(cl_uint),
                       NULL, &error);
    clCheckOrExit(error);
    data.integral_image_data.buffers[2] =
        clCreateBuffer(data.environment.context,
                   CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE,
                   (data.original_data.width + 1) * (data.original_data.height + 1) * sizeof(cl_double),
                       NULL, &error);
    clCheckOrExit(error);
    data.integral_image_data.buffers[3] =
    clCreateBuffer(data.environment.context,
                   CL_MEM_ALLOC_HOST_PTR | CL_MEM_WRITE_ONLY,
                   (data.original_data.width + 1) * (data.original_data.height + 1) * sizeof(cl_uint),
                   NULL, &error);
    clCheckOrExit(error);
    data.integral_image_data.buffers[4] =
    clCreateBuffer(data.environment.context,
                   CL_MEM_ALLOC_HOST_PTR | CL_MEM_WRITE_ONLY,
                   (data.original_data.width + 1) * (data.original_data.height + 1) * sizeof(cl_ulong),
                   NULL, &error);
    clCheckOrExit(error);
    
    // Setup bgr to gray sizes
    data.bgr_to_gray_data.global_size[0] = data.original_data.width;
    data.bgr_to_gray_data.global_size[1] = data.original_data.height;
    data.bgr_to_gray_data.local_size[0] = 64;
    data.bgr_to_gray_data.local_size[1] = 1;
    
    // Setup integral image sizes
    data.integral_image_data.global_size[0] = data.original_data.height;
    data.integral_image_data.global_size[1] = data.original_data.width;
    data.integral_image_data.local_size[0] = 32;
    data.integral_image_data.local_size[1] = 64;
    
    // Setup bgr to gray kernel args
    clSetKernelArg(data.environment.kernels[0], 0, sizeof(cl_mem), &data.bgr_to_gray_data.buffers[0]);
    clCheckOrExit(error);
    clSetKernelArg(data.environment.kernels[0], 1, sizeof(cl_mem), &data.bgr_to_gray_data.buffers[1]);
    clCheckOrExit(error);
    clSetKernelArg(data.environment.kernels[0], 2, sizeof(cl_uint), &data.original_data.width);
    clCheckOrExit(error);
    clSetKernelArg(data.environment.kernels[0], 3, sizeof(cl_uint), &data.original_data.height);
    clCheckOrExit(error);
    clSetKernelArg(data.environment.kernels[0], 4, sizeof(cl_uint), &data.original_data.stride);
    clCheckOrExit(error);
    
    // Setup integral image (sum rows) kernel args
    clSetKernelArg(data.environment.kernels[1], 0, sizeof(cl_mem), &data.integral_image_data.buffers[0]);
    clCheckOrExit(error);
    clSetKernelArg(data.environment.kernels[1], 1, sizeof(cl_mem), &data.integral_image_data.buffers[1]);
    clCheckOrExit(error);
    clSetKernelArg(data.environment.kernels[1], 2, sizeof(cl_mem), &data.integral_image_data.buffers[2]);
    clCheckOrExit(error);
    clSetKernelArg(data.environment.kernels[1], 3, sizeof(cl_uint), &data.original_data.width);
    clCheckOrExit(error);
    // NB: Set stride to width because we know window size is multiple of 4
    clSetKernelArg(data.environment.kernels[1], 4, sizeof(cl_uint), &data.original_data.width);
    
    // Setup integral image (sum rows) kernel args
    clSetKernelArg(data.environment.kernels[2], 0, sizeof(cl_mem), &data.integral_image_data.buffers[1]);
    clCheckOrExit(error);
    clSetKernelArg(data.environment.kernels[2], 1, sizeof(cl_mem), &data.integral_image_data.buffers[2]);
    clCheckOrExit(error);
    clSetKernelArg(data.environment.kernels[2], 2, sizeof(cl_mem), &data.integral_image_data.buffers[3]);
    clCheckOrExit(error);
    clSetKernelArg(data.environment.kernels[2], 3, sizeof(cl_mem), &data.integral_image_data.buffers[4]);
    clCheckOrExit(error);
    clSetKernelArg(data.environment.kernels[2], 4, sizeof(cl_uint), &data.original_data.width);
    clCheckOrExit(error);
    clSetKernelArg(data.environment.kernels[2], 5, sizeof(cl_uint), &data.original_data.height);
    clCheckOrExit(error);
    
    // Release
    for(cl_uint i = 0; i < platform_device_count; i++)
        clFreeDeviceInfo(&platform_device_list[i]);
    free(platform_device_list);
    
    // Return
    return data;
}

void
releaseCLEnvironment(CLEnvironmentData data) {
    //clEnqueueUnmapMemObject(data.environment.queue, data.dest_image, data.dest_ptr, 0, NULL, NULL);
    for(cl_uint i = 0; i < 2; i++)
        clReleaseMemObject(data.bgr_to_gray_data.buffers[i]);
    for(cl_uint i = 0; i < 5; i++)
        clReleaseMemObject(data.integral_image_data.buffers[i]);
    clFreeDeviceEnvironments(&data.environment, 1, 0);
}

// Host sequential computations
cl_uint*
integralImage(cl_uchar* source,
              cl_uint width,
              cl_uint height)
{
    cl_uint* data = (cl_uint*)malloc((width + 1) * (height + 1) * sizeof(cl_uint)); //source char, dest uint
    cl_int r = -1, c = 1;
    // Sum rows
    for(r = -1; r < height; r++) {
        cl_uint sum = 0;
        for(c = -1; c < width; c++) {
            if(r == -1 || c == -1)
                data[((r + 1) * (width + 1)) + c + 1] = 0;
            else {
                sum += source[(width * r) + c];
                data[((r + 1) * (width + 1)) + c + 1] = sum;
            }
        }
    }
    // Sum cols
    for(c = 0; c < (width + 1); c++) {
        cl_uint sum = 0;
        for(r = 0; r < (height + 1); r++) {
            data[(r * (height + 1)) + c] += sum;
            sum = data[((height + 1) * r) + c];
        }
    }
    return data;
}

cl_uchar*
bgrToGrayScale(cl_uchar* source,
               cl_uint width,
               cl_uint height,
               cl_uint stride,
               cl_uint channels)
{
    cl_uchar* data = (cl_uchar*)malloc(width * height); //source char * channels, dest char
    
    for(cl_uint row = 0; row < height; row++) {
        for(cl_uint col = 0; col < width; col++) {
            cl_uint pos = (row * stride) + (col * channels);
            float gray = (rgb_to_grayscale_coeff[2] * source[pos]) +
                         (rgb_to_grayscale_coeff[1] * source[pos + 1]) +
                         (rgb_to_grayscale_coeff[0] * source[pos + 2]);
            if(gray > 255)
                gray = 255;
            data[(row * width) + col] = (cl_uchar)gray;
        }
    }
    
    return data;
}

// OpenCL computations
cl_uchar*
clBgrToGrayscale(cl_uchar* source,
                 CLEnvironmentData data)
{
    cl_int error = CL_SUCCESS;
    
    // Init buffer
    error = clEnqueueWriteBuffer(data.environment.queue, data.bgr_to_gray_data.buffers[0], CL_FALSE, 0, data.original_data.stride * data.original_data.height, source, 0, NULL, NULL);
    clCheckOrExit(error);
    
    // Run kernel
    error = clEnqueueNDRangeKernel(data.environment.queue, data.environment.kernels[0], 2, NULL, data.bgr_to_gray_data.global_size, data.bgr_to_gray_data.local_size, 0, NULL, NULL);
    clCheckOrExit(error);
    
    // Read result
    cl_uchar* result = (cl_uchar*)clEnqueueMapBuffer(data.environment.queue, data.bgr_to_gray_data.buffers[1], CL_TRUE, CL_MAP_READ, 0, data.original_data.width * data.original_data.height, 0, NULL, NULL, &error);
    clCheckOrExit(error);
    
    data.integral_image_data.ptr = result;
    
    // Return
    return result;
}

void
clIntegralImage(cl_uchar* source,
                CLEnvironmentData data,
                cl_uint** result,
                cl_ulong** square_result)
{
    cl_int error = CL_SUCCESS;
    
    // Init buffer
    error = clEnqueueWriteBuffer(data.environment.queue, data.integral_image_data.buffers[0], CL_FALSE, 0, data.original_data.width * data.original_data.height, source, 0, NULL, NULL);
    clCheckOrExit(error);
    
    // Run sum rows kernel
    error = clEnqueueNDRangeKernel(data.environment.queue, data.environment.kernels[1], 1, NULL, &data.integral_image_data.global_size[0], &data.integral_image_data.local_size[0], 0, NULL, NULL);
    clCheckOrExit(error);
    
    // Run sum cols kernel
    error = clEnqueueNDRangeKernel(data.environment.queue, data.environment.kernels[2], 1, NULL, &data.integral_image_data.global_size[1], &data.integral_image_data.local_size[1], 0, NULL, NULL);
    clCheckOrExit(error);
    
    // Read result
    *result = (cl_uint*)clEnqueueMapBuffer(data.environment.queue, data.integral_image_data.buffers[3], CL_TRUE, CL_MAP_READ, 0, ((data.original_data.width + 1) * (data.original_data.height + 1) * sizeof(cl_uint)), 0, NULL, NULL, &error);
    clCheckOrExit(error);
    
    *square_result = (cl_ulong*)clEnqueueMapBuffer(data.environment.queue, data.integral_image_data.buffers[4], CL_TRUE, CL_MAP_READ, 0, ((data.original_data.width + 1) * (data.original_data.height + 1) * sizeof(cl_ulong)), 0, NULL, NULL, &error);
    clCheckOrExit(error);
    
    data.integral_image_data.ptr = result;
}


void
clIntegralImage(CLEnvironmentData data,
                cl_uint** result,
                cl_ulong** square_result)
{
    cl_int error = CL_SUCCESS;
    
    // Set as arg the output of greyscale    
    clSetKernelArg(data.environment.kernels[1], 0, sizeof(cl_mem), &data.bgr_to_gray_data.buffers[1]);
    clCheckOrExit(error);
    
    // Run sum rows kernel
    error = clEnqueueNDRangeKernel(data.environment.queue, data.environment.kernels[1], 1, NULL, &data.integral_image_data.global_size[0], &data.integral_image_data.local_size[0], 0, NULL, NULL);
    clCheckOrExit(error);
    
    // Run sum cols kernel
    error = clEnqueueNDRangeKernel(data.environment.queue, data.environment.kernels[2], 1, NULL, &data.integral_image_data.global_size[1], &data.integral_image_data.local_size[1], 0, NULL, NULL);
    clCheckOrExit(error);
    
    // Read result
    *result = (cl_uint*)clEnqueueMapBuffer(data.environment.queue, data.integral_image_data.buffers[3], CL_TRUE, CL_MAP_READ, 0, ((data.original_data.width + 1) * (data.original_data.height + 1) * sizeof(cl_uint)), 0, NULL, NULL, &error);
    clCheckOrExit(error);
    
    *square_result = (cl_ulong*)clEnqueueMapBuffer(data.environment.queue, data.integral_image_data.buffers[4], CL_TRUE, CL_MAP_READ, 0, ((data.original_data.width + 1) * (data.original_data.height + 1) * sizeof(cl_uint)), 0, NULL, NULL, &error);
    clCheckOrExit(error);
    
    data.integral_image_data.ptr = result;
}

 
#include "clif.h"

#define matp(matrix,stride,x,y) (matrix + ((stride) * (y)) + (x))
#define mate(matrix,stride,x,y) (*(matp(matrix,stride,x,y)))


// Private computations start
/*
void
ifIntegralImage(const cl_uchar* source,
                const cl_uint width,
                const cl_uint height,
                cl_uint** result,
                cl_ulong** square_result)
{
    cl_uint integral_image_width = 0;
    cl_uint integral_image_height = 0;
    
    cl_uint* data = (cl_uint*)malloc(integral_image_width * integral_image_height * sizeof(cl_uint)); //source char, dest uint
    cl_ulong* square_data = (cl_ulong*)malloc(integral_image_width * integral_image_height * sizeof(cl_ulong)); //source char, dest long
    cl_int x = -1, y = 1;
    
    // Sum rows
    for(y = -1; y < height; y++) {
        cl_uint sum = 0;
        cl_ulong square_sum = 0;
        for(x = -1; x < width; x++) {
            if(x == -1 || y == -1) {
                mate(data, integral_image_width, x+1, y+1) = 0;
                mate(square_data, integral_image_width, x+1, y+1) = 0;
            }
            else {
                cl_uint source_el = mate(source, width, x, y);
                sum += source_el;
                square_sum += source_el * source_el;
                mate(data, integral_image_width, x+1, y+1) = sum;
                mate(square_data, integral_image_width, x+1, y+1) = square_sum;
            }
        }
    }
    // Sum cols
    for(x = 0; x < integral_image_width; x++) {
        cl_uint sum = 0;
        cl_ulong square_sum = 0;
        for(y = 0; y < integral_image_height; y++) {
            mate(data, integral_image_width, x, y) += sum;
            mate(square_data, integral_image_width, x, y) += square_sum;
            sum = mate(data, integral_image_width, x, y);
            square_sum = mate(square_data, integral_image_width, x, y);
        }
    }
}

cl_uchar*
ifBgrToGrayScale(const cl_uchar* source,
                 const cl_uint width,
                 const cl_uint height,
                 const cl_uint stride,
                 const cl_uint channels)
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
// Private computations end
*/
// Init and release OpenCLIF environment
CLIFEnvironmentData*
clifInitEnvironment(const cl_uint device_index)
{
    CLIFEnvironmentData* data = (CLIFEnvironmentData*)malloc(sizeof(CLIFEnvironmentData));
    
    // Get available devices
    cl_uint platform_device_count;
    CLDeviceInfo* platform_device_list = clGetDeviceList(&platform_device_count);
    
    // Get selected device
    CLDeviceInfo device = platform_device_list[device_index];
    
    // Set up kernel file path and functions
    const char* kernel_path = "/Users/Gabriele/Documents/Projects/CLFaceDetection/CLFaceDetection/clif.cl";
    const char* kernel_functions[] = { "bgrToGrayscale", "integralImageSumRows", "integralImageSumCols" };
   
    // Create device environment
    char build_options[1024] = { 0 };
    clCreateDeviceEnvironment(&device, 1, kernel_path, kernel_functions, 3, build_options, 0, 0, &(data->environment));
    
    // Release
    for(cl_uint i = 0; i < platform_device_count; i++)
        clFreeDeviceInfo(&platform_device_list[i]);
    free(platform_device_list);
    
    return data;
}
    /*
     // Create source image
     cl_mem source_image = NULL;
     if(use_host_ptr) {
     source_image = clCreateImage(environment.context, CLIF_MEM_USE_HOST_PTR | CLIF_MEM_READ_ONLY, &image_format, &image_description, source, &error);
     clCheckOrExit(error);
     }
     
     // Create destination image
     cl_mem dest_image = NULL;
     dest_image = clCreateImage(environment.context, CLIF_MEM_WRITE_ONLY, &image_format, &image_description, NULL, &error);
     clCheckOrExit(error);
     */
void
clifInitBuffers(CLIFEnvironmentData* data,
                const cl_uint image_width,
                const cl_uint image_height,
                const cl_uint image_stride,
                const cl_uint image_channels)
{
    cl_int error = CL_SUCCESS;
    
    // Setup bgr to gray buffers
    data->bgr_to_gray_data.buffers[0] =
        clCreateBuffer(data->environment.context,
                       CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY,
                       image_stride * image_height,
                       NULL, &error);
    clCheckOrExit(error);
    data->bgr_to_gray_data.buffers[1] =
        clCreateBuffer(data->environment.context,
                       CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE,
                       image_stride * image_height,
                       NULL, &error);
    clCheckOrExit(error);
    
    // Setup integral image buffers
    data->integral_image_data.buffers[0] =
        clCreateBuffer(data->environment.context,
                       CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY,
                       image_stride * image_height,
                       NULL, &error);
    clCheckOrExit(error);
    data->integral_image_data.buffers[1] =
        clCreateBuffer(data->environment.context,
                       CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE,
                       (image_width + 1) * (image_height + 1) * sizeof(cl_uint),
                       NULL, &error);
    clCheckOrExit(error);
    data->integral_image_data.buffers[2] =
        clCreateBuffer(data->environment.context,
                   CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE,
                   (image_width + 1) * (image_height + 1) * sizeof(cl_double),
                       NULL, &error);
    clCheckOrExit(error);
    data->integral_image_data.buffers[3] =
    clCreateBuffer(data->environment.context,
                   CL_MEM_ALLOC_HOST_PTR | CL_MEM_WRITE_ONLY,
                   (image_width + 1) * (image_height + 1) * sizeof(cl_uint),
                   NULL, &error);
    clCheckOrExit(error);
    data->integral_image_data.buffers[4] =
    clCreateBuffer(data->environment.context,
                   CL_MEM_ALLOC_HOST_PTR | CL_MEM_WRITE_ONLY,
                   (image_width + 1) * (image_height + 1) * sizeof(cl_ulong),
                   NULL, &error);
    clCheckOrExit(error);
    
    // Setup bgr to gray sizes
    data->bgr_to_gray_data.global_size[0] = image_width;
    data->bgr_to_gray_data.global_size[1] = image_height;
    data->bgr_to_gray_data.local_size[0] = 64;
    data->bgr_to_gray_data.local_size[1] = 1;
    
    // Setup integral image sizes
    data->integral_image_data.global_size[0] = image_height;
    data->integral_image_data.global_size[1] = image_width;
    data->integral_image_data.local_size[0] = 32;
    data->integral_image_data.local_size[1] = 64;
    
    // Setup bgr to gray kernel args
    clSetKernelArg(data->environment.kernels[0], 0, sizeof(cl_mem), &(data->bgr_to_gray_data.buffers[0]));
    clCheckOrExit(error);
    clSetKernelArg(data->environment.kernels[0], 1, sizeof(cl_mem), &(data->bgr_to_gray_data.buffers[1]));
    clCheckOrExit(error);
    clSetKernelArg(data->environment.kernels[0], 2, sizeof(cl_uint), &(image_width));
    clCheckOrExit(error);
    clSetKernelArg(data->environment.kernels[0], 3, sizeof(cl_uint), &(image_height));
    clCheckOrExit(error);
    clSetKernelArg(data->environment.kernels[0], 4, sizeof(cl_uint), &(image_stride));
    clCheckOrExit(error);
    
    // Setup integral image (sum rows) kernel args
    clSetKernelArg(data->environment.kernels[1], 0, sizeof(cl_mem), &(data->integral_image_data.buffers[0]));
    clCheckOrExit(error);
    clSetKernelArg(data->environment.kernels[1], 1, sizeof(cl_mem), &(data->integral_image_data.buffers[1]));
    clCheckOrExit(error);
    clSetKernelArg(data->environment.kernels[1], 2, sizeof(cl_mem), &(data->integral_image_data.buffers[2]));
    clCheckOrExit(error);
    clSetKernelArg(data->environment.kernels[1], 3, sizeof(cl_uint), &(image_width));
    clCheckOrExit(error);
    // NB: Set stride to width because we know window size is multiple of 4
    clSetKernelArg(data->environment.kernels[1], 4, sizeof(cl_uint), &(image_width));
    
    // Setup integral image (sum rows) kernel args
    clSetKernelArg(data->environment.kernels[2], 0, sizeof(cl_mem), &(data->integral_image_data.buffers[1]));
    clCheckOrExit(error);
    clSetKernelArg(data->environment.kernels[2], 1, sizeof(cl_mem), &(data->integral_image_data.buffers[2]));
    clCheckOrExit(error);
    clSetKernelArg(data->environment.kernels[2], 2, sizeof(cl_mem), &(data->integral_image_data.buffers[3]));
    clCheckOrExit(error);
    clSetKernelArg(data->environment.kernels[2], 3, sizeof(cl_mem), &(data->integral_image_data.buffers[4]));
    clCheckOrExit(error);
    clSetKernelArg(data->environment.kernels[2], 4, sizeof(cl_uint), &(image_width));
    clCheckOrExit(error);
    clSetKernelArg(data->environment.kernels[2], 5, sizeof(cl_uint), &(image_height));
    clCheckOrExit(error);
}

void
clifReleaseBuffers(CLIFEnvironmentData* data) {
    for(cl_uint i = 0; i < 2; i++)
        clReleaseMemObject(data->bgr_to_gray_data.buffers[i]);
    for(cl_uint i = 0; i < 5; i++)
        clReleaseMemObject(data->integral_image_data.buffers[i]);
}
    
void
clifReleaseEnvironment(CLIFEnvironmentData* data) {
    //clEnqueueUnmapMemObject(data.environment.queue, data.dest_image, data.dest_ptr, 0, NULL, NULL);
    clFreeDeviceEnvironments(&(data->environment), 1, 0);
}

// OpenCLIF computations
CLIFGrayscaleResult
clifGrayscale(const IplImage* source,
              CLIFEnvironmentData* data,
              const cl_bool use_opencl)
{
    CLIFGrayscaleResult ret;
    if(!use_opencl) {
        ret.image = cvCreateImage(cvSize(source->width, source->height), IPL_DEPTH_8U, 1);
        cvCvtColor(source, ret.image, CV_BGR2GRAY);
        return ret;
    }
    
    cl_int error = CL_SUCCESS;
    
    // Init buffer
    error = clEnqueueWriteBuffer(data->environment.queue, data->bgr_to_gray_data.buffers[0], CL_FALSE, 0, source->widthStep * source->height, source->imageData, 0, NULL, NULL);
    clCheckOrExit(error);
    
    // Run kernel
    error = clEnqueueNDRangeKernel(data->environment.queue, data->environment.kernels[0], 2, NULL, data->bgr_to_gray_data.global_size, data->bgr_to_gray_data.local_size, 0, NULL, NULL);
    clCheckOrExit(error);
    
    // Read result
    cl_uchar* result = (cl_uchar*)clEnqueueMapBuffer(data->environment.queue, data->bgr_to_gray_data.buffers[1], CL_TRUE, CL_MAP_READ, 0, source->width * source->height, 0, NULL, NULL, &error);
    clCheckOrExit(error);
    
    // Return
    ret.image = cvCreateImageHeader(cvSize(source->width, source->height), IPL_DEPTH_8U, 1);
    cvSetData(ret.image, result, source->width);
    return ret;
}

CLIFIntegralResult
clifIntegral(const IplImage* source,
             CLIFEnvironmentData* data,
             const cl_bool use_opencl)
{
    CLIFIntegralResult ret;
    
    if(!use_opencl) {        
        ret.image = cvCreateMat(source->height + 1, source->width + 1, CV_32SC1);
        ret.square_image = cvCreateMat(source->height + 1, source->width + 1, CV_64FC1);
        cvIntegral(source, ret.image, ret.square_image);
        return ret;
    }
    
    cl_int error = CL_SUCCESS;
    
    // Init buffer
    error = clEnqueueWriteBuffer(data->environment.queue, data->integral_image_data.buffers[0], CL_FALSE, 0, source->width * source->height, source, 0, NULL, NULL);
    clCheckOrExit(error);
    
    // Run sum rows kernel
    error = clEnqueueNDRangeKernel(data->environment.queue, data->environment.kernels[1], 1, NULL, &(data->integral_image_data.global_size[0]), &(data->integral_image_data.local_size[0]), 0, NULL, NULL);
    clCheckOrExit(error);
    
    // Run sum cols kernel
    error = clEnqueueNDRangeKernel(data->environment.queue, data->environment.kernels[2], 1, NULL, &(data->integral_image_data.global_size[1]), &(data->integral_image_data.local_size[1]), 0, NULL, NULL);
    clCheckOrExit(error);
    
    // Read result
    cl_uint* result = (cl_uint*)clEnqueueMapBuffer(data->environment.queue, data->integral_image_data.buffers[3], CL_TRUE, CL_MAP_READ, 0, (source->width + 1) * (source->height + 1) * sizeof(cl_uint), 0, NULL, NULL, &error);
    clCheckOrExit(error);
    
    cl_ulong* square_result = (cl_ulong*)clEnqueueMapBuffer(data->environment.queue, data->integral_image_data.buffers[4], CL_TRUE, CL_MAP_READ, 0, (source->width + 1) * (source->height + 1) * sizeof(cl_ulong), 0, NULL, NULL, &error);
    clCheckOrExit(error);
    
    data->integral_image_data.ptr = result;
    
    // Return    
    ret.image = cvCreateMat(source->height + 1, source->width + 1, CV_32SC1);
    cvSetData(ret.image, result, (source->width + 1) * sizeof(cl_uint));
    ret.square_image = cvCreateMatHeader(source->height + 1, source->width + 1, CV_64FC1);
    cvSetData(ret.square_image, square_result, (source->width + 1) * sizeof(cl_ulong));
    return ret;
}


CLIFIntegralResult
clifGrayscaleIntegral(const IplImage* source,
                      CLIFEnvironmentData* data,
                      const cl_bool use_opencl)
{
    CLIFIntegralResult ret;
    
    if(!use_opencl) {
        IplImage* grayscale = cvCreateImage(cvSize(source->width, source->height), IPL_DEPTH_8U, 1);
        cvCvtColor(source, grayscale, CV_BGR2GRAY);
        ret.image = cvCreateMat(source->height + 1, source->width + 1, CV_32SC1);
        ret.square_image = cvCreateMat(source->height + 1, source->width + 1, CV_64FC1);
        cvIntegral(grayscale, ret.image, ret.square_image);
        cvReleaseImage(&grayscale);
        
        return ret;
    }
    
    cl_int error = CL_SUCCESS;
    
    // Init buffer
    error = clEnqueueWriteBuffer(data->environment.queue, data->bgr_to_gray_data.buffers[0], CL_FALSE, 0, source->widthStep * source->height, source->imageData, 0, NULL, NULL);
    clCheckOrExit(error);
    
    // Run kernel
    error = clEnqueueNDRangeKernel(data->environment.queue, data->environment.kernels[0], 2, NULL, data->bgr_to_gray_data.global_size, data->bgr_to_gray_data.local_size, 0, NULL, NULL);
    clCheckOrExit(error);
    
    // Set as arg the output of greyscale
    clSetKernelArg(data->environment.kernels[1], 0, sizeof(cl_mem), &(data->bgr_to_gray_data.buffers[1]));
    clCheckOrExit(error);
    
    // Run sum rows kernel
    error = clEnqueueNDRangeKernel(data->environment.queue, data->environment.kernels[1], 1, NULL, &(data->integral_image_data.global_size[0]), &(data->integral_image_data.local_size[0]), 0, NULL, NULL);
    clCheckOrExit(error);
    
    // Run sum cols kernel
    error = clEnqueueNDRangeKernel(data->environment.queue, data->environment.kernels[2], 1, NULL, &(data->integral_image_data.global_size[1]), &(data->integral_image_data.local_size[1]), 0, NULL, NULL);
    clCheckOrExit(error);
    
    // Read result
    cl_uint* result = (cl_uint*)clEnqueueMapBuffer(data->environment.queue, data->integral_image_data.buffers[3], CL_TRUE, CL_MAP_READ, 0, (source->width + 1) * (source->height + 1) * sizeof(cl_uint), 0, NULL, NULL, &error);
    clCheckOrExit(error);
    
    cl_ulong* square_result = (cl_ulong*)clEnqueueMapBuffer(data->environment.queue, data->integral_image_data.buffers[4], CL_TRUE, CL_MAP_READ, 0, (source->width + 1) * (source->height + 1) * sizeof(cl_ulong), 0, NULL, NULL, &error);
    clCheckOrExit(error);
    
    data->integral_image_data.ptr = result;
    
    // Return
    ret.image = cvCreateMatHeader(source->height + 1, source->width + 1, CV_32SC1);
    cvSetData(ret.image, result, source->width + 1);
    ret.square_image = cvCreateMatHeader(source->height + 1, source->width + 1, CV_64FC1);
    cvSetData(ret.square_image, square_result, source->width + 1);
    return ret;
}

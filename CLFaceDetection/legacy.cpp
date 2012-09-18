
cl_uint runStage(cl_uint* integral_image,
                 KernelStage* stage,
                 const CLODSubwindowData* win_src,
                 CLODSubwindowData* win_dst,
                 cl_uint win_src_count,
                 cl_uint* win_dst_count,
                 cl_uint scaled_window_area,
                 cl_float current_scale,
                 cl_uint integral_image_width,
                 cl_uint gid)
{
    // Win dst count must be atomic
    if(gid == 0)
        win_dst_count[0] = 0;
    
    if(gid < win_src_count) {
        const CLODSubwindowData subwindow = win_src[gid];
        
        // Iterate over classifiers
        float stage_sum = 0;
        
        for(cl_uint classifier_index = 0; classifier_index < stage->count; classifier_index++) {
            KernelClassifier classifier = stage->classifier[classifier_index];
            
            // Compute threshold normalized by window vaiance
            float norm_threshold = classifier.threshold * subwindow.variance;
            
            float rect_sum = 0;
            
            // Precalculation on rectangles (loop unroll)
            float first_rect_area = 0;
            float sum_rect_area = 0;
            KernelRect final_rect[3];
            
            // Normalize rect size
            for(cl_uint ri = 0; ri < 3; ri++) {
                if(classifier.rect[ri].weight != 0) {
                    KernelRect temp_rect = classifier.rect[ri];
                    final_rect[ri].x = (cl_uint)round(temp_rect.x * current_scale);
                    final_rect[ri].y = (cl_uint)round(temp_rect.y * current_scale);
                    final_rect[ri].width = (cl_uint)round(temp_rect.width * current_scale);
                    final_rect[ri].height = (cl_uint)round(temp_rect.height * current_scale);
                    // Normalize rect weight based on window area
                    final_rect[ri].weight = (float)(classifier.rect[ri].weight) / (float)scaled_window_area;
                    if(ri == 0)
                        first_rect_area = final_rect[ri].width * final_rect[ri].height;
                    else
                        sum_rect_area += final_rect[ri].weight * final_rect[ri].width * final_rect[ri].height;
                }
            }
            final_rect[0].weight = (float)(-sum_rect_area/first_rect_area);
            
            // Calculation on rectangles (loop unroll)
            for(cl_uint ri = 0; ri < 3; ri++) {
                if(classifier.rect[ri].weight != 0) {
                    cl_uint temp_sum = mate(integral_image,641,subwindow.x + final_rect[ri].x,subwindow.y + final_rect[ri].y);
                    temp_sum -= mate(integral_image,641,subwindow.x + final_rect[ri].x + final_rect[ri].width, subwindow.y + final_rect[ri].y);
                    temp_sum -= mate(integral_image,641,subwindow.x + final_rect[ri].x, subwindow.y + final_rect[ri].y + final_rect[ri].height);
                    temp_sum += mate(integral_image,641,subwindow.x + final_rect[ri].x + final_rect[ri].width, subwindow.y + final_rect[ri].y + final_rect[ri].height);
                    rect_sum += (float)(temp_sum * final_rect[ri].weight);
                }
            }
            
            // If rect sum less than stage_sum updated with threshold left_val else right_val
            stage_sum += classifier.alpha[rect_sum >= norm_threshold];
        }
        
        // Add subwindow to accepted list
        if(stage_sum >= stage->threshold) {
            if(subwindow.x == 114 && subwindow.y == 182) {
                printf("");
            }
            win_dst[*win_dst_count].x = subwindow.x;
            win_dst[*win_dst_count].y = subwindow.y;
            win_dst[*win_dst_count].variance = subwindow.variance;
            win_dst[*win_dst_count].offset = subwindow.offset;
            (*win_dst_count)++;
            return 0;
        }
        else {
            if(subwindow.x == 114 && subwindow.y == 182) {
                printf("");
            }
            return 1;
        }
    }
    return 1;
}


void
runKernelStage(const CLODEnvironmentData* data,
               const cl_uint input_window_count,
               const cl_uint scaled_window_area,
               const cl_float current_scale,
               const cl_uint stage_index,
               cl_uint* output_window_count)
{
    cl_int error = CL_SUCCESS;
    
    // Set source windows count
    clSetKernelArg(data->environment.kernels[0], 4, sizeof(cl_uint), &(input_window_count));
    clCheckOrExit(error);
    
    // Setup kernel sizes (Global size can be not a multiple of 64, set global size as LCM)
    size_t wavefront_size = 64;
    size_t global_size = ((input_window_count / wavefront_size) + 1) * wavefront_size;
    size_t local_size = 1;
    
    // Run kernel
    error = clEnqueueNDRangeKernel(data->environment.queue, data->environment.kernels[0], 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    /*Test on host
     cl_uint* integral_image = (cl_uint*)clEnqueueMapBuffer(data->environment.queue, data->detect_objects_data.buffers[0], CL_TRUE, CL_MAP_READ, 0, 641 * 481 * sizeof(cl_uint), 0, NULL, NULL, &error);
     KernelStage* stage = (KernelStage*)clEnqueueMapBuffer(data->environment.queue, data->detect_objects_data.buffers[1], CL_TRUE, CL_MAP_READ, 0, sizeof(KernelStage), 0, NULL, NULL, &error);
     cl_uint indx_buf_src, indx_buf_dst;
     if(stage_index & 1) {
     indx_buf_src = 3;
     indx_buf_dst = 2;
     }
     else {
     indx_buf_src = 2;
     indx_buf_dst = 3;
     }
     CLODSubwindowData* win_src = (CLODSubwindowData*)clEnqueueMapBuffer(data->environment.queue, data->detect_objects_data.buffers[indx_buf_src], CL_TRUE, CL_MAP_READ, 0, input_window_count * sizeof(CLODSubwindowData), 0, NULL, NULL, &error);
     CLODSubwindowData* win_dst = (CLODSubwindowData*)clEnqueueMapBuffer(data->environment.queue, data->detect_objects_data.buffers[indx_buf_dst], CL_TRUE, CL_MAP_WRITE, 0, input_window_count * sizeof(CLODSubwindowData), 0, NULL, NULL, &error);
     cl_uint win_dst_count = 0;
     for(cl_uint i = 0; i < input_window_count; i++) {
     cl_uint exit = runStage(integral_image,
     stage,
     win_src,
     win_dst,
     input_window_count,
     &win_dst_count,
     scaled_window_area,
     current_scale,
     641,
     i);
     if((stage_index == 0) && exit)
     i++;
     }
     clCheckOrExit(error);
     *output_window_count = win_dst_count;
     
     clEnqueueUnmapMemObject(data->environment.queue, data->detect_objects_data.buffers[0], integral_image, 0, NULL, NULL);
     clEnqueueUnmapMemObject(data->environment.queue, data->detect_objects_data.buffers[1], integral_image, 0, NULL, NULL);
     clEnqueueUnmapMemObject(data->environment.queue, data->detect_objects_data.buffers[indx_buf_src], win_src, 0, NULL, NULL);
     clEnqueueUnmapMemObject(data->environment.queue, data->detect_objects_data.buffers[indx_buf_dst], win_dst, 0, NULL, NULL);
     */
    // Read output window count
    cl_uint* p_output_window_count = (cl_uint*)clEnqueueMapBuffer(data->environment.queue, data->detect_objects_data.buffers[4], CL_TRUE, CL_MAP_READ, 0, sizeof(cl_uint), 0, NULL, NULL, &error);
    clCheckOrExit(error);
    *output_window_count = *p_output_window_count;
    clEnqueueUnmapMemObject(data->environment.queue, data->detect_objects_data.buffers[4], p_output_window_count, 0, NULL, NULL);
    clCheckOrExit(error);
}



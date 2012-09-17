#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

// Optimized for stump based (each classifier has one only feature)

typedef struct KernelOptimizedRect {
    uint left_top_offset;
    uint right_top_offset;
    uint left_bottom_offset;
    uint right_bottom_offset;
    float weight;
} KernelOptimizedRect;

typedef struct KernelClassifier {
    float alpha[2];
    KernelOptimizedRect rect[3];
    float threshold;
} KernelClassifier;

typedef struct KernelStage {
    float threshold;
    KernelClassifier classifier[MAX_STAGE_CLASSIFIER_COUNT];
    uint count;
} KernelStage;

typedef struct KernelSubwindowData {
    uint x;
    uint y;
    uint offset;
    float variance;
} KernelSubwindowData;
    
kernel void runStage(global uint* integral_image,
                     global KernelStage* stage,
                     global KernelSubwindowData* win_src,
                     global KernelSubwindowData* win_dst,
                     uint win_src_count,
                     global uint* win_dst_count,
                     uint scaled_window_area,
                     float current_scale,
                     uint integral_image_width)
{
    uint gid = get_global_id(0);
    
    // Win dst count must be atomic
    if(gid == 0)
        win_dst_count[0] = 0;
    
    if(gid < win_src_count) {
        KernelSubwindowData subwindow = win_src[gid];
        
        // Iterate over classifiers
        float stage_sum = 0;
        
        for(uint classifier_index = 0; classifier_index < stage->count; classifier_index++) {
            KernelClassifier classifier = stage->classifier[classifier_index];
            
            // Compute threshold normalized by window vaiance
            float norm_threshold = classifier.threshold * subwindow.variance;
            
            float rect_sum = 0;
            
            // Calculation on rectangles (loop unroll)
            for(uint ri = 0; ri < 3; ri++) {
                if(classifier.rect[ri].weight != 0) {
                    rect_sum += (float)(integral_image[subwindow.offset + classifier.rect[ri].left_top_offset] -
                                        integral_image[subwindow.offset + classifier.rect[ri].right_top_offset] -
                                        integral_image[subwindow.offset + classifier.rect[ri].left_bottom_offset] +
                                        integral_image[subwindow.offset + classifier.rect[ri].right_bottom_offset]) * classifier.rect[ri].weight;
                }
            }
            
            // If rect sum less than stage_sum updated with threshold left_val else right_val
            stage_sum += classifier.alpha[rect_sum >= norm_threshold];
        }
        
        // Add subwindow to accepted list
        if(stage_sum >= stage->threshold) {
            uint old_dest_count = atom_inc(win_dst_count);
            win_dst[old_dest_count].x = subwindow.x;
            win_dst[old_dest_count].y = subwindow.y;
            win_dst[old_dest_count].variance = subwindow.variance;
            win_dst[old_dest_count].offset = subwindow.offset;
        }
    }
}




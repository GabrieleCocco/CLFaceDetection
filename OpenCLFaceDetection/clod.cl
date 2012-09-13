#define mato(stride,x,y) (((stride) * (y)) + (x));
#define matp(matrix,stride,x,y) (matrix + ((stride) * (y)) + (x))
#define mate(matrix,stride,x,y) (*(matp(matrix,stride,x,y)))
#define mats(matrix,stride,x,y,w,h) \
(mate(matrix,stride,x,y) - mate(matrix,stride,x+w,y) - mate(matrix,stride,x,y+h) + mate(matrix,stride,x+w,y+h))
#defin MAX_STAGE_CLASSIFIER_COUNT 100

// Optimized for stump based (each classifier has one only feature)
typedef struct KernelRect {
    uint x;
    uint y;
    uint width;
    uint height;
    float weight;
} KernelRect;

typedef struct KernelClassifier {
    float alpha[2];
    KernelRect rect[3];    
} KernelClassifier;

typedef struct KernelStage {
    float threshold;
    KernelClassifier classifier[MAX_STAGE_CLASSIFIER_COUNT];
    uint count;
} KernelStage;

typedef struct SubwindowData {
    uint x;
    uint y;
    float variance;
} SubwindowData;
    
kernel void
detectObjects(global uint* integral_image,
              global KernelStage stage,
              global SubwindowData* win_src,
              global SubwindowData* win_dst,
              uint win_src_count,
              global uint* win_dst_count,
              uint scaled_window_area,
              float current_scale,
              uint integral_image_width)
{
    int gid = get_global_id(0);
    
    // Win dst count must be atomic
    if(gid == 0)
        *win_dst_count = 0;
    
    SubwindowData subwindow = win_src[get_global_id(0)];    
    uint offset = mato(integral_image_width, subwindow.x, subwindow.y);
    
    // Iterate over classifiers
    float stage_sum = 0;
    for(uint classifier_index = 0; classifier_index < stage.count; classifier_index++) {
        KernelClassifier classifier = stage.classifier[classifier_index];
        
        float norm_threshold = classifier.threshold * subwindow.variance;
        float rect_sum = 0;
               
        // Calculation on rectangles (loop unroll)
        rect_sum += (float)(mats(integral_image,
                                 integral_image_width,
                                 subwindow.x + classifier.rect[0].x,
                                 subwindow.y + classifier.rect[0].y,
                                 classifier.rect[0].width,
                                 classifier.rect[0].height) * classifier.rect[0].weight) +
                    (float)(mats(integral_image,
                                  integral_image_width,
                                  subwindow.x + classifier.rect[1].x,
                                  subwindow.y + classifier.rect[1].y,
                                  classifier.rect[1].width,
                                  classifier.rect[1].height) * classifier.rect[1].weight);
        if(classifier.rect[2].weight != 0)
            rect_sum +=
            (float)(mats(integral_image,
                         integral_image_width,
                         subwindow.x + classifier.rect[2].x,
                         subwindow.y + classifier.rect[2].y,
                         classifier.rect[2].width,
                         classifier.rect[2].height) * classifier.rect[2].weight);
        
        // If rect sum less than stage_sum updated with threshold left_val else right_val
        stage_sum += classifier.alpha[rect_sum >= norm_threshold];
    }
    
    // If stage sum less than threshold do nothing (atomic operation)
    if(stage_sum >= stage.threshold) {
        uint current_index = atomic_inc(win_dst_count);
        win_dst[current_index].x = subwindow.x;
        win_dst[current_index].y = subwindow.y;
        win_dst[current_index].variance = subwindow.variance;
    }    
}


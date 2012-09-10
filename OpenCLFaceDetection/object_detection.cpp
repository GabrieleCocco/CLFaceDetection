//
//  object_detection.cpp
//  OpenCLFaceDetection
//
//  Created by Gabriele Cocco on 9/8/12.
//  Copyright (c) 2012 Gabriele Cocco. All rights reserved.
//

#include <object_detection.h>

#define EPS 0.2
#define MAX_FEATURE_RECT_COUNT 3
#define MAX_CLASSIFIER_FEATURE_COUNT 200

#define mato(stride,x,y) (((stride) * (y)) + (x));
#define matp(matrix,stride,x,y) (matrix + ((stride) * (y)) + (x))
#define mate(matrix,stride,x,y) (*(matp(matrix,stride,x,y)))
#define mats(matrix,stride,x,y,w,h) \
    (mate(matrix,stride,x,y) - mate(matrix,stride,x+w,y) - mate(matrix,stride,x,y+h) + mate(matrix,stride,x+w,y+h))
#define matsp(lefttop,righttop,leftbottom,rightbottom) \
    (*(lefttop) - *(righttop) - *(leftbottom) + *(rightbottom))

typedef struct OptimizedRect {
    cl_uint *sum_left_top, *sum_left_bottom, *sum_right_top, *sum_right_bottom;
    float weight;
} OptimizedRect;

typedef struct SubwindowData {
    cl_uint x;
    cl_uint y;
    cl_float variance;
} SubwindowData;

cl_uint
areRectSimilar(const CLWeightedRect* r1,
               const CLWeightedRect* r2,
               const cl_float eps)
{
    float delta = eps * (MIN(r1->width, r2->width) + MIN(r1->height, r2->height)) * 0.5;
    return (abs(r1->x - r2->x) <= delta) &&
           (abs(r1->y - r2->y) <= delta) &&
           (abs(r1->x + r1->width - r2->x - r2->width) <= delta) &&
           (abs(r1->y + r1->height - r2->y - r2->height) <= delta);
}

cl_int
partitionData(const CLWeightedRect* data,
              const cl_uint count,
              const cl_float eps,
              int** labels)
{
    int i, j, N = count;
    const int PARENT=0;
    const int RANK=1;
    
    int* _nodes = (int*)malloc(count * 2 * sizeof(int));
    int (*nodes) [2] = (int(*)[2])&_nodes[0];
    
    // The first O(N) pass: create N single-vertex trees
    for(i = 0; i < N; i++)
    {
        nodes[i][PARENT]= -1;
        nodes[i][RANK] = 0;
    }
    
    // The main O(N^2) pass: merge connected components
    for(i = 0; i < N; i++)
    {
        int root = i;
        
        // find root
        while(nodes[root][PARENT] >= 0)
            root = nodes[root][PARENT];
        
        for(j = 0; j < N; j++)
        {
            if(i == j || !areRectSimilar(&data[i], &data[j], eps))
                continue;
            int root2 = j;
            
            while(nodes[root2][PARENT] >= 0)
                root2 = nodes[root2][PARENT];
            
            if(root2 != root) {
                // Merge trees
                int rank = nodes[root][RANK], rank2 = nodes[root2][RANK];
                if(rank > rank2)
                    nodes[root2][PARENT] = root;
                else {
                    nodes[root][PARENT] = root2;
                    nodes[root2][RANK] += rank == rank2;
                    root = root2;
                }
                int k = j;
                int parent;
                
                // compress the path from node2 to root
                while((parent = nodes[k][PARENT]) >= 0) {
                    nodes[k][PARENT] = root;
                    k = parent;
                }
                // compress the path from node to root
                k = i;
                while((parent = nodes[k][PARENT]) >= 0) {
                    nodes[k][PARENT] = root;
                    k = parent;
                }
            }
        }
    }
    
    *labels = (int*)malloc(N * sizeof(int));
    // Final O(N) pass: enumerate classes
    int nclasses = 0;
    
    for(i = 0; i < N; i++)
    {
        int root = i;
        while(nodes[root][PARENT] >= 0)
            root = nodes[root][PARENT];
        // re-use the rank as the class label
        if( nodes[root][RANK] >= 0 )
            nodes[root][RANK] = ~nclasses++;
        (*labels)[i] = ~nodes[root][RANK];
    }
    
    // Release
    free(_nodes);
    
    // Return
    return nclasses;
}

cl_uint
filterResult(CLWeightedRect* data,
             const cl_uint count,
             const int group_threshold,
             const cl_float eps)
{
    int* labels;
    int nclasses = partitionData(data, count, eps, &labels);
    CLWeightedRect* rrects = (CLWeightedRect*)malloc(nclasses * sizeof(CLWeightedRect));
    int* rweights = (int*)calloc(nclasses, sizeof(int));
    
    int i, j;
    int n_labels = (int)count;
    for(i = 0; i < n_labels; i++) {
        int cls = labels[i];
        rrects[cls].x += data[i].x;
        rrects[cls].y += data[i].y;
        rrects[cls].width += data[i].width;
        rrects[cls].height += data[i].height;
        rweights[cls]++;
    }
    
    for(i = 0; i < nclasses; i++)
    { 
        CLWeightedRect r = rrects[i];
        float s = 1.f/(float)rweights[i];
        rrects[i].x = (int)MIN(r.x * s, INT_MAX);
        rrects[i].y = (int)MIN(r.y * s, INT_MAX);
        rrects[i].width = (int)MIN(r.width * s, INT_MAX);
        rrects[i].height = (int)MIN(r.height * s, INT_MAX);
        rrects[i].weight = rweights[i];
    }
    
    memset(data, 0, count * sizeof(CLRect));
    
    cl_uint insertion_point = 0;
    for(i = 0; i < nclasses; i++) {
        CLWeightedRect r1 = rrects[i];
        int n1 = rweights[i];
        if(n1 <= group_threshold)
            continue;
        
        // filter out small face rectangles inside large rectangles
        for(j = 0; j < nclasses; j++)
        {
            int n2 = rweights[j];
            
            if(j == i || n2 <= group_threshold)
                continue;
            CLWeightedRect r2 = rrects[j];
            
            int dx = (int)MAX(r2.width * eps, INT_MAX);
            int dy = (int)MAX(r2.height * eps, INT_MAX);
            if(i != j &&
               r1.x >= r2.x - dx &&
               r1.y >= r2.y - dy &&
               r1.x + r1.width <= r2.x + r2.width + dx &&
               r1.y + r1.height <= r2.y + r2.height + dy &&
               (n2 > MAX(3, n1) || n1 < 3))
                break;
        }
        
        if(j == nclasses) {
            data[insertion_point] = r1;
            insertion_point++;
        }
    }
    
    // Release
    free(rweights);
    free(labels);
    free(rrects);
    
    // Return
    return insertion_point;
}

/*** Various implementations below ***/
void
setupImage(const IplImage* src,
           CvMat** sum,
           CvMat** square_sum)
{
    // Compute grayscale image
    IplImage* grayscale = cvCreateImage(cvSize(src->width, src->height), IPL_DEPTH_8U, 1);
    cvCvtColor(src, grayscale, CV_BGR2GRAY);

    // Compute integral images
    *sum = cvCreateMat(src->height + 1, src->width + 1, CV_32SC1);
    *square_sum = cvCreateMat(src->height + 1, src->width + 1, CV_64FC1);

    cvIntegral(grayscale, *sum, *square_sum);
    
    // Release
    cvReleaseImage(&grayscale);
}

cl_int
setupScale(const cl_float current_scale,
           const cl_uint image_width,
           const cl_uint image_height,
           const cl_uint orig_window_width,
           const cl_uint orig_window_height,
           const cl_uint min_window_width,
           const cl_uint min_window_height,
           const cl_uint max_window_width,
           const cl_uint max_window_height,
           CLRect* equ_rect,
           cl_uint* scaled_window_width,
           cl_uint* scaled_window_height,
           cl_uint* scaled_window_area,
           cl_uint* end_x,
           cl_uint* end_y,
           cl_float* step)
{
    // Compute window y shift
    *step = MAX(2.0, (float)current_scale);
    
    // Compute scaled window size
    *scaled_window_width = (cl_uint)round(orig_window_width * current_scale);
    *scaled_window_height = (cl_uint)round(orig_window_height * current_scale);
    
    // If the window is smaller than the minimum size continue
    if(*scaled_window_width < min_window_width || *scaled_window_height < min_window_height)
        return -1;
    // If the window is bigger than the maximum size continue
    if((max_window_width != 0) && (*scaled_window_width > max_window_width))
        return -1;
    if((max_window_height != 0) && (*scaled_window_height > max_window_height))
        return -1;
    
    // If the window is bigger than the image exit
    if(*scaled_window_width > image_width || *scaled_window_height > image_height)
        return -1;
    
    // Compute scaled window area (using equalized rect, not fully understood)
    equ_rect->x = (cl_uint)round(current_scale);
    equ_rect->y = equ_rect->x;
    equ_rect->width = (cl_uint)round((orig_window_width - 2) * current_scale);
    equ_rect->height = (cl_uint)round((orig_window_height - 2) * current_scale);
    *scaled_window_area = equ_rect->width * equ_rect->height;
    
    // Set init and end positions of subwindows
    *end_x = (int)lrint((image_width - *scaled_window_width) / *step);
    *end_y = (int)lrint((image_height - *scaled_window_height) / *step);
    
    return 0;
}


cl_float
computeVariance(const cl_uint* integral_image,
                const cl_double* square_integral_image,
                const cl_uint integral_image_width,
                const CLRect* equ_rect,
                const cl_uint x,
                const cl_uint y,
                const cl_uint scaled_window_area)
{
    // Sum of window pixels normalized by the window size E(x)
    float mean = (float)mats(integral_image, integral_image_width, x + equ_rect->x, y + equ_rect->y, equ_rect->width, equ_rect->height) / (float)scaled_window_area;
    // E(xˆ2) - Eˆ2(x)
    float variance = (float)mats(square_integral_image, integral_image_width, x + equ_rect->x, y + equ_rect->y, equ_rect->width, equ_rect->height);
    variance = (variance / (float)scaled_window_area) - (mean * mean);
    // Fix wrong variance
    if(variance >= 0)
        variance = sqrt(variance);
    else
        variance = 1;
    
    return variance;
    
}

void
precomputeFeatures(cl_uint* integral_image,
                   const cl_uint integral_image_width,
                   const cl_uint scaled_window_area,
                   const CvHaarClassifierCascade* casc,
                   const cl_float current_scale,
                   OptimizedRect* opt_rectangles)
{
    // Precompute feature rect offset in integral image and square integral image into a new cascade
    cl_uint opt_rect_index = 0;
    for(cl_uint stage_index = 0; stage_index < casc->count; stage_index++) {
        for(cl_uint classifier_index = 0; classifier_index < casc->stage_classifier[stage_index].count; classifier_index++) {
            // Optimized for stump based classifier (otherwise loop over features)
            CvHaarFeature feature = casc->stage_classifier[stage_index].classifier[classifier_index].haar_feature[0];
            
            // Normalize rect weight based on window area
            cl_float first_rect_area;
            cl_uint first_rect_index = opt_rect_index;
            cl_float sum_rect_area = 0;
            for(cl_uint i = 0; i < MAX_FEATURE_RECT_COUNT; i++) {
                opt_rectangles[opt_rect_index].weight = 0;
                if(feature.rect[i].weight != 0) {
                    register CvRect* temp_rect = &feature.rect[i].r;
                    register OptimizedRect* opt_rect = &opt_rectangles[opt_rect_index];
                    register cl_uint rect_x = round(temp_rect->x * current_scale);
                    register cl_uint rect_y = round(temp_rect->y * current_scale);
                    register cl_uint rect_width = round(temp_rect->width * current_scale);
                    register cl_uint rect_height = round(temp_rect->height * current_scale);
                    register cl_float rect_weight = (feature.rect[i].weight) / (float)scaled_window_area;
                    opt_rect->weight = rect_weight;
                    opt_rect->sum_left_top = matp(integral_image, integral_image_width, rect_x, rect_y);
                    opt_rect->sum_right_top = matp(integral_image, integral_image_width, rect_x + rect_width, rect_y);
                    opt_rect->sum_left_bottom = matp(integral_image, integral_image_width, rect_x, rect_y + rect_height);
                    opt_rect->sum_right_bottom = matp(integral_image, integral_image_width, rect_x + rect_width, rect_y + rect_height);
                    
                    if(i > 0)
                        sum_rect_area += rect_weight * rect_width * rect_height;
                    else
                        first_rect_area = rect_width * rect_height;
                    
                    opt_rect_index++;
                }
            }
            opt_rectangles[first_rect_index].weight = (-sum_rect_area/first_rect_area);
        }
    }
}

void
precomputeWindows(const cl_uint step,
                  const cl_uint* integral_image,
                  const cl_double* square_integral_image,
                  const cl_uint integral_image_width,
                  const CLRect* equ_rect,
                  const cl_uint start_x,
                  const cl_uint start_y,
                  const cl_uint end_x,
                  const cl_uint end_y,
                  const cl_uint scaled_window_area,
                  SubwindowData** subwindow_data,
                  cl_uint* subwindow_count)
{    
    // Precompute x and y vars for each subwindow
    *subwindow_data = (SubwindowData*)malloc((end_y - start_y + 1) * (end_x - start_x + 1) * sizeof(SubwindowData));
    cl_uint current_subwindow = 0;
    for(int y_index = start_y; y_index < end_y; y_index++) {
        for(int x_index = start_x; x_index < end_x; x_index++) {
            // Real position
            int x = (int)round(x_index * step);
            int y = (int)round(y_index * step);
            
            cl_float variance = computeVariance(integral_image, square_integral_image, integral_image_width, equ_rect, x, y, scaled_window_area);
            
            (*subwindow_data)[current_subwindow].x = x;
            (*subwindow_data)[current_subwindow].y = y;
            (*subwindow_data)[current_subwindow].variance = variance;
            current_subwindow++;
        }
    }
    *subwindow_count = current_subwindow;
}

void
runClassifier(const cl_uint* integral_image,
              const cl_uint integral_image_width,
              const CvHaarClassifier* classifier,
              const cl_uint x,
              const cl_uint y,
              const cl_float variance,
              const cl_float current_scale,
              const cl_uint scaled_window_area,
              cl_float* stage_sum)
{
    // Compute threshold normalized by window vaiance
    float norm_threshold = *classifier->threshold * variance;
    
    // Iterate over features (optimized for stump)
    CvHaarFeature feature = classifier->haar_feature[0];
    
    float rect_sum = 0;
    
    // Precalculation on rectangles (loop unroll)
    float first_rect_area = 0;
    float sum_rect_area = 0;
    CLWeightedRect final_rect[3];
    
    // Normalize rect size
    for(cl_uint ri = 0; ri < 3; ri++) {
        if(feature.rect[ri].weight != 0) {
            register CvRect* temp_rect = &feature.rect[ri].r;
            register CLWeightedRect* temp_final_rect = &final_rect[ri];
            temp_final_rect->x = (cl_uint)round(temp_rect->x * current_scale);
            temp_final_rect->y = (cl_uint)round(temp_rect->y * current_scale);
            temp_final_rect->width = (cl_uint)round(temp_rect->width * current_scale);
            temp_final_rect->height = (cl_uint)round(temp_rect->height * current_scale);
            // Normalize rect weight based on window area
            temp_final_rect->weight = (float)(feature.rect[ri].weight) / (float)scaled_window_area;
            if(ri == 0)
                first_rect_area = temp_final_rect->width * temp_final_rect->height;
            else
                sum_rect_area += temp_final_rect->weight * temp_final_rect->width * temp_final_rect->height;
        }
    }
    final_rect[0].weight = (float)(-sum_rect_area/first_rect_area);
    
    // Calculation on rectangles (loop unroll)
    for(cl_uint ri = 0; ri < 3; ri++) {
        if(feature.rect[ri].weight != 0) {
            rect_sum += (cl_float)((mats(integral_image,
                                      integral_image_width,
                                      x + final_rect[ri].x,
                                      y + final_rect[ri].y,
                                      final_rect[ri].width,
                                      final_rect[ri].height) * final_rect[ri].weight));
        }
    }
    // If rect sum less than stage_sum updated with threshold left_val else right_val
    *stage_sum += classifier->alpha[rect_sum >= norm_threshold];
}

void
runClassifierWithPrecomputedFeatures(const cl_uint* integral_image,
                                     const CvHaarClassifier* classifier,
                                     const OptimizedRect* opt_rectangles,
                                     cl_uint *opt_rect_index,
                                     const cl_uint offset,
                                     const cl_float variance,
                                     cl_float* stage_sum)
{    
    // Compute threshold normalized by window vaiance
    float norm_threshold = *classifier->threshold * variance;
    
    // Iterate over features (optimized for stump)
    //for(cl_uint feature_index = 0; feature_index < classifier.count; feature_index++) {
    CvHaarFeature feature = classifier->haar_feature[0];
    
    // Calculation on rectangles (loop unroll)
    cl_float rect_sum =
    (matsp(opt_rectangles[*opt_rect_index].sum_left_top + offset,
           opt_rectangles[*opt_rect_index].sum_right_top + offset,
           opt_rectangles[*opt_rect_index].sum_left_bottom + offset,
           opt_rectangles[*opt_rect_index].sum_right_bottom + offset) * opt_rectangles[*opt_rect_index].weight);
    (*opt_rect_index)++;
    rect_sum +=
    (matsp(opt_rectangles[*opt_rect_index].sum_left_top + offset,
           opt_rectangles[*opt_rect_index].sum_right_top + offset,
           opt_rectangles[*opt_rect_index].sum_left_bottom + offset,
           opt_rectangles[*opt_rect_index].sum_right_bottom + offset) * opt_rectangles[*opt_rect_index].weight);
    (*opt_rect_index)++;
    if(feature.rect[2].weight != 0) {
        rect_sum +=
        (matsp(opt_rectangles[*opt_rect_index].sum_left_top + offset,
               opt_rectangles[*opt_rect_index].sum_right_top + offset,
               opt_rectangles[*opt_rect_index].sum_left_bottom + offset,
               opt_rectangles[*opt_rect_index].sum_right_bottom + offset) * opt_rectangles[*opt_rect_index].weight);
        (*opt_rect_index)++;
    }
    
    // If rect sum less than stage_sum updated with threshold left_val else right_val
    *stage_sum += classifier->alpha[rect_sum >= norm_threshold];
}

cl_uint
runCascade(const cl_uint* integral_image,
           const cl_uint integral_image_width,
           CvHaarClassifierCascade* cascade,
           OptimizedRect* opt_rectangles,
           const cl_uint x,
           const cl_uint y,
           const cl_uint scaled_window_width,
           const cl_uint scaled_window_height,
           const cl_uint scaled_window_area,
           const cl_float variance,
           const cl_float current_scale,
           const cl_bool precompute_features,
           CLWeightedRect* matches,
           cl_uint* match_count)
{
    cl_uint offset = mato(integral_image_width, x, y);
    
    // Iterate over stages until skip
    cl_int exit_stage = 1;
    cl_uint opt_rect_index = 0;
    for(cl_uint stage_index = 0; stage_index < cascade->count; stage_index++)
    {
        CvHaarStageClassifier stage = cascade->stage_classifier[stage_index];
        
        // Iterate over classifiers
        float stage_sum = 0;
        for(cl_uint classifier_index = 0; classifier_index < stage.count; classifier_index++) {
            CvHaarClassifier classifier = stage.classifier[classifier_index];
            if(precompute_features)
                runClassifierWithPrecomputedFeatures(integral_image, &classifier, opt_rectangles, &opt_rect_index, offset, variance, &stage_sum);
            else
                runClassifier(integral_image, integral_image_width, &classifier, x, y, variance, current_scale, scaled_window_area, &stage_sum);
        }
        // If stage sum less than threshold exit and continue with next window
        if(stage_sum < stage.threshold) {
            exit_stage = -stage_index;
            break;
        }
    }
    
    // If exit at first stage increment by 2, else by 1
    if(exit_stage <= 0)
        return 1;
    else {
        CLWeightedRect* r = &matches[*match_count];
        r->x = x;
        r->y = y;
        r->width = scaled_window_width;
        r->height = scaled_window_height;
        r->weight = 0;
        (*match_count)++;
    }
    
    return 0;
}

void
runSubwindow(const cl_uint* integral_image,
             const cl_uint integral_image_width,
             const OptimizedRect* opt_rectangles,
             const cl_uint start_rect_index,
             cl_uint* end_rect_index,
             const CvHaarStageClassifier* stage,
             const SubwindowData* win_src,
             SubwindowData** p_win_dst,
             const cl_uint win_src_count,
             cl_uint* win_dst_count,
             const cl_uint scaled_window_area,
             const cl_float current_scale,
             const cl_bool precompute_features)
{
    *p_win_dst = (SubwindowData*)malloc(win_src_count * sizeof(SubwindowData));
    SubwindowData* win_dst = *p_win_dst;
    *win_dst_count = 0;
    
    // Parallelize this
    for(cl_uint subwindow_index = 0; subwindow_index < win_src_count; subwindow_index++) {
        SubwindowData subwindow = win_src[subwindow_index];
        
        cl_uint offset = mato(integral_image_width, subwindow.x, subwindow.y);
        
        // Iterate over classifiers
        float stage_sum = 0;
        cl_uint opt_rect_index = start_rect_index;
        for(cl_uint classifier_index = 0; classifier_index < stage->count; classifier_index++) {
            CvHaarClassifier classifier = stage->classifier[classifier_index];
            if(precompute_features)
                runClassifierWithPrecomputedFeatures(integral_image, &classifier, opt_rectangles, &opt_rect_index, offset, subwindow.variance, &stage_sum);
            else
                runClassifier(integral_image, integral_image_width, &classifier, subwindow.x, subwindow.y, subwindow.variance, current_scale, scaled_window_area, &stage_sum);
        }
        *end_rect_index = opt_rect_index;
        
        // Add subwindow to accepted list
        if(stage_sum >= stage->threshold) {
            win_dst[*win_dst_count].x = subwindow.x;
            win_dst[*win_dst_count].y = subwindow.y;
            win_dst[*win_dst_count].variance = subwindow.variance;
            (*win_dst_count)++;
        }
    }
}

    
CLWeightedRect*
detectObjects(IplImage* image,
              CvHaarClassifierCascade* cascade,
              CLEnvironmentData* data,
              cl_uint min_window_width,
              cl_uint min_window_height,
              cl_uint max_window_width,
              cl_uint max_window_height,
              cl_uint min_neighbors,
              cl_uint* final_match_count,
              cl_bool precompute_features,
              cl_bool per_stage_iteration)
{
    
    float scale_factor = 1.1;
    //ElapseTime time,time2,time3,time4;
    // Compute greyscale image
   /* cl_uchar* grayscale_image = clBgrToGrayscale((cl_uchar*)image->imageData, *data);
    
    // Compute integral image and squared integral image
    cl_uint* integral_image;
    cl_ulong *square_integral_image;
    clIntegralImage(grayscale_image, *data, &integral_image, &square_integral_image);
    */
    // Setup image
    CvMat* sum, *square_sum;
    setupImage(image, &sum, &square_sum);
    cl_uint* integral_image = (cl_uint*)sum->data.ptr;
    cl_double* square_integral_image = (cl_double*)square_sum->data.ptr;
    
    // Calculate number of different scales
    cl_uint scale_count = 0;
    for(float current_scale = 1;
        current_scale * cascade->orig_window_size.width < image->width - 10 &&
        current_scale * cascade->orig_window_size.height < image->height - 10;
        current_scale *= scale_factor) {
        scale_count++;
    }
    
    // Precompute feature rect offset in integral image and square integral image into a new cascade
    OptimizedRect* opt_rectangles = NULL;
    if(precompute_features) 
        opt_rectangles = (OptimizedRect*)malloc(cascade->count * MAX_CLASSIFIER_FEATURE_COUNT * MAX_FEATURE_RECT_COUNT * sizeof(OptimizedRect));
    
    // Vector to store positive matches
    CLWeightedRect* matches = (CLWeightedRect*)malloc(image->width * image->height * scale_count * sizeof(CLWeightedRect));
    cl_uint match_count = 0;
    
    // Iterate over scales
    cl_float current_scale = 1;
    for(cl_uint scale_index = 0; scale_index < scale_count; scale_index++, current_scale *= scale_factor) {
        
        // Setup scale-dependent variables
        cl_uint scaled_window_width, scaled_window_height, scaled_window_area;
        CLRect equ_rect;
        cl_uint start_x = 0, start_y = 0;
        cl_uint end_x, end_y;
        cl_float step;
        if(setupScale(current_scale,
                      image->width, image->height,
                      cascade->orig_window_size.width, cascade->orig_window_size.height,
                      min_window_width, min_window_height,
                      max_window_width, max_window_height,
                      &equ_rect,
                      &scaled_window_width, &scaled_window_height, &scaled_window_area, &end_x, &end_y, &step) != CL_SUCCESS) {
            continue;
        }
        
        // Precompute feature rect offset in integral image and square integral image into a new cascade
        if(precompute_features)
            precomputeFeatures(integral_image, image->width + 1, scaled_window_area, cascade, current_scale, opt_rectangles);
        
        if(!per_stage_iteration) {
            // Iterate over windows
            for(int y_index = start_y; y_index < end_y; y_index++) {
                for(int x_index = start_x; x_index < end_x; x_index++) {
                    // Real position
                    cl_uint x = (int)round(x_index * step);
                    cl_uint y = (int)round(y_index * step);
                    
                    // Sum of window pixels normalized by the window size E(x)
                    cl_float variance = computeVariance(integral_image, square_integral_image, image->width + 1, &equ_rect, x, y, scaled_window_area);
                    
                    // Run cascade on point x,y
                    cl_uint incr = runCascade(integral_image, image->width + 1,
                                              cascade,
                                              opt_rectangles,
                                              x, y,
                                              scaled_window_width, scaled_window_height, scaled_window_area,
                                              variance, current_scale, precompute_features,
                                              matches, &match_count);
                    x_index += incr;
                }
            }
        }
        else {
            // Allocate windows to be computed by successive stages
            SubwindowData* input_windows = (SubwindowData*)malloc((end_y - start_y + 1) * (end_x - start_x + 1) * sizeof(SubwindowData));
            SubwindowData* output_windows = NULL;
            cl_uint input_window_count = 0;
            cl_uint output_window_count = 0;
            
            // Precompute windows
            precomputeWindows(step, integral_image, square_integral_image, image->width + 1,
                              &equ_rect,
                              start_x, start_y, end_x, end_y,
                              scaled_window_area, &input_windows, &input_window_count);            
            
            // Iterate over stages
            cl_uint start_rect_index = 0;
            cl_uint end_rect_index = 0;
            for(cl_uint stage_index = 0; stage_index < cascade->count; stage_index++)
            {
                CvHaarStageClassifier stage = cascade->stage_classifier[stage_index];
                // Run stage on GPU for each subwindow
                runSubwindow(integral_image, image->width + 1,
                             opt_rectangles,
                             start_rect_index, &end_rect_index,
                             &stage,
                             input_windows, &output_windows,
                             input_window_count, &output_window_count,
                             scaled_window_area, current_scale, precompute_features);
                
                free(input_windows);
                input_windows = output_windows;
                input_window_count = output_window_count;
                start_rect_index = end_rect_index;
                if(output_window_count == 0)
                    break;
            }
            
            // Add to matches
            for(cl_uint i = 0; i < output_window_count; i++) {
                matches[match_count].x = output_windows[i].x;
                matches[match_count].y = output_windows[i].y;
                matches[match_count].width = scaled_window_width;
                matches[match_count].height = scaled_window_height;
                match_count++;
            }
            free(output_windows);
        }
    }
    
    // Filter out results
    if(min_neighbors != 0) 
        match_count = filterResult(matches, match_count, MAX(min_neighbors, 1), EPS);
    
    // Release
    if(precompute_features)
        free(opt_rectangles);
    cvReleaseMat(&sum);
    cvReleaseMat(&square_sum);
    
    // Return
    *final_match_count = match_count;
    return matches;
}
    

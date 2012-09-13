//
//  object_detection.cpp
//  OpenCLFaceDetection
//
//  Created by Gabriele Cocco on 9/8/12.
//  Copyright (c) 2012 Gabriele Cocco. All rights reserved.
//

#include "clod.h"

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

typedef struct CLODOptimizedRect {
    cl_uint *sum_left_top, *sum_left_bottom, *sum_right_top, *sum_right_bottom;
    float weight;
} CLODOptimizedRect;

typedef struct CLODSubwindowData {
    cl_uint x;
    cl_uint y;
    cl_uint offset;
    cl_float variance;
} CLODSubwindowData;

cl_uint
areRectSimilar(const CLODWeightedRect* r1,
               const CLODWeightedRect* r2,
               const cl_float eps)
{
    float delta = eps * (MIN(r1->rect.width, r2->rect.width) + MIN(r1->rect.height, r2->rect.height)) * 0.5;
    return (abs(r1->rect.x - r2->rect.x) <= delta) &&
           (abs(r1->rect.y - r2->rect.y) <= delta) &&
           (abs(r1->rect.x + r1->rect.width - r2->rect.x - r2->rect.width) <= delta) &&
           (abs(r1->rect.y + r1->rect.height - r2->rect.y - r2->rect.height) <= delta);
}

cl_int
partitionData(const CLODWeightedRect* data,
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
filterResult(CLODWeightedRect* data,
             const cl_uint count,
             const int group_threshold,
             const cl_float eps)
{
    int* labels;
    int nclasses = partitionData(data, count, eps, &labels);
    CLODWeightedRect* rrects = (CLODWeightedRect*)malloc(nclasses * sizeof(CLODWeightedRect));
    int* rweights = (int*)calloc(nclasses, sizeof(int));
    
    int i, j;
    int n_labels = (int)count;
    for(i = 0; i < n_labels; i++) {
        int cls = labels[i];
        rrects[cls].rect.x += data[i].rect.x;
        rrects[cls].rect.y += data[i].rect.y;
        rrects[cls].rect.width += data[i].rect.width;
        rrects[cls].rect.height += data[i].rect.height;
        rweights[cls]++;
    }
    
    for(i = 0; i < nclasses; i++)
    { 
        CLODWeightedRect r = rrects[i];
        float s = 1.f/(float)rweights[i];
        rrects[i].rect.x = (int)MIN(r.rect.x * s, INT_MAX);
        rrects[i].rect.y = (int)MIN(r.rect.y * s, INT_MAX);
        rrects[i].rect.width = (int)MIN(r.rect.width * s, INT_MAX);
        rrects[i].rect.height = (int)MIN(r.rect.height * s, INT_MAX);
        rrects[i].weight = rweights[i];
    }
    
    memset(data, 0, count * sizeof(CvRect));
    
    cl_uint insertion_point = 0;
    for(i = 0; i < nclasses; i++) {
        CLODWeightedRect r1 = rrects[i];
        int n1 = rweights[i];
        if(n1 <= group_threshold)
            continue;
        
        // filter out small face rectangles inside large rectangles
        for(j = 0; j < nclasses; j++)
        {
            int n2 = rweights[j];
            
            if(j == i || n2 <= group_threshold)
                continue;
            CLODWeightedRect r2 = rrects[j];
            
            int dx = (int)MAX(r2.rect.width * eps, INT_MAX);
            int dy = (int)MAX(r2.rect.height * eps, INT_MAX);
            if(i != j &&
               r1.rect.x >= r2.rect.x - dx &&
               r1.rect.y >= r2.rect.y - dy &&
               r1.rect.width + r1.rect.width <= r2.rect.x + r2.rect.width + dx &&
               r1.rect.height + r1.rect.height <= r2.rect.y + r2.rect.height + dy &&
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
    CLIFIntegralResult result = clifGrayscaleIntegral(src, NULL, false);
    *sum = result.image;
    *square_sum = result.square_image;
}

cl_int
setupScale(const cl_float current_scale,
           const CvSize* image_size,
           const CvSize* feature_size,
           const CvSize* min_window_size,
           const CvSize* max_window_size,
           CvRect* equ_rect,
           CvSize* scaled_window_size,
           cl_uint* scaled_window_area,
           CvPoint* end_point,
           cl_float* step)
{
    // Compute window y shift
    *step = MAX(2.0, (float)current_scale);
    
    // Compute scaled window size
    scaled_window_size->width = (cl_uint)round(feature_size->width * current_scale);
    scaled_window_size->height = (cl_uint)round(feature_size->height * current_scale);
    
    // If the window is smaller than the minimum size continue
    if(scaled_window_size->width < min_window_size->width || scaled_window_size->height < min_window_size->height)
        return -1;
    // If the window is bigger than the maximum size continue
    if((max_window_size->width != 0) && (scaled_window_size->width > max_window_size->width))
        return -1;
    if((max_window_size->height != 0) && (scaled_window_size->height > max_window_size->height))
        return -1;
    
    // If the window is bigger than the image exit
    if(scaled_window_size->width > image_size->width || scaled_window_size->height > image_size->height)
        return -1;
    
    // Compute scaled window area (using equalized rect, not fully understood)
    equ_rect->x = (cl_uint)round(current_scale);
    equ_rect->y = equ_rect->x;
    equ_rect->width = (cl_uint)round((feature_size->width - 2) * current_scale);
    equ_rect->height = (cl_uint)round((feature_size->height - 2) * current_scale);
    *scaled_window_area = equ_rect->width * equ_rect->height;
    
    // Set init and end positions of subwindows
    end_point->x = (int)lrint((image_size->width - scaled_window_size->width) / *step);
    end_point->y = (int)lrint((image_size->height - scaled_window_size->height) / *step);
    
    return 0;
}


inline cl_float
computeVariance(const CvMat* integral_image,
                const CvMat* square_integral_image,
                const CvRect* equ_rect,
                const CvPoint* point,
                const cl_uint scaled_window_area)
{
    // Sum of window pixels normalized by the window size E(x)
    float mean = (float)mats(integral_image->data.i,
                             integral_image->width,
                             point->x + equ_rect->x,
                             point->y + equ_rect->y,
                             equ_rect->width, equ_rect->height) / (float)scaled_window_area;
    // E(xˆ2) - Eˆ2(x)
    float variance = (float)mats(square_integral_image->data.db,
                                 integral_image->width,
                                 point->x + equ_rect->x,
                                 point->y + equ_rect->y,
                                 equ_rect->width, equ_rect->height);
    
    variance = (variance / (float)scaled_window_area) - (mean * mean);
    // Fix wrong variance
    if(variance >= 0)
        variance = sqrt(variance);
    else
        variance = 1;
    
    return variance;    
}

inline void
precomputeFeatures(const CvMat* integral_image,
                   const cl_uint scaled_window_area,
                   const CvHaarClassifierCascade* casc,
                   const cl_float current_scale,
                   CLODOptimizedRect* opt_rectangles)
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
                    register CLODOptimizedRect* opt_rect = &opt_rectangles[opt_rect_index];
                    register cl_uint rect_x = round(temp_rect->x * current_scale);
                    register cl_uint rect_y = round(temp_rect->y * current_scale);
                    register cl_uint rect_width = round(temp_rect->width * current_scale);
                    register cl_uint rect_height = round(temp_rect->height * current_scale);
                    register cl_float rect_weight = (feature.rect[i].weight) / (float)scaled_window_area;
                    opt_rect->weight = rect_weight;
                    opt_rect->sum_left_top = matp((cl_uint*)integral_image->data.i, integral_image->width, rect_x, rect_y);
                    opt_rect->sum_right_top = matp((cl_uint*)integral_image->data.i, integral_image->width, rect_x + rect_width, rect_y);
                    opt_rect->sum_left_bottom = matp((cl_uint*)integral_image->data.i, integral_image->width, rect_x, rect_y + rect_height);
                    opt_rect->sum_right_bottom = matp((cl_uint*)integral_image->data.i, integral_image->width, rect_x + rect_width, rect_y + rect_height);
                    
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

inline void
precomputeWindows(const cl_float step,
                  const CvMat* integral_image,
                  const CvMat* square_integral_image,
                  const CvRect* equ_rect,
                  const CvPoint* start_point,
                  const CvPoint* end_point,
                  const cl_uint scaled_window_area,
                  CLODSubwindowData** psubwindow_data,
                  cl_uint* subwindow_count)
{    
    // Precompute x and y vars for each subwindow
    *psubwindow_data = (CLODSubwindowData*)malloc((end_point->y - start_point->y) * (end_point->x - start_point->x) * sizeof(CLODSubwindowData));
    CLODSubwindowData* subwindow_data = *psubwindow_data;
    cl_uint current_subwindow = 0;
    
    for(int y_index = start_point->y; y_index < end_point->y; y_index++) {
        for(int x_index = start_point->x; x_index < end_point->x; x_index++) {
            // Real position
            CvPoint point = cvPoint((cl_uint)lrint(x_index * step), (cl_uint)lrint(y_index * step));
            
            cl_float variance = computeVariance(integral_image, square_integral_image, equ_rect, &point, scaled_window_area);
            
            subwindow_data[current_subwindow].x = point.x;
            subwindow_data[current_subwindow].y = point.y;
            subwindow_data[current_subwindow].variance = variance;
            subwindow_data[current_subwindow].offset = mato(integral_image->width, point.x, point.y);
            
            current_subwindow++;
        }
    }
    *subwindow_count = current_subwindow;
}

inline void
runClassifier(const CvMat* integral_image,
              const CvHaarClassifier* classifier,
              const CvPoint* point,
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
    CLODWeightedRect final_rect[3];
    
    // Normalize rect size
    for(cl_uint ri = 0; ri < 3; ri++) {
        if(feature.rect[ri].weight != 0) {
            register CvRect* temp_rect = &feature.rect[ri].r;
            register CLODWeightedRect* temp_final_rect = &final_rect[ri];
            temp_final_rect->rect.x = (cl_uint)round(temp_rect->x * current_scale);
            temp_final_rect->rect.y = (cl_uint)round(temp_rect->y * current_scale);
            temp_final_rect->rect.width = (cl_uint)round(temp_rect->width * current_scale);
            temp_final_rect->rect.height = (cl_uint)round(temp_rect->height * current_scale);
            // Normalize rect weight based on window area
            temp_final_rect->weight = (float)(feature.rect[ri].weight) / (float)scaled_window_area;
            if(ri == 0)
                first_rect_area = temp_final_rect->rect.width * temp_final_rect->rect.height;
            else
                sum_rect_area += temp_final_rect->weight * temp_final_rect->rect.width * temp_final_rect->rect.height;
        }
    }
    final_rect[0].weight = (float)(-sum_rect_area/first_rect_area);
    
    // Calculation on rectangles (loop unroll)
    for(cl_uint ri = 0; ri < 3; ri++) {
        if(feature.rect[ri].weight != 0) {
            rect_sum += (cl_float)((mats((cl_uint*)integral_image->data.i,
                                      integral_image->width,
                                      point->x + final_rect[ri].rect.x,
                                      point->y + final_rect[ri].rect.y,
                                      final_rect[ri].rect.width,
                                      final_rect[ri].rect.height) * final_rect[ri].weight));
        }
    }
    // If rect sum less than stage_sum updated with threshold left_val else right_val
    *stage_sum += classifier->alpha[rect_sum >= norm_threshold];
}

inline void
runClassifierWithPrecomputedFeatures(const CvHaarClassifier* classifier,
                                     const CLODOptimizedRect* opt_rectangles,
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

inline cl_int
runCascade(const CvMat* integral_image,
           const CvHaarClassifierCascade* cascade,
           CLODOptimizedRect* opt_rectangles,
           const CvPoint* point,
           const CvSize* scaled_window_size,
           const cl_uint scaled_window_area,
           const cl_float variance,
           const cl_float current_scale,
           const cl_bool precompute_features,
           CLODWeightedRect* matches,
           cl_uint* match_count)
{
    cl_uint offset = mato(integral_image->width, point->x, point->y);
    
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
                runClassifierWithPrecomputedFeatures(&classifier, opt_rectangles, &opt_rect_index, offset, variance, &stage_sum);
            else
                runClassifier(integral_image, &classifier, point, variance, current_scale, scaled_window_area, &stage_sum);
            
        }
        // If stage sum less than threshold exit and continue with next window
        if(stage_sum < stage.threshold) {
            exit_stage = -stage_index;
            break;
        }
    }
    
    // If exit at first stage increment by 2, else by 1
    if(exit_stage > 0) {
        CLODWeightedRect* r = &matches[*match_count];
        r->rect.x = point->x;
        r->rect.y = point->y;
        r->rect.width = scaled_window_size->width;
        r->rect.height = scaled_window_size->height;
        r->weight = 0;
        (*match_count)++;
    }
    
    return exit_stage;
}

inline void
runSubwindow(const CvMat* integral_image,
             const CLODOptimizedRect* opt_rectangles,
             const cl_uint start_rect_index,
             cl_uint* end_rect_index,
             const CvHaarStageClassifier* stage,
             const cl_uint stage_index,
             const CLODSubwindowData* win_src,
             CLODSubwindowData** p_win_dst,
             const cl_uint win_src_count,
             cl_uint* win_dst_count,
             const cl_uint scaled_window_area,
             const cl_float current_scale,
             const cl_bool precompute_features)
{
    *p_win_dst = (CLODSubwindowData*)malloc(win_src_count * sizeof(CLODSubwindowData));
    CLODSubwindowData* win_dst = *p_win_dst;
    *win_dst_count = 0;
    
    // Parallelize this
    cl_uint subwindow_incr = 1;
    for(cl_uint subwindow_index = 0; subwindow_index < win_src_count; subwindow_index += subwindow_incr) {
        CLODSubwindowData subwindow = win_src[subwindow_index];
        CvPoint point = cvPoint(subwindow.x, subwindow.y);
        
        // Iterate over classifiers
        float stage_sum = 0;
        
        cl_uint opt_rect_index = start_rect_index;
        for(cl_uint classifier_index = 0; classifier_index < stage->count; classifier_index++) {
            CvHaarClassifier classifier = stage->classifier[classifier_index];
            if(precompute_features)
                runClassifierWithPrecomputedFeatures(&classifier, opt_rectangles, &opt_rect_index, subwindow.offset, subwindow.variance, &stage_sum);
            else
                runClassifier(integral_image, &classifier, &point, subwindow.variance, current_scale, scaled_window_area, &stage_sum);
        }
        *end_rect_index = opt_rect_index;
        
        subwindow_incr = 1;
        
        // Add subwindow to accepted list
        if(stage_sum >= stage->threshold) {
            win_dst[*win_dst_count].x = subwindow.x;
            win_dst[*win_dst_count].y = subwindow.y;
            win_dst[*win_dst_count].variance = subwindow.variance;
            win_dst[*win_dst_count].offset = subwindow.offset;
            (*win_dst_count)++;
        }
        else {
            if(subwindow.x == 16 && subwindow.y == 0) {
                printf("");
            }
            if(stage_index == 0)
                subwindow_incr = 2;
        }
        // Performance improvement
        // Note that we skip a window (step = 2 instead of 1) if a stage fails, not if the cascade fails (linke in non-per-stage methods)
    }
}

CLODDetectObjectsResult
clodDetectObjects(const IplImage* image,
                  const CvHaarClassifierCascade* cascade,
                  const CLIFEnvironmentData* data,
                  const CvSize min_window_size,
                  const CvSize max_window_size,
                  const cl_uint min_neighbors,
                  const clod_flags flags)
{
    
    float scale_factor = 1.1;
    CLODDetectObjectsResult result;
    
    CvSize image_size = cvSize(image->width, image->height);
    
    // Setup image
    CvMat* integral_image, *square_integral_image;
    setupImage(image, &integral_image, &square_integral_image);
    
    // Calculate number of different scales
    cl_uint scale_count = 0;
    for(float current_scale = 1;
        current_scale * cascade->orig_window_size.width < image->width - 10 &&
        current_scale * cascade->orig_window_size.height < image->height - 10;
        current_scale *= scale_factor) {
        scale_count++;
    }
    
    // Precompute feature rect offset in integral image and square integral image into a new cascade
    CLODOptimizedRect* opt_rectangles = NULL;
    if(flags & CLOD_PRECOMPUTE_FEATURES)
        opt_rectangles = (CLODOptimizedRect*)malloc(cascade->count * MAX_CLASSIFIER_FEATURE_COUNT * MAX_FEATURE_RECT_COUNT * sizeof(CLODOptimizedRect));
    
    // Vector to store positive matches
    CLODWeightedRect* matches = (CLODWeightedRect*)malloc(image->width * image->height * scale_count * sizeof(CLODWeightedRect));
    cl_uint match_count = 0;
    
    // Iterate over scales
    cl_float current_scale = 1;
    for(cl_uint scale_index = 0; scale_index < scale_count; scale_index++, current_scale *= scale_factor) {
        
        // Setup scale-dependent variables
        CvSize scaled_window_size;
        cl_uint scaled_window_area;
        CvRect equ_rect;
        CvPoint start_point = cvPoint(0, 0);
        CvPoint end_point;
        cl_float step;
        if(setupScale(current_scale,
                      &image_size,
                      &cascade->orig_window_size,
                      &min_window_size,
                      &max_window_size,
                      &equ_rect,
                      &scaled_window_size,
                      &scaled_window_area,
                      &end_point, &step) != CL_SUCCESS) {
            continue;
        }
        
        // Precompute feature rect offset in integral image and square integral image into a new cascade
        if(flags & CLOD_PRECOMPUTE_FEATURES)
            precomputeFeatures(integral_image, scaled_window_area, cascade, current_scale, opt_rectangles);
        
        if(!(flags & CLOD_PER_STAGE_ITERATIONS)) {
            // Iterate over windows
            cl_uint x_incr = 1;
            for(int y_index = start_point.y; y_index < end_point.y; y_index++) {
                for(int x_index = start_point.x; x_index < end_point.x; x_index += x_incr) {
                    // Real position
                    CvPoint point = cvPoint((cl_uint)round(x_index * step), (cl_uint)round(y_index * step));
                    
                    // Sum of window pixels normalized by the window size E(x)
                    cl_float variance = computeVariance(integral_image, square_integral_image, &equ_rect, &point, scaled_window_area);
                    
                    // Run cascade on point x,y
                    cl_int exit_stage = runCascade(integral_image,
                                              cascade,
                                              opt_rectangles,
                                              &point,
                                              &scaled_window_size,
                                              scaled_window_area,
                                              variance, current_scale, (flags & CLOD_PRECOMPUTE_FEATURES),
                                              matches, &match_count);
                    x_incr = exit_stage != 0 ? 1 : 2;
                }
            }
        }
        else {
            // Allocate windows to be computed by successive stages
            CLODSubwindowData* input_windows = (CLODSubwindowData*)malloc((end_point.y - start_point.y + 1) * (end_point.x - start_point.x + 1) * sizeof(CLODSubwindowData));
            CLODSubwindowData* output_windows = NULL;
            cl_uint input_window_count = 0;
            cl_uint output_window_count = 0;
            
            // Precompute windows
            precomputeWindows(step, integral_image, square_integral_image,
                              &equ_rect,
                              &start_point,
                              &end_point,
                              scaled_window_area, &input_windows, &input_window_count);
            
            // Iterate over stages
            cl_uint start_rect_index = 0;
            cl_uint end_rect_index = 0;
            for(cl_uint stage_index = 0; stage_index < cascade->count; stage_index++)
            {
                CvHaarStageClassifier stage = cascade->stage_classifier[stage_index];
                // Run stage on GPU for each subwindow
                runSubwindow(integral_image,
                             opt_rectangles,
                             start_rect_index, &end_rect_index,
                             &stage, stage_index,
                             input_windows, &output_windows,
                             input_window_count, &output_window_count,
                             scaled_window_area, current_scale, (flags & CLOD_PRECOMPUTE_FEATURES));
                
                free(input_windows);
                input_windows = output_windows;
                input_window_count = output_window_count;
                start_rect_index = end_rect_index;
                if(output_window_count == 0)
                    break;
            }
            
            // Add to matches
            for(cl_uint i = 0; i < output_window_count; i++) {
                matches[match_count].rect.x = output_windows[i].x;
                matches[match_count].rect.y = output_windows[i].y;
                matches[match_count].rect.width = scaled_window_size.width;
                matches[match_count].rect.height = scaled_window_size.height;
                match_count++;
            }
            free(output_windows);
        }
    }
    
    // Filter out results
    if(min_neighbors != 0) 
        match_count = filterResult(matches, match_count, MAX(min_neighbors, 1), EPS);
    
    // Release
    if((flags & CLOD_PER_STAGE_ITERATIONS))
        free(opt_rectangles);
    cvReleaseMat(&integral_image);
    cvReleaseMat(&square_integral_image);
    
    // Return
    result.matches = matches;
    result.match_count = match_count;
    return result;
}


/* Code obtained unfolding function calls. Seems to be more efficient */
CLODDetectObjectsResult
clodDetectObjectsBlock(const IplImage* image,
                       const CvHaarClassifierCascade* cascade,
                       const CLIFEnvironmentData* data,
                       const CvSize min_window_size,
                       const CvSize max_window_size,
                       const cl_uint min_neighbors,
                       const clod_flags flags)
{
    CLODDetectObjectsResult result;
    float scale_factor = 1.1;
    
    // Setup image
    CvMat* sum, *square_sum;
    setupImage(image, &sum, &square_sum);
    cl_uint* integral_image = (cl_uint*)sum->data.ptr;
    cl_double* square_integral_image = (cl_double*)square_sum->data.ptr;
    cl_uint integral_image_width = image->width + 1;
    
    // Calculate number of different scales
    cl_uint scale_count = 0;
    for(float current_scale = 1;
        current_scale * cascade->orig_window_size.width < image->width - 10 &&
        current_scale * cascade->orig_window_size.height < image->height - 10;
        current_scale *= scale_factor) {
        scale_count++;
    }
    
    // Precompute feature rect offset in integral image and square integral image into a new cascade
    CLODOptimizedRect* opt_rectangles = NULL;
    if(flags & CLOD_PRECOMPUTE_FEATURES)
        opt_rectangles = (CLODOptimizedRect*)malloc(cascade->count * MAX_CLASSIFIER_FEATURE_COUNT * MAX_FEATURE_RECT_COUNT * sizeof(CLODOptimizedRect));
    
    // Vector to store positive matches
    CLODWeightedRect* matches = (CLODWeightedRect*)malloc(image->width * image->height * scale_count * sizeof(CLODWeightedRect));
    cl_uint match_count = 0;
    
    // Iterate over scales
    cl_float current_scale = 1;
    for(cl_uint scale_index = 0; scale_index < scale_count; scale_index++, current_scale *= scale_factor) {
        // Compute window y shift
        const double step = MAX((double)2.0, (double)current_scale);
        
        // Compute scaled window size
        cl_uint scaled_window_width = (cl_uint)round(cascade->orig_window_size.width * current_scale);
        cl_uint scaled_window_height = (cl_uint)round(cascade->orig_window_size.height * current_scale);
        
        // If the window is smaller than the minimum size continue
        if(scaled_window_width < min_window_size.width || scaled_window_height < min_window_size.height)
            continue;
        // If the window is bigger than the maximum size continue
        if((max_window_size.width != 0) && (scaled_window_width > max_window_size.width))
            continue;
        if((max_window_size.height != 0) && (scaled_window_height > max_window_size.height))
            continue;
        
        // If the window is bigger than the image exit
        if(scaled_window_width > image->width || scaled_window_height > image->height)
            break;
        
        // Compute scaled window area (using equalized rect, not fully understood)
        cl_uint equ_rect_x = (cl_uint)round(current_scale);
        cl_uint equ_rect_y = equ_rect_x;
        cl_uint equ_rect_width = (cl_uint)round((cascade->orig_window_size.width - 2) * current_scale);
        cl_uint equ_rect_height = (cl_uint)round((cascade->orig_window_size.height - 2) * current_scale);
        cl_uint scaled_window_area = equ_rect_width * equ_rect_height;
        
        // Set init and end positions of subwindows
        int start_x = 0, start_y = 0;
        int end_x = (int)lrint((image->width - scaled_window_width) / step);
        int end_y = (int)lrint((image->height - scaled_window_height) / step);
                
        // Precompute feature rect offset in integral image and square integral image into a new cascade
        if(flags & CLOD_PRECOMPUTE_FEATURES) {
            cl_uint opt_rect_index = 0;
            for(cl_uint stage_index = 0; stage_index < cascade->count; stage_index++) {
                for(cl_uint classifier_index = 0; classifier_index < cascade->stage_classifier[stage_index].count; classifier_index++) {
                    // Optimized for stump based classifier (otherwise loop over features)
                    CvHaarFeature feature = cascade->stage_classifier[stage_index].classifier[classifier_index].haar_feature[0];
                    
                    // Normalize rect weight based on window area
                    cl_float first_rect_area;
                    cl_uint first_rect_index = opt_rect_index;
                    cl_float sum_rect_area = 0;
                    for(cl_uint i = 0; i < MAX_FEATURE_RECT_COUNT; i++) {
                        opt_rectangles[opt_rect_index].weight = 0;
                        if(feature.rect[i].weight != 0) {
                            register CvRect* temp_rect = &feature.rect[i].r;
                            register CLODOptimizedRect* opt_rect = &opt_rectangles[opt_rect_index];
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
        // Precompute end
        
        // Iterate over windows
        if(!(flags & CLOD_PER_STAGE_ITERATIONS)) {
            cl_uint x_incr = 1;
            for(int y_index = start_y; y_index < end_y; y_index++) {
                for(int x_index = start_x; x_index < end_x; x_index += x_incr) {
                    // Real position
                    cl_uint x = (cl_uint)lrint(x_index * step);
                    cl_uint y = (cl_uint)lrint(y_index * step);
                    // Sum of window pixels normalized by the window size E(x)
                    float mean = (float)mats(integral_image, integral_image_width, x + equ_rect_x, y + equ_rect_y, equ_rect_width, equ_rect_height) / (float)scaled_window_area;
                    // E(xˆ2) - Eˆ2(x)
                    float variance = (float)mats(square_integral_image, integral_image_width, x + equ_rect_x, y + equ_rect_y, equ_rect_width, equ_rect_height);
                    variance = (variance / (float)scaled_window_area) - (mean * mean);
                    // Fix wrong variance
                    if(variance >= 0)
                        variance = sqrt(variance);
                    else
                        variance = 1;

                    // Run cascade on point x,y
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
                            
                            // Compute threshold normalized by window vaiance
                            float norm_threshold = *classifier.threshold * variance;
                            
                            // Iterate over features (optimized for stump)
                            //for(cl_uint feature_index = 0; feature_index < classifier.count; feature_index++) {
                            CvHaarFeature feature = classifier.haar_feature[0];
                            
                            // Calculation on rectangles (loop unroll)
                            cl_float rect_sum =
                            (matsp(opt_rectangles[opt_rect_index].sum_left_top + offset,
                                   opt_rectangles[opt_rect_index].sum_right_top + offset,
                                   opt_rectangles[opt_rect_index].sum_left_bottom + offset,
                                   opt_rectangles[opt_rect_index].sum_right_bottom + offset) * opt_rectangles[opt_rect_index].weight);
                            opt_rect_index++;
                            rect_sum +=
                            (matsp(opt_rectangles[opt_rect_index].sum_left_top + offset,
                                   opt_rectangles[opt_rect_index].sum_right_top + offset,
                                   opt_rectangles[opt_rect_index].sum_left_bottom + offset,
                                   opt_rectangles[opt_rect_index].sum_right_bottom + offset) * opt_rectangles[opt_rect_index].weight);
                            opt_rect_index++;
                            if(feature.rect[2].weight != 0) {
                                rect_sum +=
                                (matsp(opt_rectangles[opt_rect_index].sum_left_top + offset,
                                       opt_rectangles[opt_rect_index].sum_right_top + offset,
                                       opt_rectangles[opt_rect_index].sum_left_bottom + offset,
                                       opt_rectangles[opt_rect_index].sum_right_bottom + offset) * opt_rectangles[opt_rect_index].weight);
                                opt_rect_index++;
                            }
                            
                            // If rect sum less than stage_sum updated with threshold left_val else right_val
                            stage_sum += classifier.alpha[rect_sum >= norm_threshold];
                        }
                        // If stage sum less than threshold exit and continue with next window
                        if(stage_sum < stage.threshold) {
                            exit_stage = -stage_index;
                            break;
                        }
                    }
                    
                    // If exit at first stage increment by 2, else by 1
                    x_incr = exit_stage != 0 ? 1 : 2;
                    
                    if(exit_stage > 0) { 
                        CLODWeightedRect* r = &matches[match_count];
                        r->rect.x = x;
                        r->rect.y = y;
                        r->rect.width = scaled_window_width;
                        r->rect.height = scaled_window_height;
                        r->weight = 0;
                        match_count++;
                        x_incr = 1;
                    }
                }
            }
        }
        else {
            // Allocate windows to be computed by successive stages
            CLODSubwindowData* input_windows = (CLODSubwindowData*)malloc((end_y - start_y + 1) * (end_x - start_x + 1) * sizeof(CLODSubwindowData));
            CLODSubwindowData* output_windows = NULL;
            
            // Precompute windows
            cl_uint current_subwindow = 0;
            for(int y_index = start_y; y_index < end_y; y_index++) {
                for(int x_index = start_x; x_index < end_x; x_index++) {
                    // Real position
                    CvPoint point = cvPoint((cl_uint)round(x_index * step), (cl_uint)round(y_index * step));
                    
                    // Sum of window pixels normalized by the window size E(x)
                    float mean = (float)mats(integral_image, integral_image_width,
                                             point.x + equ_rect_x,
                                             point.y + equ_rect_y,
                                             equ_rect_width,
                                             equ_rect_height) / (float)scaled_window_area;
                    // E(xˆ2) - Eˆ2(x)
                    float variance = (float)mats(square_integral_image, integral_image_width,
                                                 point.x + equ_rect_x,
                                                 point.y + equ_rect_y,
                                                 equ_rect_width,
                                                 equ_rect_height);
                    variance = (variance / (float)scaled_window_area) - (mean * mean);
                    // Fix wrong variance
                    if(variance >= 0)
                        variance = sqrt(variance);
                    else
                        variance = 1;
                    
                    input_windows[current_subwindow].x = point.x;
                    input_windows[current_subwindow].y = point.y;
                    input_windows[current_subwindow].variance = variance;
                    input_windows[current_subwindow].offset = mato(integral_image_width, point.x, point.y);
                    
                    current_subwindow++;
                }
            }
            cl_uint input_window_count = current_subwindow;
            cl_uint output_window_count = 0;
            
            // Iterate over stages
            cl_uint start_rect_index = 0;
            cl_uint end_rect_index = 0;
            for(cl_uint stage_index = 0; stage_index < cascade->count; stage_index++)
            {
                output_windows = (CLODSubwindowData*)malloc(input_window_count * sizeof(CLODSubwindowData));
                output_window_count = 0;
                
                CvHaarStageClassifier stage = cascade->stage_classifier[stage_index];
                // Run stage on GPU for each subwindow
                cl_uint subwindow_incr = 1;
                for(cl_uint subwindow_index = 0; subwindow_index < input_window_count; subwindow_index += subwindow_incr) {
                    CLODSubwindowData subwindow = input_windows[subwindow_index];
                    
                    // Iterate over classifiers
                    float stage_sum = 0;
                    
                    cl_uint opt_rect_index = start_rect_index;
                    for(cl_uint classifier_index = 0; classifier_index < stage.count; classifier_index++) {
                        CvHaarClassifier classifier = stage.classifier[classifier_index];
                        
                        // Compute threshold normalized by window vaiance
                        float norm_threshold = *classifier.threshold * subwindow.variance;
                        
                        // Iterate over features (optimized for stump)
                        //for(cl_uint feature_index = 0; feature_index < classifier.count; feature_index++) {
                        CvHaarFeature feature = classifier.haar_feature[0];
                        
                        // Calculation on rectangles (loop unroll)
                        cl_float rect_sum =
                        (matsp(opt_rectangles[opt_rect_index].sum_left_top + subwindow.offset,
                               opt_rectangles[opt_rect_index].sum_right_top + subwindow.offset,
                               opt_rectangles[opt_rect_index].sum_left_bottom + subwindow.offset,
                               opt_rectangles[opt_rect_index].sum_right_bottom + subwindow.offset) * opt_rectangles[opt_rect_index].weight);
                        opt_rect_index++;
                        rect_sum +=
                        (matsp(opt_rectangles[opt_rect_index].sum_left_top + subwindow.offset,
                               opt_rectangles[opt_rect_index].sum_right_top + subwindow.offset,
                               opt_rectangles[opt_rect_index].sum_left_bottom + subwindow.offset,
                               opt_rectangles[opt_rect_index].sum_right_bottom + subwindow.offset) * opt_rectangles[opt_rect_index].weight);
                        opt_rect_index++;
                        if(feature.rect[2].weight != 0) {
                            rect_sum +=
                            (matsp(opt_rectangles[opt_rect_index].sum_left_top + subwindow.offset,
                                   opt_rectangles[opt_rect_index].sum_right_top + subwindow.offset,
                                   opt_rectangles[opt_rect_index].sum_left_bottom + subwindow.offset,
                                   opt_rectangles[opt_rect_index].sum_right_bottom + subwindow.offset) * opt_rectangles[opt_rect_index].weight);
                            opt_rect_index++;
                        }
                        
                        // If rect sum less than stage_sum updated with threshold left_val else right_val
                        stage_sum += classifier.alpha[rect_sum >= norm_threshold];
                    }
                    end_rect_index = opt_rect_index;
                    
                    subwindow_incr = 1;
                    
                    // Add subwindow to accepted list
                    if(stage_sum >= stage.threshold) {
                        output_windows[output_window_count].x = subwindow.x;
                        output_windows[output_window_count].y = subwindow.y;
                        output_windows[output_window_count].variance = subwindow.variance;
                        output_windows[output_window_count].offset = subwindow.offset;
                        output_window_count++;
                    }
                    else {
                        if(stage_index == 0)
                            subwindow_incr = 2;
                    }
                    // Performance improvement
                    // Note that we skip a window (step = 2 instead of 1) if a stage fails, not if the cascade fails (linke in non-per-stage methods)
                }

                free(input_windows);
                input_windows = output_windows;
                input_window_count = output_window_count;
                start_rect_index = end_rect_index;
                if(output_window_count == 0)
                    break;
            }
            
            // Add to matches
            for(cl_uint i = 0; i < output_window_count; i++) {
                matches[match_count].rect.x = output_windows[i].x;
                matches[match_count].rect.y = output_windows[i].y;
                matches[match_count].rect.width = scaled_window_width;
                matches[match_count].rect.height = scaled_window_height;
                match_count++;
            }
            free(output_windows);            
        }
    }
    
    // Filter out results
    if(min_neighbors != 0)
        match_count = filterResult(matches, match_count, MAX(min_neighbors, 1), EPS);
    
    // Release
    if(flags & CLOD_PRECOMPUTE_FEATURES) 
        free(opt_rectangles);
    cvReleaseMat(&sum);
    cvReleaseMat(&square_sum);
    
    // Return
    result.matches = matches;
    result.match_count = match_count;
    return result;
}

//
//  main.c
//  OpenCLFaceDetection
//
//  Created by Gabriele Cocco on 8/21/12.
//  Copyright (c) 2012 Gabriele Cocco. All rights reserved.
//

#include "clod.h"
char file_xml[] = "/Users/Gabriele/Desktop/OpenCLFaceDetection/OpenCLFaceDetection/haarcascade_frontalface_default.xml";

char win_face[] = "FaceDetect";
static CvMemStorage* storage = 0;
static CvHaarClassifierCascade* cascade = 0;

void find_faces_rect_opencv(IplImage* img, CvSize min_window_size, CvSize max_window_size);
void find_faces_rect_opencl(IplImage* img, CLODEnvironmentData* data, CvSize min_window_size, CvSize max_window_size, clod_flags, cl_bool);

int main( int argc, char** argv )
{
	int wcam = -1;
	CvCapture* capture = 0;
	IplImage *frame = 0;
	IplImage *frame_resized = 0;
	IplImage *frame_resized2 = 0;
	
    CvSize min_window_size, max_window_size;
    min_window_size.width = 40;
    min_window_size.height = 40;
    max_window_size.width = 0;
    max_window_size.height = 0;
    
	cvNamedWindow(win_face, 1);
    
	// Carico il file con le informazioni su cosa trovare
	cascade = (CvHaarClassifierCascade*)cvLoad(file_xml, 0, 0, 0);
    
	// Alloco la memoria per elaborare i dati
	storage = cvCreateMemStorage(0);
	
	/*if(!(capture = cvCaptureFromCAM(wcam)))
	{
		printf("Impossibile aprire la webcam.\n");
		return -1;
	}*/
    
    CvSize window_size = cvSize(640, 480);
    frame = cvLoadImage("/Users/Gabriele/Desktop/jobs.jpeg");
    frame_resized = cvCreateImage(window_size, frame->depth, 3);
    frame_resized2 = cvCreateImage(window_size, frame->depth, 3);
    cvResize(frame, frame_resized);
    
    CLODEnvironmentData* data = clodInitEnvironment(0);
    clodInitBuffers(data, &window_size);
    
    ElapseTime t;;
    
    cvCopyImage(frame_resized, frame_resized2);
    t.start();
    find_faces_rect_opencv(frame_resized2, min_window_size, max_window_size);
    printf("OpenCV: %8.4f ms\n", t.get());
    cvShowImage("Sample OpenCV", frame_resized2);
        
    cvCopyImage(frame_resized, frame_resized2);
    t.start();
    find_faces_rect_opencl(frame_resized2, data, min_window_size, max_window_size, CLOD_PER_STAGE_ITERATIONS | CLOD_PRECOMPUTE_FEATURES, CL_FALSE);
    printf("OpenCL (optimized): %8.4f ms\n", t.get());
    cvShowImage("Sample OpenCL (optimized)", frame_resized2);
    cvCopyImage(frame_resized, frame_resized2);
    t.start();
    find_faces_rect_opencl(frame_resized2, data, min_window_size, max_window_size, CLOD_PRECOMPUTE_FEATURES, CL_TRUE);
    printf("                    %8.4f ms (block)\n", t.get());
    cvShowImage("Sample OpenCL (optimized, block)", frame_resized2);
    
    cvCopyImage(frame_resized, frame_resized2);
    t.start();
    find_faces_rect_opencl(frame_resized2, data, min_window_size, max_window_size, CLOD_PRECOMPUTE_FEATURES | CLOD_PER_STAGE_ITERATIONS, CL_FALSE);
    printf("OpenCL (per-stage): %8.4f ms\n", t.get());
    cvShowImage("Sample OpenCL (per-stage)", frame_resized2);
    cvCopyImage(frame_resized, frame_resized2);
    t.start();
    find_faces_rect_opencl(frame_resized2, data, min_window_size, max_window_size, CLOD_PRECOMPUTE_FEATURES | CLOD_PER_STAGE_ITERATIONS, CL_TRUE);
    printf("                    %8.4f ms (block)\n", t.get());
    cvShowImage("Sample OpenCL (per-stage, block)", frame_resized2);
    
    //frame_resized->imageData =
    //printf("OpenCL (per-stage, optimized): %8.4f ms\n", t.get());
    //cvShowImage("Sample OpenCL (per-stage, optimized)", frame2);
    
    //detect_faces(frame, &data);
    /*
	while(1)
	{
		// Cerco le facce
        
        
        char fps[1024] = { 0 };
        sprintf(fps, "FPS: %4.2f", (1000.0) / (double)(end - start));
        CvFont font;
        cvInitFont(&font, CV_FONT_HERSHEY_PLAIN, 1.0f, 1.0f);
        CvPoint point;
        point.x = 10;
        point.y = frame->height - 10;
        CvScalar scalar = { 255,255,255,1 };
        cvPutText(frame, fps, point, &font, scalar);
		cvShowImage(win_face, frame);
		
		frame = cvQueryFrame(capture);
        
		if( (cvWaitKey(10) & 255) == 27 ) break;
	}
    */
    clodReleaseBuffers(data);
    clodReleaseEnvironment(data);
    free(data);
    
	cvReleaseImage(&frame);
	cvReleaseImage(&frame_resized);
	cvReleaseCapture(&capture);
	cvDestroyWindow(win_face);
    
	return 0;
}

void find_faces_rect_opencv(IplImage* img, CvSize min_window_size, CvSize max_window_size)
{
	CvPoint pt1, pt2;
	int i;
    
	// Libero la memoria
	cvClearMemStorage(storage);	// Ci potrebbero essere più oggetti. Quindi li salvo in sequenza
	CvSeq* faces = cvHaarDetectObjects(img, cascade, storage, 1.1, 0, 0, cvSize(min_window_size.width, min_window_size.height));

    // Disegno un rettangolo per ogni oggetto trovato
	for(i=0; i<(faces ? faces->total : 0); i++)
	{
		CvRect* r = (CvRect*)cvGetSeqElem( faces, i );
		pt1.x = r->x;
		pt2.x = (r->x+r->width);
		pt1.y = r->y;
		pt2.y = (r->y+r->height);        
		cvRectangle(img, pt1, pt2, CV_RGB(255,0,0), 3, 8, 0 );
	}
}

void find_faces_rect_opencl(IplImage* img, CLODEnvironmentData* data, CvSize min_window_size, CvSize max_window_size, clod_flags flags, cl_bool block)
{
	CvPoint pt1, pt2;
    CLODDetectObjectsResult result;
    
    if(block) {
        result = clodDetectObjects(img, cascade, data, min_window_size, max_window_size, 0, flags | CLOD_BLOCK_IMPLEMENTATION, CL_TRUE);
    }
    else {
        result = clodDetectObjects(img, cascade, data, min_window_size, max_window_size, 0, flags, CL_TRUE);
    }
    
    // Disegno un rettangolo per ogni oggetto trovato
	for(cl_uint i = 0; i< result.match_count; i++)
	{
        CvRect r = result.matches[i].rect;

		pt1.x = r.x;
		pt2.x = r.x + r.width;
		pt1.y = r.y;
		pt2.y = r.y + r.height;
        
		cvRectangle(img, pt1, pt2, CV_RGB(255,0,0), 3, 8, 0 );
	}
    free(result.matches);
}




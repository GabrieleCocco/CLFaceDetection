//
//  main.c
//  OpenCLFaceDetection
//
//  Created by Gabriele Cocco on 8/21/12.
//  Copyright (c) 2012 Gabriele Cocco. All rights reserved.
//

#include "object_detection.h"
char file_xml[] = "/Users/Gabriele/Desktop/OpenCLFaceDetection/OpenCLFaceDetection/haarcascade_frontalface_default.xml";

char win_face[] = "FaceDetect";
static CvMemStorage* storage = 0;
static CvHaarClassifierCascade* cascade = 0;

void find_faces_rect_opencv(IplImage* img);
void find_faces_rect_opencl(IplImage* img, CLEnvironmentData* data, cl_bool, cl_bool);

int main( int argc, char** argv )
{
	int wcam = -1;
	CvCapture* capture = 0;
	IplImage *tframe = 0;
	IplImage *frame = 0;
	IplImage *frame2 = 0;
	
	cvNamedWindow(win_face, 1);
    
	// Carico il file con le informazioni su cosa trovare
	cascade = (CvHaarClassifierCascade*)cvLoad(file_xml, 0, 0, 0 );
    
	// Alloco la memoria per elaborare i dati
	storage = cvCreateMemStorage(0);
	
	if(!(capture = cvCaptureFromCAM(wcam)))
	{
		printf("Impossibile aprire la webcam.\n");
		return -1;
	}
    
    tframe = cvLoadImage("/Users/Gabriele/Desktop/jobs.jpeg");
    frame = cvCreateImage(cvSize(640, 480), tframe->depth, 3);
    frame2 = cvCreateImage(cvSize(640, 480), tframe->depth, 3);
    cvResize(tframe, frame);
    
    cvCopyImage(frame, frame2);
    
    CLEnvironmentData data = initCLEnvironment(frame->width, frame->height, frame->widthStep, 3, 0);
    
    ElapseTime t;
    t.start();
    find_faces_rect_opencv(frame2);
    printf("OpenCV: %8.4f ms\n", t.get());
    cvShowImage("Sample OpenCV", frame2);
    
    cvCopyImage(frame, frame2);
    t.start();
    find_faces_rect_opencl(frame2, &data, CL_TRUE, CL_TRUE);
    printf("OpenCL (per-stage, optimized): %8.4f ms\n", t.get());
    cvShowImage("Sample OpenCL (per-stage, optimized)", frame2); 
    
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
    releaseCLEnvironment(data);
    
	cvReleaseImage(&tframe);
	cvReleaseImage(&frame);
	cvReleaseImage(&frame2);
	cvReleaseCapture(&capture);
	cvDestroyWindow(win_face);
    
	return 0;
}

void find_faces_rect_opencv ( IplImage* img )
{
	CvPoint pt1, pt2;
	int i;
    
	// Libero la memoria
	cvClearMemStorage(storage);	// Ci potrebbero essere più oggetti. Quindi li salvo in sequenza
	CvSeq* faces = cvHaarDetectObjects(img, cascade, storage, 1.1, 0, 0, cvSize(40, 40));
    
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

void find_faces_rect_opencl(IplImage* img, CLEnvironmentData* data, cl_bool precompute_rect, cl_bool per_stage_iteration)
{
	CvPoint pt1, pt2;
    
    cl_uint match_count;
    CLWeightedRect* faces = detectObjects(img, cascade, data, 40, 40, 0, 0, 0, &match_count, precompute_rect, per_stage_iteration);
    
    // Disegno un rettangolo per ogni oggetto trovato
	for(cl_uint i = 0; i< match_count; i++)
	{
        CLWeightedRect r = faces[i];

		pt1.x = r.x;
		pt2.x = r.x + r.width;
		pt1.y = r.y;
		pt2.y = r.y + r.height;
        
		cvRectangle(img, pt1, pt2, CV_RGB(255,0,0), 3, 8, 0 );
	}
    free(faces);
}




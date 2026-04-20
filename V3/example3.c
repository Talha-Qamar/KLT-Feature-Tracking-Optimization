/**********************************************************************
Finds the 150 best features in an image and tracks them through the 
next two images.  The sequential mode is set in order to speed
processing.  The features are stored in a feature table, which is then
saved to a text file; each feature list is also written to a PPM file.
**********************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include "pnmio.h"
#include "klt.h"
#include "gpu_kernels.h"
#include "klt_timing.h"

/* #define REPLACE */

#ifdef WIN32
int RunExample3()
#else
int main()
#endif
{
  unsigned char *img1, *img2;
  char fnamein[100], fnameout[100];
  KLT_TrackingContext tc;
  KLT_FeatureList fl;
  KLT_FeatureTable ft;
  KLT_TimingContext timing;
  int nFeatures = 150, nFrames = 10;
  int ncols, nrows;
  int i;

  timing = KLT_CreateTimingContext();
  KLT_StartTimer(timing, KLT_TIMER_TOTAL);

  tc = KLTCreateTrackingContext();
  fl = KLTCreateFeatureList(nFeatures);
  ft = KLTCreateFeatureTable(nFrames, nFeatures);
  tc->sequentialMode = TRUE;
  tc->writeInternalImages = FALSE;
  tc->affineConsistencyCheck = -1;  /* set this to 2 to turn on affine consistency check */
 
  KLT_StartTimer(timing, KLT_TIMER_IO);
  img1 = pgmReadFile("../images/img0.pgm", NULL, &ncols, &nrows);
  img2 = (unsigned char *) malloc(ncols*nrows*sizeof(unsigned char));
  KLT_StopTimer(timing, KLT_TIMER_IO);

  KLT_StartTimer(timing, KLT_TIMER_SELECT);
  KLTSelectGoodFeatures(tc, img1, ncols, nrows, fl);
  KLT_StopTimer(timing, KLT_TIMER_SELECT);
  
  KLTStoreFeatureList(fl, ft, 0);
  KLTWriteFeatureListToPPM(fl, img1, ncols, nrows, "feat0.ppm");

  KLT_StartTimer(timing, KLT_TIMER_TRACK);
  for (i = 1 ; i < nFrames ; i++)  {
    sprintf(fnamein, "../images/img%d.pgm", i);
    
    KLT_StartTimer(timing, KLT_TIMER_IO);
    pgmReadFile(fnamein, img2, &ncols, &nrows);
    KLT_StopTimer(timing, KLT_TIMER_IO);
    
    KLTTrackFeatures(tc, img1, img2, ncols, nrows, fl);
#ifdef REPLACE
    KLTReplaceLostFeatures(tc, img2, ncols, nrows, fl);
#endif
    KLTStoreFeatureList(fl, ft, i);
    sprintf(fnameout, "feat%d.ppm", i);
    KLTWriteFeatureListToPPM(fl, img2, ncols, nrows, fnameout);
  }
  KLT_StopTimer(timing, KLT_TIMER_TRACK);

  // Capture GPU kernel timing (measured via CUDA events)
  float gpu_kernel_time = getTotalGPUTime();          // Total kernel execution
  float gpu_memory_time = getTotalMemoryTime();       // Total memory operations
  KLT_SetGPUTime(timing, gpu_kernel_time);
  KLT_SetMemoryTime(timing, gpu_memory_time);
  
  KLTWriteFeatureTable(ft, "features.txt", "%5.1f");
  KLTWriteFeatureTable(ft, "features.ft", NULL);

  KLT_StopTimer(timing, KLT_TIMER_TOTAL);
  
  printGPUKernelTimes();
  KLT_PrintTimingResults(timing, "V3 - Optimized GPU Implementation (Example 3)");
  KLT_SaveTimingToFile(timing, "V3-Ex3", "timing_results.csv");
  KLT_FreeTimingContext(timing);

  KLTFreeFeatureTable(ft);
  KLTFreeFeatureList(fl);
  KLTFreeTrackingContext(tc);
  free(img1);
  free(img2);

  return 0;
}


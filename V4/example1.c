/**********************************************************************
Finds the 100 best features in an image, and tracks these
features to the next image.  Saves the feature
locations (before and after tracking) to text files and to PPM files, 
and prints the features to the screen.
**********************************************************************/

#include "pnmio.h"
#include "klt.h"
#include "klt_timing.h"
#ifdef WIN32
int RunExample1()
#else
int main()
#endif
{
  unsigned char *img1, *img2;
  KLT_TrackingContext tc;
  KLT_FeatureList fl;
  KLT_TimingContext timing;
  int nFeatures = 100;
  int ncols, nrows;
  int i;

  // Create timing context
  timing = KLT_CreateTimingContext();
  KLT_StartTimer(timing, KLT_TIMER_TOTAL);

  tc = KLTCreateTrackingContext();
  KLTPrintTrackingContext(tc);
  fl = KLTCreateFeatureList(nFeatures);

  // Time I/O operations
  KLT_StartTimer(timing, KLT_TIMER_IO);
  img1 = pgmReadFile("../images/img0.pgm", NULL, &ncols, &nrows);
  img2 = pgmReadFile("../images/img1.pgm", NULL, &ncols, &nrows);
  KLT_StopTimer(timing, KLT_TIMER_IO);

  // Time feature selection
  KLT_StartTimer(timing, KLT_TIMER_SELECT);
  KLTSelectGoodFeatures(tc, img1, ncols, nrows, fl);
  KLT_StopTimer(timing, KLT_TIMER_SELECT);

  printf("\nIn first image:\n");
  for (i = 0 ; i < fl->nFeatures ; i++)  {
    printf("Feature #%d:  (%f,%f) with value of %d\n",
           i, fl->feature[i]->x, fl->feature[i]->y,
           fl->feature[i]->val);
  }

  KLTWriteFeatureListToPPM(fl, img1, ncols, nrows, "feat1.ppm");
  KLTWriteFeatureList(fl, "feat1.txt", "%3d");

  // Time feature tracking
  KLT_StartTimer(timing, KLT_TIMER_TRACK);
  KLTTrackFeatures(tc, img1, img2, ncols, nrows, fl);
  KLT_StopTimer(timing, KLT_TIMER_TRACK);

  printf("\nIn second image:\n");
  for (i = 0 ; i < fl->nFeatures ; i++)  {
    printf("Feature #%d:  (%f,%f) with value of %d\n",
           i, fl->feature[i]->x, fl->feature[i]->y,
           fl->feature[i]->val);
  }

  KLTWriteFeatureListToPPM(fl, img2, ncols, nrows, "feat2.ppm");
  KLTWriteFeatureList(fl, "feat2.fl", NULL);      /* binary file */
  KLTWriteFeatureList(fl, "feat2.txt", "%5.1f");  /* text file   */

  // Stop total timer and print results
  KLT_StopTimer(timing, KLT_TIMER_TOTAL);
  KLT_PrintTimingResults(timing, "V1 - CPU Implementation");
  
  // Save timing to CSV for analysis
  KLT_SaveTimingToFile(timing, "V1", "timing_results.csv");
  
  // Clean up
  KLT_FreeTimingContext(timing);

  return 0;
}


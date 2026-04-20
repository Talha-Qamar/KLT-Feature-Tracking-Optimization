/**********************************************************************
Finds the 100 best features in an image, tracks these
features to the next image, and replaces the lost features with new
features in the second image.  Saves the feature
locations (before and after tracking) to text files and to PPM files.
**********************************************************************/

#include "pnmio.h"
#include "klt.h"
#include "klt_timing.h"

#ifdef WIN32
int RunExample2()
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

  timing = KLT_CreateTimingContext();
  KLT_StartTimer(timing, KLT_TIMER_TOTAL);

  tc = KLTCreateTrackingContext();
  fl = KLTCreateFeatureList(nFeatures);

  KLT_StartTimer(timing, KLT_TIMER_IO);
  img1 = pgmReadFile("../images/img0.pgm", NULL, &ncols, &nrows);
  img2 = pgmReadFile("../images/img1.pgm", NULL, &ncols, &nrows);
  KLT_StopTimer(timing, KLT_TIMER_IO);

  KLT_StartTimer(timing, KLT_TIMER_SELECT);
  KLTSelectGoodFeatures(tc, img1, ncols, nrows, fl);
  KLT_StopTimer(timing, KLT_TIMER_SELECT);

  KLTWriteFeatureListToPPM(fl, img1, ncols, nrows, "feat1.ppm");
  KLTWriteFeatureList(fl, "feat1.txt", "%3d");

  KLT_StartTimer(timing, KLT_TIMER_TRACK);
  KLTTrackFeatures(tc, img1, img2, ncols, nrows, fl);
  KLTReplaceLostFeatures(tc, img2, ncols, nrows, fl);
  KLT_StopTimer(timing, KLT_TIMER_TRACK);

  KLTWriteFeatureListToPPM(fl, img2, ncols, nrows, "feat2.ppm");
  KLTWriteFeatureList(fl, "feat2.txt", "%3d");

  KLT_StopTimer(timing, KLT_TIMER_TOTAL);
  KLT_PrintTimingResults(timing, "V1 - CPU Implementation (Example 2)");
  KLT_SaveTimingToFile(timing, "V1-Ex2", "timing_results.csv");
  KLT_FreeTimingContext(timing);

  return 0;
}


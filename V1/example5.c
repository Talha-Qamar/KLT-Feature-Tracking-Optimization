/**********************************************************************
Demonstrates manually tweaking the tracking context parameters.
**********************************************************************/

#include "pnmio.h"
#include "klt.h"
#include "klt_timing.h"

#ifdef WIN32
int RunExample5()
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
  tc->mindist = 20;
  tc->window_width  = 9;
  tc->window_height = 9;
  KLTChangeTCPyramid(tc, 15);
  KLTUpdateTCBorder(tc);
  fl = KLTCreateFeatureList(nFeatures);

  KLT_StartTimer(timing, KLT_TIMER_IO);
  img1 = pgmReadFile("../images/img0.pgm", NULL, &ncols, &nrows);
  img2 = pgmReadFile("../images/img2.pgm", NULL, &ncols, &nrows);
  KLT_StopTimer(timing, KLT_TIMER_IO);

  KLT_StartTimer(timing, KLT_TIMER_SELECT);
  KLTSelectGoodFeatures(tc, img1, ncols, nrows, fl);
  KLT_StopTimer(timing, KLT_TIMER_SELECT);

  KLTWriteFeatureListToPPM(fl, img1, ncols, nrows, "feat1b.ppm");

  KLT_StartTimer(timing, KLT_TIMER_TRACK);
  KLTTrackFeatures(tc, img1, img2, ncols, nrows, fl);
  KLT_StopTimer(timing, KLT_TIMER_TRACK);

  KLTWriteFeatureListToPPM(fl, img2, ncols, nrows, "feat2b.ppm");

  KLT_StopTimer(timing, KLT_TIMER_TOTAL);
  KLT_PrintTimingResults(timing, "V1 - CPU Implementation (Example 5)");
  KLT_SaveTimingToFile(timing, "V1-Ex5", "timing_results.csv");
  KLT_FreeTimingContext(timing);

  return 0;
}


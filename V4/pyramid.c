/*********************************************************************
 * pyramid.c
 *
 *********************************************************************/

/* Standard includes */
#include <assert.h>
#include <stdlib.h>		/* malloc() ? */
#include <string.h>		/* memset() ? */
#include <math.h>		/* */

/* Our includes */
#include "base.h"
#include "error.h"
#include "convolve.h"	/* for computing pyramid */
#include "pyramid.h"


/*********************************************************************
 *
 */

_KLT_Pyramid _KLTCreatePyramid(
  int ncols,
  int nrows,
  int subsampling,
  int nlevels)
{
  _KLT_Pyramid pyramid;
  int nbytes = sizeof(_KLT_PyramidRec) +
    nlevels * sizeof(_KLT_FloatImage *) +
    nlevels * sizeof(int) +
    nlevels * sizeof(int);
  int i;

  if (subsampling != 2 && subsampling != 4 &&
      subsampling != 8 && subsampling != 16 && subsampling != 32)
    KLTError("(_KLTCreatePyramid)  Pyramid's subsampling must "
             "be either 2, 4, 8, 16, or 32");

  pyramid = (_KLT_Pyramid) malloc(nbytes);
  if (pyramid == NULL)
    KLTError("(_KLTCreatePyramid)  Out of memory");

  pyramid->subsampling = subsampling;
  pyramid->nLevels = nlevels;
  pyramid->img = (_KLT_FloatImage *) (pyramid + 1);
  pyramid->ncols = (int *) (pyramid->img + nlevels);
  pyramid->nrows = (int *) (pyramid->ncols + nlevels);

  for (i = 0 ; i < nlevels ; i++)  {
    pyramid->img[i] =  _KLTCreateFloatImage(ncols, nrows);
    pyramid->ncols[i] = ncols;  pyramid->nrows[i] = nrows;
    ncols /= subsampling;  nrows /= subsampling;
  }

  return pyramid;
}


/*********************************************************************
 *
 */

void _KLTFreePyramid(
  _KLT_Pyramid pyramid)
{
  int i;

  /* Free images */
  for (i = 0 ; i < pyramid->nLevels ; i++)
    _KLTFreeFloatImage(pyramid->img[i]);

  /* Free structure */
  free(pyramid);
}


/*********************************************************************
 *
 */

void _KLTComputePyramid(
  _KLT_FloatImage img, 
  _KLT_Pyramid pyramid,
  float sigma_fact)
{
  _KLT_FloatImage currimg, tmpimg;
  int ncols = img->ncols, nrows = img->nrows;
  int subsampling = pyramid->subsampling;
  int subhalf = subsampling / 2;
  float sigma = subsampling * sigma_fact;  /* empirically determined */
  int oldncols, oldnrows;
  int i, x, y;
	
  if (subsampling != 2 && subsampling != 4 && 
      subsampling != 8 && subsampling != 16 && subsampling != 32)
    KLTError("(_KLTComputePyramid)  Pyramid's subsampling must "
             "be either 2, 4, 8, 16, or 32");

  assert(pyramid->ncols[0] == img->ncols);
  assert(pyramid->nrows[0] == img->nrows);

  /* Copy original image to level 0 of pyramid */
  memcpy(pyramid->img[0]->data, img->data, ncols*nrows*sizeof(float));

  currimg = img;
  for (i = 1 ; i < pyramid->nLevels ; i++)  {
    tmpimg = _KLTCreateFloatImage(ncols, nrows);
    _KLTComputeSmoothedImage(currimg, sigma, tmpimg);


    /* Subsample */
    oldncols = ncols;
    oldnrows = nrows;
    ncols /= subsampling;  nrows /= subsampling;
    /* Parallel subsampling per level */
#ifdef USE_OPENACC
    {
      const int dst_elems = ncols * nrows;
      const int src_elems = oldncols * oldnrows;
      float *dst = pyramid->img[i]->data;
      float *src = tmpimg->data;
      if (dst_elems > 0 && src_elems > 0) {
#pragma acc data copyin(src[0:src_elems]) copyout(dst[0:dst_elems])
        {
#pragma acc parallel loop gang vector collapse(2)
          for (y = 0 ; y < nrows ; y++) {
            for (x = 0 ; x < ncols ; x++) {
              dst[y*ncols+x] =
                src[(subsampling*y+subhalf)*oldncols + (subsampling*x+subhalf)];
            }
          }
        }
      }
    }
#else
    for (y = 0 ; y < nrows ; y++) {
      for (x = 0 ; x < ncols ; x++) {
        pyramid->img[i]->data[y*ncols+x] =
          tmpimg->data[(subsampling*y+subhalf)*oldncols + (subsampling*x+subhalf)];
      }
    }
#endif

    /* Reassign current image */
    currimg = pyramid->img[i];
				
    _KLTFreeFloatImage(tmpimg);
  }
}
 












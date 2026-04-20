/*********************************************************************
 * convolve.c
// CPU stubs for horizontal/vertical convolution removed; using GPU wrappers
/* Standard includes needed for assert/exp/fabs */
#include <assert.h>
#include <math.h>

#include "base.h"
#include "error.h"
#include "convolve.h"
#include "klt_util.h"   /* printing */
#include "gpu_kernels.h"

/* _KLT_FloatImage is declared in klt_util.h */

#define MAX_KERNEL_WIDTH 	71


typedef struct  {
  int width;
  float data[MAX_KERNEL_WIDTH];
}  ConvolutionKernel;

/* Kernels */
static ConvolutionKernel gauss_kernel;
static ConvolutionKernel gaussderiv_kernel;
static float sigma_last = -10.0;


/*********************************************************************
 * _KLTToFloatImage
 *
 * Given a pointer to image data (probably unsigned chars), copy
 * data to a float image.
 */

void _KLTToFloatImage(
  unsigned char *img,
  int ncols, int nrows,
  _KLT_FloatImage floatimg)
{
  unsigned char *ptrend = img + ncols*nrows;
  float *ptrout = floatimg->data;

  /* Output image must be large enough to hold result */
  assert(floatimg->ncols >= ncols);
  assert(floatimg->nrows >= nrows);

  floatimg->ncols = ncols;
  floatimg->nrows = nrows;

  while (img < ptrend)  *ptrout++ = (float) *img++;
}


/*********************************************************************
 * _computeKernels
 */

static void _computeKernels(
  float sigma,
  ConvolutionKernel *gauss,
  ConvolutionKernel *gaussderiv)
{
  const float factor = 0.01f;   /* for truncating tail */
  int i;

  assert(MAX_KERNEL_WIDTH % 2 == 1);
  assert(sigma >= 0.0);

  /* Compute kernels, and automatically determine widths */
  {
    const int hw = MAX_KERNEL_WIDTH / 2;
    float max_gauss = 1.0f, max_gaussderiv = (float) (sigma*exp(-0.5f));
	
    /* Compute gauss and deriv */
    for (i = -hw ; i <= hw ; i++)  {
      gauss->data[i+hw]      = (float) exp(-i*i / (2*sigma*sigma));
      gaussderiv->data[i+hw] = -i * gauss->data[i+hw];
    }

    /* Compute widths */
    gauss->width = MAX_KERNEL_WIDTH;
    for (i = -hw ; fabs(gauss->data[i+hw] / max_gauss) < factor ; 
         i++, gauss->width -= 2);
    gaussderiv->width = MAX_KERNEL_WIDTH;
    for (i = -hw ; fabs(gaussderiv->data[i+hw] / max_gaussderiv) < factor ; 
         i++, gaussderiv->width -= 2);
    if (gauss->width == MAX_KERNEL_WIDTH || 
        gaussderiv->width == MAX_KERNEL_WIDTH)
      KLTError("(_computeKernels) MAX_KERNEL_WIDTH %d is too small for "
               "a sigma of %f", MAX_KERNEL_WIDTH, sigma);
  }

  /* Shift if width less than MAX_KERNEL_WIDTH */
  for (i = 0 ; i < gauss->width ; i++)
    gauss->data[i] = gauss->data[i+(MAX_KERNEL_WIDTH-gauss->width)/2];
  for (i = 0 ; i < gaussderiv->width ; i++)
    gaussderiv->data[i] = gaussderiv->data[i+(MAX_KERNEL_WIDTH-gaussderiv->width)/2];
  /* Normalize gauss and deriv */
  {
    const int hw = gaussderiv->width / 2;
    float den;
			
    den = 0.0;
    for (i = 0 ; i < gauss->width ; i++)  den += gauss->data[i];
    for (i = 0 ; i < gauss->width ; i++)  gauss->data[i] /= den;
    den = 0.0;
    for (i = -hw ; i <= hw ; i++)  den -= i*gaussderiv->data[i+hw];
    for (i = -hw ; i <= hw ; i++)  gaussderiv->data[i+hw] /= den;
  }

  sigma_last = sigma;
}
	

/*********************************************************************
 * _KLTGetKernelWidths
 *
 */

void _KLTGetKernelWidths(
  float sigma,
  int *gauss_width,
  int *gaussderiv_width)
{
  _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
  *gauss_width = gauss_kernel.width;
  *gaussderiv_width = gaussderiv_kernel.width;
}


/*********************************************************************
 * _convolveImageHoriz
 */



/*********************************************************************
 * _convolveImageVert
 */



/*********************************************************************
 * _convolveSeparate
 */

static void _convolveSeparate(
  _KLT_FloatImage imgin,
  ConvolutionKernel horiz_kernel,
  ConvolutionKernel vert_kernel,
  _KLT_FloatImage imgout)
{
  /* Create temporary image */
  _KLT_FloatImage tmpimg;
  tmpimg = _KLTCreateFloatImage(imgin->ncols, imgin->nrows);
  
  /* GPU-accelerated separable convolution */
  launchConvolveImageHoriz(
    imgin->data, tmpimg->data,
    imgin->ncols, imgin->nrows,
    horiz_kernel.data, horiz_kernel.width);
  launchConvolveImageVert(
    tmpimg->data, imgout->data,
    imgin->ncols, imgin->nrows,
    vert_kernel.data, vert_kernel.width);

  /* Free memory */
  _KLTFreeFloatImage(tmpimg);
}

	
/*********************************************************************
 * _KLTComputeGradients
 */

void _KLTComputeGradients(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage gradx,
  _KLT_FloatImage grady)
{
				
  /* Output images must be large enough to hold result */
  assert(gradx->ncols >= img->ncols);
  assert(gradx->nrows >= img->nrows);
  assert(grady->ncols >= img->ncols);
  assert(grady->nrows >= img->nrows);

  /* Compute kernels, if necessary */
  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
	
  /* GPU-accelerated gradient convolution */
  {
    _KLT_FloatImage tmpimg = _KLTCreateFloatImage(img->ncols, img->nrows);
    launchConvolveImageHoriz(
        img->data, tmpimg->data,
        img->ncols, img->nrows,
        gaussderiv_kernel.data, gaussderiv_kernel.width);
    launchConvolveImageVert(
        tmpimg->data, gradx->data,
        img->ncols, img->nrows,
        gauss_kernel.data, gauss_kernel.width);
    _KLTFreeFloatImage(tmpimg);
  }
  {
    _KLT_FloatImage tmpimg = _KLTCreateFloatImage(img->ncols, img->nrows);
    launchConvolveImageHoriz(
        img->data, tmpimg->data,
        img->ncols, img->nrows,
        gauss_kernel.data, gauss_kernel.width);
    launchConvolveImageVert(
        tmpimg->data, grady->data,
        img->ncols, img->nrows,
        gaussderiv_kernel.data, gaussderiv_kernel.width);
    _KLTFreeFloatImage(tmpimg);
  }

}
	

/*********************************************************************
 * _KLTComputeSmoothedImage
 */

void _KLTComputeSmoothedImage(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage smooth)
{
  /* Output image must be large enough to hold result */
  assert(smooth->ncols >= img->ncols);
  assert(smooth->nrows >= img->nrows);

  /* Compute kernel, if necessary; gauss_deriv is not used */
  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

  /* GPU-accelerated smoothing */
  {
    _KLT_FloatImage tmpimg = _KLTCreateFloatImage(img->ncols, img->nrows);
    launchConvolveImageHoriz(
        img->data, tmpimg->data,
        img->ncols, img->nrows,
        gauss_kernel.data, gauss_kernel.width);
    launchConvolveImageVert(
        tmpimg->data, smooth->data,
        img->ncols, img->nrows,
        gauss_kernel.data, gauss_kernel.width);
    _KLTFreeFloatImage(tmpimg);
  }
}




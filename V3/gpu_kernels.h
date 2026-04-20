#ifndef GPU_KERNELS_H
#define GPU_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif


// GPU wrappers for convolution
void launchConvolveImageHoriz(const float* imgin, float* imgout, int ncols, int nrows, const float* kernel, int klen);
void launchConvolveImageVert(const float* imgin, float* imgout, int ncols, int nrows, const float* kernel, int klen);

// SelectGoodFeatures: compute (x,y,val) pointlist on GPU for sampled pixels
// Only the outer two loops are parallelized; inner window loops remain serial per thread
void launchSelectGoodFeaturesComputePoints(const float* gradx, const float* grady, int ncols, int nrows, int borderx, int bordery, int window_hw, int window_hh, int step, int int_limit, int* host_pointlist_out, int* npoints_out);

// Timing function
void printGPUKernelTimes(void);

// Get total GPU time and reset timers for timing integration
float getTotalGPUTime(void);              // Kernel execution time only
float getTotalGPUOperationTime(void);     // Complete GPU operation time (kernels + transfers)
float getTotalMemoryTime(void);           // Memory setup + transfer time
void resetGPUTimers(void);

// Cleanup function - call at program exit to free persistent GPU resources
void cleanupGPUResources(void);

#ifdef __cplusplus
}
#endif
#endif // GPU_KERNELS_H
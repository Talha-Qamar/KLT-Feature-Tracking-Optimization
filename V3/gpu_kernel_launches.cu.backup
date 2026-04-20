#include "gpu_kernels.h"
#include <cuda.h>
#include <math.h>
#include <stdio.h>

// Global timing accumulators (in milliseconds)
static float g_time_horiz_ms = 0.0f;
static float g_time_vert_ms = 0.0f;
static float g_time_select_ms = 0.0f;

// Detailed timing breakdowns
static float g_time_horiz_h2d_ms = 0.0f;
static float g_time_horiz_d2h_ms = 0.0f;
static float g_time_horiz_malloc_ms = 0.0f;
static float g_time_vert_h2d_ms = 0.0f;
static float g_time_vert_d2h_ms = 0.0f;
static float g_time_vert_malloc_ms = 0.0f;
static float g_time_select_h2d_ms = 0.0f;
static float g_time_select_d2h_ms = 0.0f;
static float g_time_select_malloc_ms = 0.0f;

// GPU kernel for horizontal convolution
__global__ void convolveImageHorizKernel(const float* imgin, float* imgout, int ncols, int nrows, const float* kernel, int klen) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int radius = klen / 2;
    if (row >= nrows || col >= ncols) return;
    float sum = 0.0f;
    if (col < radius || col >= ncols - radius) {
        imgout[row * ncols + col] = 0.0f;
        return;
    }
    for (int k = 0; k < klen; ++k) {
        int inCol = col - radius + k;
        sum += imgin[row * ncols + inCol] * kernel[k];
    }
    imgout[row * ncols + col] = sum;
}

// GPU kernel for vertical convolution
__global__ void convolveImageVertKernel(const float* imgin, float* imgout, int ncols, int nrows, const float* kernel, int klen) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int radius = klen / 2;
    if (row >= nrows || col >= ncols) return;
    float sum = 0.0f;
    if (row < radius || row >= nrows - radius) {
        imgout[row * ncols + col] = 0.0f;
        return;
    }
    for (int k = 0; k < klen; ++k) {
        int inRow = row - radius + k;
        sum += imgin[inRow * ncols + col] * kernel[k];
    }
    imgout[row * ncols + col] = sum;
}

// Host wrapper for horizontal convolution
void launchConvolveImageHoriz(const float* imgin, float* imgout, int ncols, int nrows, const float* kernel, int klen) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;
    
    float *d_imgin, *d_imgout, *d_kernel;
    size_t imgSize = ncols * nrows * sizeof(float);
    size_t kernelSize = klen * sizeof(float);
    
    // Time memory allocation
    cudaEventRecord(start, 0);
    cudaMalloc(&d_imgin, imgSize);
    cudaMalloc(&d_imgout, imgSize);
    cudaMalloc(&d_kernel, kernelSize);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    g_time_horiz_malloc_ms += ms;
    
    // Time H2D transfer
    cudaEventRecord(start, 0);
    cudaMemcpy(d_imgin, imgin, imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernelSize, cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    g_time_horiz_h2d_ms += ms;
    
    dim3 block(16, 16);
    dim3 grid((ncols + 15) / 16, (nrows + 15) / 16);
    
    // Time kernel execution
    cudaEventRecord(start, 0);
    convolveImageHorizKernel<<<grid, block>>>(d_imgin, d_imgout, ncols, nrows, d_kernel, klen);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    g_time_horiz_ms += ms;
    
    // Time D2H transfer
    cudaEventRecord(start, 0);
    cudaMemcpy(imgout, d_imgout, imgSize, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    g_time_horiz_d2h_ms += ms;
    
    cudaFree(d_imgin);
    cudaFree(d_imgout);
    cudaFree(d_kernel);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Host wrapper for vertical convolution
void launchConvolveImageVert(const float* imgin, float* imgout, int ncols, int nrows, const float* kernel, int klen) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;
    
    float *d_imgin, *d_imgout, *d_kernel;
    size_t imgSize = ncols * nrows * sizeof(float);
    size_t kernelSize = klen * sizeof(float);
    
    // Time memory allocation
    cudaEventRecord(start, 0);
    cudaMalloc(&d_imgin, imgSize);
    cudaMalloc(&d_imgout, imgSize);
    cudaMalloc(&d_kernel, kernelSize);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    g_time_vert_malloc_ms += ms;
    
    // Time H2D transfer
    cudaEventRecord(start, 0);
    cudaMemcpy(d_imgin, imgin, imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernelSize, cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    g_time_vert_h2d_ms += ms;
    
    dim3 block(16, 16);
    dim3 grid((ncols + 15) / 16, (nrows + 15) / 16);
    
    // Time kernel execution
    cudaEventRecord(start, 0);
    convolveImageVertKernel<<<grid, block>>>(d_imgin, d_imgout, ncols, nrows, d_kernel, klen);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    g_time_vert_ms += ms;
    
    // Time D2H transfer
    cudaEventRecord(start, 0);
    cudaMemcpy(imgout, d_imgout, imgSize, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    g_time_vert_d2h_ms += ms;
    
    cudaFree(d_imgin);
    cudaFree(d_imgout);
    cudaFree(d_kernel);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ---------------------------------------------------------------
// KLT SelectGoodFeatures outer-loop GPU parallelization
// Each thread handles one sampled (x, y) position; inner window loops remain serial

__device__ inline float deviceMinEigenvalue(float gxx, float gxy, float gyy) {
    float a = gxx + gyy;
    float b = gxx - gyy;
    float disc = sqrtf(b * b + 4.0f * gxy * gxy);
    return 0.5f * (a - disc);
}

__global__ void _KLTSelectGoodFeaturesKernel(const float* gradx, const float* grady, int ncols, int nrows, int borderx, int bordery, int window_hw, int window_hh, int step, int int_limit, int countX, int countY, int* pointlist) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // 0..countX-1
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // 0..countY-1
    if (ix >= countX || iy >= countY) return;

    int x = borderx + ix * step;
    int y = bordery + iy * step;

    // Accumulate gradients in window around (x,y)
    float gxx = 0.0f, gxy = 0.0f, gyy = 0.0f;
    for (int yy = y - window_hh; yy <= y + window_hh; ++yy) {
        if (yy < 0 || yy >= nrows) continue; // safety guard
        int rowOffset = yy * ncols;
        for (int xx = x - window_hw; xx <= x + window_hw; ++xx) {
            if (xx < 0 || xx >= ncols) continue; // safety guard
            float gx = gradx[rowOffset + xx];
            float gy = grady[rowOffset + xx];
            gxx += gx * gx;
            gxy += gx * gy;
            gyy += gy * gy;
        }
    }

    float valf = deviceMinEigenvalue(gxx, gxy, gyy);
    if (valf > (float)int_limit) valf = (float)int_limit;

    int base = (iy * countX + ix) * 3;
    pointlist[base + 0] = x;
    pointlist[base + 1] = y;
    pointlist[base + 2] = (int)valf;
}

void launchSelectGoodFeaturesComputePoints(
    const float* gradx_h,
    const float* grady_h,
    int ncols,
    int nrows,
    int borderx,
    int bordery,
    int window_hw,
    int window_hh,
    int step,
    int int_limit,
    int* host_pointlist_out,
    int* npoints_out
) {
    if (ncols <= 0 || nrows <= 0) { *npoints_out = 0; return; }
    int bx = borderx < window_hw ? window_hw : borderx;
    int by = bordery < window_hh ? window_hh : bordery;
    int maxXInclusive = (ncols - bx) - 1;
    int maxYInclusive = (nrows - by) - 1;
    if (maxXInclusive < bx || maxYInclusive < by) {
        *npoints_out = 0;
        return;
    }

    int countX = (maxXInclusive - bx) / step + 1;
    int countY = (maxYInclusive - by) / step + 1;
    int npoints = countX * countY;
    *npoints_out = npoints;

    size_t imgSize = (size_t)ncols * (size_t)nrows * sizeof(float);
    size_t plistSize = (size_t)npoints * 3 * sizeof(int);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;
    
    float *d_gradx = NULL, *d_grady = NULL;
    int *d_pointlist = NULL;
    
    // Time memory allocation
    cudaEventRecord(start, 0);
    cudaMalloc(&d_gradx, imgSize);
    cudaMalloc(&d_grady, imgSize);
    cudaMalloc(&d_pointlist, plistSize);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    g_time_select_malloc_ms += ms;
    
    // Time H2D transfer
    cudaEventRecord(start, 0);
    cudaMemcpy(d_gradx, gradx_h, imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_grady, grady_h, imgSize, cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    g_time_select_h2d_ms += ms;

    dim3 block(16, 16);
    dim3 grid((countX + block.x - 1) / block.x, (countY + block.y - 1) / block.y);
    
    // Time kernel execution
    cudaEventRecord(start, 0);
    _KLTSelectGoodFeaturesKernel<<<grid, block>>>(
        d_gradx, d_grady, ncols, nrows, bx, by,
        window_hw, window_hh, step, int_limit, countX, countY, d_pointlist);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    g_time_select_ms += ms;
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        *npoints_out = 0;
        cudaFree(d_gradx);
        cudaFree(d_grady);
        cudaFree(d_pointlist);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return;
    }

    // Time D2H transfer
    cudaEventRecord(start, 0);
    cudaMemcpy(host_pointlist_out, d_pointlist, plistSize, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    g_time_select_d2h_ms += ms;
    
    cudaFree(d_gradx);
    cudaFree(d_grady);
    cudaFree(d_pointlist);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Print accumulated kernel execution times
void printGPUKernelTimes() {
    printf("\n========== V2 GPU TIMING BREAKDOWN ==========\n");
    printf("HORIZONTAL CONVOLUTION:\n");
    printf("  Memory Allocation:  %.3f ms\n", g_time_horiz_malloc_ms);
    printf("  Host-to-Device:     %.3f ms\n", g_time_horiz_h2d_ms);
    printf("  Kernel Execution:   %.3f ms\n", g_time_horiz_ms);
    printf("  Device-to-Host:     %.3f ms\n", g_time_horiz_d2h_ms);
    printf("  Subtotal:           %.3f ms\n", g_time_horiz_malloc_ms + g_time_horiz_h2d_ms + g_time_horiz_ms + g_time_horiz_d2h_ms);
    
    printf("\nVERTICAL CONVOLUTION:\n");
    printf("  Memory Allocation:  %.3f ms\n", g_time_vert_malloc_ms);
    printf("  Host-to-Device:     %.3f ms\n", g_time_vert_h2d_ms);
    printf("  Kernel Execution:   %.3f ms\n", g_time_vert_ms);
    printf("  Device-to-Host:     %.3f ms\n", g_time_vert_d2h_ms);
    printf("  Subtotal:           %.3f ms\n", g_time_vert_malloc_ms + g_time_vert_h2d_ms + g_time_vert_ms + g_time_vert_d2h_ms);
    
    printf("\nFEATURE SELECTION:\n");
    printf("  Memory Allocation:  %.3f ms\n", g_time_select_malloc_ms);
    printf("  Host-to-Device:     %.3f ms\n", g_time_select_h2d_ms);
    printf("  Kernel Execution:   %.3f ms\n", g_time_select_ms);
    printf("  Device-to-Host:     %.3f ms\n", g_time_select_d2h_ms);
    printf("  Subtotal:           %.3f ms\n", g_time_select_malloc_ms + g_time_select_h2d_ms + g_time_select_ms + g_time_select_d2h_ms);
    
    float total_malloc = g_time_horiz_malloc_ms + g_time_vert_malloc_ms + g_time_select_malloc_ms;
    float total_h2d = g_time_horiz_h2d_ms + g_time_vert_h2d_ms + g_time_select_h2d_ms;
    float total_kernel = g_time_horiz_ms + g_time_vert_ms + g_time_select_ms;
    float total_d2h = g_time_horiz_d2h_ms + g_time_vert_d2h_ms + g_time_select_d2h_ms;
    float grand_total = total_malloc + total_h2d + total_kernel + total_d2h;
    
    printf("\n----- GRAND TOTALS -----\n");
    printf("Total Memory Allocation: %.3f ms (%.1f%%)\n", total_malloc, 100.0f * total_malloc / grand_total);
    printf("Total Host-to-Device:    %.3f ms (%.1f%%)\n", total_h2d, 100.0f * total_h2d / grand_total);
    printf("Total Kernel Execution:  %.3f ms (%.1f%%)\n", total_kernel, 100.0f * total_kernel / grand_total);
    printf("Total Device-to-Host:    %.3f ms (%.1f%%)\n", total_d2h, 100.0f * total_d2h / grand_total);
    printf("GRAND TOTAL:             %.3f ms\n", grand_total);
    printf("=============================================\n\n");
}

// Get total GPU kernel time only
float getTotalGPUTime() {
    return g_time_horiz_ms + g_time_vert_ms + g_time_select_ms;
}

// Get total GPU operation time (including memory transfers)
float getTotalGPUOperationTime() {
    float total_malloc = g_time_horiz_malloc_ms + g_time_vert_malloc_ms + g_time_select_malloc_ms;
    float total_h2d = g_time_horiz_h2d_ms + g_time_vert_h2d_ms + g_time_select_h2d_ms;
    float total_kernel = g_time_horiz_ms + g_time_vert_ms + g_time_select_ms;
    float total_d2h = g_time_horiz_d2h_ms + g_time_vert_d2h_ms + g_time_select_d2h_ms;
    return total_malloc + total_h2d + total_kernel + total_d2h;
}

// Get total memory transfer time
float getTotalMemoryTime() {
    float total_malloc = g_time_horiz_malloc_ms + g_time_vert_malloc_ms + g_time_select_malloc_ms;
    float total_h2d = g_time_horiz_h2d_ms + g_time_vert_h2d_ms + g_time_select_h2d_ms;
    float total_d2h = g_time_horiz_d2h_ms + g_time_vert_d2h_ms + g_time_select_d2h_ms;
    return total_malloc + total_h2d + total_d2h;
}

// Reset GPU timing counters
void resetGPUTimers() {
    g_time_horiz_ms = 0.0f;
    g_time_vert_ms = 0.0f;
    g_time_select_ms = 0.0f;
    g_time_horiz_h2d_ms = 0.0f;
    g_time_horiz_d2h_ms = 0.0f;
    g_time_horiz_malloc_ms = 0.0f;
    g_time_vert_h2d_ms = 0.0f;
    g_time_vert_d2h_ms = 0.0f;
    g_time_vert_malloc_ms = 0.0f;
    g_time_select_h2d_ms = 0.0f;
    g_time_select_d2h_ms = 0.0f;
    g_time_select_malloc_ms = 0.0f;
}
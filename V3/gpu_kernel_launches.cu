#include "gpu_kernels.h"
#include <cuda.h>
#include <math.h>
#include <stdio.h>

// Preprocessor Defines
#define MAX_KERNEL_SIZE 128
#define ALIGNMENT 128
#define TILE_WIDTH 16
#define TILE_HEIGHT 16
#define PADDING 1

// Constant Memory
__constant__ float d_kernel_const[MAX_KERNEL_SIZE];

// Global Timing Variables
static float g_time_horiz_ms = 0.0f, g_time_vert_ms = 0.0f, g_time_select_ms = 0.0f;
static float g_time_horiz_h2d_ms = 0.0f, g_time_horiz_d2h_ms = 0.0f, g_time_horiz_memsetup_ms = 0.0f;
static float g_time_vert_h2d_ms = 0.0f, g_time_vert_d2h_ms = 0.0f, g_time_vert_memsetup_ms = 0.0f;
static float g_time_select_h2d_ms = 0.0f, g_time_select_d2h_ms = 0.0f, g_time_select_memsetup_ms = 0.0f;

// Global Memory Pointers
static float *d_persistent_img1 = NULL, *d_persistent_img2 = NULL, *h_pinned_buffer = NULL;

// Global Size Variables
static size_t persistent_capacity = 0, pinned_capacity = 0;

// Global CUDA Resources
static cudaStream_t stream1 = 0, stream2 = 0;
static bool streams_initialized = false;

static void initGPUResources() {
    if (!streams_initialized) {
        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);
        streams_initialized = true;
    }
}

static void ensurePersistentMemory(size_t required_bytes) {
    required_bytes = ((required_bytes + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
    
    if (required_bytes > persistent_capacity) {
        if (d_persistent_img1) cudaFree(d_persistent_img1);
        if (d_persistent_img2) cudaFree(d_persistent_img2);
        cudaMalloc(&d_persistent_img1, required_bytes);
        cudaMalloc(&d_persistent_img2, required_bytes);
        persistent_capacity = required_bytes;
    }
}

static void ensurePinnedMemory(size_t required_bytes) {
    if (required_bytes > pinned_capacity) {
        if (h_pinned_buffer) cudaFreeHost(h_pinned_buffer);
        cudaMallocHost(&h_pinned_buffer, required_bytes);
        pinned_capacity = required_bytes;
    }
}

__global__ void convolveImageHorizKernel_Optimized(
    const float* __restrict__ imgin,
    float* __restrict__ imgout, 
    int ncols, 
    int nrows, 
    int klen) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int radius = klen / 2;
    
    __shared__ float tile[TILE_HEIGHT][TILE_WIDTH + 2 * (MAX_KERNEL_SIZE/2) + PADDING];
    
    int sharedCol = threadIdx.x + radius;
    if (row < nrows) {
        if (col < ncols) {
            tile[threadIdx.y][sharedCol] = imgin[row * ncols + col];
        }
        
        if (threadIdx.x < radius) {
            int leftCol = col - radius;
            tile[threadIdx.y][threadIdx.x] = (leftCol >= 0) ? imgin[row * ncols + leftCol] : 0.0f;
        }
        
        if (threadIdx.x < radius) {
            int rightCol = col + TILE_WIDTH;
            if (rightCol < ncols) {
                tile[threadIdx.y][sharedCol + TILE_WIDTH] = imgin[row * ncols + rightCol];
            } else {
                tile[threadIdx.y][sharedCol + TILE_WIDTH] = 0.0f;
            }
        }
    }
    
    __syncthreads();
    
    if (row >= nrows || col >= ncols) return;
    
    float sum = 0.0f;
    bool isValid = (col >= radius && col < ncols - radius);
    
    if (isValid) {
        for (int k = 0; k < klen; ++k) {
            sum += tile[threadIdx.y][sharedCol - radius + k] * d_kernel_const[k];
        }
    }
    
    imgout[row * ncols + col] = isValid ? sum : 0.0f;
}

__global__ void convolveImageVertKernel_Optimized(
    const float* __restrict__ imgin, 
    float* __restrict__ imgout, 
    int ncols, 
    int nrows, 
    int klen) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int radius = klen / 2;
    
    __shared__ float tile[TILE_HEIGHT + 2 * (MAX_KERNEL_SIZE/2)][TILE_WIDTH + PADDING];
    
    int sharedRow = threadIdx.y + radius;
    if (col < ncols) {
        if (row < nrows) {
            tile[sharedRow][threadIdx.x] = imgin[row * ncols + col];
        }
        
        if (threadIdx.y < radius) {
            int topRow = row - radius;
            tile[threadIdx.y][threadIdx.x] = (topRow >= 0) ? imgin[topRow * ncols + col] : 0.0f;
        }
        
        if (threadIdx.y < radius) {
            int bottomRow = row + TILE_HEIGHT;
            if (bottomRow < nrows) {
                tile[sharedRow + TILE_HEIGHT][threadIdx.x] = imgin[bottomRow * ncols + col];
            } else {
                tile[sharedRow + TILE_HEIGHT][threadIdx.x] = 0.0f;
            }
        }
    }
    
    __syncthreads();
    
    if (row >= nrows || col >= ncols) return;
    
    float sum = 0.0f;
    bool isValid = (row >= radius && row < nrows - radius);
    
    if (isValid) {
        for (int k = 0; k < klen; ++k) {
            sum += tile[sharedRow - radius + k][threadIdx.x] * d_kernel_const[k];
        }
    }
    
    imgout[row * ncols + col] = isValid ? sum : 0.0f;
}

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

__global__ void convolveSeparable2DFused_Optimized(
    const float* __restrict__ imgin,
    float* __restrict__ imgout,
    int ncols,
    int nrows,
    int klen) {
    
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int row = blockIdx.y * TILE_HEIGHT + threadIdx.y;
    int radius = klen / 2;
    
    __shared__ float horizResult[TILE_HEIGHT][TILE_WIDTH + PADDING];
    __shared__ float inputTile[TILE_HEIGHT][TILE_WIDTH + 2 * (MAX_KERNEL_SIZE/2) + PADDING];
    
    int sharedCol = threadIdx.x + radius;
    if (row < nrows) {
        if (col < ncols) {
            inputTile[threadIdx.y][sharedCol] = imgin[row * ncols + col];
        }
        if (threadIdx.x < radius) {
            int leftCol = col - radius;
            inputTile[threadIdx.y][threadIdx.x] = (leftCol >= 0) ? imgin[row * ncols + leftCol] : 0.0f;
        }
        if (threadIdx.x < radius) {
            int rightCol = col + TILE_WIDTH;
            inputTile[threadIdx.y][sharedCol + TILE_WIDTH] = 
                (rightCol < ncols) ? imgin[row * ncols + rightCol] : 0.0f;
        }
    }
    
    __syncthreads();
    
    if (row < nrows && col < ncols) {
        float sum = 0.0f;
        bool validH = (col >= radius && col < ncols - radius);
        if (validH) {
            for (int k = 0; k < klen; ++k) {
                sum += inputTile[threadIdx.y][sharedCol - radius + k] * d_kernel_const[k];
            }
        }
        horizResult[threadIdx.y][threadIdx.x] = validH ? sum : 0.0f;
    }
    
    __syncthreads();

    __shared__ float vertTile[TILE_HEIGHT + 2 * (MAX_KERNEL_SIZE/2)][TILE_WIDTH + PADDING];
    
    if (row < nrows && col < ncols) {
        vertTile[threadIdx.y + radius][threadIdx.x] = horizResult[threadIdx.y][threadIdx.x];
    }
    
    __syncthreads();
    
    if (row < nrows && col < ncols) {
        float sum = 0.0f;
        bool validV = (row >= radius && row < nrows - radius);
        if (validV) {
            for (int k = 0; k < klen; ++k) {
                sum += vertTile[threadIdx.y + radius - radius + k][threadIdx.x] * d_kernel_const[k];
            }
        }
        imgout[row * ncols + col] = validV ? sum : 0.0f;
    }
}

void launchConvolveImageHoriz(const float* imgin, float* imgout, int ncols, int nrows, const float* kernel, int klen) {
    cudaEvent_t start, stop, start2, stop2;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    float ms = 0.0f;
    
    initGPUResources();
    
    size_t imgSize = ncols * nrows * sizeof(float);
    
    // Time memory setup (persistent + pinned allocation if needed)
    cudaEventRecord(start, 0);
    ensurePersistentMemory(imgSize);
    ensurePinnedMemory(imgSize);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    g_time_horiz_memsetup_ms += ms;
    
    // Copy to pinned buffer first (NOT timed as part of H2D - it's CPU work)
    memcpy(h_pinned_buffer, imgin, imgSize);
    
    if (klen <= MAX_KERNEL_SIZE) {
        cudaMemcpyToSymbol(d_kernel_const, kernel, klen * sizeof(float));
    }
    
    // Time H2D transfer (only GPU transfer, not host memcpy)
    cudaEventRecord(start, stream1);
    cudaMemcpyAsync(d_persistent_img1, h_pinned_buffer, imgSize, cudaMemcpyHostToDevice, stream1);
    cudaEventRecord(stop, stream1);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    g_time_horiz_h2d_ms += ms;
    
    dim3 block(TILE_WIDTH, TILE_HEIGHT);
    dim3 grid((ncols + TILE_WIDTH - 1) / TILE_WIDTH, (nrows + TILE_HEIGHT - 1) / TILE_HEIGHT);
    
    // Time kernel execution
    cudaEventRecord(start2, stream1);
    convolveImageHorizKernel_Optimized<<<grid, block, 0, stream1>>>(
        d_persistent_img1, d_persistent_img2, ncols, nrows, klen);
    cudaEventRecord(stop2, stream1);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&ms, start2, stop2);
    g_time_horiz_ms += ms;
    
    // Time D2H transfer (only GPU transfer, not host memcpy)
    cudaEventRecord(start, stream1);
    cudaMemcpyAsync(h_pinned_buffer, d_persistent_img2, imgSize, cudaMemcpyDeviceToHost, stream1);
    cudaEventRecord(stop, stream1);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    g_time_horiz_d2h_ms += ms;
    
    // Copy from pinned buffer to output (NOT timed - it's CPU work)
    memcpy(imgout, h_pinned_buffer, imgSize);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);
}

void launchConvolveImageVert(const float* imgin, float* imgout, int ncols, int nrows, const float* kernel, int klen) {
    cudaEvent_t start, stop, start2, stop2;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    float ms = 0.0f;
    
    initGPUResources();
    
    size_t imgSize = ncols * nrows * sizeof(float);
    
    // Time memory setup
    cudaEventRecord(start, 0);
    ensurePersistentMemory(imgSize);
    ensurePinnedMemory(imgSize);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    g_time_vert_memsetup_ms += ms;
    
    // Copy to pinned buffer first (NOT timed as part of H2D)
    memcpy(h_pinned_buffer, imgin, imgSize);
    
    if (klen <= MAX_KERNEL_SIZE) {
        cudaMemcpyToSymbol(d_kernel_const, kernel, klen * sizeof(float));
    }
    
    // Time H2D transfer (only GPU transfer)
    cudaEventRecord(start, stream1);
    cudaMemcpyAsync(d_persistent_img1, h_pinned_buffer, imgSize, cudaMemcpyHostToDevice, stream1);
    cudaEventRecord(stop, stream1);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    g_time_vert_h2d_ms += ms;
    
    dim3 block(TILE_WIDTH, TILE_HEIGHT);
    dim3 grid((ncols + TILE_WIDTH - 1) / TILE_WIDTH, (nrows + TILE_HEIGHT - 1) / TILE_HEIGHT);
    
    // Time kernel execution
    cudaEventRecord(start2, stream1);
    convolveImageVertKernel_Optimized<<<grid, block, 0, stream1>>>(
        d_persistent_img1, d_persistent_img2, ncols, nrows, klen);
    cudaEventRecord(stop2, stream1);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&ms, start2, stop2);
    g_time_vert_ms += ms;
    
    // Time D2H transfer (only GPU transfer)
    cudaEventRecord(start, stream1);
    cudaMemcpyAsync(h_pinned_buffer, d_persistent_img2, imgSize, cudaMemcpyDeviceToHost, stream1);
    cudaEventRecord(stop, stream1);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    g_time_vert_d2h_ms += ms;
    
    // Copy from pinned buffer to output (NOT timed)
    memcpy(imgout, h_pinned_buffer, imgSize);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);
}

__device__ inline float deviceMinEigenvalue(float gxx, float gxy, float gyy) {
    float a = gxx + gyy;
    float b = gxx - gyy;
    float disc = sqrtf(b * b + 4.0f * gxy * gxy);
    return 0.5f * (a - disc);
}


#define FEATURE_TILE_SIZE 16
#define MAX_WINDOW_SIZE 10

__global__ void _KLTSelectGoodFeaturesKernel_Optimized(
    const float* __restrict__ gradx, 
    const float* __restrict__ grady, 
    int ncols, 
    int nrows, 
    int borderx, 
    int bordery, 
    int window_hw, 
    int window_hh, 
    int step, 
    int int_limit, 
    int countX, 
    int countY, 
    int* __restrict__ pointlist) {
    
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (ix >= countX || iy >= countY) return;
    
    int x = borderx + ix * step;
    int y = bordery + iy * step;
    
    const int TILE_WITH_HALO = FEATURE_TILE_SIZE + 2 * MAX_WINDOW_SIZE;
    __shared__ float s_gradx[TILE_WITH_HALO][TILE_WITH_HALO + PADDING];
    __shared__ float s_grady[TILE_WITH_HALO][TILE_WITH_HALO + PADDING];
    
    int blockStartX = blockIdx.x * FEATURE_TILE_SIZE * step + borderx - MAX_WINDOW_SIZE;
    int blockStartY = blockIdx.y * FEATURE_TILE_SIZE * step + bordery - MAX_WINDOW_SIZE;
    
    for (int ty = threadIdx.y; ty < TILE_WITH_HALO; ty += blockDim.y) {
        for (int tx = threadIdx.x; tx < TILE_WITH_HALO; tx += blockDim.x) {
            int globalX = blockStartX + tx;
            int globalY = blockStartY + ty;
            
            bool valid = (globalX >= 0 && globalX < ncols && globalY >= 0 && globalY < nrows);
            int idx = globalY * ncols + globalX;
            
            s_gradx[ty][tx] = valid ? gradx[idx] : 0.0f;
            s_grady[ty][tx] = valid ? grady[idx] : 0.0f;
        }
    }
    
    __syncthreads();
    
    int localX = (x - blockStartX);
    int localY = (y - blockStartY);
    
    float gxx = 0.0f, gxy = 0.0f, gyy = 0.0f;
    
    for (int yy = -window_hh; yy <= window_hh; ++yy) {
        int sy = localY + yy;
        if (sy < 0 || sy >= TILE_WITH_HALO) continue;
        
        for (int xx = -window_hw; xx <= window_hw; ++xx) {
            int sx = localX + xx;
            if (sx < 0 || sx >= TILE_WITH_HALO) continue;
            
            float gx = s_gradx[sy][sx];
            float gy = s_grady[sy][sx];
            
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
    cudaEvent_t start, stop, start2, stop2;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    float ms = 0.0f;
    
    initGPUResources();
    
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

    static float *d_gradx_persist = NULL;
    static float *d_grady_persist = NULL;
    static int *d_pointlist_persist = NULL;
    static size_t grad_capacity = 0;
    static size_t plist_capacity = 0;
    
    // Time memory setup
    cudaEventRecord(start, 0);
    if (imgSize > grad_capacity) {
        if (d_gradx_persist) cudaFree(d_gradx_persist);
        if (d_grady_persist) cudaFree(d_grady_persist);
        cudaMalloc(&d_gradx_persist, imgSize);
        cudaMalloc(&d_grady_persist, imgSize);
        grad_capacity = imgSize;
    }
    
    if (plistSize > plist_capacity) {
        if (d_pointlist_persist) cudaFree(d_pointlist_persist);
        cudaMalloc(&d_pointlist_persist, plistSize);
        plist_capacity = plistSize;
    }
    
    static float *h_pinned_gradx = NULL;
    static float *h_pinned_grady = NULL;
    static int *h_pinned_plist = NULL;
    static size_t pinned_grad_cap = 0;
    static size_t pinned_plist_cap = 0;
    
    if (imgSize > pinned_grad_cap) {
        if (h_pinned_gradx) cudaFreeHost(h_pinned_gradx);
        if (h_pinned_grady) cudaFreeHost(h_pinned_grady);
        cudaMallocHost(&h_pinned_gradx, imgSize);
        cudaMallocHost(&h_pinned_grady, imgSize);
        pinned_grad_cap = imgSize;
    }
    
    if (plistSize > pinned_plist_cap) {
        if (h_pinned_plist) cudaFreeHost(h_pinned_plist);
        cudaMallocHost(&h_pinned_plist, plistSize);
        pinned_plist_cap = plistSize;
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    g_time_select_memsetup_ms += ms;
    
    // Copy to pinned buffers first (NOT timed as part of H2D - it's CPU work)
    memcpy(h_pinned_gradx, gradx_h, imgSize);
    memcpy(h_pinned_grady, grady_h, imgSize);
    
    // Time H2D transfer (only GPU transfers, not host memcpy)
    cudaEventRecord(start2, stream2);
    cudaMemcpyAsync(d_gradx_persist, h_pinned_gradx, imgSize, cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_grady_persist, h_pinned_grady, imgSize, cudaMemcpyHostToDevice, stream2);
    cudaEventRecord(stop2, stream2);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&ms, start2, stop2);
    g_time_select_h2d_ms += ms;

    dim3 block(16, 16);
    dim3 grid((countX + block.x - 1) / block.x, (countY + block.y - 1) / block.y);
    
    // Time kernel execution
    cudaEventRecord(start, stream2);
    
    _KLTSelectGoodFeaturesKernel_Optimized<<<grid, block, 0, stream2>>>(
        d_gradx_persist, d_grady_persist, ncols, nrows, bx, by,
        window_hw, window_hh, step, int_limit, countX, countY, d_pointlist_persist);
    
    cudaEventRecord(stop, stream2);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    g_time_select_ms += ms;
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        *npoints_out = 0;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaEventDestroy(start2);
        cudaEventDestroy(stop2);
        return;
    }

    // Time D2H transfer (only GPU transfer)
    cudaEventRecord(start2, stream2);
    cudaMemcpyAsync(h_pinned_plist, d_pointlist_persist, plistSize, cudaMemcpyDeviceToHost, stream2);
    cudaEventRecord(stop2, stream2);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&ms, start2, stop2);
    g_time_select_d2h_ms += ms;
    
    // Copy from pinned buffer to output (NOT timed)
    memcpy(host_pointlist_out, h_pinned_plist, plistSize);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);
}

// Print accumulated kernel execution times
void printGPUKernelTimes() {
    printf("\n========== V3 GPU TIMING BREAKDOWN ==========\n");
    printf("HORIZONTAL CONVOLUTION:\n");
    printf("  Memory Setup:       %.3f ms (persistent+pinned alloc)\n", g_time_horiz_memsetup_ms);
    printf("  Host-to-Device:     %.3f ms (async+pinned)\n", g_time_horiz_h2d_ms);
    printf("  Kernel Execution:   %.3f ms (optimized)\n", g_time_horiz_ms);
    printf("  Device-to-Host:     %.3f ms (async+pinned)\n", g_time_horiz_d2h_ms);
    printf("  Subtotal:           %.3f ms\n", g_time_horiz_memsetup_ms + g_time_horiz_h2d_ms + g_time_horiz_ms + g_time_horiz_d2h_ms);
    
    printf("\nVERTICAL CONVOLUTION:\n");
    printf("  Memory Setup:       %.3f ms (persistent+pinned alloc)\n", g_time_vert_memsetup_ms);
    printf("  Host-to-Device:     %.3f ms (async+pinned)\n", g_time_vert_h2d_ms);
    printf("  Kernel Execution:   %.3f ms (optimized)\n", g_time_vert_ms);
    printf("  Device-to-Host:     %.3f ms (async+pinned)\n", g_time_vert_d2h_ms);
    printf("  Subtotal:           %.3f ms\n", g_time_vert_memsetup_ms + g_time_vert_h2d_ms + g_time_vert_ms + g_time_vert_d2h_ms);
    
    printf("\nFEATURE SELECTION:\n");
    printf("  Memory Setup:       %.3f ms (persistent+pinned alloc)\n", g_time_select_memsetup_ms);
    printf("  Host-to-Device:     %.3f ms (async+pinned)\n", g_time_select_h2d_ms);
    printf("  Kernel Execution:   %.3f ms (optimized)\n", g_time_select_ms);
    printf("  Device-to-Host:     %.3f ms (async+pinned)\n", g_time_select_d2h_ms);
    printf("  Subtotal:           %.3f ms\n", g_time_select_memsetup_ms + g_time_select_h2d_ms + g_time_select_ms + g_time_select_d2h_ms);
    
    float total_memsetup = g_time_horiz_memsetup_ms + g_time_vert_memsetup_ms + g_time_select_memsetup_ms;
    float total_h2d = g_time_horiz_h2d_ms + g_time_vert_h2d_ms + g_time_select_h2d_ms;
    float total_kernel = g_time_horiz_ms + g_time_vert_ms + g_time_select_ms;
    float total_d2h = g_time_horiz_d2h_ms + g_time_vert_d2h_ms + g_time_select_d2h_ms;
    float grand_total = total_memsetup + total_h2d + total_kernel + total_d2h;
    
    printf("\n----- GRAND TOTALS -----\n");
    printf("Total Memory Setup:      %.3f ms (%.1f%%)\n", total_memsetup, 100.0f * total_memsetup / grand_total);
    printf("Total Host-to-Device:    %.3f ms (%.1f%%)\n", total_h2d, 100.0f * total_h2d / grand_total);
    printf("Total Kernel Execution:  %.3f ms (%.1f%%)\n", total_kernel, 100.0f * total_kernel / grand_total);
    printf("Total Device-to-Host:    %.3f ms (%.1f%%)\n", total_d2h, 100.0f * total_d2h / grand_total);
    printf("GRAND TOTAL:             %.3f ms\n", grand_total);
    printf("=============================================\n\n");
}

extern "C" float getTotalGPUTime() {
    return g_time_horiz_ms + g_time_vert_ms + g_time_select_ms;
}

extern "C" float getTotalGPUOperationTime() {
    float total_memsetup = g_time_horiz_memsetup_ms + g_time_vert_memsetup_ms + g_time_select_memsetup_ms;
    float total_h2d = g_time_horiz_h2d_ms + g_time_vert_h2d_ms + g_time_select_h2d_ms;
    float total_kernel = g_time_horiz_ms + g_time_vert_ms + g_time_select_ms;
    float total_d2h = g_time_horiz_d2h_ms + g_time_vert_d2h_ms + g_time_select_d2h_ms;
    return total_memsetup + total_h2d + total_kernel + total_d2h;
}

extern "C" float getTotalMemoryTime() {
    float total_memsetup = g_time_horiz_memsetup_ms + g_time_vert_memsetup_ms + g_time_select_memsetup_ms;
    float total_h2d = g_time_horiz_h2d_ms + g_time_vert_h2d_ms + g_time_select_h2d_ms;
    float total_d2h = g_time_horiz_d2h_ms + g_time_vert_d2h_ms + g_time_select_d2h_ms;
    return total_memsetup + total_h2d + total_d2h;
}

extern "C" void resetGPUTimers() {
    g_time_horiz_ms = 0.0f;
    g_time_vert_ms = 0.0f;
    g_time_select_ms = 0.0f;
    g_time_horiz_h2d_ms = 0.0f;
    g_time_horiz_d2h_ms = 0.0f;
    g_time_horiz_memsetup_ms = 0.0f;
    g_time_vert_h2d_ms = 0.0f;
    g_time_vert_d2h_ms = 0.0f;
    g_time_vert_memsetup_ms = 0.0f;
    g_time_select_h2d_ms = 0.0f;
    g_time_select_d2h_ms = 0.0f;
    g_time_select_memsetup_ms = 0.0f;
}

void cleanupGPUResources() {
    if (d_persistent_img1) cudaFree(d_persistent_img1);
    if (d_persistent_img2) cudaFree(d_persistent_img2);
    if (h_pinned_buffer) cudaFreeHost(h_pinned_buffer);
    if (streams_initialized) {
        cudaStreamDestroy(stream1);
        cudaStreamDestroy(stream2);
    }
    d_persistent_img1 = NULL;
    d_persistent_img2 = NULL;
    h_pinned_buffer = NULL;
    persistent_capacity = 0;
    pinned_capacity = 0;
    streams_initialized = false;
}
/*
 * klt_timing.h - Comprehensive timing infrastructure for KLT tracker (V3 - Optimized GPU)
 *
 * This header provides timing utilities to measure and compare performance
 * across different versions of the KLT implementation (CPU, GPU, Optimized GPU).
 *
 * Timing Points Measured:
 * 1. Feature Selection - Time to find good features in an image
 * 2. Feature Tracking - Time to track features from one image to another
 * 3. GPU Kernel Time - Time spent in GPU kernels (from existing GPU timing)
 * 4. Memory Transfers - Time spent transferring data to/from GPU
 * 5. Total Execution - Overall runtime including I/O and initialization
 *
 * V3 improvements measured:
 * - Persistent GPU memory allocation benefits
 * - Reduced memory transfer overhead
 * - Optimized kernel performance
 */

#ifndef KLT_TIMING_H
#define KLT_TIMING_H

#include <stdint.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Timer types */
typedef enum {
    KLT_TIMER_TOTAL = 0,
    KLT_TIMER_SELECT,
    KLT_TIMER_TRACK,
    KLT_TIMER_IO,
    KLT_TIMER_GPU_COMPUTE,
    KLT_TIMER_MEMORY_OPS,
    KLT_NUM_TIMERS
} KLT_TimerType;

/* Timing context structure */
typedef struct {
    uint64_t start_times[KLT_NUM_TIMERS];
    uint64_t elapsed_times[KLT_NUM_TIMERS];
    int timer_active[KLT_NUM_TIMERS];
    const char* timer_names[KLT_NUM_TIMERS];
} KLT_TimingContextRec, *KLT_TimingContext;

/* Get current time in microseconds (wall-clock time) */
static inline uint64_t KLT_GetTimeUs(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000ULL + (uint64_t)tv.tv_usec;
}

/* Create timing context */
static inline KLT_TimingContext KLT_CreateTimingContext(void) {
    KLT_TimingContext tc = (KLT_TimingContext)malloc(sizeof(KLT_TimingContextRec));
    if (!tc) return NULL;
    
    memset(tc->start_times, 0, sizeof(tc->start_times));
    memset(tc->elapsed_times, 0, sizeof(tc->elapsed_times));
    memset(tc->timer_active, 0, sizeof(tc->timer_active));
    
    tc->timer_names[KLT_TIMER_TOTAL] = "Total Execution Time";
    tc->timer_names[KLT_TIMER_SELECT] = "Feature Selection";
    tc->timer_names[KLT_TIMER_TRACK] = "Feature Tracking";
    tc->timer_names[KLT_TIMER_IO] = "I/O Operations";
    tc->timer_names[KLT_TIMER_GPU_COMPUTE] = "GPU Kernel Time";
    tc->timer_names[KLT_TIMER_MEMORY_OPS] = "Memory Operations";
    
    return tc;
}

/* Free timing context */
static inline void KLT_FreeTimingContext(KLT_TimingContext tc) {
    if (tc) free(tc);
}

/* Start a timer */
static inline void KLT_StartTimer(KLT_TimingContext tc, KLT_TimerType timer) {
    if (!tc) return;
    tc->start_times[timer] = KLT_GetTimeUs();
    tc->timer_active[timer] = 1;
}

/* Stop a timer and accumulate elapsed time */
static inline void KLT_StopTimer(KLT_TimingContext tc, KLT_TimerType timer) {
    if (!tc || !tc->timer_active[timer]) return;
    uint64_t end_time = KLT_GetTimeUs();
    tc->elapsed_times[timer] += (end_time - tc->start_times[timer]);
    tc->timer_active[timer] = 0;
}

/* Get elapsed time for a timer in milliseconds */
static inline double KLT_GetElapsedMs(KLT_TimingContext tc, KLT_TimerType timer) {
    if (!tc) return 0.0;
    return tc->elapsed_times[timer] / 1000.0;
}

/* Set GPU kernel time from external source (e.g., CUDA events) */
static inline void KLT_SetGPUTime(KLT_TimingContext tc, double gpu_ms) {
    if (!tc) return;
    tc->elapsed_times[KLT_TIMER_GPU_COMPUTE] = (uint64_t)(gpu_ms * 1000.0);
}

/* Set GPU memory operation time from external source */
static inline void KLT_SetMemoryTime(KLT_TimingContext tc, double memory_ms) {
    if (!tc) return;
    tc->elapsed_times[KLT_TIMER_MEMORY_OPS] = (uint64_t)(memory_ms * 1000.0);
}

/* Print timing results */
static inline void KLT_PrintTimingResults(KLT_TimingContext tc, const char* version_name) {
    if (!tc) return;
    
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  %-58s  ║\n", version_name);
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    
    for (int i = 0; i < KLT_NUM_TIMERS; i++) {
        if (tc->elapsed_times[i] > 0) {
            printf("║  %-35s: %10.3f ms  ║\n", 
                   tc->timer_names[i], 
                   KLT_GetElapsedMs(tc, (KLT_TimerType)i));
        }
    }
    
    // Calculate efficiency metrics
    double total = KLT_GetElapsedMs(tc, KLT_TIMER_TOTAL);
    double gpu = KLT_GetElapsedMs(tc, KLT_TIMER_GPU_COMPUTE);
    double mem = KLT_GetElapsedMs(tc, KLT_TIMER_MEMORY_OPS);
    
    if (gpu > 0 && total > 0) {
        printf("║  %-35s: %10.3f ms  ║\n", "CPU Overhead", total - gpu - mem);
        printf("║  %-35s: %10.1f %%   ║\n", "GPU Compute Efficiency", (gpu / total) * 100.0);
    }
    if (mem > 0 && total > 0) {
        printf("║  %-35s: %10.1f %%   ║\n", "Memory Transfer Overhead", (mem / total) * 100.0);
    }
    
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("\n");
}

/* Print comparative results (for comparison between versions) */
static inline void KLT_PrintComparison(KLT_TimingContext tc1, const char* name1,
                                       KLT_TimingContext tc2, const char* name2) {
    if (!tc1 || !tc2) return;
    
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  Performance Comparison: %-18s vs %-18s  ║\n", name1, name2);
    printf("╠═══════════════════════════════════════════════════════════════════════════╣\n");
    printf("║  %-25s  %12s  %12s  %10s  ║\n", "Operation", name1, name2, "Speedup");
    printf("╠═══════════════════════════════════════════════════════════════════════════╣\n");
    
    for (int i = 0; i < KLT_NUM_TIMERS; i++) {
        if (tc1->elapsed_times[i] > 0 && tc2->elapsed_times[i] > 0) {
            double time1 = KLT_GetElapsedMs(tc1, (KLT_TimerType)i);
            double time2 = KLT_GetElapsedMs(tc2, (KLT_TimerType)i);
            double speedup = time1 / time2;
            
            printf("║  %-25s  %9.3f ms  %9.3f ms  %8.2fx  ║\n",
                   tc1->timer_names[i], time1, time2, speedup);
        }
    }
    
    printf("╚═══════════════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
}

/* Save timing results to file for later analysis */
static inline void KLT_SaveTimingToFile(KLT_TimingContext tc, const char* version_name, const char* filename) {
    if (!tc) return;
    
    FILE* fp = fopen(filename, "a");
    if (!fp) return;
    
    fprintf(fp, "%s,", version_name);
    for (int i = 0; i < KLT_NUM_TIMERS; i++) {
        fprintf(fp, "%.3f", KLT_GetElapsedMs(tc, (KLT_TimerType)i));
        if (i < KLT_NUM_TIMERS - 1) fprintf(fp, ",");
    }
    fprintf(fp, "\n");
    
    fclose(fp);
}

#endif /* KLT_TIMING_H */

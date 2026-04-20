/*
 * timing.h - lightweight CPU time measurement helpers
 *
 * Usage:
 *   #include "timing.h"
 *   uint64_t t0 = klt_cpu_time_us();
 *   ... work ...
 *   uint64_t t1 = klt_cpu_time_us();
 *   fprintf(stderr, "took %.3f ms\n", (t1 - t0) / 1000.0);
 *
 * We use getrusage() to obtain process CPU (user+sys) time, which
 * accumulates CPU consumption and excludes idle/wait. Measuring deltas
 * around a function approximates CPU used by that function.
 */

#ifndef KLT_TIMING_H
#define KLT_TIMING_H

#include <stdint.h>
#include <sys/resource.h>
#include <sys/time.h>

static inline uint64_t klt_cpu_time_us(void) {
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  uint64_t user_us = (uint64_t)ru.ru_utime.tv_sec * 1000000ull + (uint64_t)ru.ru_utime.tv_usec;
  uint64_t sys_us  = (uint64_t)ru.ru_stime.tv_sec * 1000000ull + (uint64_t)ru.ru_stime.tv_usec;
  return user_us + sys_us;
}

#endif /* KLT_TIMING_H */

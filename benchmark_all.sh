#!/bin/bash
################################################################################
# KLT Feature Tracker - Complete 4-Version Benchmark Suite
# Compiles and benchmarks V1 (CPU), V2 (CUDA), V3 (Optimized CUDA), V4 (OpenACC)
# Generates comprehensive performance analysis with graphs and tables
################################################################################

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "════════════════════════════════════════════════════════════════════════════"
echo "  KLT Feature Tracker - 4-Version Performance Benchmark"
echo "  Processing: 10 frames × 4K images (3840×2160) × 150 features"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

# Check for NVIDIA GPU
echo -e "${BLUE}[1/8] Checking GPU availability...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}✗ nvidia-smi not found. NVIDIA GPU required!${NC}"
    exit 1
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo -e "${GREEN}✓ GPU detected${NC}"
echo ""

# Check for required compilers
echo -e "${BLUE}[2/8] Checking compilers...${NC}"
COMPILERS_OK=true

if ! command -v gcc &> /dev/null; then
    echo -e "${RED}✗ gcc not found${NC}"
    COMPILERS_OK=false
else
    echo -e "${GREEN}✓ gcc: $(gcc --version | head -1)${NC}"
fi

if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}✗ nvcc not found (required for V2/V3)${NC}"
    COMPILERS_OK=false
else
    echo -e "${GREEN}✓ nvcc: $(nvcc --version | grep release)${NC}"
fi

if ! command -v nvc &> /dev/null; then
    echo -e "${YELLOW}⚠ nvc not found (required for V4 OpenACC)${NC}"
    echo -e "${YELLOW}  Install NVIDIA HPC SDK: https://developer.nvidia.com/hpc-sdk${NC}"
    COMPILERS_OK=false
else
    echo -e "${GREEN}✓ nvc: $(nvc --version | head -1)${NC}"
fi

if [ "$COMPILERS_OK" = false ]; then
    echo -e "${RED}✗ Missing required compilers. Exiting.${NC}"
    exit 1
fi
echo ""

# Clean previous builds and results
echo -e "${BLUE}[3/8] Cleaning previous builds...${NC}"
make -C V1 clean &>/dev/null || true
make -C V2 clean &>/dev/null || true
make -C V3 clean &>/dev/null || true
make -C V4 clean &>/dev/null || true
rm -f timing_results.csv benchmark_results.txt
echo -e "${GREEN}✓ Clean complete${NC}"
echo ""

# Build all versions
echo -e "${BLUE}[4/8] Building all versions...${NC}"
BUILD_LOG="build_log.txt"
> $BUILD_LOG

echo "════════════════════════════════════════════════════════════════════════════"
echo "Building V1 (CPU - gcc)"
echo "════════════════════════════════════════════════════════════════════════════"
if make -C V1 lib example3 >> $BUILD_LOG 2>&1; then
    echo -e "${GREEN}✓ V1 (CPU) built successfully${NC}"
else
    echo -e "${RED}✗ V1 build failed. Check $BUILD_LOG${NC}"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "Building V2 (GPU CUDA - nvcc)"
echo "════════════════════════════════════════════════════════════════════════════"
if make -C V2 MODE=gpu lib example3 >> $BUILD_LOG 2>&1; then
    echo -e "${GREEN}✓ V2 (CUDA) built successfully${NC}"
else
    echo -e "${RED}✗ V2 build failed. Check $BUILD_LOG${NC}"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "Building V3 (GPU Optimized CUDA - nvcc)"
echo "════════════════════════════════════════════════════════════════════════════"
if make -C V3 MODE=gpu lib example3 >> $BUILD_LOG 2>&1; then
    echo -e "${GREEN}✓ V3 (Optimized CUDA) built successfully${NC}"
else
    echo -e "${RED}✗ V3 build failed. Check $BUILD_LOG${NC}"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "Building V4 (GPU OpenACC - nvc)"
echo "════════════════════════════════════════════════════════════════════════════"
cd V4
make clean >> ../$BUILD_LOG 2>&1

# Compile V4 library with OpenACC
echo "  Compiling OpenACC library..."
OPENACC_FLAGS="-O3 -DNDEBUG -DKLT_PROFILE -pg -DUSE_OPENACC -acc=gpu -Minfo=accel,inline"
for src in convolve.c error.c pnmio.c pyramid.c selectGoodFeatures.c storeFeatures.c trackFeatures.c klt.c klt_util.c writeFeatures.c; do
    echo "    - $src"
    nvc -c $OPENACC_FLAGS $src >> ../$BUILD_LOG 2>&1
done

ar ruv libklt.a *.o >> ../$BUILD_LOG 2>&1
ranlib libklt.a >> ../$BUILD_LOG 2>&1

echo "  Compiling example3..."
nvc $OPENACC_FLAGS -o example3 example3.c -L. -lklt -lm >> ../$BUILD_LOG 2>&1

cd ..
echo -e "${GREEN}✓ V4 (OpenACC) built successfully${NC}"
echo ""

# Verify executables
echo -e "${BLUE}[5/8] Verifying executables...${NC}"
for version in V1 V2 V3 V4; do
    if [ -f "$version/example3" ]; then
        echo -e "${GREEN}✓ $version/example3 exists${NC}"
    else
        echo -e "${RED}✗ $version/example3 not found${NC}"
        exit 1
    fi
done
echo ""

# Run benchmarks
echo -e "${BLUE}[6/8] Running benchmarks (5 iterations each)...${NC}"
ITERATIONS=5

echo "════════════════════════════════════════════════════════════════════════════"
echo "Benchmarking V1 (CPU)"
echo "════════════════════════════════════════════════════════════════════════════"
V1_TIMES=()
for i in $(seq 1 $ITERATIONS); do
    echo -n "  Iteration $i... "
    START=$(date +%s.%N)
    (cd V1 && ./example3 > /dev/null 2>&1)
    END=$(date +%s.%N)
    TIME=$(echo "$END - $START" | bc)
    V1_TIMES+=($TIME)
    printf "%.3fs\n" $TIME
done
V1_AVG=$(echo "${V1_TIMES[@]}" | awk '{sum=0; for(i=1;i<=NF;i++)sum+=$i; print sum/NF}')
printf "${GREEN}✓ V1 Average: %.3fs${NC}\n" $V1_AVG

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "Benchmarking V2 (CUDA)"
echo "════════════════════════════════════════════════════════════════════════════"
V2_TIMES=()
for i in $(seq 1 $ITERATIONS); do
    echo -n "  Iteration $i... "
    START=$(date +%s.%N)
    (cd V2 && ./example3 > /dev/null 2>&1)
    END=$(date +%s.%N)
    TIME=$(echo "$END - $START" | bc)
    V2_TIMES+=($TIME)
    printf "%.3fs\n" $TIME
done
V2_AVG=$(echo "${V2_TIMES[@]}" | awk '{sum=0; for(i=1;i<=NF;i++)sum+=$i; print sum/NF}')
V2_SPEEDUP=$(echo "scale=2; $V1_AVG / $V2_AVG" | bc)
printf "${GREEN}✓ V2 Average: %.3fs (${V2_SPEEDUP}x speedup)${NC}\n" $V2_AVG

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "Benchmarking V3 (Optimized CUDA)"
echo "════════════════════════════════════════════════════════════════════════════"
V3_TIMES=()
for i in $(seq 1 $ITERATIONS); do
    echo -n "  Iteration $i... "
    START=$(date +%s.%N)
    (cd V3 && ./example3 > /dev/null 2>&1)
    END=$(date +%s.%N)
    TIME=$(echo "$END - $START" | bc)
    V3_TIMES+=($TIME)
    printf "%.3fs\n" $TIME
done
V3_AVG=$(echo "${V3_TIMES[@]}" | awk '{sum=0; for(i=1;i<=NF;i++)sum+=$i; print sum/NF}')
V3_SPEEDUP=$(echo "scale=2; $V1_AVG / $V3_AVG" | bc)
printf "${GREEN}✓ V3 Average: %.3fs (${V3_SPEEDUP}x speedup)${NC}\n" $V3_AVG

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "Benchmarking V4 (OpenACC)"
echo "════════════════════════════════════════════════════════════════════════════"
V4_TIMES=()
for i in $(seq 1 $ITERATIONS); do
    echo -n "  Iteration $i... "
    START=$(date +%s.%N)
    (cd V4 && ./example3 > /dev/null 2>&1)
    END=$(date +%s.%N)
    TIME=$(echo "$END - $START" | bc)
    V4_TIMES+=($TIME)
    printf "%.3fs\n" $TIME
done
V4_AVG=$(echo "${V4_TIMES[@]}" | awk '{sum=0; for(i=1;i<=NF;i++)sum+=$i; print sum/NF}')
V4_SPEEDUP=$(echo "scale=2; $V1_AVG / $V4_AVG" | bc)
printf "${GREEN}✓ V4 Average: %.3fs (${V4_SPEEDUP}x speedup)${NC}\n" $V4_AVG

echo ""

# Generate detailed analysis report
echo -e "${BLUE}[7/8] Generating analysis report...${NC}"

cat > benchmark_results.txt << EOF
================================================================================
KLT FEATURE TRACKER - PERFORMANCE ANALYSIS REPORT
================================================================================
Generated: $(date)
GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
Dataset: 10 frames × 4K images (3840×2160 pixels) × 150 feature points
Iterations: $ITERATIONS per version
================================================================================

EXECUTION TIME RESULTS:
--------------------------------------------------------------------------------
Version  | Compiler | Type              | Avg Time (s) | Speedup vs V1
---------|----------|-------------------|--------------|---------------
V1       | gcc      | CPU Baseline      | $(printf "%.3f" $V1_AVG)        | 1.00x
V2       | nvcc     | GPU (CUDA)        | $(printf "%.3f" $V2_AVG)        | ${V2_SPEEDUP}x
V3       | nvcc     | GPU (Opt. CUDA)   | $(printf "%.3f" $V3_AVG)        | ${V3_SPEEDUP}x
V4       | nvc      | GPU (OpenACC)     | $(printf "%.3f" $V4_AVG)        | ${V4_SPEEDUP}x
================================================================================

INDIVIDUAL ITERATION TIMES:
--------------------------------------------------------------------------------
EOF

echo "V1 (CPU):     ${V1_TIMES[@]}" >> benchmark_results.txt
echo "V2 (CUDA):    ${V2_TIMES[@]}" >> benchmark_results.txt
echo "V3 (Opt):     ${V3_TIMES[@]}" >> benchmark_results.txt
echo "V4 (OpenACC): ${V4_TIMES[@]}" >> benchmark_results.txt

cat >> benchmark_results.txt << EOF

================================================================================
PERFORMANCE ANALYSIS:
--------------------------------------------------------------------------------
Best GPU Implementation: $(echo -e "V2\nV3\nV4" | while read v; do
    var="${v}_AVG"
    echo "$v ${!var}"
done | sort -k2 -n | head -1 | awk '{print $1}')

Maximum Speedup: $(echo -e "$V2_SPEEDUP\n$V3_SPEEDUP\n$V4_SPEEDUP" | sort -rn | head -1)x

Efficiency Comparison (relative to best GPU):
EOF

BEST_GPU=$(echo -e "$V2_AVG $V3_AVG $V4_AVG" | tr ' ' '\n' | sort -n | head -1)
V2_EFF=$(echo "scale=1; 100 * $BEST_GPU / $V2_AVG" | bc)
V3_EFF=$(echo "scale=1; 100 * $BEST_GPU / $V3_AVG" | bc)
V4_EFF=$(echo "scale=1; 100 * $BEST_GPU / $V4_AVG" | bc)

cat >> benchmark_results.txt << EOF
  V2 (CUDA):         ${V2_EFF}%
  V3 (Optimized):    ${V3_EFF}%
  V4 (OpenACC):      ${V4_EFF}%

================================================================================
DETAILED TIMING BREAKDOWN (from timing_results.csv):
--------------------------------------------------------------------------------
EOF

if [ -f "timing_results.csv" ]; then
    cat timing_results.csv >> benchmark_results.txt
fi

cat >> benchmark_results.txt << EOF

================================================================================
IMPLEMENTATION NOTES:
--------------------------------------------------------------------------------
V1: CPU baseline using standard C with gcc compiler
V2: GPU acceleration with basic CUDA kernels for convolution and tracking
V3: Optimized CUDA with:
    - Shared memory tiling for convolutions
    - Constant memory for kernel coefficients
    - Pinned host memory for faster transfers
    - Persistent device memory allocation
    - CUDA streams for async operations
V4: OpenACC directives for automatic GPU offloading:
    - #pragma acc parallel loop for data parallelism
    - #pragma acc data for optimized memory management
    - Compiler-driven optimization with nvc

================================================================================
HARDWARE SPECIFICATIONS:
--------------------------------------------------------------------------------
GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
Driver Version: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
CUDA Version: $(nvcc --version | grep release | awk '{print $5}' | tr -d ',')
GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)

CPU: $(lscpu | grep "Model name" | cut -d: -f2 | xargs)
Cores: $(nproc)
RAM: $(free -h | awk '/^Mem:/ {print $2}')

================================================================================
BUILD CONFIGURATION:
--------------------------------------------------------------------------------
V1 Flags: -O3 -DNDEBUG -DKLT_PROFILE -pg
V2/V3 Flags: -O3 -DNDEBUG -DKLT_PROFILE -pg -arch=sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
V4 Flags: -O3 -DNDEBUG -DKLT_PROFILE -pg -DUSE_OPENACC -acc=gpu -Minfo=accel

================================================================================
EOF

echo -e "${GREEN}✓ Analysis report saved to: benchmark_results.txt${NC}"
echo ""

# Generate Python visualization script
echo -e "${BLUE}[8/8] Generating visualization...${NC}"

cat > visualize_results.py << 'PYEOF'
#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import sys

# Data from benchmark
versions = ['V1\n(CPU)', 'V2\n(CUDA)', 'V3\n(Opt)', 'V4\n(OpenACC)']
PYEOF

cat >> visualize_results.py << PYEOF
times = [$V1_AVG, $V2_AVG, $V3_AVG, $V4_AVG]
speedups = [1.0, $V2_SPEEDUP, $V3_SPEEDUP, $V4_SPEEDUP]
PYEOF

cat >> visualize_results.py << 'PYEOF'
compilers = ['gcc', 'nvcc', 'nvcc', 'nvc']
types = ['CPU', 'GPU\n(CUDA)', 'GPU\n(Opt)', 'GPU\n(OpenACC)']

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Title
fig.suptitle('KLT Feature Tracker: 4-Version Performance Analysis\n' + 
             '10 Frames × 4K Images (3840×2160) × 150 Features',
             fontsize=16, fontweight='bold')

# 1. Execution Time Bar Chart
ax1 = fig.add_subplot(gs[0, :2])
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
bars1 = ax1.bar(range(len(versions)), times, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
ax1.set_title('Execution Time Comparison', fontsize=13, fontweight='bold')
ax1.set_xticks(range(len(versions)))
ax1.set_xticklabels(versions, fontsize=11)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

for i, (bar, time) in enumerate(zip(bars1, times)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{time:.3f}s',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# 2. Speedup Chart
ax2 = fig.add_subplot(gs[0, 2])
bars2 = ax2.bar(range(len(versions)), speedups, color=colors, edgecolor='black', linewidth=1.5)
ax2.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.7, label='Baseline')
ax2.set_ylabel('Speedup Factor', fontsize=12, fontweight='bold')
ax2.set_title('Speedup vs V1', fontsize=13, fontweight='bold')
ax2.set_xticks(range(len(versions)))
ax2.set_xticklabels(versions, fontsize=11)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.legend(loc='upper left', fontsize=9)

for i, (bar, speedup) in enumerate(zip(bars2, speedups)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{speedup:.2f}x',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# 3. Performance Comparison (horizontal bars)
ax3 = fig.add_subplot(gs[1, :])
y_pos = np.arange(len(versions))
bars3 = ax3.barh(y_pos, times, color=colors, edgecolor='black', linewidth=1.5)
ax3.set_yticks(y_pos)
ax3.set_yticklabels(versions, fontsize=11)
ax3.set_xlabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
ax3.set_title('Performance Ranking (Lower is Better)', fontsize=13, fontweight='bold')
ax3.grid(axis='x', alpha=0.3, linestyle='--')
ax3.invert_yaxis()

for i, (bar, time, speedup) in enumerate(zip(bars3, times, speedups)):
    width = bar.get_width()
    ax3.text(width, bar.get_y() + bar.get_height()/2.,
             f'  {time:.3f}s ({speedup:.2f}x)',
             ha='left', va='center', fontsize=10, fontweight='bold')

# 4. Summary Table
ax4 = fig.add_subplot(gs[2, :2])
ax4.axis('tight')
ax4.axis('off')

table_data = [
    ['V1', 'gcc', 'CPU', f'{times[0]:.3f}s', '1.00x', '100%'],
    ['V2', 'nvcc', 'GPU (CUDA)', f'{times[1]:.3f}s', f'{speedups[1]:.2f}x', f'{100*times[0]/times[1]:.0f}%'],
    ['V3', 'nvcc', 'GPU (Optimized)', f'{times[2]:.3f}s', f'{speedups[2]:.2f}x', f'{100*times[0]/times[2]:.0f}%'],
    ['V4', 'nvc', 'GPU (OpenACC)', f'{times[3]:.3f}s', f'{speedups[3]:.2f}x', f'{100*times[0]/times[3]:.0f}%'],
]

table = ax4.table(cellText=table_data,
                  colLabels=['Version', 'Compiler', 'Type', 'Avg Time', 'Speedup', 'Efficiency'],
                  cellLoc='center',
                  loc='center',
                  colWidths=[0.1, 0.12, 0.18, 0.15, 0.12, 0.13])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header
for i in range(6):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)

# Style rows
for i in range(1, 5):
    for j in range(6):
        table[(i, j)].set_facecolor(['#ecf0f1', '#d5dbdb', '#bdc3c7', '#95a5a6'][i-1])

# 5. Key Insights
ax5 = fig.add_subplot(gs[2, 2])
ax5.axis('off')

best_gpu_idx = times[1:].index(min(times[1:])) + 1
best_version = ['V2', 'V3', 'V4'][best_gpu_idx - 1]
max_speedup = max(speedups[1:])

insights_text = f"""
KEY INSIGHTS:

🏆 Best GPU: {best_version}
   {times[best_gpu_idx]:.3f}s

⚡ Max Speedup: {max_speedup:.2f}x
   (vs CPU baseline)

📊 GPU Advantage:
   {100*(1-min(times[1:])/times[0]):.1f}% faster

💡 Recommendation:
   {'V4 (OpenACC) - Best performance' if best_gpu_idx == 3 else 
    'V2 (CUDA) - Simple & fast' if best_gpu_idx == 1 else
    'V3 (Opt) - Advanced features'}
"""

ax5.text(0.1, 0.5, insights_text, fontsize=11, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
print('✓ Visualization saved to: benchmark_results.png')

try:
    plt.show()
except:
    print('  (Display not available, saved to file only)')
PYEOF

chmod +x visualize_results.py

if command -v python3 &> /dev/null; then
    python3 visualize_results.py
    echo -e "${GREEN}✓ Graphs generated: benchmark_results.png${NC}"
else
    echo -e "${YELLOW}⚠ Python3 not found. Run 'python3 visualize_results.py' manually to generate graphs${NC}"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo -e "${GREEN}✓✓✓ BENCHMARK COMPLETE ✓✓✓${NC}"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Results saved to:"
echo "  📄 benchmark_results.txt    - Detailed analysis report"
echo "  📊 benchmark_results.png    - Performance visualization"
echo "  📝 timing_results.csv       - Detailed timing breakdown"
echo "  🔧 build_log.txt           - Build compilation log"
echo ""
echo "Summary:"
printf "  V1 (CPU):           %.3fs  (baseline)\n" $V1_AVG
printf "  V2 (CUDA):          %.3fs  (${V2_SPEEDUP}x speedup)\n" $V2_AVG
printf "  V3 (Optimized):     %.3fs  (${V3_SPEEDUP}x speedup)\n" $V3_AVG
printf "  V4 (OpenACC):       %.3fs  (${V4_SPEEDUP}x speedup)\n" $V4_AVG
echo ""
echo "════════════════════════════════════════════════════════════════════════════"

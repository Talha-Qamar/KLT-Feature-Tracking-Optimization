#on terminal first run "make" and then "make profileX" where X is 1,2,3,4,5 and then "make graphX" to get the call graph in png format.
######################################################################
# Choose your favorite C compiler

# [CHANGE] — Added MODE selector for CPU/GPU flexibility
MODE ?= cpu

# [CHANGE] — Conditional compiler setup
ifeq ($(MODE),gpu)
    CC = nvcc
    GPUFLAG = -DUSE_GPU -lcudart
else
    CC = gcc
    GPUFLAG =
endif

######################################################################
FLAG1 = -DNDEBUG
PGFLAGS = -pg
# FLAG2 = -DKLT_USE_QSORT

CFLAGS = $(FLAG1) $(FLAG2) $(PGFLAGS) $(GPUFLAG)

EXAMPLES = example1.c example2.c example3.c example4.c example5.c
ARCH = convolve.c error.c pnmio.c pyramid.c selectGoodFeatures.c \
       storeFeatures.c trackFeatures.c klt.c klt_util.c writeFeatures.c

# GPU source files (only gpu_kernel_launches.cu contains everything)
ifeq ($(MODE),gpu)
GPU_ARCH = gpu_kernel_launches.cu
GPU_OBJS = $(GPU_ARCH:.cu=.o)
endif

LIB = -L/usr/local/lib -L/usr/lib

.SUFFIXES: .c .o .cu

all: lib $(EXAMPLES:.c=)

.c.o:
	$(CC) -c $(CFLAGS) $< -o $@

.cu.o:
	$(CC) -c $(CFLAGS) $< -o $@

# Build the library (make sure GPU objects are compiled first)
lib: $(ARCH:.c=.o) $(GPU_OBJS)
ifeq ($(MODE),gpu)
	@echo "Compiling GPU objects..."
	$(CC) -c $(CFLAGS) $(GPU_ARCH)
endif
	rm -f libklt.a
	ar ruv libklt.a $(ARCH:.c=.o) $(GPU_OBJS)
	rm -f *.o

# Build examples
example1: libklt.a
	$(CC) -O3 $(CFLAGS) -pg -o $@ $@.c -L. -lklt $(LIB) -lm

example2: libklt.a
	$(CC) -O3 $(CFLAGS) -pg -o $@ $@.c -L. -lklt $(LIB) -lm

example3: libklt.a
	$(CC) -O3 $(CFLAGS) -pg -o $@ $@.c -L. -lklt $(LIB) -lm

example4: libklt.a
	$(CC) -O3 $(CFLAGS) -pg -o $@ $@.c -L. -lklt $(LIB) -lm

example5: libklt.a
	$(CC) -O3 $(CFLAGS) -pg -o $@ $@.c -L. -lklt $(LIB) -lm

# Profiling and graph rules remain the same...
profile1: example1
	./example1
	gprof example1 gmon.out > analysis_eg1.txt

profile2: example2
	./example2
	gprof example2 gmon.out > analysis_eg2.txt

profile3: example3
	./example3
	gprof example3 gmon.out > analysis_eg3.txt

profile4: example4
	./example4
	gprof example4 gmon.out > analysis_eg4.txt

profile5: example5
	./example5
	gprof example5 gmon.out > analysis_eg5.txt

graph1: profile1
	gprof2dot analysis_eg1.txt -o callgraph_eg1.dot
	dot -Tpng callgraph_eg1.dot -o callgraph_eg1.png

graph2: profile2
	gprof2dot analysis_eg2.txt -o callgraph_eg2.dot
	dot -Tpng callgraph_eg2.dot -o callgraph_eg2.png

graph3: profile3
	gprof2dot analysis_eg3.txt -o callgraph_eg3.dot
	dot -Tpng callgraph_eg3.dot -o callgraph_eg3.png

graph4: profile4
	gprof2dot analysis_eg4.txt -o callgraph_eg4.dot
	dot -Tpng callgraph_eg4.dot -o callgraph_eg4.png

graph5: profile5
	gprof2dot analysis_eg5.txt -o callgraph_eg5.dot
	dot -Tpng callgraph_eg5.dot -o callgraph_eg5.png

depend:
	makedepend $(ARCH) $(EXAMPLES)

clean:
	rm -f *.o *.a $(EXAMPLES:.c=) *.tar *.tar.gz libklt.a \
	      feat*.ppm features.ft features.txt

# by writing make gpuX (X=0,1,2,3,4,5) it will clean, compile and execute in gpu mode.
gpu1: clean
	make MODE=gpu lib example1 && ./example1

gpu2: clean
	make MODE=gpu lib example2 && ./example2

gpu3: clean
	make MODE=gpu lib example3 && ./example3

gpu4: clean
	make MODE=gpu lib example4 && ./example4

gpu5: clean
	make MODE=gpu lib example5 && ./example5

# CPU equivalents
cpu1: clean
	make lib example1 && ./example1

cpu2: clean
	make lib example2 && ./example2

cpu3: clean
	make lib example3 && ./example3

cpu4: clean
	make lib example4 && ./example4

cpu5: clean
	make lib example5 && ./example5

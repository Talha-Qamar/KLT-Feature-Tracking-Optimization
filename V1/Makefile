#on terminal first run "make" and then "make profileX" where X is 1,2,3,4,5 and then "make graphX" to get the call graph in png format.
######################################################################
# Choose your favorite C compiler
CC = gcc

######################################################################
# -DNDEBUG prevents the assert() statements from being included in 
# the code.  If you are having problems running the code, you might 
# want to comment this line to see if an assert() statement fires.
FLAG1 = -DNDEBUG
# Enable lightweight CPU timing prints by uncommenting the line below
FLAG3 = -DKLT_PROFILE
PGFLAGS = -pg #necessery for profiling using gprof command
######################################################################
# -DKLT_USE_QSORT forces the code to use the standard qsort() 
# routine.  Otherwise it will use a quicksort routine that takes
# advantage of our specific data structure to greatly reduce the
# running time on some machines.  Uncomment this line if for some
# reason you are unhappy with the special routine.
# FLAG2 = -DKLT_USE_QSORT

######################################################################
# Add your favorite C flags here.
CFLAGS = $(FLAG1) $(FLAG2) $(FLAG3) $(PGFLAGS) #added PGFLAGS


######################################################################
# There should be no need to modify anything below this line (but
# feel free to if you want).

EXAMPLES = example1.c example2.c example3.c example4.c example5.c
ARCH = convolve.c error.c pnmio.c pyramid.c selectGoodFeatures.c \
       storeFeatures.c trackFeatures.c klt.c klt_util.c writeFeatures.c
LIB = -L/usr/local/lib -L/usr/lib

.SUFFIXES:  .c .o

all:  lib $(EXAMPLES:.c=)

.c.o:
	$(CC) -c $(CFLAGS) $<

lib: $(ARCH:.c=.o)
	rm -f libklt.a
	ar ruv libklt.a $(ARCH:.c=.o)
	rm -f *.o

#called pg flag to compile for profiling as well
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
	
#Following are commands for making profiles of all 5 examples:
#typing make profile1 will run example1 and then will create a profiled text file of that example which will have a call graph and flat profile.

profile1: example1
	./example1 #runs the image on code.
	gprof example1 gmon.out > analysis_eg1.txt #creates a an executable file "gmon.out" which has all the data for analysis but is binary in nature so it writes the data in a txt file for readibility.
#the same is for all profiling below.

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

#this will create a png file of the call graph for better understanding.
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


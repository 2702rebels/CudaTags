#CFLAGS=-I/usr/local/include/opencv4 -I.. -O3 -pg
CFLAGS=-I/usr/local/include/opencv4 -I.. -g
OBJS=tst.o

.SUFFIXES: .cu

.cu.o:
	nvcc $(CFLAGS) -c $<
	ar r libcudatags.a $@

.PHONY: libcudatags

all: libcudatags tst
#all: tst tst1

tst: $(OBJS) ../libcudatags.a
	nvcc -pg -o tst $(OBJS) -L.. -lcudatags  -L/usr/local/lib -lopencv_core -lopencv_imgcodecs

tst1: tst1.o ../libcudatags.a
	nvcc -pg -o tst1 tst1.o -L.. -lcudatags  -L/usr/local/lib -lopencv_core -lopencv_imgcodecs

libcudatags:
	make -C ..

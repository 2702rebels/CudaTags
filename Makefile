OBJS=apriltag.o apriltag_quad_thresh.o tag36h11.o

.SUFFIXES: .cu

.cu.o:
	nvcc -DNDEBUG -I. -pg -O3 -c  $<
	ar r libcudatags.a $@

.c.o:
	nvcc -I. -O3 -pg -c $<
	ar r libcudatags.a $@

.PHONY: common

all: common $(OBJS)

apriltag.o: apriltag.cu apriltag.cuh common/mempool.cuh common/mempool.cu
apriltag_quad_thresh.o: apriltag_quad_thresh.cu apriltag.cuh common/mempool.cuh common/mempool.cu common/zarray_d.cuh

common:
	make -C common

OBJS=apriltag.o apriltag_quad_thresh.o tag36h11.o

.SUFFIXES: .cu

.cu.o:
	nvcc -I. -O3 -pg -c $<
	ar r libcudatags.a $@

.c.o:
	nvcc -I. -O3 -pg -c $<
	ar r libcudatags.a $@

all: $(OBJS)

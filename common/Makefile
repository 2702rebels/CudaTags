#OBJS=cudamalloc.o g2d.o image_u8.o image_u8x3.o matd.o pnm.o svd22.o time_util.o workerpool.o zarray.o zmaxheap.o
OBJS=g2d.o image_u8.o image_u8x3.o matd.o pnm.o svd22.o time_util.o workerpool.o zarray.o zarray_d.o zmaxheap.o mempool.o

.SUFFIXES: .cu

.cu.o:
	nvcc -DNDEBUG -arch=sm_52 -dc -pg -I.. -g -c $<
	ar r ../libcudatags.a $@

.c.o:
	nvcc -pg -I.. -g -c $<
	ar r ../libcudatags.a $@

../libthf.a: $(OBJS)

zarray_d.o: zarray_d.cu zarray_d.cuh

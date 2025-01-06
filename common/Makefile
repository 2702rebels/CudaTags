#OBJS=cudamalloc.o g2d.o image_u8.o image_u8x3.o matd.o pnm.o svd22.o time_util.o workerpool.o zarray.o zmaxheap.o
OBJS=g2d.o image_u8.o image_u8x3.o matd.o pnm.o svd22.o time_util.o workerpool.o zarray.o zmaxheap.o

.SUFFIXES: .cu

.cu.o:
	nvcc -pg -I.. -O3 -c $<
	ar r ../libcudatags.a $@

.c.o:
	nvcc -pg -I.. -O3 -c $<
	ar r ../libcudatags.a $@

../libthf.a: $(OBJS)

CFLAGS=-I/usr/local/include/opencv4 -I.. -O3 -pg
OBJS=tst.o

.SUFFIXES: .cu

.cu.o:
	nvcc $(CFLAGS) -c $<
	ar r libcudatags.a $@

tst: $(OBJS) ../libcudatags.a
	nvcc -pg -o tst $(OBJS) -L.. -lcudatags  -L/usr/local/lib -lopencv_core -lopencv_imgcodecs

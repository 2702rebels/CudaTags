#include <unistd.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>

#include <string>
#include <stdexcept>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "common/image_types.cuh"
#include "common/image_u8.cuh"

#include "apriltag.cuh"
#include "tag36h11.cuh"

using namespace cv;

int main( int argc, char **argv )
{
	std::string st( "/data/terry/r00056.png" );
	if (argc !=2) {
#if 0
		printf( "Usage: %s input.png\n", argv[0] );
		return 1;
#else
#endif
	}
	else {
		st = argv[1];
	}
	Mat mIn = cv::imread( st, IMREAD_COLOR ); 
	Mat m( mIn.rows, mIn.cols, CV_8UC1 );

	uint8_t *pu8In = mIn.ptr();
	uint8_t *pu8Out = m.ptr();
	for (int r = 0; r < mIn.rows; r++) {
		for (int c = 0; c < mIn.cols; c++) {
			unsigned u= pu8In[0] + pu8In[1] + pu8In[2];
			pu8In += 3;
			*pu8Out++ = (u / 3 );
		}
	}

    apriltag_family_t *tf = tag36h11_create();
    apriltag_detector_t *td = apriltag_detector_create();
    apriltag_detector_add_family_bits(td, tf, 2);

	td->quad_decimate = 2;
	td->nthreads = 1;
	td->debug = 1;
	image_u8_t *pim = image_u8_create_stride( m.cols, m.rows, m.cols );
	pim->buf = m.ptr();

	image_u8_t *pcuda = image_u8_copy_cuda( td->pcp, pim );
	for (int i = 0; i < 1000; i++) {
		zarray_t *detections = apriltag_detector_detect(td, pcuda);
		zarray_destroy( detections );
	}
//	printf( "Detected %d\n", detections->size );
	return 0;
}


#pragma once

struct cudaPool;

#ifdef __cplusplus
extern "C" {
#endif
cudaPool *cudaPoolCreate( int nTasks, size_t stMem );
void vCheckPool( cudaPool *pcp, int nTask );
void cudaPoolReinit( cudaPool *pcp );
void cudaPoolAttachHost( cudaPool *pcp );
void cudaPoolAttachGlobal( cudaPool *pcp );

__host__ __device__ void cudaPoolFree( void *p );
__host__ __device__ void *cudaPoolMalloc( cudaPool *pcp, size_t s );
__host__ __device__ void *cudaPoolCalloc( cudaPool *pcp, size_t n, size_t s);
__host__ __device__ void *cudaPoolRealloc( cudaPool *pcp, void *p, size_t s);
#ifdef __cplusplus
};
#endif

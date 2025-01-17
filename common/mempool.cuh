#pragma once

#define CUDAPOOL_DEBUG	1

struct cudaPool;

#ifdef __cplusplus
extern "C" {
#endif

cudaPool *cudaPoolCreate( int nTasks, size_t stTaskMem, size_t stHostMem );
void vCheckPool( cudaPool *pcp, int nTask );
void cudaPoolReinit( cudaPool *pcp );

#if (!CUDAPOOL_DEBUG)
__host__ __device__ void cudaPoolFree( void *p );
__host__ __device__ void *cudaPoolMalloc( cudaPool *pcp, size_t s );
__host__ __device__ void *cudaPoolCalloc( cudaPool *pcp, size_t n, size_t s);
__host__ __device__ void *cudaPoolRealloc( cudaPool *pcp, void *p, size_t s);

void cudaPoolAttachHost( cudaPool *pcp );
void cudaPoolAttachGlobal( cudaPool *pcp );
#else
#define cudaPoolFree( p) cudaPoolFreeDbg( p, __FILE__, __LINE__ )
#define cudaPoolMalloc( pcp, s ) cudaPoolMallocDbg( pcp, s, __FILE__, __LINE__ )
#define cudaPoolCalloc( pcp, n, s ) cudaPoolCallocDbg( pcp, n, s, __FILE__, __LINE__ )
#define cudaPoolRealloc( pcp, p, s ) cudaPoolReallocDbg( pcp, p, s, __FILE__, __LINE__ )
#define cudaPoolAttachHost( pcp ) cudaPoolAttachHostDbg( pcp, __FILE__, __LINE__ )
#define cudaPoolAttachGlobal( pcp ) cudaPoolAttachGlobalDbg( pcp, __FILE__, __LINE__ )

__host__ __device__ void cudaPoolFreeDbg( void *p, char const *pszFile, int nLine );
__host__ __device__ void *cudaPoolMallocDbg( cudaPool *pcp, size_t s, char const *pszFile, int nLine );
__host__ __device__ void *cudaPoolCallocDbg( cudaPool *pcp, size_t n, size_t s, char const *pszFile, int nLine);
__host__ __device__ void *cudaPoolReallocDbg( cudaPool *pcp, void *p, size_t s, char const *pszFile, int nLine);

void cudaPoolAttachHostDbg( cudaPool *pcp, char const *pszFile, int nLine );
void cudaPoolAttachGlobalDbg( cudaPool *pcp, char const *pszFile, int nLine );
#endif

#ifdef __cplusplus
};
#endif

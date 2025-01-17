//#include "common/mempool.cuh"
#include <stdint.h>
#include <stdio.h>

#include <cuda_runtime.h>
#include "common/mempool.cuh"

#define MAX_CUDA_THREADS	128

struct MPElmnt;
struct MPool;
struct cudaPool;

struct mplistelmnt {
	MPElmnt	*pNext;
	MPElmnt	*pPrev;
};

struct MPElmnt {
	uint16_t		u16HeadStone;
	uint8_t			u8Flags;
	uint8_t			u8TID;
	uint32_t		u32Len;
	MPool			*pmp;
	mplistelmnt		mlfree;
};

struct MPFootStone {
	uint32_t	u32FootStone;
	uint32_t	u32Len;
	
};

struct mplist {
	MPElmnt *pHead;
	MPElmnt *pTail;
};
struct MPool {
	mplist		mlfree;
	size_t		stTID;
	size_t		stAllocd;
	size_t		stMaxAllocd;
};


struct cudaPool {
	size_t		m_stTaskMem;
	size_t		m_stHostMem;
	size_t		m_stTasks;
	size_t		m_stAlloc;
};


inline void _cuda_check(int v, char const *pszFunc, char const *pszFile, int line ) {
	if (v) {
		fprintf( stderr, "CUDA error at %s:%d code=%d \"%s\"\n", pszFile, line, v, pszFunc );
		exit(1);
	}
}
#define cuda_check(v) _cuda_check((v), #v, __FILE__, __LINE__ )

#define MEM_HS			0xFAED
#define MEM_FS			0xDEADBEEF

#define MEM_FREE		0x00
#define	MEM_INUSE		0x01
#define MEM_RELEASED	0x02
#define MEM_HEADBLOCK	0x04
#define MEM_TAILBLOCK	0x08

static inline __host__ __device__ void vSetFootStone( MPElmnt *p )
{
	MPFootStone *pFS = (MPFootStone *)(((uint8_t *)p) + p->u32Len - sizeof(MPFootStone));
	pFS->u32FootStone = MEM_FS;
	pFS->u32Len = p->u32Len;
}

static inline __host__ __device__ MPool *getMemPool( cudaPool *pcp, int nTask )
{
	uint8_t *pu8 = (uint8_t *)&pcp[1];
	if (nTask == 0) {
		return (MPool *)pu8;
	}
	else {
		return (MPool *)(pu8 + pcp->m_stHostMem + (nTask - 1) * pcp->m_stTaskMem);
	}
}

static void MPoolInit( MPool *p, size_t s, int tid )
{
	s -= sizeof(MPool);
	p->stTID = tid;

	// Create a tiny block at the start of the pool
	MPElmnt *pHead = (MPElmnt *)&p[1];
	pHead->pmp = p;
	pHead->u16HeadStone = MEM_HS;
	pHead->u8Flags = MEM_HEADBLOCK;
	pHead->u8TID = tid;
	pHead->u32Len = sizeof(MPElmnt) + sizeof(MPFootStone);
	vSetFootStone( pHead );
	
	// now create one at the end of the pool
	MPElmnt *pTail = (MPElmnt*)(((uint8_t*)pHead) + s - sizeof(MPElmnt) - sizeof(MPFootStone));
	pTail->pmp = p;
	pTail->u16HeadStone = MEM_HS;
	pTail->u8Flags = MEM_TAILBLOCK;
	pTail->u8TID = tid;
	pTail->u32Len = sizeof(MPElmnt) + sizeof(MPFootStone);
	vSetFootStone( pTail );

	// Advance to the middle, larget block
	pHead = (MPElmnt *)(((uint8_t *)pHead) + pHead->u32Len);
	pHead->pmp = p;
	pHead->u16HeadStone = MEM_HS;
	pHead->u8Flags = MEM_FREE;
	pHead->u8TID = tid;
	pHead->u32Len = s - 2 *(sizeof(MPElmnt) + sizeof(MPFootStone));
	pHead->mlfree.pNext = 0;
	pHead->mlfree.pPrev = 0;
	vSetFootStone( pHead );

	p->mlfree.pHead = pHead;
	p->mlfree.pTail = pHead;
}

static __host__ __device__ void vCheckBlock( MPool *pmp, MPElmnt *pel, char const *pszDesc )
{
#ifndef __CUDA_ARCH__
	if (!pszDesc)
		pszDesc = "";

	if (pel->pmp != pmp) {
		printf( "%spmp not correct\n", pszDesc );
		exit(1);
	}
	if (pel->u16HeadStone != MEM_HS) {
		printf( "%sHeadstone Invalid\n", pszDesc );
		exit(1);
	}

	MPFootStone *pfs = (MPFootStone *)(((uint8_t *)pel) + pel->u32Len - sizeof(MPFootStone));
	if (pfs->u32FootStone != MEM_FS) {
		printf( "%sFootstone Invalid\n", pszDesc );
		exit(1);
	}
	if (pfs->u32Len != pel->u32Len) {
		printf( "%sHead/Foot size Mismatch: %d/%d\n", pszDesc, pel->u32Len, pfs->u32Len );
		exit(1);
	}
#endif
}

void vCheckMPool( MPool *pmp )
{
	MPElmnt *pel = (MPElmnt *)&pmp[1];
	
	vCheckBlock( pmp, pel, "vCheckMPool:" );
	do {
		printf( "Mem %p: %d  %u\n", pel, pel->u8Flags, pel->u32Len );

		pel = (MPElmnt *)(((uint8_t *)pel) + pel->u32Len);
		if (pel->u16HeadStone != MEM_HS) {
			printf( "Incorrect HeadStone\n" );
			exit ( 1 );
		}
		vCheckBlock( pmp, pel, "vCheckMpool:" );
	} while (pel->u8Flags != MEM_TAILBLOCK);

	printf( "Free List:\n" );
	pel = pmp->mlfree.pHead;
	while( pel ) {
		printf( "Mem %p: %d  %u\n", pel, pel->u8Flags, pel->u32Len );
		vCheckBlock( pmp, pel, "vCheckMPool:" );
		pel = pel->mlfree.pNext;
	}
}

void vCheckPool( cudaPool *pcp, int nTask )
{
	printf( "Memory Pool %d:\n", nTask );
	vCheckMPool( getMemPool( pcp, nTask ) );
}

#if 0
MPool *mpInit( char const *pszFile, int nLine, size_t s )
{
	MPool *pPool;
	cudaMallocManaged( &pPool, s + sizeof(MPool), cudaMemAttachHost );
	pPool->pszFile = pszFile;
	pPool->nLine = nLine;
	pPool->sMem = s;
	pPool->sAllocd = 0;
	pPool->sMaxAllocd = 0;

	MPElmnt *pHead = (MPElmnt *)&pPool[1];
	pHead->pszFile = pszFile;
	pHead->nLine = nLine;
	pHead->pPrev = 0;
	pHead->pNext = 0;
	pHead->sLen = s - sizeof(MPool) - sizeof(MPElmnt);

	pPool->list.pHead = pHead;
	pPool->list.pTail = pHead;

	return pPool;
}
#endif

void cudaPoolReinit( cudaPool *p )
{
	// Data immediately follows Pool.
	// There is one extra task for the HOST
	MPoolInit( getMemPool( p, 0 ), p->m_stHostMem, 0 );
	for (size_t t = 1; t <= p->m_stTasks; t++) {
		MPoolInit( getMemPool( p, t ), p->m_stTaskMem, t );
	}
}

cudaPool *cudaPoolCreate( int nTasks, size_t stTaskMem, size_t stHostMem )
{
	size_t s = stTaskMem * nTasks + stHostMem + sizeof(cudaPool);
	cudaPool *p;

	cudaMallocManaged( &p, s, cudaMemAttachHost );
	p->m_stTaskMem = stTaskMem;
	p->m_stHostMem = stHostMem;
	p->m_stTasks = nTasks;
	p->m_stAlloc = s;

	cudaPoolReinit( p );

	return p;
};

// must be power of 2
#define MEM_ALIGNMENT 8

#if 0
__host__ __device__ void cudaPoolFree( void *p )
{
}
#else
__host__ __device__ void cudaPoolFree( void *p )
{
	MPElmnt *pE = (MPElmnt *)(((uint8_t *)p) - offsetof(MPElmnt, mlfree));
	if (pE->u16HeadStone != MEM_HS) {
#ifdef __CUDA_ARCH__
		__threadfence();
		asm("trap;");
#else
		fprintf( stderr, "%d Headstone Not Valid\n",__LINE__ );
		exit(1);
#endif
	}
#ifdef __CUDA_ARCH__
	unsigned uTID = threadIdx.x + 1;
#else
	unsigned uTID = 0;
#endif
	// If the current task is not the host and the membory block doesn't belong to
	// this task, just mark it as free... It will get cleaned up later...
	if (pE->u8TID != 0 && pE->u8TID != uTID)  {
		pE->u8Flags = MEM_RELEASED;
		return;
	}

	MPool *pmp = pE->pmp;
	// The memory belongs to us, so release it and merge if possible
	bool bMerged;

	do {
		bMerged = false;

		MPElmnt *pNext = (MPElmnt *)(((uint8_t *)pE) + pE->u32Len);
		if (pNext->u16HeadStone != MEM_HS) {
#ifdef __CUDA_ARCH__
			__threadfence();
			asm("trap;");
#else
			fprintf( stderr, "Headstone (next) Not Valid\n" );
			exit(1);
#endif
		}
		// If next block has been released by another thread,
		// Merge the two and try again.
		if (pNext->u8Flags == MEM_RELEASED) {
			pE->u32Len += pNext->u32Len;
			vSetFootStone( pE );
			vCheckBlock( pmp, pE, "MWNR:" );
			bMerged = true;
			continue;
		}
		
		// If previous block has been freed
		// Merge the two and try again.
		if (pNext->u8Flags == MEM_FREE) {
			pE->u32Len += pNext->u32Len;
			vSetFootStone( pE );
			vCheckBlock( pmp, pE, "MWNF:" );
			if (pNext->mlfree.pNext) {
				pNext->mlfree.pNext->mlfree.pPrev = pNext->mlfree.pPrev;
			}
			else {
				pmp->mlfree.pTail = pNext->mlfree.pPrev;
			}
			if (pNext->mlfree.pPrev) {
				pNext->mlfree.pPrev->mlfree.pNext = pNext->mlfree.pNext;
			}
			else {
				pmp->mlfree.pHead = pNext->mlfree.pNext;
			}
			bMerged = true;
			continue;
		}

		// Make sure the Footstone of the previous block is valid
		MPFootStone *pFS = (MPFootStone *)(((uint8_t *)pE) - sizeof(MPFootStone));
		if (pFS->u32FootStone != MEM_FS) {
#ifdef __CUDA_ARCH__
			__threadfence();
			asm("trap;");
#else
			fprintf( stderr, "Footstone Not Valid\n" );
			exit(1);
#endif
		}

		MPElmnt *pPrev = (MPElmnt *)(((uint8_t *)pE) - pFS->u32Len);
		if (pPrev->u16HeadStone != MEM_HS) {
#ifdef __CUDA_ARCH__
			__threadfence();
			asm("trap;");
#else
			fprintf( stderr, "%d Headstone Not Valid\n",__LINE__ );
			exit(1);
#endif
		}

		// If previous block has been released by another thread,
		// Merge the two and try again.
		if (pPrev->u8Flags == MEM_RELEASED) {
			pPrev->u32Len += pE->u32Len;
			vSetFootStone( pPrev );
			vCheckBlock( pmp, pPrev, "MWPR:" );
			pE = pPrev;
			bMerged = true;
			continue;
		}
		
		// If previous block has been freed
		// Merge the two and try again.
		if (pPrev->u8Flags == MEM_FREE) {
			if (pPrev->mlfree.pNext) {
				pPrev->mlfree.pNext->mlfree.pPrev = pPrev->mlfree.pPrev;
			}
			else {
				pmp->mlfree.pTail = pPrev->mlfree.pPrev;
			}
			if (pPrev->mlfree.pPrev) {
				pPrev->mlfree.pPrev->mlfree.pNext = pPrev->mlfree.pNext;
			}
			else {
				pmp->mlfree.pHead = pPrev->mlfree.pNext;
			}
			pPrev->u32Len += pE->u32Len;
			vSetFootStone( pPrev );
			vCheckBlock( pmp, pPrev, "MWPF:" );
			pE = pPrev;
			bMerged = true;
			continue;
		}
	} while( bMerged );

	pE->u8Flags = MEM_FREE;
	pE->mlfree.pNext = pE->pmp->mlfree.pHead;
	pE->mlfree.pPrev = 0;
	pE->pmp->mlfree.pHead = pE;
	if (pE->mlfree.pNext) {
		pE->mlfree.pNext->mlfree.pPrev = pE;
	}
	else {
		pE->pmp->mlfree.pTail = pE;
	}
}
#endif

static __host__ __device__ void *mpAlloc( MPool *p, size_t s )
{
	size_t sReq = (s + sizeof(MPElmnt) + sizeof(MPFootStone) + (MEM_ALIGNMENT -1)) & ~(MEM_ALIGNMENT - 1);
	MPElmnt *pE = p->mlfree.pHead;

	while (pE && pE->u32Len < sReq) {
		pE = pE->mlfree.pNext;
	}
	if (!pE) {
#ifdef __CUDA_ARCH__
		__threadfence();
		asm("trap;");
#else
		fprintf( stderr, "Out of mem %ld\n", s);
		vCheckMPool( p );
		exit(1);
#endif
	}

	// if block is too small to split
	if (pE->u32Len < (sReq + 2 * sizeof(MPElmnt))) {
		if (pE->mlfree.pPrev) {
			// Unlink from free list
			pE->mlfree.pPrev->mlfree.pNext = pE->mlfree.pNext;
		}
		else {
			// first element in free list, move head to next
			p->mlfree.pHead = pE->mlfree.pNext;
		}
		if (pE->mlfree.pNext) {
			// Unlink from list
			pE->mlfree.pNext->mlfree.pPrev = pE->mlfree.pPrev;
		}
		else {
			// last element in free list, move tail to prev
			p->mlfree.pTail = pE->mlfree.pPrev;
		}
		
		pE->u8Flags = MEM_INUSE;
		pE->u8TID = p->stTID;
		return (void *)(&pE->mlfree);
	}
	else {
		// We have a larger block that we wish to split
		// Take the new block from the end of the current block
		MPElmnt *pNE = (MPElmnt *)(((uint8_t *)pE) + pE->u32Len - sReq);
		pNE->pmp = p;
		pNE->u32Len = sReq;
		pNE->u16HeadStone = MEM_HS;
		pNE->u8Flags = MEM_INUSE;
		pNE->u8TID = p->stTID;
		vSetFootStone( pNE );

		pE->u32Len -= sReq;
		vSetFootStone( pE );
		
		return (void *)(&pNE->mlfree);
	}
}

__host__ __device__ void *cudaPoolMalloc( cudaPool *pcp, size_t s )
{
#ifdef __CUDA_ARCH__
	int nTID = threadIdx.x + 1;
#else
	int nTID = 0;
#endif
	MPool *pmp = getMemPool( pcp, nTID );
	return mpAlloc( pmp, s );
}

__host__ __device__ void *cudaPoolCalloc( cudaPool *pcp, size_t n, size_t s)
{
	size_t sTot = n * s;
	void *p = cudaPoolMalloc( pcp, sTot );
	memset( p, 0, sTot );
	return p;
}

__host__ __device__ void *cudaPoolRealloc( cudaPool *pcp, void *p, size_t s)
{
	if (!p) {
		return cudaPoolMalloc( pcp, s );
	}
	size_t sReq = (s + sizeof(MPElmnt) + sizeof(MPFootStone) + (MEM_ALIGNMENT -1)) & ~(MEM_ALIGNMENT - 1);

	MPElmnt *pE = (MPElmnt *)(((uint8_t *)p) - offsetof(MPElmnt, mlfree));
	if (pE->u16HeadStone != MEM_HS) {
#ifdef __CUDA_ARCH__
		__threadfence();
		asm("trap;");
#else
		fprintf( stderr, "%d Headstone Not Valid\n",__LINE__ );
		exit(1);
#endif
	}

	if (pE->u8Flags != MEM_INUSE) {
#ifdef __CUDA_ARCH__
		__threadfence();
		asm("trap;");
#else
		fprintf( stderr, "Realloc: memory is free\n" );
		exit(1);
#endif
		
	}
	if (pE->u32Len >= sReq) {
		return (void *)(&pE->mlfree);
	}

	void *pNew = cudaPoolMalloc( pcp, s );
	memcpy( pNew, p, pE->u32Len - sizeof(MPElmnt) - sizeof(MPFootStone) );

	cudaPoolFree( p );
	return pNew;

}

void cudaPoolAttachHost( cudaPool *pcp )
{
	// attach memory pools to HOST CPU
	cudaDeviceSynchronize();
	cudaStreamAttachMemAsync( 0, pcp, pcp->m_stAlloc, cudaMemAttachHost );
}

void cudaPoolAttachGlobal( cudaPool *pcp )
{
	cudaStreamAttachMemAsync( 0, pcp, pcp->m_stAlloc, cudaMemAttachGlobal );
	cudaDeviceSynchronize();
}

#if 0
__global__ void do_malloc( cudaPool *pcp )
{
	if (threadIdx.x >= MAX_CUDA_THREADS)
		return;

	for( int i = 0; i < 5; i++ )
		cudaPoolMalloc( pcp, 24 );
}

int main( int argc, char **argv )
{
	cudaStream_t main_stream = {};
	cudaStreamCreate(&main_stream);

	printf( "sizeof(size_t) = %lu\n", sizeof(size_t) );
	printf( "sizeof(MPElmnt) = %lu\n", sizeof(MPElmnt) );
	printf( "sizeof(MPElmnt*) = %lu\n", sizeof(MPElmnt *) );

	cudaPool *cp = cudaPoolCreate( 128, 768 * 1024 );

	void *grpMem[5];
	for (int i = 0; i < 5; i++) {
		grpMem[i] = cudaPoolMalloc( cp, 24 );
	}
	vCheckPool( cp, 0 );
	for (int i = 0; i < 5; i++) {
		cudaPoolFree( grpMem[i] );
		vCheckPool( cp, 0 );
	}
	cudaStreamAttachMemAsync( 0, cp, cp->m_stAlloc, cudaMemAttachGlobal );
	do_malloc<<<1,MAX_CUDA_THREADS>>>( cp );
	cudaDeviceSynchronize();
	cudaStreamAttachMemAsync( 0, cp, cp->m_stAlloc, cudaMemAttachHost );
//	cudaDeviceSynchronize();
	for (int i = 1; i <= 128; i++) {
		vCheckPool( cp, i );
	}
	return 1;
}
#endif

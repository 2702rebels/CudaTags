/* Copyright (C) 2013-2016, The Regents of The University of Michigan.
All rights reserved.
This software was developed in the APRIL Robotics Lab under the
direction of Edwin Olson, ebolson@umich.edu. This software may be
available under alternative licensing terms; contact the address above.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the Regents of The University of Michigan.
*/

#pragma once

#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>

#include "common/mempool.cuh"

#ifdef __cplusplus
extern "C" {
#endif

//#define cudaPoolCalloc( p, n, s ) ((void *)0)
/**
 * Defines a structure which acts as a resize-able array ala Java's ArrayList.
 */
typedef struct zarray_d zarray_d_t;
struct zarray_d
{
    size_t el_sz; // size of each element

    int size; // how many elements?
    int alloc; // we've allocated storage for how many elements?
    char *data;
};

/**
 * Creates and returns a variable array structure capable of holding elements of
 * the specified size. It is the caller's responsibility to call zarray_destroy()
 * on the returned array when it is no longer needed.
 */
static __host__ __device__ zarray_d_t *zarray_d_create(cudaPool *pcp, size_t el_sz)
{
//    assert(el_sz > 0);

    zarray_d_t *za = (zarray_d_t *)cudaPoolCalloc( pcp, 1, sizeof(zarray_d_t) );
//    zarray_d_t *za = (zarray_d_t *)cudaPoolMalloc( pcp, sizeof(zarray_d_t) );
    za->el_sz = el_sz;
    return za;
}

/**
 * Frees all resources associated with the variable array structure which was
 * created by zarray_create(). After calling, 'za' will no longer be valid for storage.
 */
static inline __host__ __device__ void zarray_d_destroy(cudaPool *pcp, zarray_d_t *za)
{
    if (za == NULL)
        return;

	if (za->data) {
		cudaPoolFree( za->data );
	}
	cudaPoolFree( za );
}

/** Allocate a new zarray that contains a copy of the data in the argument. **/
static inline __host__ __device__ zarray_d_t *zarray_d_copy(cudaPool *pcp, const zarray_d_t *za)
{
    assert(za != NULL);

    zarray_d_t *zb = (zarray_d_t *)cudaPoolMalloc( pcp, sizeof(zarray_d_t) );

    zb->el_sz = za->el_sz;
    zb->size = za->size;
    zb->alloc = za->alloc;
	zb->data = (char *)cudaPoolMalloc( pcp, za->alloc * za->el_sz );
    memcpy(zb->data, za->data, za->size * za->el_sz);

    return zb;
}

static inline int iceillog2_d(int v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

/**
 * Allocate a new zarray that contains a subset of the original
 * elements. NOTE: end index is EXCLUSIVE, that is one past the last
 * element you want.
 */
static __device__ inline zarray_d_t *zarray_d_copy_subset( cudaPool *pcp,
							 const zarray_d_t *za,
                             int start_idx,
                             int end_idx_exclusive)
{
    zarray_d_t *out = (zarray_d_t *)cudaPoolMalloc( pcp, sizeof(zarray_d_t) );
    out->el_sz = za->el_sz;
    out->size = end_idx_exclusive - start_idx;
    out->alloc = iceillog2_d(out->size); // round up pow 2
	out->data = (char *)cudaPoolMalloc( pcp, out->alloc * out->el_sz );
    memcpy(out->data,  za->data +(start_idx*out->el_sz), out->size*out->el_sz);
    return out;
}

/**
 * Retrieves the number of elements currently being contained by the passed
 * array, which may be different from its capacity. The index of the last element
 * in the array will be one less than the returned value.
 */
static inline __host__ __device__ int zarray_d_size(const zarray_d_t *za)
{
    assert(za != NULL);

    return za->size;
}

/**
 * Returns 1 if zarray_size(za) == 0,
 * returns 0 otherwise.
 */
/*
JUST CALL zarray_size
int zarray_isempty(const zarray_t *za)
{
    assert(za != NULL);
    if (za->size <= 0)
        return 1;
    else
        return 0;
}
*/


/**
 * Allocates enough internal storage in the supplied variable array structure to
 * guarantee that the supplied number of elements (capacity) can be safely stored.
 */
static inline __host__ __device__ void zarray_d_ensure_capacity( cudaPool *pcp, zarray_d_t *za, int capacity)
{
    assert(za != NULL);

    if (capacity <= za->alloc)
        return;
    while (za->alloc < capacity) {
        za->alloc *= 2;
        if (za->alloc < 8)
            za->alloc = 8;
    }
	
	za->data = (char *)cudaPoolRealloc( pcp, za->data, za->alloc * za->el_sz );
}

/**
 * Adds a new element to the end of the supplied array, and sets its value
 * (by copying) from the data pointed to by the supplied pointer 'p'.
 * Automatically ensures that enough storage space is available for the new element.
 */
static inline __host__ __device__ void zarray_d_add(cudaPool *pcp, zarray_d_t *za, const void *p)
{
    assert(za != NULL);
    assert(p != NULL);

    zarray_d_ensure_capacity(pcp, za, za->size + 1);

	memcpy( &za->data[za->size*za->el_sz], p, za->el_sz );
    za->size++;
}

/**
 * Retrieves the element from the supplied array located at the zero-based
 * index of 'idx' and copies its value into the variable pointed to by the pointer
 * 'p'.
 */
static inline __host__ __device__ void zarray_d_get(cudaPool *pcp, const zarray_d_t *za, int idx, void *p)
{
    assert(za != NULL);
    assert(p != NULL);
    assert(idx >= 0);

    assert(idx < za->size);
    memcpy(p, &za->data[idx*za->el_sz], za->el_sz);
}

/**
 * Similar to zarray_get(), but returns a "live" pointer to the internal
 * storage, avoiding a memcpy. This pointer is not valid across
 * operations which might move memory around (i.e. zarray_remove_value(),
 * zarray_remove_index(), zarray_insert(), zarray_sort(), zarray_clear()).
 * 'p' should be a pointer to the pointer which will be set to the internal address.
 */
inline static __host__ __device__ void zarray_d_get_volatile(cudaPool *pcp, const zarray_d_t *za, int idx, void *p)
{
    assert(za != NULL);
    assert(p != NULL);
    assert(idx >= 0);
    assert(idx < za->size);

    *((void**) p) = &za->data[idx*za->el_sz];
}

inline static __device__ __host__ void zarray_d_truncate(zarray_d_t *za, int sz)
{
   assert(za != NULL);

   assert(sz <= za->size);
   za->size = sz;
}

static inline __host__ __device__ void memMove(void *dst, void *src, size_t len)
{
	if (dst < src) {
		memcpy(dst, src, len);
	}
	else {
		char *d = (char *)dst;
		const char *s = (char const *)src;

		for (int i=len; i>0; i--) {
			d[i-1] = s[i-1];
		}
	}
}

/**
 * Removes the entry at index 'idx'.
 * If shuffle is true, the last element in the array will be placed in
 * the newly-open space; if false, the zarray is compacted.
 */
static inline __host__ __device__ void zarray_d_remove_index(zarray_d_t *za, int idx, int shuffle)
{
    assert(za != NULL);
    assert(idx >= 0);
    assert(idx < za->size);

    if (shuffle) {
        if (idx < za->size-1)
            memcpy(&za->data[idx*za->el_sz], &za->data[(za->size-1)*za->el_sz], za->el_sz);
        za->size--;
        return;
    } else {
        // size = 10, idx = 7. Should copy 2 entries (at idx=8 and idx=9).
        // size = 10, idx = 9. Should copy 0 entries.
        int ncopy = za->size - idx - 1;
        if (ncopy > 0)
            memMove(&za->data[idx*za->el_sz], &za->data[(idx+1)*za->el_sz], ncopy*za->el_sz);
        za->size--;
        return;
    }
}

/**
 * Remove the entry whose value is equal to the value pointed to by 'p'.
 * If shuffle is true, the last element in the array will be placed in
 * the newly-open space; if false, the zarray is compacted. At most
 * one element will be removed.
 *
 * Note that objects will be compared using memcmp over the full size
 * of the value. If the value is a struct that contains padding,
 * differences in the padding bytes can cause comparisons to
 * fail. Thus, it remains best practice to bzero all structs so that
 * the padding is set to zero.
 *
 * Returns the number of elements removed (0 or 1).
 */
// remove the entry whose value is equal to the value pointed to by p.
// if shuffle is true, the last element in the array will be placed in
// the newly-open space; if false, the zarray is compacted.
static inline __host__ __device__ int zarray_d_remove_value(zarray_d_t *za, const void *p, int shuffle)
{
    assert(za != NULL);
    assert(p != NULL);

    for (int idx = 0; idx < za->size; idx++) {
        if (!memcmp(p, &za->data[idx*za->el_sz], za->el_sz)) {
            zarray_d_remove_index(za, idx, shuffle);
            return 1;
        }
    }

    return 0;
}


/**
 * Creates a new entry and inserts it into the array so that it will have the
 * index 'idx' (i.e. before the item which currently has that index). The value
 * of the new entry is set to (copied from) the data pointed to by 'p'. 'idx'
 * can be one larger than the current max index to place the new item at the end
 * of the array, or zero to add it to an empty array.
 */
static inline __host__ __device__ void zarray_d_insert(cudaPool *pcp, zarray_d_t *za, int idx, const void *p)
{
    assert(za != NULL);
    assert(p != NULL);
    assert(idx >= 0);
    assert(idx <= za->size);

    zarray_d_ensure_capacity(pcp, za, za->size + 1);
    // size = 10, idx = 7. Should copy three entries (idx=7, idx=8, idx=9)
    int ncopy = za->size - idx;

    memMove(&za->data[(idx+1)*za->el_sz], &za->data[idx*za->el_sz], ncopy*za->el_sz);
    memcpy(&za->data[idx*za->el_sz], p, za->el_sz);

    za->size++;
}


/**
 * Sets the value of the current element at index 'idx' by copying its value from
 * the data pointed to by 'p'. The previous value of the changed element will be
 * copied into the data pointed to by 'outp' if it is not null.
 */
static inline __device__ void zarray_d_set(zarray_d_t *za, int idx, const void *p, void *outp)
{
    assert(za != NULL);
    assert(p != NULL);
    assert(idx >= 0);
    assert(idx < za->size);

    if (outp != NULL)
        memcpy(outp, &za->data[idx*za->el_sz], za->el_sz);

    memcpy(&za->data[idx*za->el_sz], p, za->el_sz);
}

/**
 * Calls the supplied function for every element in the array in index order.
 * The map function will be passed a pointer to each element in turn and must
 * have the following format:
 *
 * void map_function(element_type *element)
 */
static inline __device__ void zarray_d_map(zarray_d_t *za, void (*f)(void*))
{
    assert(za != NULL);
    assert(f != NULL);

    for (int idx = 0; idx < za->size; idx++)
        f(&za->data[idx*za->el_sz]);
}

/**
 * Calls the supplied function for every element in the array in index order.
 * HOWEVER values are passed to the function, not pointers to values. In the
 * case where the zarray stores object pointers, zarray_vmap allows you to
 * pass in the object's destroy function (or free) directly. Can only be used
 * with zarray's which contain pointer data. The map function should have the
 * following format:
 *
 * void map_function(element_type *element)
 */
    void zarray_d_vmap(zarray_d_t *za, void (*f)(void *));

/**
 * Removes all elements from the array and sets its size to zero. Pointers to
 * any data elements obtained i.e. by zarray_get_volatile() will no longer be
 * valid.
 */
static inline __host__ __device__ void zarray_d_clear(zarray_d_t *za)
{
    assert(za != NULL);
    za->size = 0;
}

/**
 * Determines whether any element in the array has a value which matches the
 * data pointed to by 'p'.
 *
 * Returns 1 if a match was found anywhere in the array, else 0.
 */
static inline __host__ __device__ int zarray_d_contains(const zarray_d_t *za, const void *p)
{
    assert(za != NULL);
    assert(p != NULL);

    for (int idx = 0; idx < za->size; idx++) {
        if (!memcmp(p, &za->data[idx*za->el_sz], za->el_sz)) {
            return 1;
        }
    }

    return 0;
}

/**
 * Uses qsort() to sort the elements contained by the array in ascending order.
 * Uses the supplied comparison function to determine the appropriate order.
 *
 * The comparison function will be passed a pointer to two elements to be compared
 * and should return a measure of the difference between them (see strcmp()).
 * I.e. it should return a negative number if the first element is 'less than'
 * the second, zero if they are equivalent, and a positive number if the first
 * element is 'greater than' the second. The function should have the following format:
 *
 * int comparison_function(const element_type *first, const element_type *second)
 *
 * zstrcmp() can be used as the comparison function for string elements, which
 * will call strcmp() internally.
 */
static inline __host__ __device__ void zarray_d_sort(zarray_d_t *za, int (*compar)(const void*, const void*))
{
    assert(za != NULL);
    assert(compar != NULL);
    if (za->size == 0)
        return;

    qsort(za->data, za->size, za->el_sz, compar);
}

/**
 * A comparison function for comparing strings which can be used by zarray_sort()
 * to sort arrays with char* elements.
 */
    int zstrcmp(const void * a_pp, const void * b_pp);

/**
  * Find the index of an element, or return -1 if not found. Remember that p is
  * a pointer to the element.
 **/
// returns -1 if not in array. Remember p is a pointer to the item.
static inline __host__ __device__ int zarray_d_index_of(const zarray_d_t *za, const void *p)
{
    assert(za != NULL);
    assert(p != NULL);

    for (int i = 0; i < za->size; i++) {
        if (!memcmp(p, &za->data[i*za->el_sz], za->el_sz))
            return i;
    }

    return -1;
}

/**
 * Add elements from start up to and excluding end from 'source' into 'dest'.
 * el_sz must be the same for both lists
 **/
static inline __host__ __device__ void zarray_d_add_range(cudaPool *pcp, zarray_d_t *dest, const zarray_d_t *source, int start, int end)
{
    assert(dest->el_sz == source->el_sz);
    assert(dest != NULL);
    assert(source != NULL);
    assert(start >= 0);
    assert(end <= source->size);
    if (start == end) {
        return;
    }
    assert(start < end);

    int count = end - start;
    zarray_d_ensure_capacity(pcp, dest, dest->size + count);

    memcpy(&dest->data[dest->size*dest->el_sz], &source->data[source->el_sz*start], dest->el_sz*count);
    dest->size += count;
}

#ifdef __cplusplus
}
#endif

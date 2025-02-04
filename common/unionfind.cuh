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

#include <string.h>
#include <stdint.h>
#include <stdlib.h>

#include "common/mempool.cuh"

typedef struct unionfind unionfind_t;

struct unionfind
{
    uint32_t maxid;

    // Parent node for each. Initialized to 0xffffffff
    uint32_t *parent;

    // The size of the tree excluding the root
    uint32_t *size;
};

static inline __host__ __device__ unionfind_t *unionfind_create(cudaPool *pcp, uint32_t maxid)
{
    unionfind_t *uf = (unionfind_t *)cudaPoolCalloc( pcp, 1, sizeof(unionfind_t) );
	uf->maxid = maxid;

	uf->parent = (uint32_t *)cudaPoolMalloc(pcp, (maxid+1) * sizeof(uint32_t) * 2);
	memset(uf->parent, 0xff, (maxid+1) * sizeof(uint32_t));
    uf->size = uf->parent + (maxid+1);
    memset(uf->size, 0, (maxid+1) * sizeof(uint32_t));

    return uf;
}

static inline __host__ __device__ void unionfind_destroy(cudaPool *pcp, unionfind_t *uf)
{
	cudaPoolFree(uf->parent);
	cudaPoolFree(uf);
}

/*
static inline uint32_t unionfind_get_representative(unionfind_t *uf, uint32_t id)
{
    // base case: a node is its own parent
    if (uf->parent[id] == id)
        return id;

    // otherwise, recurse
    uint32_t root = unionfind_get_representative(uf, uf->parent[id]);

    // short circuit the path. [XXX This write prevents tail recursion]
    uf->parent[id] = root;

    return root;
}
*/

// this one seems to be every-so-slightly faster than the recursive
// version above.
static inline __host__ __device__ uint32_t unionfind_get_representative(unionfind_t *uf, uint32_t id)
{
    uint32_t root = uf->parent[id];
    // unititialized node, so set to self
    if (root == 0xffffffff) {
        uf->parent[id] = id;
        return id;
    }

    // chase down the root
    while (uf->parent[root] != root) {
        root = uf->parent[root];
    }

    // go back and collapse the tree.
    while (uf->parent[id] != root) {
        uint32_t tmp = uf->parent[id];
        uf->parent[id] = root;
        id = tmp;
    }

    return root;
}

static inline __host__ __device__ uint32_t unionfind_get_set_size(unionfind_t *uf, uint32_t id)
{
    uint32_t repid = unionfind_get_representative(uf, id);
    return uf->size[repid] + 1;
}

static inline __host__ __device__ uint32_t unionfind_connect(unionfind_t *uf, uint32_t aid, uint32_t bid)
{
    uint32_t aroot = unionfind_get_representative(uf, aid);
    uint32_t broot = unionfind_get_representative(uf, bid);

    if (aroot == broot)
        return aroot;

    // we don't perform "union by rank", but we perform a similar
    // operation (but probably without the same asymptotic guarantee):
    // We join trees based on the number of *elements* (as opposed to
    // rank) contained within each tree. I.e., we use size as a proxy
    // for rank.  In my testing, it's often *faster* to use size than
    // rank, perhaps because the rank of the tree isn't that critical
    // if there are very few nodes in it.
    uint32_t asize = uf->size[aroot] + 1;
    uint32_t bsize = uf->size[broot] + 1;

    // optimization idea: We could shortcut some or all of the tree
    // that is grafted onto the other tree. Pro: those nodes were just
    // read and so are probably in cache. Con: it might end up being
    // wasted effort -- the tree might be grafted onto another tree in
    // a moment!
    if (asize > bsize) {
        uf->parent[broot] = aroot;
        uf->size[aroot] += bsize;
        return aroot;
    } else {
        uf->parent[aroot] = broot;
        uf->size[broot] += asize;
        return broot;
    }
}

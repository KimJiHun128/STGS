///*
// * Copyright (C) 2023, Inria
// * GRAPHDECO research group, https://team.inria.fr/graphdeco
// * All rights reserved.
// *
// * This software is free for non-commercial, research and evaluation use
// * under the terms of the LICENSE.md file.
// *
// * For inquiries contact  george.drettakis@inria.fr
// */
//#include <vector_types.h>
//#ifndef SIMPLEKNN_H_INCLUDED
//#define SIMPLEKNN_H_INCLUDED
//
//class SimpleKNN
//{
//public:
//	static void knn(int P, float3* points, float* meanDists);
//};
//
//#endif

/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef SIMPLEKNN_H_INCLUDED
#define SIMPLEKNN_H_INCLUDED

#include <torch/extension.h>   // 🔹 PyTorch tensor types
#include <vector_types.h>      // 🔹 float3 정의 (CUDA)
#include <cuda_runtime.h>      // 🔹 CUDA 관련 타입 보장

class SimpleKNN {
public:
    static void knn(int P, float3* points, float* meanDists);
};

#endif

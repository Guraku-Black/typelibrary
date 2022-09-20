#define CUDA_LIBRARY_KERNEL

#include "cudalibrary.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#ifndef CUDAMEMORY_H
#define CUDAMEMORY_H

long cudaMemoryResetDevice()
{
    cudaError_t cudaStatus;

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaDeviceReset failed! %s\n", cudaGetErrorString(cudaStatus));

        return 0;
    }

    return 1;
}

long cudaMemorySetDevice(unsigned long deviceID)
{
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(deviceID);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaSetDevice failed! %s\n", cudaGetErrorString(cudaStatus));

        return 0;
    }

    return 1;
}

long cudaMemoryAllocateMemory(void** result, unsigned long length)
{
    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void**)result, length);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMemoryAllocateMemory failed! %s\n", cudaGetErrorString(cudaStatus));

        return 0;
    }

    return 1;
}

long cudaMemorySetDeviceMemory(void* devicememory, void* hostmemory, unsigned long length)
{
    cudaError_t cudaStatus;

    cudaStatus = cudaMemcpy(devicememory, hostmemory, length, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMemorySetDeviceMemory failed! %s\n", cudaGetErrorString(cudaStatus));

        return 0;
    }

    return 1;
}

long cudaMemoryGetDeviceMemory(void* devicememory, void* hostmemory, unsigned long length)
{
    cudaError_t cudaStatus;

    cudaStatus = cudaMemcpy(hostmemory, devicememory, length, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMemoryGetDeviceMemory failed! %s\n", cudaGetErrorString(cudaStatus));

        return 0;
    }

    return 1;
}

long cudaMemoryCopyDeviceMemory(void* devicedestination, void* devicesource, unsigned long length)
{
    cudaError_t cudaStatus;

    cudaStatus = cudaMemcpy(devicedestination, devicesource, length, cudaMemcpyDeviceToDevice);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMemoryCopyDeviceMemory failed! %s\n", cudaGetErrorString(cudaStatus));

        return 0;
    }

    return 1;
}

long cudaMemoryDeviceSynchronize()
{
    cudaError_t cudaStatus;

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMemoryDeviceSynchronize failed! %s\n", cudaGetErrorString(cudaStatus));

        return 0;
    }

    return 1;
}

long cudaMemoryFreeMemory(void* source)
{
    cudaError_t cudaStatus;

    cudaStatus = cudaFree(source);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemoryFreeMemory failed! %s\n", cudaGetErrorString(cudaStatus));

        return 0;
    }

    return 1;
}

#endif // CUDAMEMORY_H

#ifndef CUDANEURALARRAY_H
#define CUDANEURALARRAY_H

#include "cudalibrary.h"

#include <stdlib.h>

long cudaNeuralArrayCreate(cudaNeuralArray* parent, unsigned long width, unsigned long height, unsigned long depth)
{
    cudaNeuralUnit* data;
    unsigned long   length;
    unsigned long   datalength;

    if (parent == 0)
        return 0;

    typeShapeAssign(&parent->arrayShape, width, height, depth);
    length = typeShapeGetLength(&parent->arrayShape);
    datalength = sizeof(cudaNeuralUnit) * length;

    if (cudaMemoryAllocateMemory((void**)&data, datalength))
    {
        parent->arrayLength = length;
        parent->arraySize = datalength;
        parent->arrayData = data;

        return 1;
    }

    typeShapeZero(&parent->arrayShape);
    parent->arrayLength = 0;
    parent->arraySize = 0;
    parent->arrayData = 0;

    return 0;
}

long cudaNeuralArraySetData(cudaNeuralArray* parent, cudaNeuralUnit* buffer, unsigned long buffersize)
{
    if ((parent == 0) || (buffer == 0))
        return 0;

    if (buffersize != parent->arrayLength)
        return 0;

    return cudaMemorySetDeviceMemory(parent->arrayData, buffer, buffersize * sizeof(cudaNeuralUnit));
}

long cudaNeuralArrayGetData(cudaNeuralArray* parent, cudaNeuralUnit* buffer, unsigned long buffersize)
{
    if ((parent == 0) || (buffer == 0))
        return 0;

    if (buffersize != parent->arrayLength)
        return 0;

    return cudaMemoryGetDeviceMemory(parent->arrayData, buffer, buffersize * sizeof(cudaNeuralUnit));
}

long cudaNeuralArrayCopy(cudaNeuralArray* parent, cudaNeuralArray* source)
{
    if ((parent == 0) || (source == 0))
        return 0;

    if (parent->arrayLength != source->arrayLength)
        return 0;

    return cudaMemoryCopyDeviceMemory(parent->arrayData, source->arrayData, parent->arrayLength * sizeof(cudaNeuralUnit));
}

long cudaNeuralArraySetImageRGB(cudaNeuralArray* parent, unsigned long imagewidth, unsigned long imageheight, unsigned long* imagedata)
{
    unsigned long   I, J;
    unsigned long   A, B, C;
    unsigned long   length;
    unsigned char*  color;
    cudaNeuralUnit* buffer;

    if ((parent == 0) || (imagedata == 0))
        return 0;

    if (typeShapeCompareValue(&parent->arrayShape, imagewidth, imageheight, 3) == 0)
        return 0;

    color = (unsigned char*)imagedata;
    buffer = (cudaNeuralUnit*)malloc(parent->arraySize);
    if (buffer == 0)
        return 0;

    length = imagewidth * imageheight;
    A = length * 0;
    B = length * 1;
    C = length * 2;

    J = 0;
    for (I = 0; I < length; I++)
    {
        buffer[A + I] = (cudaNeuralUnit)color[J + 0] / 255;
        buffer[B + I] = (cudaNeuralUnit)color[J + 1] / 255;
        buffer[C + I] = (cudaNeuralUnit)color[J + 2] / 255;

        J += 4;
    }

    cudaNeuralArraySetData(parent, buffer, parent->arrayLength);

    free(buffer);

    return 1;
}

double neural_maxout(double a)
{
    if (a < 0)
        return 0;

    if (a > 1)
        return 1;

    return a;
}

long cudaNeuralArrayGetImageRGB(cudaNeuralArray* parent, unsigned long imagewidth, unsigned long imageheight, unsigned long* imagedata)
{
    unsigned long   I, J;
    unsigned long   A, B, C;
    unsigned long   length;
    unsigned char* color;
    cudaNeuralUnit* buffer;

    if ((parent == 0) || (imagedata == 0))
        return 0;

    if (typeShapeCompareValue(&parent->arrayShape, imagewidth, imageheight, 3) == 0)
        return 0;

    color = (unsigned char*)imagedata;
    buffer = (cudaNeuralUnit*)malloc(parent->arraySize);
    if (buffer == 0)
        return 0;

    cudaNeuralArraySetData(parent, buffer, parent->arrayLength);

    length = imagewidth * imageheight;
    A = length * 0;
    B = length * 1;
    C = length * 2;

    J = 0;
    for (I = 0; I < length; I++)
    {
        color[J + 0] = (unsigned char)(neural_maxout(buffer[A + I]) * 255);
        color[J + 1] = (unsigned char)(neural_maxout(buffer[B + I]) * 255);
        color[J + 2] = (unsigned char)(neural_maxout(buffer[C + I]) * 255);

        J += 4;
    }

    free(buffer);

    return 1;
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelFillZero(cudaNeuralUnit* result, unsigned long length)
{
    unsigned long  I = blockDim.x * blockIdx.x + threadIdx.x;

    if (I >= length)
        return;

    result[I] = 0;
}

long cudaNeuralArrayFillZero(cudaNeuralArray* result)
{
    if (result == 0)
        return 0;

    dim3 threadsPerBlock(512);
    dim3 blocksPerGrid((result->arrayLength + threadsPerBlock.x - 1) / threadsPerBlock.x);

    kernelFillZero << < blocksPerGrid, threadsPerBlock >> > (result->arrayData, result->arrayLength);

    return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelFillOnes(cudaNeuralUnit* result, unsigned long length)
{
    unsigned long  I = blockDim.x * blockIdx.x + threadIdx.x;

    if (I >= length)
        return;

    result[I] = 1;
}

long cudaNeuralArrayFillOnes(cudaNeuralArray* result)
{
    if (result == 0)
        return 0;

    dim3 threadsPerBlock(512);
    dim3 blocksPerGrid((result->arrayLength + threadsPerBlock.x - 1) / threadsPerBlock.x);

    kernelFillOnes << < blocksPerGrid, threadsPerBlock >> > (result->arrayData, result->arrayLength);

    return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>

__global__ void kernelFillRandom(cudaNeuralUnit* result, unsigned long seed, unsigned long length)
{
    curandState    state;
    unsigned long  I;

    curand_init(seed, 0, 0, &state);

    for (I = 0; I < length; I++)
    {
        result[I] = curand_uniform(&state);
    }
}

long cudaNeuralArrayFillRandom(cudaNeuralArray* result)
{
    if (result == 0)
        return 0;

    dim3 threadsPerBlock(512);
    dim3 blocksPerGrid((result->arrayLength + threadsPerBlock.x - 1) / threadsPerBlock.x);

    kernelFillRandom << < 1, 1 >> > (result->arrayData, clock(), result->arrayLength);

    return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelAdd(cudaNeuralUnit* result, cudaNeuralUnit* value1, cudaNeuralUnit* value2, unsigned long length)
{
    unsigned long  I = blockDim.x * blockIdx.x + threadIdx.x;

    if (I >= length)
        return;

    result[I] = value1[I] + value2[I];
}

long cudaNeuralArrayAdd(cudaNeuralArray* result, cudaNeuralArray* value1, cudaNeuralArray* value2)
{
    if ((result == 0) || (value1 == 0) || (value2 == 0))
        return 0;

    if ((result->arrayLength != value1->arrayLength) ||
        (result->arrayLength != value2->arrayLength))
        return 0;

    dim3 threadsPerBlock(512);
    dim3 blocksPerGrid((result->arrayLength + threadsPerBlock.x - 1) / threadsPerBlock.x);

    kernelAdd << < blocksPerGrid, threadsPerBlock >> > (result->arrayData, value1->arrayData, value2->arrayData, result->arrayLength);

    return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelSubtract(cudaNeuralUnit* result, cudaNeuralUnit* value1, cudaNeuralUnit* value2, unsigned long length)
{
    unsigned long  I = blockDim.x * blockIdx.x + threadIdx.x;

    if (I >= length)
        return;

    result[I] = value1[I] - value2[I];
}

long cudaNeuralArraySubtract(cudaNeuralArray* result, cudaNeuralArray* value1, cudaNeuralArray* value2)
{
    if ((result == 0) || (value1 == 0) || (value2 == 0))
        return 0;

    if ((result->arrayLength != value1->arrayLength) ||
        (result->arrayLength != value2->arrayLength))
        return 0;

    dim3 threadsPerBlock(512);
    dim3 blocksPerGrid((result->arrayLength + threadsPerBlock.x - 1) / threadsPerBlock.x);

    kernelSubtract << < blocksPerGrid, threadsPerBlock >> > (result->arrayData, value1->arrayData, value2->arrayData, result->arrayLength);

    return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelMultiply(cudaNeuralUnit* result, cudaNeuralUnit* value1, cudaNeuralUnit* value2, unsigned long length)
{
    unsigned long  I = blockDim.x * blockIdx.x + threadIdx.x;

    if (I >= length)
        return;

    result[I] = value1[I] * value2[I];
}

long cudaNeuralArrayMultiply(cudaNeuralArray* result, cudaNeuralArray* value1, cudaNeuralArray* value2)
{
    if ((result == 0) || (value1 == 0) || (value2 == 0))
        return 0;

    if ((result->arrayLength != value1->arrayLength) ||
        (result->arrayLength != value2->arrayLength))
        return 0;

    dim3 threadsPerBlock(512);
    dim3 blocksPerGrid((result->arrayLength + threadsPerBlock.x - 1) / threadsPerBlock.x);

    kernelMultiply << < blocksPerGrid, threadsPerBlock >> > (result->arrayData, value1->arrayData, value2->arrayData, result->arrayLength);

    return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelDivide(cudaNeuralUnit* result, cudaNeuralUnit* value1, cudaNeuralUnit* value2, unsigned long length)
{
    unsigned long  I = blockDim.x * blockIdx.x + threadIdx.x;

    if (I >= length)
        return;

    if (value2[I] != 0)
        result[I] = value1[I] * value2[I];
    else
        result[I] = 0;
}

long cudaNeuralArrayDivide(cudaNeuralArray* result, cudaNeuralArray* value1, cudaNeuralArray* value2)
{
    if ((result == 0) || (value1 == 0) || (value2 == 0))
        return 0;

    if ((result->arrayLength != value1->arrayLength) ||
        (result->arrayLength != value2->arrayLength))
        return 0;

    dim3 threadsPerBlock(512);
    dim3 blocksPerGrid((result->arrayLength + threadsPerBlock.x - 1) / threadsPerBlock.x);

    kernelDivide << < blocksPerGrid, threadsPerBlock >> > (result->arrayData, value1->arrayData, value2->arrayData, result->arrayLength);

    return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelReverse(cudaNeuralUnit* result, cudaNeuralUnit* source, unsigned long length)
{
    unsigned long  I = blockDim.x * blockIdx.x + threadIdx.x;

    if (I >= length)
        return;

    result[I] = source[length - I - 1];
}

long cudaNeuralArrayReverse(cudaNeuralArray* result, cudaNeuralArray* source)
{
    if ((result == 0) || (source == 0))
        return 0;

    if (result->arrayLength != source->arrayLength)
        return 0;

    dim3 threadsPerBlock(512);
    dim3 blocksPerGrid((result->arrayLength + threadsPerBlock.x - 1) / threadsPerBlock.x);

    kernelReverse << < blocksPerGrid, threadsPerBlock >> > (result->arrayData, source->arrayData, result->arrayLength);

    return cudaMemoryDeviceSynchronize();
}

long cudaNeuralArrayDestroy(cudaNeuralArray* parent)
{
    if (parent == 0)
        return 0;

    cudaMemoryFreeMemory((void*)parent->arrayData);

    parent->arrayLength = 0;
    parent->arraySize = 0;
    parent->arrayData = 0;

    return 1;
}

#endif // CUDANEURALARRAY_H


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

#ifndef CUDASHAPE_H
#define CUDASHAPE_H

#include "cudalibrary.h"

void cudaShapeZero(cudaShape* parent)
{
	parent->shapeWidth = 0;
	parent->shapeHeight = 0;
	parent->shapeDepth = 0;
}

void cudaShapeAssign(cudaShape* parent, unsigned long X, unsigned long Y, unsigned long Z)
{
	parent->shapeWidth = X;
	parent->shapeHeight = Y;
	parent->shapeDepth = Z;
}

long cudaShapeCompare(cudaShape* parent, cudaShape* target)
{
	if ((parent->shapeWidth != target->shapeWidth) ||
		(parent->shapeHeight != target->shapeHeight) ||
		(parent->shapeDepth != target->shapeDepth))
		return 0;

	return 1;
}

long cudaShapeCompareValue(cudaShape* parent, unsigned long X, unsigned long Y, unsigned long Z)
{
	if ((parent->shapeWidth != X) ||
		(parent->shapeHeight != Y) ||
		(parent->shapeDepth != Z))
		return 0;

	return 1;
}

unsigned long cudaShapeGetLength(cudaShape* parent)
{
	return (parent->shapeWidth * parent->shapeHeight * parent->shapeDepth);
}

#endif // CUDASHAPE_H

#ifndef CUDAINDEXMAP_H
#define CUDAINDEXMAP_H

#include "cudalibrary.h"

long cudaIndexMapCreateCopy(cudaIndexMap* parent, unsigned long width, unsigned long height, unsigned long* source)
{
	unsigned long  length;
	unsigned long  datalength;
	unsigned long* data;

	if ((parent == 0) || (source == 0))
		return 0;

	length = width * height;
	datalength = length * sizeof(unsigned long);
	if (cudaMemoryAllocateMemory((void**)&data, datalength))
	{
		cudaMemorySetDeviceMemory(data, source, datalength);

		parent->indexWidth = width;
		parent->indexHeight = height;
		parent->indexLength = length;
		parent->indexSize = datalength;
		parent->indexData = data;

		return 1;
	}

	parent->indexWidth = 0;
	parent->indexHeight = 0;
	parent->indexLength = 0;
	parent->indexSize = 0;
	parent->indexData = 0;

	return 0;
}

long cudaIndexMapDestroy(cudaIndexMap* parent)
{
	cudaMemoryFreeMemory(parent->indexData);

	parent->indexWidth = 0;
	parent->indexHeight = 0;
	parent->indexLength = 0;
	parent->indexSize = 0;
	parent->indexData = 0;

	return 1;
}

#endif //  CUDAINDEXMAP_H

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

	cudaShapeAssign(&parent->arrayShape, width, height, depth);
	length = cudaShapeGetLength(&parent->arrayShape);
	datalength = sizeof(cudaNeuralUnit) * length;

	if (cudaMemoryAllocateMemory((void**)&data, datalength))
	{
		parent->arrayLength = length;
		parent->arraySize = datalength;
		parent->arrayData = data;

		return 1;
	}

	cudaShapeZero(&parent->arrayShape);
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
	unsigned char* color;
	cudaNeuralUnit* buffer;

	if ((parent == 0) || (imagedata == 0))
		return 0;

	if (cudaShapeCompareValue(&parent->arrayShape, imagewidth, imageheight, 3) == 0)
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

	if (cudaShapeCompareValue(&parent->arrayShape, imagewidth, imageheight, 3) == 0)
		return 0;

	color = (unsigned char*)imagedata;
	buffer = (cudaNeuralUnit*)malloc(parent->arraySize);
	if (buffer == 0)
		return 0;

	cudaNeuralArrayGetData(parent, buffer, parent->arrayLength);

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
__global__ void kernelArrayFillZero(cudaNeuralUnit* result, unsigned long length)
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

	kernelArrayFillZero << < blocksPerGrid, threadsPerBlock >> > (result->arrayData, result->arrayLength);

	return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelArrayFillOnes(cudaNeuralUnit* result, unsigned long length)
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

	kernelArrayFillOnes << < blocksPerGrid, threadsPerBlock >> > (result->arrayData, result->arrayLength);

	return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>

__global__ void kernelArrayFillRandomUniform(cudaNeuralUnit* result, unsigned long seed, unsigned long length)
{
	curandState    state;
	unsigned long  I;

	curand_init(seed, 0, 0, &state);

	for (I = 0; I < length; I++)
	{
		result[I] = curand_uniform(&state);
	}
}

long cudaNeuralArrayFillRandomUniform(cudaNeuralArray* result)
{
	if (result == 0)
		return 0;

	kernelArrayFillRandomUniform << < 1, 1 >> > (result->arrayData, clock(), result->arrayLength);

	return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelArrayFillRandomNormal(cudaNeuralUnit* result, unsigned long seed, unsigned long length)
{
	curandState    state;
	unsigned long  I;

	curand_init(seed, 0, 0, &state);

	for (I = 0; I < length; I++)
	{
		result[I] = curand_normal(&state);
	}
}

long cudaNeuralArrayFillRandomNormal(cudaNeuralArray* result)
{
	if (result == 0)
		return 0;

	kernelArrayFillRandomNormal << < 1, 1 >> > (result->arrayData, clock(), result->arrayLength);

	return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelArrayFillRandomXavier(cudaNeuralUnit* result, unsigned long seed, unsigned long fanIn, unsigned long fanOut, unsigned long length)
{
	curandState    state;
	cudaNeuralUnit value;
	cudaNeuralUnit avg;
	unsigned long  I;

	curand_init(seed, 0, 0, &state);

	for (I = 0; I < length; I++)
	{
		value = curand_normal(&state);

		avg = ((double)fanIn + fanOut) / 2;
		result[I] = value * (1.0 / sqrt(avg)); //  XAVIER
		//result[I] = value * (2.0 / sqrt((double)fanIn)); //  HE
		//result[I] = value * (1.0 / sqrt((double)fanIn)); //  LeCun 

	}
}

long cudaNeuralArrayFillRandomXavier(cudaNeuralArray* result, unsigned long fanIn, unsigned long fanOut)
{
	if (result == 0)
		return 0;

	kernelArrayFillRandomXavier << < 1, 1 >> > (result->arrayData, clock(), fanIn, fanOut, result->arrayLength);

	return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelArrayFillOneHot(cudaNeuralUnit* result, unsigned long index, unsigned long length)
{
	unsigned long  I;

	for (I = 0; I < length; I++)
	{
		if (I == index)
			result[I] = 1;
		else
			result[I] = 0;
	}
}

long cudaNeuralArrayFillOneHot(cudaNeuralArray* result, unsigned long index)
{
	if (result == 0)
		return 0;

	kernelArrayFillOneHot << < 1, 1 >> > (result->arrayData, index, result->arrayLength);

	return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelArrayAdd(cudaNeuralUnit* result, cudaNeuralUnit* value1, cudaNeuralUnit* value2, unsigned long length)
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

	kernelArrayAdd << < blocksPerGrid, threadsPerBlock >> > (result->arrayData, value1->arrayData, value2->arrayData, result->arrayLength);

	return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelArrayAddValue(cudaNeuralUnit* result, cudaNeuralUnit* value1, cudaNeuralUnit value2, unsigned long length)
{
	unsigned long  I = blockDim.x * blockIdx.x + threadIdx.x;

	if (I >= length)
		return;

	result[I] = value1[I] + value2;
}

long cudaNeuralArrayAddValue(cudaNeuralArray* result, cudaNeuralArray* value1, cudaNeuralUnit value2)
{
	if ((result == 0) || (value1 == 0))
		return 0;

	if (result->arrayLength != value1->arrayLength)
		return 0;

	dim3 threadsPerBlock(512);
	dim3 blocksPerGrid((result->arrayLength + threadsPerBlock.x - 1) / threadsPerBlock.x);

	kernelArrayAddValue << < blocksPerGrid, threadsPerBlock >> > (result->arrayData, value1->arrayData, value2, result->arrayLength);

	return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelArraySubtract(cudaNeuralUnit* result, cudaNeuralUnit* value1, cudaNeuralUnit* value2, unsigned long length)
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

	kernelArraySubtract << < blocksPerGrid, threadsPerBlock >> > (result->arrayData, value1->arrayData, value2->arrayData, result->arrayLength);

	return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelArraySubtractValue(cudaNeuralUnit* result, cudaNeuralUnit* value1, cudaNeuralUnit value2, unsigned long length)
{
	unsigned long  I = blockDim.x * blockIdx.x + threadIdx.x;

	if (I >= length)
		return;

	result[I] = value1[I] - value2;
}

long cudaNeuralArraySubtractValue(cudaNeuralArray* result, cudaNeuralArray* value1, cudaNeuralUnit value2)
{
	if ((result == 0) || (value1 == 0))
		return 0;

	if (result->arrayLength != value1->arrayLength)
		return 0;

	dim3 threadsPerBlock(512);
	dim3 blocksPerGrid((result->arrayLength + threadsPerBlock.x - 1) / threadsPerBlock.x);

	kernelArraySubtractValue << < blocksPerGrid, threadsPerBlock >> > (result->arrayData, value1->arrayData, value2, result->arrayLength);

	return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelArrayMultiply(cudaNeuralUnit* result, cudaNeuralUnit* value1, cudaNeuralUnit* value2, unsigned long length)
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

	kernelArrayMultiply << < blocksPerGrid, threadsPerBlock >> > (result->arrayData, value1->arrayData, value2->arrayData, result->arrayLength);

	return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelArrayMultiplyValue(cudaNeuralUnit* result, cudaNeuralUnit* value1, cudaNeuralUnit value2, unsigned long length)
{
	unsigned long  I = blockDim.x * blockIdx.x + threadIdx.x;

	if (I >= length)
		return;

	result[I] = value1[I] * value2;
}

long cudaNeuralArrayMultiplyValue(cudaNeuralArray* result, cudaNeuralArray* value1, cudaNeuralUnit value2)
{
	if ((result == 0) || (value1 == 0))
		return 0;

	if (result->arrayLength != value1->arrayLength)
		return 0;

	dim3 threadsPerBlock(512);
	dim3 blocksPerGrid((result->arrayLength + threadsPerBlock.x - 1) / threadsPerBlock.x);

	kernelArrayMultiplyValue << < blocksPerGrid, threadsPerBlock >> > (result->arrayData, value1->arrayData, value2, result->arrayLength);

	return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelArrayDivide(cudaNeuralUnit* result, cudaNeuralUnit* value1, cudaNeuralUnit* value2, unsigned long length)
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

	kernelArrayDivide << < blocksPerGrid, threadsPerBlock >> > (result->arrayData, value1->arrayData, value2->arrayData, result->arrayLength);

	return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelArrayDivideValue(cudaNeuralUnit* result, cudaNeuralUnit* value1, cudaNeuralUnit value2, unsigned long length)
{
	unsigned long  I = blockDim.x * blockIdx.x + threadIdx.x;

	if (I >= length)
		return;

	if (value2 != 0)
		result[I] = value1[I] * value2;
	else
		result[I] = 0;
}

long cudaNeuralArrayDivideValue(cudaNeuralArray* result, cudaNeuralArray* value1, cudaNeuralUnit value2)
{
	if ((result == 0) || (value1 == 0))
		return 0;

	if (result->arrayLength != value1->arrayLength)
		return 0;

	dim3 threadsPerBlock(512);
	dim3 blocksPerGrid((result->arrayLength + threadsPerBlock.x - 1) / threadsPerBlock.x);

	kernelArrayDivideValue << < blocksPerGrid, threadsPerBlock >> > (result->arrayData, value1->arrayData, value2, result->arrayLength);

	return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
__device__ unsigned long deviceShapeGetIndex(long X, long Y, long Z, unsigned long width, unsigned long height, unsigned long depth)
{
	if ((X < 0) || (X >= (long)width) ||
		(Y < 0) || (Y >= (long)height) ||
		(Z < 0) || (Z >= (long)depth))
		return CUDA_INVALID_INDEX;

	return ((width * height) * Z) + (Y * width + X);
}

__global__ void kernelArrayFlip3D(cudaNeuralUnit* result, cudaNeuralUnit* source, long A, long B, long C, unsigned long width, unsigned long height, unsigned long depth)
{
	unsigned long  I = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long  J = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned long  K = blockDim.z * blockIdx.z + threadIdx.z;
	unsigned long  resultindex;
	unsigned long  sourceindex;
	long		   X, Y, Z;

	if (I >= width)
		return;
	if (J >= height)
		return;
	if (K >= depth)
		return;

	resultindex = deviceShapeGetIndex(I, J, K, width, height, depth);

	X = A ? (width - I - 1) : I;
	Y = B ? (height - J - 1) : J;
	Z = C ? (depth - K - 1) : K;

	sourceindex = deviceShapeGetIndex(X, Y, Z, width, height, depth);

	if (sourceindex != CUDA_INVALID_INDEX)
		result[resultindex] = source[sourceindex];
	else
		result[resultindex] = 0;
}

long cudaNeuralArrayFlip3D(cudaNeuralArray* result, cudaNeuralArray* source, long A, long B, long C)
{
	if ((result == 0) || (source == 0))
		return 0;

	if (cudaShapeCompare(&result->arrayShape, &source->arrayShape) == 0)
		return 0;

	dim3 threadsPerBlock(8, 8, 8);
	dim3 blocksPerGrid(
		(result->arrayShape.shapeWidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(result->arrayShape.shapeHeight + threadsPerBlock.y - 1) / threadsPerBlock.y,
		(result->arrayShape.shapeDepth + threadsPerBlock.z - 1) / threadsPerBlock.z);

	kernelArrayFlip3D << < blocksPerGrid, threadsPerBlock >> > (
		result->arrayData,
		source->arrayData,
		A, B, C,
		result->arrayShape.shapeWidth,
		result->arrayShape.shapeHeight,
		result->arrayShape.shapeDepth);

	return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelArrayShift3D(cudaNeuralUnit* result, cudaNeuralUnit* source, long A, long B, long C, unsigned long width, unsigned long height, unsigned long depth)
{
	unsigned long  I = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long  J = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned long  K = blockDim.z * blockIdx.z + threadIdx.z;
	unsigned long  resultindex;
	unsigned long  sourceindex;
	long		   X, Y, Z;

	if (I >= width)
		return;
	if (J >= height)
		return;
	if (K >= depth)
		return;

	resultindex = deviceShapeGetIndex(I, J, K, width, height, depth);

	X = (long)I + A;
	Y = (long)J + B;
	Z = (long)K + C;

	sourceindex = deviceShapeGetIndex(X, Y, Z, width, height, depth);

	if (sourceindex != CUDA_INVALID_INDEX)
		result[resultindex] = source[sourceindex];
	else
		result[resultindex] = 0;
}

long cudaNeuralArrayShift3D(cudaNeuralArray* result, cudaNeuralArray* source, long A, long B, long C)
{
	if ((result == 0) || (source == 0))
		return 0;

	if (cudaShapeCompare(&result->arrayShape, &source->arrayShape) == 0)
		return 0;

	dim3 threadsPerBlock(8, 8, 8);
	dim3 blocksPerGrid(
		(result->arrayShape.shapeWidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(result->arrayShape.shapeHeight + threadsPerBlock.y - 1) / threadsPerBlock.y,
		(result->arrayShape.shapeDepth + threadsPerBlock.z - 1) / threadsPerBlock.z);

	kernelArrayShift3D << < blocksPerGrid, threadsPerBlock >> > (
		result->arrayData,
		source->arrayData,
		A, B, C,
		result->arrayShape.shapeWidth,
		result->arrayShape.shapeHeight,
		result->arrayShape.shapeDepth);

	return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelArrayZoom3D(cudaNeuralUnit* result, cudaNeuralUnit* source, double A, double B, double C, unsigned long width, unsigned long height, unsigned long depth)
{
	unsigned long  I = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long  J = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned long  K = blockDim.z * blockIdx.z + threadIdx.z;
	unsigned long  resultindex;
	unsigned long  sourceindex;
	long		   X, Y, Z;
	double         XC, YC, ZC;

	if (I >= width)
		return;
	if (J >= height)
		return;
	if (K >= depth)
		return;

	XC = (double)width / 2;
	YC = (double)height / 2;
	ZC = (double)depth / 2;

	resultindex = deviceShapeGetIndex(I, J, K, width, height, depth);

	X = (long)((I - XC) * A + XC);
	Y = (long)((J - YC) * B + YC);
	Z = (long)((K - ZC) * C + ZC);

	sourceindex = deviceShapeGetIndex(X, Y, Z, width, height, depth);

	if (sourceindex != CUDA_INVALID_INDEX)
		result[resultindex] = source[sourceindex];
	else
		result[resultindex] = 0;
}

long cudaNeuralArrayZoom3D(cudaNeuralArray* result, cudaNeuralArray* source, double A, double B, double C)
{
	if ((result == 0) || (source == 0))
		return 0;

	if (cudaShapeCompare(&result->arrayShape, &source->arrayShape) == 0)
		return 0;

	dim3 threadsPerBlock(8, 8, 8);
	dim3 blocksPerGrid(
		(result->arrayShape.shapeWidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(result->arrayShape.shapeHeight + threadsPerBlock.y - 1) / threadsPerBlock.y,
		(result->arrayShape.shapeDepth + threadsPerBlock.z - 1) / threadsPerBlock.z);

	kernelArrayZoom3D << < blocksPerGrid, threadsPerBlock >> > (
		result->arrayData,
		source->arrayData,
		A, B, C,
		result->arrayShape.shapeWidth,
		result->arrayShape.shapeHeight,
		result->arrayShape.shapeDepth);

	return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
__device__ const auto CUDA_PI = 3.141592653589793238462643383279502884197169399375105820974944;

__device__ double cuda_sin(double a)
{
	return sin(a / 180.0 * CUDA_PI);
}

__device__ double cuda_cos(double a)
{
	return cos(a / 180.0 * CUDA_PI);
}

__global__ void kernelArrayRotate2D(cudaNeuralUnit* result, cudaNeuralUnit* source, double angle, unsigned long width, unsigned long height, unsigned long depth)
{
	unsigned long  I = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long  J = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned long  K = blockDim.z * blockIdx.z + threadIdx.z;
	unsigned long  resultindex;
	unsigned long  sourceindex;
	long           X, Y;
	double         XC, YC;
	double         XT, YT;
	double         xsin, xcos;

	if (I >= width)
		return;
	if (J >= height)
		return;
	if (K >= depth)
		return;

	XC = (double)width / 2;
	YC = (double)height / 2;

	xsin = cuda_sin(angle);
	xcos = cuda_cos(angle);

	resultindex = deviceShapeGetIndex(I, J, K, width, height, depth);

	XT = I - XC;
	YT = J - YC;

	X = (long)((XT * xcos + YT * xsin) + XC);
	Y = (long)((YT * xcos - XT * xsin) + YC);

	sourceindex = deviceShapeGetIndex(X, Y, K, width, height, depth);

	if (sourceindex != CUDA_INVALID_INDEX)
		result[resultindex] = source[sourceindex];
	else
		result[resultindex] = 0;
}

long cudaNeuralArrayRotate2D(cudaNeuralArray* result, cudaNeuralArray* source, double angle)
{
	if ((result == 0) || (source == 0))
		return 0;

	if (cudaShapeCompare(&result->arrayShape, &source->arrayShape) == 0)
		return 0;

	dim3 threadsPerBlock(8, 8, 8);
	dim3 blocksPerGrid(
		(result->arrayShape.shapeWidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(result->arrayShape.shapeHeight + threadsPerBlock.y - 1) / threadsPerBlock.y,
		(result->arrayShape.shapeDepth + threadsPerBlock.z - 1) / threadsPerBlock.z);

	kernelArrayRotate2D << < blocksPerGrid, threadsPerBlock >> > (
		result->arrayData,
		source->arrayData,
		angle,
		result->arrayShape.shapeWidth,
		result->arrayShape.shapeHeight,
		result->arrayShape.shapeDepth);

	return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
__managed__ cudaNeuralUnit     cudaRegister1;

__global__ void kernelArrayGetMeanSquaredError(cudaNeuralUnit* source, cudaNeuralUnit* target, unsigned long length)
{
	unsigned long   I;
	cudaNeuralUnit  sum;

	sum = 0;
	for (I = 0; I < length; I++)
	{
		sum += (source[I] - target[I]) * (source[I] - target[I]);
	}

	if (length > 1)
		sum = sum / length;

	cudaRegister1 = sum;
}

long cudaNeuralArrayGetMeanSquaredError(cudaNeuralArray* source, cudaNeuralArray* target, cudaNeuralUnit* result)
{
	if ((source == 0) || (target == 0) || (result == 0))
		return 0;

	if (source->arrayLength != target->arrayLength)
		return 0;

	kernelArrayGetMeanSquaredError << < 1, 1 >> > (source->arrayData, target->arrayData, source->arrayLength);

	if (cudaMemoryDeviceSynchronize())
	{
		*result = cudaRegister1;

		return 1;

	}

	return 0;
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelArrayGetSum(cudaNeuralUnit* source, unsigned long length)
{
	unsigned long   I;
	cudaNeuralUnit  sum;

	sum = 0;
	for (I = 0; I < length; I++)
	{
		sum += source[I];
	}

	cudaRegister1 = sum;
}

long cudaNeuralArrayGetSum(cudaNeuralArray* source, cudaNeuralUnit* result)
{
	if ((source == 0) || (result == 0))
		return 0;

	kernelArrayGetSum << < 1, 1 >> > (source->arrayData, source->arrayLength);

	if (cudaMemoryDeviceSynchronize())
	{
		*result = cudaRegister1;

		return 1;
	}

	return 0;
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelArrayGetMean(cudaNeuralUnit* source, unsigned long length)
{
	unsigned long   I;
	cudaNeuralUnit  sum;

	sum = 0;
	for (I = 0; I < length; I++)
	{
		sum += source[I];
	}

	if (length > 1)
		sum = sum / length;

	cudaRegister1 = sum;
}

long cudaNeuralArrayGetMean(cudaNeuralArray* source, cudaNeuralUnit* result)
{
	if ((source == 0) || (result == 0))
		return 0;

	kernelArrayGetMean << < 1, 1 >> > (source->arrayData, source->arrayLength);

	if (cudaMemoryDeviceSynchronize())
	{
		*result = cudaRegister1;

		return 1;
	}

	return 0;
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelArrayGetMagnitude(cudaNeuralUnit* source, unsigned long length)
{
	unsigned long   I;
	cudaNeuralUnit  sum;

	sum = 0;
	for (I = 0; I < length; I++)
	{
		sum += source[I] * source[I];
	}

	cudaRegister1 = sqrt(sum);
}

long cudaNeuralArrayGetMagnitude(cudaNeuralArray* source, cudaNeuralUnit* result)
{
	if ((source == 0) || (result == 0))
		return 0;

	kernelArrayGetMagnitude << < 1, 1 >> > (source->arrayData, source->arrayLength);

	if (cudaMemoryDeviceSynchronize())
	{
		*result = cudaRegister1;

		return 1;
	}

	return 0;
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelArrayGetDotProduct(cudaNeuralUnit* source1, cudaNeuralUnit* source2, unsigned long length)
{
	unsigned long   I;
	cudaNeuralUnit  sum;

	sum = 0;
	for (I = 0; I < length; I++)
	{
		sum += source1[I] * source2[I];
	}

	cudaRegister1 = sum;
}

long cudaNeuralArrayGetDotProduct(cudaNeuralArray* source1, cudaNeuralArray* source2, cudaNeuralUnit* result)
{
	if ((source1 == 0) || (source2 == 0) || (result == 0))
		return 0;

	if (source1->arrayLength != source2->arrayLength)
		return 0;

	kernelArrayGetDotProduct << < 1, 1 >> > (source1->arrayData, source2->arrayData, source2->arrayLength);

	if (cudaMemoryDeviceSynchronize())
	{
		*result = cudaRegister1;

		return 1;
	}

	return 0;
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelArrayGetEucledeanDistance(cudaNeuralUnit* source1, cudaNeuralUnit* source2, unsigned long length)
{
	unsigned long   I;
	cudaNeuralUnit  sum;

	sum = 0;
	for (I = 0; I < length; I++)
	{
		sum += (source1[I] - source2[I]) * (source1[I] - source2[I]);
	}

	cudaRegister1 = sqrt(sum);
}

long cudaNeuralArrayGetEucledeanDistance(cudaNeuralArray* source1, cudaNeuralArray* source2, cudaNeuralUnit* result)
{
	if ((source1 == 0) || (source2 == 0) || (result == 0))
		return 0;

	if (source1->arrayLength != source2->arrayLength)
		return 0;

	kernelArrayGetEucledeanDistance << < 1, 1 >> > (source1->arrayData, source2->arrayData, source2->arrayLength);

	if (cudaMemoryDeviceSynchronize())
	{
		*result = cudaRegister1;

		return 1;
	}

	return 0;
}

//////////////////////////////////////////////////////////////////////////////////
long cudaNeuralArrayGetCosineSimilarity(cudaNeuralArray* source1, cudaNeuralArray* source2, cudaNeuralUnit* result)
{
	cudaNeuralUnit   mag1, mag2, dot;

	if ((source1 == 0) || (source2 == 0) || (result == 0))
		return 0;

	if (source1->arrayLength != source2->arrayLength)
		return 0;

	if (cudaNeuralArrayGetMagnitude(source1, &mag1))
		if (cudaNeuralArrayGetMagnitude(source2, &mag2))
			if (cudaNeuralArrayGetDotProduct(source1, source2, &dot))
			{
				*result = dot / (mag1 * mag2);

				return 1;
			}

	return 0;
}

//////////////////////////////////////////////////////////////////////////////////
long cudaNeuralArrayNormalize(cudaNeuralArray* result, cudaNeuralArray* source)
{
	cudaNeuralUnit   mag;

	if ((result == 0) || (source == 0))
		return 0;

	if (result->arrayLength != source->arrayLength)
		return 0;

	if (cudaNeuralArrayGetMagnitude(source, &mag))
		if (cudaNeuralArrayDivideValue(result, source, mag))
			{
				return 1;
			}

	return 0;
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelArrayShuffle(cudaNeuralUnit* result, unsigned long seed, unsigned long length)
{
	unsigned long   I, J;
	cudaNeuralUnit  temp;
	curandState     state;

	curand_init(seed, 0, 0, &state);

	for (I = 0; I < length; I++)
	{
		J = curand(&state);

		temp = result[I];
		result[I] = result[J];
		result[J] = temp;
	}
}

long cudaNeuralArrayShuffle(cudaNeuralArray* result)
{
	if (result == 0)
		return 0;

	kernelArrayShuffle << < 1, 1 >> > (result->arrayData, clock(), result->arrayLength);

	return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelArrayConcatenate(cudaNeuralUnit* result, cudaNeuralUnit* source1, cudaNeuralUnit* source2, unsigned long length0, unsigned long length1, unsigned long length2)
{
	unsigned long  I = blockDim.x * blockIdx.x + threadIdx.x;

	if (I >= length0)
		return;

	if (I < length1)
		result[I] = source1[I];
	else
		result[I] = source2[I - length1];
}

long cudaNeuralArrayConcatenate(cudaNeuralArray* result, cudaNeuralArray* source1, cudaNeuralArray* source2)
{
	if ((result == 0) || (source1 == 0) || (source2 == 0))
		return 0;

	if (result->arrayLength != (source1->arrayLength + source2->arrayLength))
		return 0;

	dim3 threadsPerBlock(512);
	dim3 blocksPerGrid((result->arrayLength + threadsPerBlock.x - 1) / threadsPerBlock.x);

	kernelArrayConcatenate << < blocksPerGrid, threadsPerBlock >> > (result->arrayData, source1->arrayData, source2->arrayData, result->arrayLength, source1->arrayLength, source2->arrayLength);

	return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelArraySplit(cudaNeuralUnit* result, cudaNeuralUnit* source1, cudaNeuralUnit* source2, unsigned long length0, unsigned long length1, unsigned long length2)
{
	unsigned long  I = blockDim.x * blockIdx.x + threadIdx.x;

	if (I >= length0)
		return;

	if (I < length1)
		source1[I] = result[I];
	else
		source2[I - length1] = result[I];
}

long cudaNeuralArraySplit(cudaNeuralArray* result, cudaNeuralArray* source1, cudaNeuralArray* source2)
{
	if ((result == 0) || (source1 == 0) || (source2 == 0))
		return 0;

	if (result->arrayLength != (source1->arrayLength + source2->arrayLength))
		return 0;

	dim3 threadsPerBlock(512);
	dim3 blocksPerGrid((result->arrayLength + threadsPerBlock.x - 1) / threadsPerBlock.x);

	kernelArraySplit << < blocksPerGrid, threadsPerBlock >> > (result->arrayData, source1->arrayData, source2->arrayData, result->arrayLength, source1->arrayLength, source2->arrayLength);

	return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelArrayIndexMap2DCopy(
	cudaNeuralUnit* result,
	cudaNeuralUnit* source,
	unsigned long   indexwidth,
	unsigned long   indexheight,
	unsigned long* indexmap,
	unsigned long resultlength,
	unsigned long resultcount,
	unsigned long sourcelength,
	unsigned long sourcecount)
{
	cudaNeuralUnit sum;
	unsigned long  counter;
	unsigned long  index;

	unsigned long  I = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long  J = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned long  K;

	if (I >= resultlength)
		return;

	if (J >= resultcount)
		return;

	unsigned long   resultindex = J * resultlength;
	unsigned long   sourceindex = J * sourcelength;

	sum = 0;
	counter = 0;
	K = I;

	for (J = 0; J < indexheight; J++)
	{
		index = indexmap[K];
		if ((index != CUDA_INVALID_INDEX) &&
			(index < sourcelength))
		{
			sum += source[sourceindex + index];
			counter++;
		}

		K += indexwidth;
	}

	if (counter > 1)
	{
		sum /= counter;
	}

	result[resultindex + I] = sum;
}

long cudaNeuralArrayIndexMap2DCopy(cudaNeuralArray* result, cudaNeuralArray* source, cudaIndexMap* indexmap)
{
	unsigned long   resultlength;
	unsigned long   resultcount;
	unsigned long   sourcelength;
	unsigned long   sourcecount;

	if ((result == 0) || (source == 0) || (indexmap == 0))
		return 0;

	resultlength = (result->arrayShape.shapeWidth * result->arrayShape.shapeHeight);
	resultcount = result->arrayShape.shapeDepth;
	sourcelength = (source->arrayShape.shapeWidth * source->arrayShape.shapeHeight);
	sourcecount = source->arrayShape.shapeDepth;

	if (indexmap->indexWidth != resultlength)
		return 0;

	if (resultcount != sourcecount)
		return 0;

	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid(
		(resultlength + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(resultcount + threadsPerBlock.y - 1) / threadsPerBlock.y);

	kernelArrayIndexMap2DCopy << < blocksPerGrid, threadsPerBlock >> > (
		result->arrayData,
		source->arrayData,
		indexmap->indexWidth,
		indexmap->indexHeight,
		indexmap->indexData,
		resultlength,
		resultcount,
		sourcelength,
		sourcecount);


	return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelArrayIndexMap2DPoolingForward(
	cudaNeuralUnit* result,
	cudaNeuralUnit* source,
	unsigned long   indexwidth,
	unsigned long   indexheight,
	unsigned long* indexmap,
	unsigned long resultlength,
	unsigned long resultcount,
	unsigned long sourcelength,
	unsigned long sourcecount)
{
	cudaNeuralUnit sum;
	unsigned long  index;

	unsigned long  I = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long  J = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned long  K;

	if (I >= resultlength)
		return;

	if (J >= resultcount)
		return;

	unsigned long   resultindex = J * resultlength;
	unsigned long   sourceindex = J * sourcelength;

	sum = 0;
	K = I;

	for (J = 0; J < indexheight; J++)
	{
		index = indexmap[K];
		if ((index != CUDA_INVALID_INDEX) &&
			(index < sourcelength))
		{
			if (fabsf(source[sourceindex + index]) > fabsf(sum))
				sum = source[sourceindex + index];
		}

		K += indexwidth;
	}


	result[resultindex + I] = sum;
}

long cudaNeuralArrayIndexMap2DPoolingForward(cudaNeuralArray* result, cudaNeuralArray* source, cudaIndexMap* indexmap)
{
	unsigned long   resultlength;
	unsigned long   resultcount;
	unsigned long   sourcelength;
	unsigned long   sourcecount;

	if ((result == 0) || (source == 0) || (indexmap == 0))
		return 0;

	resultlength = (result->arrayShape.shapeWidth * result->arrayShape.shapeHeight);
	resultcount = result->arrayShape.shapeDepth;
	sourcelength = (source->arrayShape.shapeWidth * source->arrayShape.shapeHeight);
	sourcecount = source->arrayShape.shapeDepth;

	if (indexmap->indexWidth != resultlength)
		return 0;

	if (resultcount != sourcecount)
		return 0;

	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid(
		(resultlength + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(resultcount + threadsPerBlock.y - 1) / threadsPerBlock.y);

	kernelArrayIndexMap2DPoolingForward << < blocksPerGrid, threadsPerBlock >> > (
		result->arrayData,
		source->arrayData,
		indexmap->indexWidth,
		indexmap->indexHeight,
		indexmap->indexData,
		resultlength,
		resultcount,
		sourcelength,
		sourcecount);

	return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelArrayIndexMap2DPoolingReverse(
	cudaNeuralUnit* result,
	cudaNeuralUnit* source,
	cudaNeuralUnit* outputs,
	unsigned long   indexwidth,
	unsigned long   indexheight,
	unsigned long* indexmap,
	unsigned long resultlength,
	unsigned long resultcount,
	unsigned long sourcelength,
	unsigned long sourcecount)
{
	cudaNeuralUnit sum;
	unsigned long  counter;
	unsigned long  index;

	unsigned long  I = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long  J = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned long  K;

	if (I >= resultlength)
		return;

	if (J >= resultcount)
		return;

	unsigned long   resultindex = J * resultlength;
	unsigned long   sourceindex = J * sourcelength;

	sum = 0;
	counter = 0;
	K = I;

	for (J = 0; J < indexheight; J++)
	{
		index = indexmap[K];
		if ((index != CUDA_INVALID_INDEX) &&
			(index < sourcelength))
		{
			if (fabsf(outputs[sourceindex + index]) > fabsf(sum))
			{
				sum = outputs[sourceindex + index];
				counter = J;
			}
		}

		K += indexwidth;
	}

	K = I;
	for (J = 0; J < indexheight; J++)
	{
		index = indexmap[K];
		if ((index != CUDA_INVALID_INDEX) &&
			(index < sourcelength))
		{
			if (counter == J)
				source[sourceindex + index] = result[resultindex + I];
			else
				source[sourceindex + index] = 0;
		}

		K += indexwidth;
	}
}

long cudaNeuralArrayIndexMap2DPoolingReverse(cudaNeuralArray* result, cudaNeuralArray* source, cudaNeuralArray* outputs, cudaIndexMap* indexmap)
{
	unsigned long   resultlength;
	unsigned long   resultcount;
	unsigned long   sourcelength;
	unsigned long   sourcecount;

	if ((result == 0) || (source == 0) || (outputs == 0) || (indexmap == 0))
		return 0;

	resultlength = (result->arrayShape.shapeWidth * result->arrayShape.shapeHeight);
	resultcount = result->arrayShape.shapeDepth;
	sourcelength = (source->arrayShape.shapeWidth * source->arrayShape.shapeHeight);
	sourcecount = source->arrayShape.shapeDepth;

	if (indexmap->indexWidth != resultlength)
		return 0;

	if (resultcount != sourcecount)
		return 0;

	if (cudaShapeCompare(&source->arrayShape, &outputs->arrayShape))
		return 0;

	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid(
		(resultlength + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(resultcount + threadsPerBlock.y - 1) / threadsPerBlock.y);

	kernelArrayIndexMap2DPoolingReverse << < blocksPerGrid, threadsPerBlock >> > (
		result->arrayData,
		source->arrayData,
		outputs->arrayData,
		indexmap->indexWidth,
		indexmap->indexHeight,
		indexmap->indexData,
		resultlength,
		resultcount,
		sourcelength,
		sourcecount);

	return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
__device__ cudaNeuralUnit indexMap2DConvolutionValue(
	cudaNeuralUnit* source,
	cudaNeuralUnit* filter,
	unsigned long   indexwidth,
	unsigned long   indexheight,
	unsigned long* indexmap,
	unsigned long  indexoffset,
	unsigned long  sourceindex,
	unsigned long  sourcelength,
	unsigned long  filterindex,
	unsigned long  filterlength)
{
	unsigned long  I, J;
	unsigned long  index;
	cudaNeuralUnit sum;

	sum = 0;
	J = indexoffset;
	for (I = 0; I < indexheight; I++)
	{
		index = indexmap[J];
		if ((index != CUDA_INVALID_INDEX) &&
			(index < sourcelength))
		{
			sum += source[sourceindex + index] * filter[filterindex + I];
		}

		J += indexwidth;
	}

	return sum;
}

__global__ void kernelArrayIndexMap2DConvolutionForward(
	cudaNeuralUnit* result,
	cudaNeuralUnit* source,
	cudaNeuralUnit* filter,
	unsigned long   indexwidth,
	unsigned long   indexheight,
	unsigned long* indexmap,
	unsigned long resultlength,
	unsigned long resultcount,
	unsigned long sourcelength,
	unsigned long sourcecount,
	unsigned long filterlength,
	unsigned long filtercount)
{
	unsigned long  I;
	unsigned long  J = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long  K = blockDim.y * blockIdx.y + threadIdx.y;
	cudaNeuralUnit sum;

	if (J >= resultcount)
		return;
	if (K >= resultlength)
		return;

	unsigned long   resultindex = J * resultlength;
	unsigned long   sourceindex = 0;
	unsigned long   filterindex = (J * sourcecount) * filterlength;

	sum = 0;
	for (I = 0; I < sourcecount; I++)
	{
		sum += indexMap2DConvolutionValue(
			source,
			filter,
			indexwidth,
			indexheight,
			indexmap,
			K,
			sourceindex,
			sourcelength,
			filterindex,
			filterlength);

		sourceindex += sourcelength;
		filterindex += filterlength;
	}

	if ((sum < -3) || (sum > 3))
	{
		sum = 0;
		filterindex = (J * sourcecount) * filterlength;
		for (I = 0; I < sourcecount; I++)
		{
			for (J = 0; J < filterlength; J++)
				filter[filterindex + J] *= 0.9;

			filterindex += filterlength;
		}
	}

	result[resultindex + K] = sum;
}

long cudaNeuralArrayIndexMap2DConvolutionForward(cudaNeuralArray* result, cudaNeuralArray* source, cudaNeuralArray* filter, cudaIndexMap* indexmap)
{
	unsigned long   resultlength;
	unsigned long   resultcount;
	unsigned long   sourcelength;
	unsigned long   sourcecount;
	unsigned long   filterlength;
	unsigned long   filtercount;

	if ((result == 0) || (source == 0) || (filter == 0) || (indexmap == 0))
		return 0;

	resultlength = (result->arrayShape.shapeWidth * result->arrayShape.shapeHeight);
	resultcount = result->arrayShape.shapeDepth;
	sourcelength = (source->arrayShape.shapeWidth * source->arrayShape.shapeHeight);
	sourcecount = source->arrayShape.shapeDepth;
	filterlength = (filter->arrayShape.shapeWidth * filter->arrayShape.shapeHeight);
	filtercount = filter->arrayShape.shapeDepth;

	if (filtercount != (resultcount * sourcecount))
		return 0;

	if ((indexmap->indexWidth != resultlength) ||
		(indexmap->indexHeight != filterlength))
		return 0;

	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid(
		(resultcount + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(resultlength + threadsPerBlock.y - 1) / threadsPerBlock.y);

	kernelArrayIndexMap2DConvolutionForward << < blocksPerGrid, threadsPerBlock >> > (
		result->arrayData,
		source->arrayData,
		filter->arrayData,
		indexmap->indexWidth,
		indexmap->indexHeight,
		indexmap->indexData,
		resultlength,
		resultcount,
		sourcelength,
		sourcecount,
		filterlength,
		filtercount);

	return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelArrayIndexMap2DConvolutionReverse(
	cudaNeuralUnit* result,
	cudaNeuralUnit* source,
	cudaNeuralUnit* filter,
	unsigned long   indexwidth,
	unsigned long   indexheight,
	unsigned long* indexmap,
	unsigned long resultlength,
	unsigned long resultcount,
	unsigned long sourcelength,
	unsigned long sourcecount,
	unsigned long filterlength,
	unsigned long filtercount)
{
	unsigned long  I = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long  J;
	unsigned long  K = blockDim.y * blockIdx.y + threadIdx.y;
	cudaNeuralUnit sum;

	if (I >= sourcecount)
		return;
	if (K >= sourcelength)
		return;

	unsigned long   resultindex = 0;
	unsigned long   sourceindex = I * sourcelength;
	unsigned long   filterindex = I * filterlength;
	unsigned long   filterjump = sourcecount * filterlength;

	sum = 0;
	for (J = 0; J < resultcount; J++)
	{
		sum += indexMap2DConvolutionValue(
			result,
			filter,
			indexwidth,
			indexheight,
			indexmap,
			K,
			resultindex,
			resultlength,
			filterindex,
			filterlength);

		resultindex += resultlength;
		filterindex += filterjump;
	}

	source[sourceindex + K] = sum;
}

long cudaNeuralArrayIndexMap2DConvolutionReverse(cudaNeuralArray* result, cudaNeuralArray* source, cudaNeuralArray* filter, cudaIndexMap* indexmap)
{
	unsigned long   resultlength;
	unsigned long   resultcount;
	unsigned long   sourcelength;
	unsigned long   sourcecount;
	unsigned long   filterlength;
	unsigned long   filtercount;

	if ((result == 0) || (source == 0) || (filter == 0) || (indexmap == 0))
		return 0;

	resultlength = (result->arrayShape.shapeWidth * result->arrayShape.shapeHeight);
	resultcount = result->arrayShape.shapeDepth;
	sourcelength = (source->arrayShape.shapeWidth * source->arrayShape.shapeHeight);
	sourcecount = source->arrayShape.shapeDepth;
	filterlength = (filter->arrayShape.shapeWidth * filter->arrayShape.shapeHeight);
	filtercount = filter->arrayShape.shapeDepth;

	if (filtercount != (resultcount * sourcecount))
		return 0;

	if ((indexmap->indexWidth != sourcelength) ||
		(indexmap->indexHeight != filterlength))
		return 0;

	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid(
		(sourcecount + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(sourcelength + threadsPerBlock.y - 1) / threadsPerBlock.y);

	kernelArrayIndexMap2DConvolutionReverse << < blocksPerGrid, threadsPerBlock >> > (
		result->arrayData,
		source->arrayData,
		filter->arrayData,
		indexmap->indexWidth,
		indexmap->indexHeight,
		indexmap->indexData,
		resultlength,
		resultcount,
		sourcelength,
		sourcecount,
		filterlength,
		filtercount);

	return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelArrayIndexMap2DConvolutionGetDerivatives(
	cudaNeuralUnit* result,
	cudaNeuralUnit* source,
	cudaNeuralUnit* filter,
	unsigned long   indexwidth,
	unsigned long   indexheight,
	unsigned long* indexmap,
	unsigned long resultlength,
	unsigned long resultcount,
	unsigned long sourcelength,
	unsigned long sourcecount,
	unsigned long filterlength,
	unsigned long filtercount)
{
	unsigned long  I = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long  J = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned long  K = blockDim.z * blockIdx.z + threadIdx.z;

	if (I >= sourcecount)
		return;
	if (J >= resultcount)
		return;
	if (K >= filterlength)
		return;

	unsigned long   sourceindex = I * sourcelength;
	unsigned long   resultindex = J * resultlength;
	unsigned long   filterindex = (J * sourcecount + I) * filterlength;

	unsigned long  X, Y;
	unsigned long  index;
	unsigned long  counter;
	cudaNeuralUnit sum;

	sum = 0;
	counter = 0;
	Y = K * indexwidth;
	for (X = 0; X < indexwidth; X++)
	{
		index = indexmap[Y + X];
		if ((index != CUDA_INVALID_INDEX) &&
			(index < sourcelength))
		{
			sum += source[sourceindex + index] * result[resultindex + X];
			counter++;
		}
	}

	if (counter > 1)
		sum = sum / counter;

	filter[filterindex + K] = sum;

}

long cudaNeuralArrayIndexMap2DConvolutionGetDerivatives(cudaNeuralArray* result, cudaNeuralArray* source, cudaNeuralArray* filter, cudaIndexMap* indexmap)
{
	unsigned long   resultlength;
	unsigned long   resultcount;
	unsigned long   sourcelength;
	unsigned long   sourcecount;
	unsigned long   filterlength;
	unsigned long   filtercount;

	if ((result == 0) || (source == 0) || (filter == 0) || (indexmap == 0))
		return 0;

	resultlength = (result->arrayShape.shapeWidth * result->arrayShape.shapeHeight);
	resultcount = result->arrayShape.shapeDepth;
	sourcelength = (source->arrayShape.shapeWidth * source->arrayShape.shapeHeight);
	sourcecount = source->arrayShape.shapeDepth;;
	filterlength = (filter->arrayShape.shapeWidth * filter->arrayShape.shapeHeight);
	filtercount = filter->arrayShape.shapeDepth;;

	if (filtercount != (resultcount * sourcecount))
		return 0;

	if ((indexmap->indexWidth != resultlength) ||
		(indexmap->indexHeight != filterlength))
		return 0;

	dim3 threadsPerBlock(8, 8, 8);
	dim3 blocksPerGrid(
		(sourcecount + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(resultcount + threadsPerBlock.y - 1) / threadsPerBlock.y,
		(filterlength + threadsPerBlock.z - 1) / threadsPerBlock.z);

	kernelArrayIndexMap2DConvolutionGetDerivatives << < blocksPerGrid, threadsPerBlock >> > (
		result->arrayData,
		source->arrayData,
		filter->arrayData,
		indexmap->indexWidth,
		indexmap->indexHeight,
		indexmap->indexData,
		resultlength,
		resultcount,
		sourcelength,
		sourcecount,
		filterlength,
		filtercount);

	return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelArrayMatrixForward(cudaNeuralUnit* result, cudaNeuralUnit* source, cudaNeuralUnit* weights, unsigned long resultlength, unsigned long sourcelength)
{
	unsigned long  I;
	unsigned long  J = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long  K;
	cudaNeuralUnit sum;

	if (J >= resultlength)
		return;

	sum = 0;
	K = J * sourcelength;
	for (I = 0; I < sourcelength; I++)
	{
		sum += source[I] * weights[K + I];
	}

	result[J] = sum;
}

long cudaNeuralArrayMatrixForward(cudaNeuralArray* result, cudaNeuralArray* source, cudaNeuralArray* weights)
{
	if ((result == 0) || (source == 0) || (weights == 0))
		return 0;

	if (weights->arrayLength != (result->arrayLength * source->arrayLength))
		return 0;

	dim3 threadsPerBlock(512);
	dim3 blocksPerGrid((result->arrayLength + threadsPerBlock.x - 1) / threadsPerBlock.x);

	kernelArrayMatrixForward << < blocksPerGrid, threadsPerBlock >> >
		(result->arrayData, source->arrayData, weights->arrayData, result->arrayLength, source->arrayLength);

	return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelArrayMatrixReverse(cudaNeuralUnit* result, cudaNeuralUnit* source, cudaNeuralUnit* weights, unsigned long resultlength, unsigned long sourcelength)
{
	unsigned long  I = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long  J;
	unsigned long  K;
	cudaNeuralUnit sum;

	if (I >= sourcelength)
		return;

	sum = 0;
	K = I;
	for (J = 0; J < resultlength; J++)
	{
		sum += result[J] * weights[K];

		K += sourcelength;
	}

	source[I] = sum;
}

long cudaNeuralArrayMatrixReverse(cudaNeuralArray* result, cudaNeuralArray* source, cudaNeuralArray* weights)
{
	if ((result == 0) || (source == 0) || (weights == 0))
		return 0;

	if (weights->arrayLength != (result->arrayLength * source->arrayLength))
		return 0;

	dim3 threadsPerBlock(512);
	dim3 blocksPerGrid((source->arrayLength + threadsPerBlock.x - 1) / threadsPerBlock.x);

	kernelArrayMatrixReverse << < blocksPerGrid, threadsPerBlock >> >
		(result->arrayData, source->arrayData, weights->arrayData, result->arrayLength, source->arrayLength);

	return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelArrayMatrixGetDerivatives(cudaNeuralUnit* result, cudaNeuralUnit* source, cudaNeuralUnit* weights, unsigned long resultlength, unsigned long sourcelength)
{
	unsigned long  I = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long  J = blockDim.y * blockIdx.y + threadIdx.y;

	if (I >= sourcelength)
		return;
	if (J >= resultlength)
		return;

	weights[J * sourcelength + I] = result[J] * source[I];
}

long cudaNeuralArrayMatrixGetDerivatives(cudaNeuralArray* result, cudaNeuralArray* source, cudaNeuralArray* weights)
{
	if ((result == 0) || (source == 0) || (weights == 0))
		return 0;

	if (weights->arrayLength != (result->arrayLength * source->arrayLength))
		return 0;

	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid(
		(source->arrayLength + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(result->arrayLength + threadsPerBlock.y - 1) / threadsPerBlock.y);

	kernelArrayMatrixGetDerivatives << < blocksPerGrid, threadsPerBlock >> >
		(result->arrayData, source->arrayData, weights->arrayData, result->arrayLength, source->arrayLength);

	return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
__device__ double cuda_reluFunction(double a)
{
	if (a > 0)
		return a;

	return a * 0.01;
}

__device__ double cuda_reluDerivative(double a)
{
	if (a < 0)
		return 0.01;

	return 1;
}

__device__ double cuda_sigmoidFunction(double a)
{
	return 1 / (1 + exp(-a));
}

__device__ double cuda_sigmoidDerivative(double a)
{
	return (cuda_sigmoidFunction(a) * (1 - cuda_sigmoidFunction(a)));
}

__device__ double cuda_tanhFunction(double a)
{
	return tanh(a);
}

__device__ double cuda_tanhDerivative(double a)
{
	return 1 - (tanh(a) * tanh(a));
}

__global__ void kernelArrayTransferForward(cudaNeuralUnit* result, cudaNeuralUnit* source, unsigned long length, unsigned long transfer)
{
	unsigned long  I = blockDim.x * blockIdx.x + threadIdx.x;

	if (I >= length)
		return;

	switch (transfer)
	{
	case CUDA_TRANSFER_RELU:
		result[I] = cuda_reluFunction(source[I]);
	case CUDA_TRANSFER_SIGMOID:
		result[I] = cuda_sigmoidFunction(source[I]);
	case CUDA_TRANSFER_TANH:
		result[I] = cuda_tanhFunction(source[I]);
	}
}

long cudaNeuralArrayTransferForward(cudaNeuralArray* result, cudaNeuralArray* source, unsigned long transfer)
{
	if ((result == 0) || (source == 0))
		return 0;

	if (result->arrayLength != source->arrayLength)
		return 0;

	dim3 threadsPerBlock(512);
	dim3 blocksPerGrid((result->arrayLength + threadsPerBlock.x - 1) / threadsPerBlock.x);

	kernelArrayTransferForward << < blocksPerGrid, threadsPerBlock >> > (result->arrayData, source->arrayData, result->arrayLength, transfer);

	return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelArrayTransferReverse(cudaNeuralUnit* result, cudaNeuralUnit* source, cudaNeuralUnit* outputs, unsigned long length, unsigned long transfer)
{
	unsigned long  I = blockDim.x * blockIdx.x + threadIdx.x;

	if (I >= length)
		return;

	switch (transfer)
	{
	case CUDA_TRANSFER_RELU:
		source[I] = result[I] * cuda_reluDerivative(outputs[I]);
	case CUDA_TRANSFER_SIGMOID:
		source[I] = result[I] * cuda_sigmoidDerivative(outputs[I]);
	case CUDA_TRANSFER_TANH:
		source[I] = result[I] * cuda_tanhDerivative(outputs[I]);
	}
}

long cudaNeuralArrayTransferReverse(cudaNeuralArray* result, cudaNeuralArray* source, cudaNeuralArray* outputs, unsigned long transfer)
{
	if ((result == 0) || (source == 0) || (outputs == 0))
		return 0;

	if ((result->arrayLength != source->arrayLength) ||
		(result->arrayLength != outputs->arrayLength))
		return 0;

	dim3 threadsPerBlock(512);
	dim3 blocksPerGrid((result->arrayLength + threadsPerBlock.x - 1) / threadsPerBlock.x);

	kernelArrayTransferReverse << < blocksPerGrid, threadsPerBlock >> > (result->arrayData, source->arrayData, outputs->arrayData, result->arrayLength, transfer);

	return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelArrayTransferFocus(
	cudaNeuralUnit* result,
	cudaNeuralUnit* source,
	cudaNeuralUnit* outputs,
	cudaNeuralUnit* target,
	cudaNeuralUnit  lambda,
	cudaNeuralUnit  range,
	unsigned long length)
{
	unsigned long  I = blockDim.x * blockIdx.x + threadIdx.x;

	if (I >= length)
		return;

	cudaNeuralUnit   gap, grad;

	gap = (target[I] - outputs[I]);
	grad = (gap / range) * (gap / range);
	source[I] = result[I] + (grad * lambda) * gap;
}

long cudaNeuralArrayTransferFocus(cudaNeuralArray* result, cudaNeuralArray* source, cudaNeuralArray* outputs, cudaNeuralArray* target, cudaNeuralUnit lambda, cudaNeuralUnit range)
{
	if ((result == 0) || (source == 0) || (outputs == 0) || (target == 0))
		return 0;

	if ((result->arrayLength != source->arrayLength) ||
		(result->arrayLength != outputs->arrayLength) ||
		(result->arrayLength != target->arrayLength))
		return 0;

	dim3 threadsPerBlock(512);
	dim3 blocksPerGrid((result->arrayLength + threadsPerBlock.x - 1) / threadsPerBlock.x);

	kernelArrayTransferFocus << < blocksPerGrid, threadsPerBlock >> > (result->arrayData, source->arrayData, outputs->arrayData, target->arrayData, lambda, range, result->arrayLength);

	return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelArrayUpdateMomentum(cudaNeuralUnit* weights, cudaNeuralUnit* vectors, cudaNeuralUnit* deltas, cudaNeuralUnit learningrate, cudaNeuralUnit momentum, unsigned long length)
{
	unsigned long  I = blockDim.x * blockIdx.x + threadIdx.x;

	if (I >= length)
		return;

	vectors[I] = (vectors[I] * momentum) + (1 - momentum) * (deltas[I]);

	weights[I] = weights[I] + learningrate * vectors[I];
}

long cudaNeuralArrayUpdateMomentum(cudaNeuralArray* weights, cudaNeuralArray* vectors, cudaNeuralArray* deltas, cudaNeuralUnit learningrate, cudaNeuralUnit momentum)
{
	if ((weights == 0) || (vectors == 0) || (deltas == 0))
		return 0;

	if ((weights->arrayLength != vectors->arrayLength) ||
		(weights->arrayLength != deltas->arrayLength))
		return 0;

	dim3 threadsPerBlock(512);
	dim3 blocksPerGrid((weights->arrayLength + threadsPerBlock.x - 1) / threadsPerBlock.x);

	kernelArrayUpdateMomentum << < blocksPerGrid, threadsPerBlock >> >
		(weights->arrayData, vectors->arrayData, deltas->arrayData, learningrate, momentum, weights->arrayLength);

	return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelArrayUpdateAdagrad(cudaNeuralUnit* weights, cudaNeuralUnit* vectors, cudaNeuralUnit* gammas, cudaNeuralUnit* deltas, cudaNeuralUnit learningrate, cudaNeuralUnit momentum, unsigned long length)
{
	unsigned long  I = blockDim.x * blockIdx.x + threadIdx.x;
	cudaNeuralUnit epsilon = 0.00000001;
	cudaNeuralUnit m2 = 0.999;
	cudaNeuralUnit grad;

	if (I >= length)
		return;

	gammas[I] = (gammas[I] * m2) + (1 - m2) * (deltas[I] * deltas[I]);
	grad = (learningrate / (sqrt(gammas[I]) + epsilon));

	vectors[I] = (vectors[I] * momentum) + (1 - momentum) * (deltas[I]);

	weights[I] = weights[I] + grad * vectors[I];
}

long cudaNeuralArrayUpdateAdagrad(cudaNeuralArray* weights, cudaNeuralArray* vectors, cudaNeuralArray* gammas, cudaNeuralArray* deltas, cudaNeuralUnit learningrate, cudaNeuralUnit momentum)
{
	if ((weights == 0) || (vectors == 0) || (deltas == 0))
		return 0;

	if ((weights->arrayLength != vectors->arrayLength) ||
		(weights->arrayLength != gammas->arrayLength) ||
		(weights->arrayLength != deltas->arrayLength))
		return 0;

	dim3 threadsPerBlock(512);
	dim3 blocksPerGrid((weights->arrayLength + threadsPerBlock.x - 1) / threadsPerBlock.x);

	kernelArrayUpdateAdagrad << < blocksPerGrid, threadsPerBlock >> >
		(weights->arrayData, vectors->arrayData, gammas->arrayData, deltas->arrayData, learningrate, momentum, weights->arrayLength);

	return cudaMemoryDeviceSynchronize();
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelArrayUpdateAdam(cudaNeuralUnit* weights, cudaNeuralUnit* vectors, cudaNeuralUnit* gammas, cudaNeuralUnit* deltas, cudaNeuralUnit learningrate, cudaNeuralUnit momentum, unsigned long length)
{
	unsigned long  I = blockDim.x * blockIdx.x + threadIdx.x;
	cudaNeuralUnit epsilon = 0.00000001;
	cudaNeuralUnit m2 = 0.999;
	cudaNeuralUnit vt, mt;
	cudaNeuralUnit grad;

	if (I >= length)
		return;

	gammas[I] = (gammas[I] * m2) + (1 - m2) * (deltas[I] * deltas[I]);
	vt = gammas[I] / (1 - m2);

	vectors[I] = (vectors[I] * momentum) + (1 - momentum) * (deltas[I]);
	mt = vectors[I] / (1 - momentum);

	grad = (learningrate / (sqrt(vt) + epsilon));

	weights[I] = weights[I] + (mt * grad);
}

long cudaNeuralArrayUpdateAdam(cudaNeuralArray* weights, cudaNeuralArray* vectors, cudaNeuralArray* gammas, cudaNeuralArray* deltas, cudaNeuralUnit learningrate, cudaNeuralUnit momentum)
{
	if ((weights == 0) || (vectors == 0) || (deltas == 0))
		return 0;

	if ((weights->arrayLength != vectors->arrayLength) ||
		(weights->arrayLength != gammas->arrayLength) ||
		(weights->arrayLength != deltas->arrayLength))
		return 0;

	dim3 threadsPerBlock(512);
	dim3 blocksPerGrid((weights->arrayLength + threadsPerBlock.x - 1) / threadsPerBlock.x);

	kernelArrayUpdateAdam << < blocksPerGrid, threadsPerBlock >> >
		(weights->arrayData, vectors->arrayData, gammas->arrayData, deltas->arrayData, learningrate, momentum, weights->arrayLength);

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


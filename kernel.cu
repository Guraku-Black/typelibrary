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

#ifndef CUDAINDEXMAP_H
#define CUDAINDEXMAP_H

#include "cudalibrary.h"

long cudaIndexMapCreateTypeCopy(cudaIndexMap* parent, typeIndexMap* source)
{
	unsigned long* data;

	if ((parent == 0) || (source == 0))
		return 0;

	if (cudaMemoryAllocateMemory((void**)&data, source->indexSize))
	{
		cudaMemorySetDeviceMemory(data, source->indexData, source->indexSize);

		parent->indexWidth = source->indexWidth;
		parent->indexHeight = source->indexHeight;
		parent->indexLength = source->indexLength;
		parent->indexSize = source->indexSize;
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
__managed__ cudaNeuralUnit     cudaRegister1;

__global__ void kernelGetMeanSquaredError(cudaNeuralUnit* source, cudaNeuralUnit* target, unsigned long length)
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

	kernelGetMeanSquaredError << < 1, 1 >> > (source->arrayData, target->arrayData, source->arrayLength);

	cudaMemoryDeviceSynchronize();

	*result = cudaRegister1;

	return 1;
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void kernelArrayIndexMapCopy2D(
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
		if ((index != TYPE_INVALID_INDEX) &&
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

long cudaNeuralArrayIndexMapCopy2D(cudaNeuralArray* result, cudaNeuralArray* source, cudaIndexMap* indexmap)
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

	kernelArrayIndexMapCopy2D << < blocksPerGrid, threadsPerBlock >> > (
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
__global__ void kernelArrayIndexMapPooling2DForward(
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
		if ((index != TYPE_INVALID_INDEX) &&
			(index < sourcelength))
		{
			if (fabsf(source[sourceindex + index]) > fabsf(sum))
				sum = source[sourceindex + index];
		}

		K += indexwidth;
	}


	result[resultindex + I] = sum;
}

long cudaNeuralArrayIndexMapPooling2DForward(cudaNeuralArray* result, cudaNeuralArray* source, cudaIndexMap* indexmap)
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

	kernelArrayIndexMapPooling2DForward << < blocksPerGrid, threadsPerBlock >> > (
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
__global__ void kernelArrayIndexMapPooling2DReverse(
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
		if ((index != TYPE_INVALID_INDEX) &&
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
		if ((index != TYPE_INVALID_INDEX) &&
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

long cudaNeuralArrayIndexMapPooling2DReverse(cudaNeuralArray* result, cudaNeuralArray* source, cudaNeuralArray* outputs, cudaIndexMap* indexmap)
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

	if (typeShapeCompare(&source->arrayShape, &outputs->arrayShape))
		return 0;

	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid(
		(resultlength + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(resultcount + threadsPerBlock.y - 1) / threadsPerBlock.y);

	kernelArrayIndexMapPooling2DReverse << < blocksPerGrid, threadsPerBlock >> > (
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
__device__ cudaNeuralUnit indexMapConvolution2DValue(
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
		if ((index != TYPE_INVALID_INDEX) &&
			(index < sourcelength))
		{
			sum += source[sourceindex + index] * filter[filterindex + I];
		}

		J += indexwidth;
	}

	return sum;
}

__global__ void kernelArrayIndexMapConvolution2DForward(
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
		sum += indexMapConvolution2DValue(
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

	result[resultindex + K] = sum;
}

long cudaNeuralArrayConvolution2DForward(cudaNeuralArray* result, cudaNeuralArray* source, cudaNeuralArray* filter, cudaIndexMap* indexmap)
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

	kernelArrayIndexMapConvolution2DForward << < blocksPerGrid, threadsPerBlock >> > (
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
__global__ void kernelArrayIndexMapConvolution2DReverse(
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
		sum += indexMapConvolution2DValue(
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

long cudaNeuralArrayConvolution2DReverse(cudaNeuralArray* result, cudaNeuralArray* source, cudaNeuralArray* filter, cudaIndexMap* indexmap)
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

	kernelArrayIndexMapConvolution2DReverse << < blocksPerGrid, threadsPerBlock >> > (
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
__global__ void kernelArrayIndexMapConvolution2DGetDerivatives(
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
		if ((index != TYPE_INVALID_INDEX) &&
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

long cudaNeuralArrayConvolution2DGetDerivatives(cudaNeuralArray* result, cudaNeuralArray* source, cudaNeuralArray* filter, cudaIndexMap* indexmap)
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

	kernelArrayIndexMapConvolution2DGetDerivatives << < blocksPerGrid, threadsPerBlock >> > (
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


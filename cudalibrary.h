#ifndef CUDALIBRARY_H
#define CUDALIBRARY_H

#ifdef CUDA_LIBRARY_KERNEL

#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C"
{
#endif

	DLLEXPORT long cudaMemoryResetDevice();
	DLLEXPORT long cudaMemorySetDevice(unsigned long deviceID);
	DLLEXPORT long cudaMemoryAllocateMemory(void** result, unsigned long length);
	DLLEXPORT long cudaMemorySetDeviceMemory(void* devicememory, void* hostmemory, unsigned long length);
	DLLEXPORT long cudaMemoryGetDeviceMemory(void* devicememory, void* hostmemory, unsigned long length);
	DLLEXPORT long cudaMemoryCopyDeviceMemory(void* devicedestination, void* devicesource, unsigned long length);
	DLLEXPORT long cudaMemoryDeviceSynchronize();
	DLLEXPORT long cudaMemoryFreeMemory(void* source);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C"
{
#endif

	#include "E:/Projects/Visual Studio/typelibrary/typeshape.h"

	typedef double     cudaNeuralUnit;

	typedef struct
	{
		typeShape       arrayShape;
		unsigned long   arrayLength;
		unsigned long   arraySize;
		cudaNeuralUnit* arrayData;
	} cudaNeuralArray;

	DLLEXPORT long cudaNeuralArrayCreate(cudaNeuralArray* parent, unsigned long length);
	DLLEXPORT long cudaNeuralArraySetData(cudaNeuralArray* parent, cudaNeuralUnit* buffer, unsigned long buffersize);
	DLLEXPORT long cudaNeuralArrayGetData(cudaNeuralArray* parent, cudaNeuralUnit* buffer, unsigned long buffersize);
	DLLEXPORT long cudaNeuralArrayCopy(cudaNeuralArray* parent, cudaNeuralArray* source);
	DLLEXPORT long cudaNeuralArrayReverse(cudaNeuralArray* parent, cudaNeuralArray* source);
	DLLEXPORT long cudaNeuralArrayDestroy(cudaNeuralArray* parent);

#ifdef __cplusplus
}
#endif

#endif // CUDALIBRARY_H
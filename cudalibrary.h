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

	const unsigned long CUDA_INVALID_INDEX = 0xFFFFFFFF;

	typedef struct
	{
		unsigned long     shapeWidth;
		unsigned long     shapeHeight;
		unsigned long     shapeDepth;
	} cudaShape;

	typedef struct
	{
		unsigned long   indexWidth;
		unsigned long   indexHeight;
		unsigned long   indexLength;
		unsigned long   indexSize;
		unsigned long*  indexData;
	} cudaIndexMap;

	DLLEXPORT long cudaIndexMapCreateCopy(cudaIndexMap* parent, unsigned long width, unsigned long height, unsigned long* data);
	DLLEXPORT long cudaIndexMapDestroy(cudaIndexMap* parent);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C"
{
#endif

	enum cuda_transferType
	{
		CUDA_TRANSFER_RELU,
		CUDA_TRANSFER_SIGMOID,
		CUDA_TRANSFER_TANH
	};

	typedef double      cudaNeuralUnit;

	typedef struct
	{
		cudaShape       arrayShape;
		unsigned long   arrayLength;
		unsigned long   arraySize;
		cudaNeuralUnit* arrayData;
	} cudaNeuralArray;

	DLLEXPORT long cudaNeuralArrayCreate(cudaNeuralArray* parent, unsigned long width, unsigned long height, unsigned long depth);
	DLLEXPORT long cudaNeuralArraySetData(cudaNeuralArray* parent, cudaNeuralUnit* buffer, unsigned long buffersize);
	DLLEXPORT long cudaNeuralArrayGetData(cudaNeuralArray* parent, cudaNeuralUnit* buffer, unsigned long buffersize);
	DLLEXPORT long cudaNeuralArrayCopy(cudaNeuralArray* parent, cudaNeuralArray* source);
	DLLEXPORT long cudaNeuralArraySetImageRGB(cudaNeuralArray* parent, unsigned long imagewidth, unsigned long imageheight, unsigned long* imagedata);
	DLLEXPORT long cudaNeuralArrayGetImageRGB(cudaNeuralArray* parent, unsigned long imagewidth, unsigned long imageheight, unsigned long* imagedata);
	DLLEXPORT long cudaNeuralArrayFillZero(cudaNeuralArray* result);
	DLLEXPORT long cudaNeuralArrayFillOnes(cudaNeuralArray* result);
	DLLEXPORT long cudaNeuralArrayFillRandomUniform(cudaNeuralArray* result);
	DLLEXPORT long cudaNeuralArrayFillRandomNormal(cudaNeuralArray* result);
	DLLEXPORT long cudaNeuralArrayFillRandomXavier(cudaNeuralArray* result, unsigned long fanIn, unsigned long fanOut);
	DLLEXPORT long cudaNeuralArrayFillOneHot(cudaNeuralArray* result, unsigned long index);
	DLLEXPORT long cudaNeuralArrayAdd(cudaNeuralArray* result, cudaNeuralArray* value1, cudaNeuralArray* value2);
	DLLEXPORT long cudaNeuralArraySubtract(cudaNeuralArray* result, cudaNeuralArray* value1, cudaNeuralArray* value2);
	DLLEXPORT long cudaNeuralArrayMultiply(cudaNeuralArray* result, cudaNeuralArray* value1, cudaNeuralArray* value2);
	DLLEXPORT long cudaNeuralArrayDivide(cudaNeuralArray* result, cudaNeuralArray* value1, cudaNeuralArray* value2);
	DLLEXPORT long cudaNeuralArrayFlip3D(cudaNeuralArray* result, cudaNeuralArray* source, long A, long B, long C);
	DLLEXPORT long cudaNeuralArrayShift3D(cudaNeuralArray* result, cudaNeuralArray* source, long A, long B, long C);
	DLLEXPORT long cudaNeuralArrayZoom3D(cudaNeuralArray* result, cudaNeuralArray* source, double A, double B, double C);
	DLLEXPORT long cudaNeuralArrayRotate2D(cudaNeuralArray* result, cudaNeuralArray* source, double angle);
	DLLEXPORT long cudaNeuralArrayGetMeanSquaredError(cudaNeuralArray* source, cudaNeuralArray* target, cudaNeuralUnit* result);
	DLLEXPORT long cudaNeuralArrayIndexMap2DCopy(cudaNeuralArray* result, cudaNeuralArray* source, cudaIndexMap* indexmap);
	DLLEXPORT long cudaNeuralArrayIndexMap2DPoolingForward(cudaNeuralArray* result, cudaNeuralArray* source, cudaIndexMap* indexmap);
	DLLEXPORT long cudaNeuralArrayIndexMap2DPoolingReverse(cudaNeuralArray* result, cudaNeuralArray* source, cudaNeuralArray* outputs, cudaIndexMap* indexmap);
	DLLEXPORT long cudaNeuralArrayIndexMap2DConvolutionForward(cudaNeuralArray* result, cudaNeuralArray* source, cudaNeuralArray* filter, cudaIndexMap* indexmap);
	DLLEXPORT long cudaNeuralArrayIndexMap2DConvolutionReverse(cudaNeuralArray* result, cudaNeuralArray* source, cudaNeuralArray* filter, cudaIndexMap* indexmap);
	DLLEXPORT long cudaNeuralArrayIndexMap2DConvolutionGetDerivatives(cudaNeuralArray* result, cudaNeuralArray* source, cudaNeuralArray* filter, cudaIndexMap* indexmap);
	DLLEXPORT long cudaNeuralArrayMatrixForward(cudaNeuralArray* result, cudaNeuralArray* source, cudaNeuralArray* weights);
	DLLEXPORT long cudaNeuralArrayMatrixReverse(cudaNeuralArray* result, cudaNeuralArray* source, cudaNeuralArray* weights);
	DLLEXPORT long cudaNeuralArrayMatrixGetDerivatives(cudaNeuralArray* result, cudaNeuralArray* source, cudaNeuralArray* weights);
	DLLEXPORT long cudaNeuralArrayTransferForward(cudaNeuralArray* result, cudaNeuralArray* source, unsigned long transfer);
	DLLEXPORT long cudaNeuralArrayTransferReverse(cudaNeuralArray* result, cudaNeuralArray* source, cudaNeuralArray* outputs, unsigned long transfer);
	DLLEXPORT long cudaNeuralArrayTransferFocus(cudaNeuralArray* result, cudaNeuralArray* source, cudaNeuralArray* outputs, cudaNeuralArray* target, cudaNeuralUnit  lambda, cudaNeuralUnit range);
	DLLEXPORT long cudaNeuralArrayUpdateMomentum(cudaNeuralArray* weights, cudaNeuralArray* vectors, cudaNeuralArray* deltas, cudaNeuralUnit learningrate, cudaNeuralUnit momentum);
	DLLEXPORT long cudaNeuralArrayUpdateAdagrad(cudaNeuralArray* weights, cudaNeuralArray* vectors, cudaNeuralArray* gammas, cudaNeuralArray* deltas, cudaNeuralUnit learningrate, cudaNeuralUnit momentum);
	DLLEXPORT long cudaNeuralArrayUpdateAdam(cudaNeuralArray* weights, cudaNeuralArray* vectors, cudaNeuralArray* gammas, cudaNeuralArray* deltas, cudaNeuralUnit learningrate, cudaNeuralUnit momentum);
	DLLEXPORT long cudaNeuralArrayDestroy(cudaNeuralArray* parent);

#ifdef __cplusplus
}
#endif

#endif // CUDALIBRARY_H
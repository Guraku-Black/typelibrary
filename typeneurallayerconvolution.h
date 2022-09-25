#ifndef TYPENEURALLAYERCONVOLUTION_H
#define TYPENEURALLAYERCONVOLUTION_H

#include "typeneurallayer.h"

long typeNeuralLayerConvolutionCreate(typeNeuralLayer* parent,
	unsigned long sourcewidth, unsigned long sourceheight, unsigned long sourcedepth,
	unsigned long resultwidth, unsigned long resultheight, unsigned long resultdepth,
	unsigned long filterwidth, unsigned long filterheight)
{
	typeShape     result;
	typeShape     source;
	typeIndexMap  indexmap1;
	typeIndexMap  indexmap2;
	unsigned long filtercount;

	if (typeNeuralLayerCreate(parent, sourcewidth, sourceheight, sourcedepth))
	{
		parent->layerType = TYPE_NEURAL_LAYER_CONVOLUTION;

		filtercount = (resultdepth * sourcedepth);

		cudaNeuralArrayCreate(&parent->layerWeights, filterwidth, filterheight, filtercount);
		cudaNeuralArrayCreate(&parent->layerVectors, filterwidth, filterheight, filtercount);
		cudaNeuralArrayCreate(&parent->layerAlphas, filterwidth, filterheight, filtercount);
		cudaNeuralArrayCreate(&parent->layerGammas, filterwidth, filterheight, filtercount);

		//cudaNeuralArrayFillRandomXavier(&parent->layerWeights, sourcewidth*sourceheight*sourcedepth, resultwidth*resultheight*resultdepth);
		cudaNeuralArrayFillRandomUniform(&parent->layerWeights);
		cudaNeuralArrayFillZero(&parent->layerVectors);
		cudaNeuralArrayFillZero(&parent->layerGammas);

		typeShapeAssign(&source, sourcewidth, sourceheight, 1);
		typeShapeAssign(&result, resultwidth, resultheight, 1);

		typeIndexMapCreateConvolution2D(&indexmap1, &result, &source, filterwidth, filterheight);
		typeIndexMapCreateConvolution2DTranspose(&indexmap2, &source, &result, filterwidth, filterheight);

		cudaIndexMapCreateCopy(&parent->layerMap1, indexmap1.indexWidth, indexmap1.indexHeight, indexmap1.indexData);
		cudaIndexMapCreateCopy(&parent->layerMap2, indexmap2.indexWidth, indexmap2.indexHeight, indexmap2.indexData);

		typeIndexMapDestroy(&indexmap1);
		typeIndexMapDestroy(&indexmap2);

		return 1;
	}

	return 0;
}
long typeNeuralLayerConvolutionFeedForward(typeNeuralLayer* parent, typeNeuralLayer* next)
{
	if (parent == 0)
		return 0;

	return cudaNeuralArrayIndexMap2DConvolutionForward(&next->layerOutputs, &parent->layerOutputs, &parent->layerWeights, &parent->layerMap1);
}

long typeNeuralLayerConvolutionBackPropagate(typeNeuralLayer* parent, typeNeuralLayer* next)
{
	if (parent == 0)
		return 0;

	return cudaNeuralArrayIndexMap2DConvolutionReverse(&next->layerDeltas, &parent->layerDeltas, &parent->layerWeights, &parent->layerMap2);
}

long typeNeuralLayerConvolutionUpdateWeights(typeNeuralLayer* parent, typeNeuralLayer* next, cudaNeuralUnit learningrate, cudaNeuralUnit momentum, unsigned long optimizer)
{
	cudaNeuralArrayIndexMap2DConvolutionGetDerivatives(&next->layerDeltas, &parent->layerOutputs, &parent->layerAlphas, &parent->layerMap1);

	switch (optimizer)
	{
	case TYPE_OPTIMIZER_MOMENTUM:
		cudaNeuralArrayUpdateMomentum(&parent->layerWeights, &parent->layerVectors, &parent->layerAlphas, learningrate, momentum);
		break;
	case TYPE_OPTIMIZER_ADAGRAD:
		cudaNeuralArrayUpdateAdagrad(&parent->layerWeights, &parent->layerVectors, &parent->layerGammas, &parent->layerAlphas, learningrate, momentum);
		break;
	case TYPE_OPTIMIZER_ADAM:
		cudaNeuralArrayUpdateAdam(&parent->layerWeights, &parent->layerVectors, &parent->layerGammas, &parent->layerAlphas, learningrate, momentum);
		break;
	}

	return 1;
}

long typeNeuralLayerConvolutionDestroy(typeNeuralLayer* parent)
{
	if (parent == 0)
		return 0;

	cudaNeuralArrayDestroy(&parent->layerWeights);
	cudaNeuralArrayDestroy(&parent->layerVectors);
	cudaNeuralArrayDestroy(&parent->layerAlphas);

	cudaIndexMapDestroy(&parent->layerMap1);
	cudaIndexMapDestroy(&parent->layerMap2);

	return typeNeuralLayerDestroy(parent);
}

#endif // TYPENEURALLAYERCONVOLUTION_H

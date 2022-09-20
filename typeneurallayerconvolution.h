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
	unsigned long filtercount;

	if (typeNeuralLayerCreate(parent, sourcewidth, sourceheight, sourcedepth))
	{
		parent->layerType = TYPE_NEURAL_LAYER_CONVOLUTION;

		filtercount = (resultdepth * sourcedepth);

		typeNeuralArrayCreate(&parent->layerWeights, filterwidth, filterheight, filtercount);
		typeNeuralArrayCreate(&parent->layerVectors, filterwidth, filterheight, filtercount);
		typeNeuralArrayCreate(&parent->layerAlphas, filterwidth, filterheight, filtercount);
		typeNeuralArrayCreate(&parent->layerGammas, filterwidth, filterheight, filtercount);

		typeNeuralArrayFillRandom(&parent->layerWeights);
		typeNeuralArrayFillZero(&parent->layerVectors);
		typeNeuralArrayFillZero(&parent->layerGammas);

		typeShapeAssign(&source, sourcewidth, sourceheight, 1);
		typeShapeAssign(&result, resultwidth, resultheight, 1);

		typeIndexMapCreateConvolution2D(&parent->layerMap1, &result, &source, filterwidth, filterheight);
		typeIndexMapCreateConvolution2DTranspose(&parent->layerMap2, &source, &result, filterwidth, filterheight);

		return 1;
	}

	return 0;
}
long typeNeuralLayerConvolutionFeedForward(typeNeuralLayer* parent, typeNeuralLayer* next)
{
	if (parent == 0)
		return 0;

	return typeNeuralArrayConvolution2DForward(&next->layerOutputs, &parent->layerOutputs, &parent->layerWeights, &parent->layerMap1);
}

long typeNeuralLayerConvolutionBackPropagate(typeNeuralLayer* parent, typeNeuralLayer* next)
{
	if (parent == 0)
		return 0;

	return typeNeuralArrayConvolution2DReverse(&next->layerDeltas, &parent->layerDeltas, &parent->layerWeights, &parent->layerMap2);
}

long typeNeuralLayerConvolutionUpdateWeights(typeNeuralLayer* parent, typeNeuralLayer* next, typeNeuralUnit learningrate, typeNeuralUnit momentum)
{
	typeNeuralArrayConvolution2DGetDerivatives(&next->layerDeltas, &parent->layerOutputs, &parent->layerAlphas, &parent->layerMap1);

	//typeNeuralArrayUpdateMomentum(&parent->layerWeights, &parent->layerVectors, &parent->layerAlphas, learningrate, momentum);
	//typeNeuralArrayUpdateAdagrad(&parent->layerWeights, &parent->layerVectors, &parent->layerGammas, &parent->layerAlphas, learningrate, momentum);
	typeNeuralArrayUpdateAdam(&parent->layerWeights, &parent->layerVectors, &parent->layerGammas, &parent->layerAlphas, learningrate, momentum);

	return 1;
}

long typeNeuralLayerConvolutionDestroy(typeNeuralLayer* parent)
{
	if (parent == 0)
		return 0;

	typeNeuralArrayDestroy(&parent->layerWeights);
	typeNeuralArrayDestroy(&parent->layerVectors);
	typeNeuralArrayDestroy(&parent->layerAlphas);

	typeIndexMapDestroy(&parent->layerMap1);
	typeIndexMapDestroy(&parent->layerMap2);

	return typeNeuralLayerDestroy(parent);
}

#endif // TYPENEURALLAYERCONVOLUTION_H

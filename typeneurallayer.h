#ifndef TYPENEURALLAYER_H
#define TYPENEURALLAYER_H

#include "typeindexmap.h"
#include <string.h>

enum layerType
{
	TYPE_NEURAL_LAYER_NULL,
	TYPE_NEURAL_LAYER_BIAS,
	TYPE_NEURAL_LAYER_SCALING,
	TYPE_NEURAL_LAYER_POOLING,
	TYPE_NEURAL_LAYER_CONVOLUTION,
	TYPE_NEURAL_LAYER_TRANSFER
};

typedef struct
{
	unsigned long     layerType;
	unsigned long     layerValue1;
	cudaIndexMap      layerMap1;
	cudaIndexMap      layerMap2;
	cudaNeuralArray   layerOutputs;
	cudaNeuralArray   layerWeights;
	cudaNeuralArray   layerVectors;
	cudaNeuralArray   layerDeltas;
	cudaNeuralArray   layerAlphas;
	cudaNeuralArray   layerGammas;
} typeNeuralLayer;

long typeNeuralLayerCreate(typeNeuralLayer* parent, unsigned long width, unsigned long height, unsigned long depth);
long typeNeuralLayerFeedForward(typeNeuralLayer* parent, typeNeuralLayer* next);
long typeNeuralLayerBackPropagate(typeNeuralLayer* parent, typeNeuralLayer* next);
long typeNeuralLayerDestroy(typeNeuralLayer* parent);

long typeNeuralLayerCreate(typeNeuralLayer* parent, unsigned long width, unsigned long height, unsigned long depth)
{
	if (parent == 0)
		return 0;

	memset(parent, 0, sizeof(typeNeuralLayer));

	if ((cudaNeuralArrayCreate(&parent->layerOutputs, width, height, depth)) &&
		(cudaNeuralArrayCreate(&parent->layerDeltas, width, height, depth)))
	{
		parent->layerType = TYPE_NEURAL_LAYER_NULL;

		return 1;
	}

	cudaNeuralArrayDestroy(&parent->layerOutputs);
	cudaNeuralArrayDestroy(&parent->layerDeltas);

	return 0;
}

long typeNeuralLayerFeedForward(typeNeuralLayer* parent, typeNeuralLayer* next)
{
	if (parent == 0)
		return 0;

	if (parent->layerOutputs.arrayLength != next->layerOutputs.arrayLength)
		return 1;

	return cudaNeuralArrayCopy(&next->layerOutputs, &parent->layerOutputs);
}

long typeNeuralLayerBackPropagate(typeNeuralLayer* parent, typeNeuralLayer* next)
{
	if (parent == 0)
		return 0;

	if (parent->layerOutputs.arrayLength != next->layerOutputs.arrayLength)
		return 1;

	return cudaNeuralArrayCopy(&parent->layerDeltas, &next->layerDeltas);
}

long typeNeuralLayerDestroy(typeNeuralLayer* parent)
{
	if (parent == 0)
		return 0;

	cudaNeuralArrayDestroy(&parent->layerOutputs);
	cudaNeuralArrayDestroy(&parent->layerDeltas);

	memset(parent, 0, sizeof(typeNeuralLayer));

	return 1;
}

#endif // TYPENEURALLAYER_H

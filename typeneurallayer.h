#ifndef TYPENEURALLAYER_H
#define TYPENEURALLAYER_H

#include "typeneuralarray.h"
#include "typeindexmap.h"
#include <string.h>

enum neuralType
{
	TYPE_NEURAL_LAYER_NULL,
	TYPE_NEURAL_LAYER_BIAS,
	TYPE_NEURAL_LAYER_SCALING,
	TYPE_NEURAL_LAYER_POOLING
};

typedef struct
{
	unsigned long     layerType;
	typeIndexMap      layerMap1;
	typeIndexMap      layerMap2;
	typeNeuralArray   layerOutputs;
	typeNeuralArray   layerWeights;
	typeNeuralArray   layerVectors;
	typeNeuralArray   layerDeltas;
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

	if ((typeNeuralArrayCreate(&parent->layerOutputs, width, height, depth)) &&
		(typeNeuralArrayCreate(&parent->layerDeltas, width, height, depth)))
	{
		parent->layerType = TYPE_NEURAL_LAYER_NULL;

		return 1;
	}

	typeNeuralArrayDestroy(&parent->layerOutputs);
	typeNeuralArrayDestroy(&parent->layerDeltas);

	return 0;
}

long typeNeuralLayerFeedForward(typeNeuralLayer* parent, typeNeuralLayer* next)
{
	if (parent == 0)
		return 0;

	if (parent->layerOutputs.arrayLength != next->layerOutputs.arrayLength)
		return 1;

	return typeNeuralArrayCopy(&next->layerOutputs, &parent->layerOutputs);
}

long typeNeuralLayerBackPropagate(typeNeuralLayer* parent, typeNeuralLayer* next)
{
	if (parent == 0)
		return 0;

	if (parent->layerOutputs.arrayLength != next->layerOutputs.arrayLength)
		return 1;

	return typeNeuralArrayCopy(&parent->layerDeltas, &next->layerDeltas);
}

long typeNeuralLayerDestroy(typeNeuralLayer* parent)
{
	if (parent == 0)
		return 0;

	typeNeuralArrayDestroy(&parent->layerOutputs);
	typeNeuralArrayDestroy(&parent->layerDeltas);

	memset(parent, 0, sizeof(typeNeuralLayer));

	return 1;
}

#endif // TYPENEURALLAYER_H
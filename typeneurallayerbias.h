#ifndef TYPENEURALLAYERBIAS_H
#define TYPENEURALLAYERBIAS_H

#include "typeneurallayer.h"

long typeNeuralLayerBiasCreate(typeNeuralLayer* parent, unsigned long width, unsigned long height, unsigned long depth)
{
	if (typeNeuralLayerCreate(parent, width, height, depth))
	{
		parent->layerType = TYPE_NEURAL_LAYER_BIAS;

		typeNeuralArrayCreate(&parent->layerWeights, width, height, depth);
		typeNeuralArrayCreate(&parent->layerVectors, width, height, depth);

		typeNeuralArrayFillRandom(&parent->layerWeights);
		typeNeuralArrayFillZero(&parent->layerVectors);

		return 1;
	}

	return 0;
}
long typeNeuralLayerBiasFeedForward(typeNeuralLayer* parent, typeNeuralLayer* next)
{
	if (parent == 0)
		return 0;

	if (parent->layerOutputs.arrayLength != next->layerOutputs.arrayLength)
		return 0;

	return typeNeuralArrayAdd(&next->layerOutputs, &parent->layerOutputs, &parent->layerWeights);
}

long typeNeuralLayerBiasBackPropagate(typeNeuralLayer* parent, typeNeuralLayer* next)
{
	if (parent == 0)
		return 0;

	if (parent->layerOutputs.arrayLength != next->layerOutputs.arrayLength)
		return 0;

	return typeNeuralArrayCopy(&parent->layerDeltas, &next->layerDeltas);
}

long typeNeuralLayerBiasUpdateWeights(typeNeuralLayer* parent, typeNeuralLayer* next, typeNeuralUnit learningrate, typeNeuralUnit momentum)
{
	typeNeuralArrayUpdateMomentum(&parent->layerWeights, &parent->layerVectors, &parent->layerDeltas, learningrate, momentum);

	return 1;
}

long typeNeuralLayerBiasDestroy(typeNeuralLayer* parent)
{
	if (parent == 0)
		return 0;

	typeNeuralArrayDestroy(&parent->layerWeights);
	typeNeuralArrayDestroy(&parent->layerVectors);

	return typeNeuralLayerDestroy(parent);
}

#endif // TYPENEURALLAYERBIAS_H


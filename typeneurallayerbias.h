#ifndef TYPENEURALLAYERBIAS_H
#define TYPENEURALLAYERBIAS_H

#include "typeneurallayer.h"

long typeNeuralLayerBiasCreate(typeNeuralLayer* parent, unsigned long width, unsigned long height, unsigned long depth)
{
	if (typeNeuralLayerCreate(parent, width, height, depth))
	{
		parent->layerType = TYPE_NEURAL_LAYER_BIAS;

		cudaNeuralArrayCreate(&parent->layerWeights, width, height, depth);
		cudaNeuralArrayCreate(&parent->layerVectors, width, height, depth);
		cudaNeuralArrayCreate(&parent->layerGammas, width, height, depth);

		cudaNeuralArrayFillZero(&parent->layerWeights);
		cudaNeuralArrayFillZero(&parent->layerVectors);
		cudaNeuralArrayFillZero(&parent->layerGammas);

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

	return cudaNeuralArrayAdd(&next->layerOutputs, &parent->layerOutputs, &parent->layerWeights);
}

long typeNeuralLayerBiasBackPropagate(typeNeuralLayer* parent, typeNeuralLayer* next)
{
	if (parent == 0)
		return 0;

	if (parent->layerOutputs.arrayLength != next->layerOutputs.arrayLength)
		return 0;

	return cudaNeuralArrayCopy(&parent->layerDeltas, &next->layerDeltas);
}

long typeNeuralLayerBiasUpdateWeights(typeNeuralLayer* parent, typeNeuralLayer* next, cudaNeuralUnit learningrate, cudaNeuralUnit momentum)
{
	cudaNeuralArrayUpdateAdagrad(&parent->layerWeights, &parent->layerVectors, &parent->layerGammas, &parent->layerDeltas, learningrate, momentum);

	return 1;
}

long typeNeuralLayerBiasDestroy(typeNeuralLayer* parent)
{
	if (parent == 0)
		return 0;

	cudaNeuralArrayDestroy(&parent->layerWeights);
	cudaNeuralArrayDestroy(&parent->layerVectors);

	return typeNeuralLayerDestroy(parent);
}

#endif // TYPENEURALLAYERBIAS_H


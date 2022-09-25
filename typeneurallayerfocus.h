#ifndef TYPENEURALLAYERFOCUS_H
#define TYPENEURALLAYERFOCUS_H

#include "typeneurallayer.h"

long typeNeuralLayerFocusCreate(typeNeuralLayer* parent, unsigned long width, unsigned long height, unsigned long depth, cudaNeuralUnit lambda, cudaNeuralUnit range)
{
	if (typeNeuralLayerCreate(parent, width, height, depth))
	{
		parent->layerType = TYPE_NEURAL_LAYER_FOCUS;
		parent->layerValue1 = lambda;
		parent->layerValue2 = range;

		cudaNeuralArrayCreate(&parent->layerAlphas, width, height, depth);

		cudaNeuralArrayFillZero(&parent->layerAlphas);

		return 1;
	}

	return 0;
}

long typeNeuralLayerFocusFeedForward(typeNeuralLayer* parent, typeNeuralLayer* next)
{
	if (parent == 0)
		return 0;

	if (parent->layerOutputs.arrayLength != next->layerOutputs.arrayLength)
		return 0;

	return cudaNeuralArrayCopy(&next->layerOutputs, &parent->layerOutputs);
}

long typeNeuralLayerFocusBackPropagate(typeNeuralLayer* parent, typeNeuralLayer* next)
{
	if (parent == 0)
		return 0;

	if (parent->layerOutputs.arrayLength != next->layerOutputs.arrayLength)
		return 0;

	return cudaNeuralArrayTransferFocus(&next->layerDeltas, &parent->layerDeltas, &parent->layerOutputs, &parent->layerAlphas, parent->layerValue1, parent->layerValue2);
}

long typeNeuralLayerFocusDestroy(typeNeuralLayer* parent)
{
	if (parent == 0)
		return 0;

	cudaNeuralArrayDestroy(&parent->layerAlphas);

	return typeNeuralLayerDestroy(parent);
}

#endif // TYPENEURALLAYERFOCUS_H



#ifndef TYPENEURALLAYERTRANSFER_H
#define TYPENEURALLAYERTRANSFER_H

#include "typeneurallayer.h"

long typeNeuralLayerTransferCreate(typeNeuralLayer* parent, unsigned long width, unsigned long height, unsigned long depth, unsigned long transfer)
{
	if (typeNeuralLayerCreate(parent, width, height, depth))
	{
		parent->layerType = TYPE_NEURAL_LAYER_TRANSFER;
		parent->layerValue1 = transfer;

		return 1;
	}

	return 0;
}
long typeNeuralLayerTransferFeedForward(typeNeuralLayer* parent, typeNeuralLayer* next)
{
	if (parent == 0)
		return 0;

	return typeNeuralArrayTransferForward(&next->layerOutputs, &parent->layerOutputs, parent->layerValue1);
}

long typeNeuralLayerTransferBackPropagate(typeNeuralLayer* parent, typeNeuralLayer* next)
{
	if (parent == 0)
		return 0;

	return typeNeuralArrayTransferReverse(&next->layerDeltas, &parent->layerDeltas, &parent->layerOutputs, parent->layerValue1);
}

long typeNeuralLayerTransferDestroy(typeNeuralLayer* parent)
{
	if (parent == 0)
		return 0;

	return typeNeuralLayerDestroy(parent);
}

#endif // TYPENEURALLAYERTRANSFER_H


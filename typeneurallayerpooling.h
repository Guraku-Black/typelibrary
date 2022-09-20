#ifndef TYPENEURALLAYERPOOLING_H
#define TYPENEURALLAYERPOOLING_H

#include "typeneurallayer.h"

long typeNeuralLayerPoolingCreate(typeNeuralLayer* parent,
	unsigned long sourcewidth, unsigned long sourceheight, unsigned long sourcedepth,
	unsigned long resultwidth, unsigned long resultheight, unsigned long resultdepth)
{
	typeShape   result;
	typeShape   source;

	if (typeNeuralLayerCreate(parent, sourcewidth, sourceheight, sourcedepth))
	{
		parent->layerType = TYPE_NEURAL_LAYER_POOLING;

		typeShapeAssign(&source, sourcewidth, sourceheight, 1);
		typeShapeAssign(&result, resultwidth, resultheight, 1);

		typeIndexMapCreateScale2D(&parent->layerMap1, &result, &source);

		return 1;
	}

	return 0;
}
long typeNeuralLayerPoolingFeedForward(typeNeuralLayer* parent, typeNeuralLayer* next)
{
	if (parent == 0)
		return 0;

	return typeNeuralArrayIndexMapPooling2DForward(&next->layerOutputs, &parent->layerOutputs, &parent->layerMap1);
}

long typeNeuralLayerPoolingBackPropagate(typeNeuralLayer* parent, typeNeuralLayer* next)
{
	if (parent == 0)
		return 0;

	return typeNeuralArrayIndexMapPooling2DReverse(&next->layerDeltas, &parent->layerDeltas, &parent->layerOutputs, &parent->layerMap1);
}

long typeNeuralLayerPoolingDestroy(typeNeuralLayer* parent)
{
	if (parent == 0)
		return 0;

	typeIndexMapDestroy(&parent->layerMap1);

	return typeNeuralLayerDestroy(parent);
}

#endif // TYPENEURALLAYERPOOLING_H

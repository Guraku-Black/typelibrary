#ifndef TYPENEURALLAYERPOOLING_H
#define TYPENEURALLAYERPOOLING_H

#include "typeneurallayer.h"

long typeNeuralLayerPoolingCreate(typeNeuralLayer* parent,
	unsigned long sourcewidth, unsigned long sourceheight, unsigned long sourcedepth,
	unsigned long resultwidth, unsigned long resultheight, unsigned long resultdepth)
{
	typeShape    result;
	typeShape    source;
	typeIndexMap indexmap1;

	if (typeNeuralLayerCreate(parent, sourcewidth, sourceheight, sourcedepth))
	{
		parent->layerType = TYPE_NEURAL_LAYER_POOLING;

		typeShapeAssign(&source, sourcewidth, sourceheight, 1);
		typeShapeAssign(&result, resultwidth, resultheight, 1);

		typeIndexMapCreateScale2D(&indexmap1, &result, &source);

		cudaIndexMapCreateCopy(&parent->layerMap1, indexmap1.indexWidth, indexmap1.indexHeight, indexmap1.indexData);

		typeIndexMapDestroy(&indexmap1);

		return 1;
	}

	return 0;
}
long typeNeuralLayerPoolingFeedForward(typeNeuralLayer* parent, typeNeuralLayer* next)
{
	if (parent == 0)
		return 0;

	return cudaNeuralArrayIndexMap2DPoolingForward(&next->layerOutputs, &parent->layerOutputs, &parent->layerMap1);
}

long typeNeuralLayerPoolingBackPropagate(typeNeuralLayer* parent, typeNeuralLayer* next)
{
	if (parent == 0)
		return 0;

	return cudaNeuralArrayIndexMap2DPoolingReverse(&next->layerDeltas, &parent->layerDeltas, &parent->layerOutputs, &parent->layerMap1);
}

long typeNeuralLayerPoolingDestroy(typeNeuralLayer* parent)
{
	if (parent == 0)
		return 0;

	cudaIndexMapDestroy(&parent->layerMap1);

	return typeNeuralLayerDestroy(parent);
}

#endif // TYPENEURALLAYERPOOLING_H

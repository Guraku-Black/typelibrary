#ifndef TYPENEURALLAYERSCALING_H
#define TYPENEURALLAYERSCALING_H

#include "typeneurallayer.h"

long typeNeuralLayerScalingCreate(typeNeuralLayer* parent, 
	unsigned long sourcewidth, unsigned long sourceheight, unsigned long sourcedepth,
	unsigned long resultwidth, unsigned long resultheight, unsigned long resultdepth)
{
	typeShape    result;
	typeShape    source;
	typeIndexMap indexmap1;
	typeIndexMap indexmap2;

	if (typeNeuralLayerCreate(parent, sourcewidth, sourceheight, sourcedepth))
	{
		parent->layerType = TYPE_NEURAL_LAYER_SCALING;

		typeShapeAssign(&source, sourcewidth, sourceheight, 1);
		typeShapeAssign(&result, resultwidth, resultheight, 1);

		typeIndexMapCreateScale2D(&indexmap1, &result, &source);
		typeIndexMapCreateScale2D(&indexmap2, &source, &result);

		cudaIndexMapCreateCopy(&parent->layerMap1, indexmap1.indexWidth, indexmap1.indexHeight, indexmap1.indexData);
		cudaIndexMapCreateCopy(&parent->layerMap2, indexmap2.indexWidth, indexmap2.indexHeight, indexmap2.indexData);

		typeIndexMapDestroy(&indexmap1);
		typeIndexMapDestroy(&indexmap2);

		return 1;
	}

	return 0;
}
long typeNeuralLayerScalingFeedForward(typeNeuralLayer* parent, typeNeuralLayer* next)
{
	if (parent == 0)
		return 0;

	return cudaNeuralArrayIndexMap2DCopy(&next->layerOutputs, &parent->layerOutputs, &parent->layerMap1);
}

long typeNeuralLayerScalingBackPropagate(typeNeuralLayer* parent, typeNeuralLayer* next)
{
	if (parent == 0)
		return 0;

	return cudaNeuralArrayIndexMap2DCopy(&parent->layerDeltas, &next->layerDeltas, &parent->layerMap2);
}


long typeNeuralLayerScalingDestroy(typeNeuralLayer* parent)
{
	if (parent == 0)
		return 0;

	cudaIndexMapDestroy(&parent->layerMap1);
	cudaIndexMapDestroy(&parent->layerMap2);

	return typeNeuralLayerDestroy(parent);
}

#endif // TYPENEURALLAYERSCALING_H
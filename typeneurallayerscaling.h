#ifndef TYPENEURALLAYERSCALING_H
#define TYPENEURALLAYERSCALING_H

#include "typeneurallayer.h"

long typeNeuralLayerScalingCreate(typeNeuralLayer* parent, 
	unsigned long sourcewidth, unsigned long sourceheight, unsigned long sourcedepth,
	unsigned long resultwidth, unsigned long resultheight, unsigned long resultdepth)
{
	typeShape   result;
	typeShape   source;

	if (typeNeuralLayerCreate(parent, sourcewidth, sourceheight, sourcedepth))
	{
		parent->layerType = TYPE_NEURAL_LAYER_SCALING;

		typeShapeAssign(&source, sourcewidth, sourceheight, 1);
		typeShapeAssign(&result, resultwidth, resultheight, 1);

		typeIndexMapCreateScale2D(&parent->layerMap1, &result, &source);
		typeIndexMapCreateScale2D(&parent->layerMap2, &source, &result);

		return 1;
	}

	return 0;
}
long typeNeuralLayerScalingFeedForward(typeNeuralLayer* parent, typeNeuralLayer* next)
{
	if (parent == 0)
		return 0;

	return typeNeuralArrayIndexMapCopy2D(&next->layerOutputs, &parent->layerOutputs, &parent->layerMap1);
}

long typeNeuralLayerScalingBackPropagate(typeNeuralLayer* parent, typeNeuralLayer* next)
{
	if (parent == 0)
		return 0;

	return typeNeuralArrayIndexMapCopy2D(&parent->layerDeltas, &next->layerDeltas, &parent->layerMap2);
}


long typeNeuralLayerScalingDestroy(typeNeuralLayer* parent)
{
	if (parent == 0)
		return 0;

	return typeNeuralLayerDestroy(parent);
}



#endif // TYPENEURALLAYERSCALING_H
#ifndef TYPENEURALNETWORK_H
#define TYPENEURALNETWORK_H

#include "typedynamicarray.h"
#include "typeneurallayer.h"
#include "typeneurallayerbias.h"
#include "typeneurallayerscaling.h"
#include "typeneurallayerpooling.h"

typedef struct
{
	unsigned long    networkLength;
	typeDynamicArray networkData;
} typeNeuralNetwork;

long typeNeuralNetworkCreate(typeNeuralNetwork* parent);
long typeNeuralNetworkAddLayerNull(typeNeuralNetwork* parent, unsigned long width, unsigned height, unsigned long depth);
long typeNeuralNetworkAddLayerBias(typeNeuralNetwork* parent, unsigned long width, unsigned height, unsigned long depth);
long typeNeuralNetworkAddLayerScaling(typeNeuralNetwork* parent, unsigned long sourcewidth, unsigned long sourceheight, unsigned long sourcedepth, unsigned long resultwidth, unsigned long resultheight, unsigned long resultdepth);
long typeNeuralNetworkAddLayerPooling(typeNeuralNetwork* parent, unsigned long sourcewidth, unsigned long sourceheight, unsigned long sourcedepth, unsigned long resultwidth, unsigned long resultheight, unsigned long resultdepth);
long typeNeuralNetworkFeedForwardArray(typeNeuralNetwork* parent, typeNeuralArray* source);
long typeNeuralNetworkBackPropagateArray(typeNeuralNetwork* parent, typeNeuralArray* target);
long typeNeuralNetworkGetOutputs(typeNeuralNetwork* parent, typeNeuralArray* result);
long typeNeuralNetworkFeedForward(typeNeuralNetwork* parent);
long typeNeuralNetworkBackPropagate(typeNeuralNetwork* parent);
long typeNeuralNetworkDestroy(typeNeuralNetwork* parent);

const int  neuralnetwork_page = sizeof(typeNeuralLayer) * 10;

long typeNeuralNetworkCreate(typeNeuralNetwork* parent)
{
	if (parent == 0)
		return 0;

	if (typeDynamicArrayCreate(&parent->networkData, neuralnetwork_page))
	{
		parent->networkLength = 0;

		return 1;
	}

	parent->networkLength = 0;

	return 0;
}

static typeNeuralLayer* typeNeuralNetworkAddLayer(typeNeuralNetwork* parent)
{
	typeNeuralLayer* layers;

	if (parent == 0)
		return 0;

	if (typeDynamicArrayResize(&parent->networkData, sizeof(typeNeuralLayer) * (parent->networkLength + 1), neuralnetwork_page))
	{
		parent->networkLength++;

		layers = (typeNeuralLayer*)parent->networkData.arrayData;

		return &layers[parent->networkLength - 1];
	}

	return 0;
}

long typeNeuralNetworkAddLayerNull(typeNeuralNetwork* parent, unsigned long width, unsigned height, unsigned long depth)
{
	typeNeuralLayer* layers;

	if (parent == 0)
		return 0;

	layers = typeNeuralNetworkAddLayer(parent);

	return typeNeuralLayerCreate(layers, width, height, depth);
}

long typeNeuralNetworkAddLayerBias(typeNeuralNetwork* parent, unsigned long width, unsigned height, unsigned long depth)
{
	typeNeuralLayer* layers;

	if (parent == 0)
		return 0;

	layers = typeNeuralNetworkAddLayer(parent);

	return typeNeuralLayerBiasCreate(layers, width, height, depth);
}

long typeNeuralNetworkAddLayerScaling(typeNeuralNetwork* parent, 
	unsigned long sourcewidth, unsigned long sourceheight, unsigned long sourcedepth,
	unsigned long resultwidth, unsigned long resultheight, unsigned long resultdepth)
{
	typeNeuralLayer* layers;

	if (parent == 0)
		return 0;

	layers = typeNeuralNetworkAddLayer(parent);

	return typeNeuralLayerScalingCreate(layers,
		sourcewidth, sourceheight, sourcedepth,
		resultwidth, resultheight, resultdepth);
}

long typeNeuralNetworkAddLayerPooling(typeNeuralNetwork* parent, 
	unsigned long sourcewidth, unsigned long sourceheight, unsigned long sourcedepth,
	unsigned long resultwidth, unsigned long resultheight, unsigned long resultdepth)
{
	typeNeuralLayer* layers;

	if (parent == 0)
		return 0;

	layers = typeNeuralNetworkAddLayer(parent);

	return typeNeuralLayerPoolingCreate(layers,
		sourcewidth, sourceheight, sourcedepth,
		resultwidth, resultheight, resultdepth);
}

long typeNeuralNetworkFeedForwardArray(typeNeuralNetwork* parent, typeNeuralArray* source)
{
	typeNeuralLayer* layers;

	if (parent == 0)
		return 0;

	layers = (typeNeuralLayer*)parent->networkData.arrayData;

	if (layers)
	{
		typeNeuralArrayCopy(&layers[0].layerOutputs, source);

		return typeNeuralNetworkFeedForward(parent);
	}

	return 0;
}

long typeNeuralNetworkBackPropagateArray(typeNeuralNetwork* parent, typeNeuralArray* target)
{
	typeNeuralLayer* layers;

	if (parent == 0)
		return 0;

	layers = (typeNeuralLayer*)parent->networkData.arrayData;

	if (layers)
	{
		typeNeuralArraySubtract(
			&layers[parent->networkLength - 1].layerDeltas, 
			target, 
			&layers[parent->networkLength - 1].layerOutputs);

		return typeNeuralNetworkBackPropagate(parent);
	}

	return 0;
}

long typeNeuralNetworkGetOutputs(typeNeuralNetwork* parent, typeNeuralArray* result)
{
	typeNeuralLayer* layers;

	if (parent == 0)
		return 0;

	layers = (typeNeuralLayer*)parent->networkData.arrayData;

	if (layers)
	{
		return typeNeuralArrayCopy(result, &layers[parent->networkLength - 1].layerOutputs);
	}

	return 0;
}

long typeNeuralNetworkFeedForward(typeNeuralNetwork* parent)
{
	typeNeuralLayer* layers;
	unsigned long    I;

	if (parent == 0)
		return 0;

	if (parent->networkLength < 2)
		return 0;

	layers = (typeNeuralLayer*)parent->networkData.arrayData;

	if (layers)
	{
		for (I = 0; I < (parent->networkLength - 1); I++)
		{
			switch (layers[I].layerType)
			{
			case TYPE_NEURAL_LAYER_NULL:
				typeNeuralLayerFeedForward(&layers[I], &layers[I + 1]);
				break;
			case TYPE_NEURAL_LAYER_BIAS:
				typeNeuralLayerBiasFeedForward(&layers[I], &layers[I + 1]);
				break;
			case TYPE_NEURAL_LAYER_SCALING:
				typeNeuralLayerScalingFeedForward(&layers[I], &layers[I + 1]);
				break;
			case TYPE_NEURAL_LAYER_POOLING:
				typeNeuralLayerPoolingFeedForward(&layers[I], &layers[I + 1]);
				break;
			}
		}
	}

	return 1;
}

long typeNeuralNetworkBackPropagate(typeNeuralNetwork* parent)
{
	typeNeuralLayer* layers;
	long             I;

	if (parent == 0)
		return 0;

	if (parent->networkLength < 2)
		return 0;

	layers = (typeNeuralLayer*)parent->networkData.arrayData;

	if (layers)
	{
		for (I = (parent->networkLength - 2); I >= 0; I--)
		{
			switch (layers[I].layerType)
			{
			case TYPE_NEURAL_LAYER_NULL:
				typeNeuralLayerBackPropagate(&layers[I], &layers[I + 1]);
				break;
			case TYPE_NEURAL_LAYER_BIAS:
				typeNeuralLayerBiasBackPropagate(&layers[I], &layers[I + 1]);
				break;
			case TYPE_NEURAL_LAYER_SCALING:
				typeNeuralLayerScalingBackPropagate(&layers[I], &layers[I + 1]);
				break;
			case TYPE_NEURAL_LAYER_POOLING:
				typeNeuralLayerPoolingBackPropagate(&layers[I], &layers[I + 1]);
				break;
			}
		}
	}

	return 1;
}

long typeNeuralNetworkUpdateWeights(typeNeuralNetwork* parent)
{
	typeNeuralLayer* layers;
	unsigned long    I;

	if (parent == 0)
		return 0;

	if (parent->networkLength < 2)
		return 0;

	layers = (typeNeuralLayer*)parent->networkData.arrayData;

	if (layers)
	{
		for (I = 0; I < (parent->networkLength - 1); I++)
		{
			switch (layers[I].layerType)
			{
			case TYPE_NEURAL_LAYER_BIAS:
				typeNeuralLayerBiasUpdateWeights(&layers[I], &layers[I + 1], 0.01, 0.9);
				break;
			}
		}
	}

	return 1;
}

long typeNeuralNetworkDestroy(typeNeuralNetwork* parent)
{
	typeNeuralLayer* layers;
	unsigned long    I;

	if (parent == 0)
		return 0;

	layers = (typeNeuralLayer*)parent->networkData.arrayData;

	if (layers)
	{
		for (I = 0; I < parent->networkLength; I++)
		{
			switch (layers[I].layerType)
			{
			case TYPE_NEURAL_LAYER_NULL:
				typeNeuralLayerDestroy(&layers[I]);
				break;
			case TYPE_NEURAL_LAYER_BIAS:
				typeNeuralLayerBiasDestroy(&layers[I]);
				break;
			case TYPE_NEURAL_LAYER_SCALING:
				typeNeuralLayerScalingDestroy(&layers[I]);
				break;
			case TYPE_NEURAL_LAYER_POOLING:
				typeNeuralLayerPoolingDestroy(&layers[I]);
				break;
			}
		}
	}

	if (typeDynamicArrayDestroy(&parent->networkData))
	{
		parent->networkLength = 0;

		return 1;
	}

	return 0;
}
#endif // TYPENEURALNETWORK_H

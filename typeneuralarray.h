#ifndef TYPENEURALARRAY_H
#define TYPENEURALARRAY_H

#include "typeshape.h"
#include "typeindexmap.h"

typedef double     typeNeuralUnit;

typedef struct
{
	typeShape       arrayShape;
	unsigned long   arrayLength;
	unsigned long   arraySize;
	typeNeuralUnit* arrayData;
} typeNeuralArray;

long typeNeuralArrayCreate(typeNeuralArray* parent, unsigned long width, unsigned long height, unsigned long depth);
long typeNeuralArraySetImageRGB(typeNeuralArray* parent, unsigned long imagewidth, unsigned long imageheight, unsigned long* imagedata);
long typeNeuralArrayGetImageRGB(typeNeuralArray* parent, unsigned long imagewidth, unsigned long imageheight, unsigned long* imagedata);
long typeNeuralArrayCopy(typeNeuralArray* parent, typeNeuralArray* source);
long typeNeuralArrayFillZero(typeNeuralArray* parent);
long typeNeuralArrayFillOne(typeNeuralArray* parent);
long typeNeuralArrayFillRandom(typeNeuralArray* parent);
long typeNeuralArrayAdd(typeNeuralArray* parent, typeNeuralArray* value1, typeNeuralArray* value2);
long typeNeuralArraySubtract(typeNeuralArray* parent, typeNeuralArray* value1, typeNeuralArray* value2);
long typeNeuralArrayMultiply(typeNeuralArray* parent, typeNeuralArray* value1, typeNeuralArray* value2);
long typeNeuralArrayDivide(typeNeuralArray* parent, typeNeuralArray* value1, typeNeuralArray* value2);
long typeNeuralArrayIndexMapCopy2D(typeNeuralArray* parent, typeNeuralArray* source, typeIndexMap* indexmap);
long typeNeuralArrayIndexMapPooling2DForward(typeNeuralArray* parent, typeNeuralArray* source, typeIndexMap* indexmap);
long typeNeuralArrayIndexMapPooling2DReverse(typeNeuralArray* parent, typeNeuralArray* source, typeNeuralArray* outputs, typeIndexMap* indexmap);
long typeNeuralArrayMatrixForward(typeNeuralArray* result, typeNeuralArray* source, typeNeuralArray* weights);
long typeNeuralArrayMatrixReverse(typeNeuralArray* result, typeNeuralArray* source, typeNeuralArray* outputs, typeNeuralArray* weights);
long typeNeuralArrayUpdateMomentum(typeNeuralArray* weights, typeNeuralArray* vectors, typeNeuralArray* deltas, typeNeuralUnit learningrate, typeNeuralUnit momentum);
long typeNeuralArrayDestroy(typeNeuralArray* parent);

long typeNeuralArrayCreate(typeNeuralArray* parent, unsigned long width, unsigned long height, unsigned long depth)
{
	typeNeuralUnit* data;
	unsigned long   length;
	unsigned long   datalength;

	if (parent == 0)
		return 0;

	typeShapeAssign(&parent->arrayShape, width, height, depth);
	length = typeShapeGetLength(&parent->arrayShape);
	datalength = sizeof(typeNeuralUnit) * length;
	data = (typeNeuralUnit*)malloc(datalength);

	if (data)
	{
		parent->arrayLength = length;
		parent->arraySize = datalength;
		parent->arrayData = data;

		return 1;
	}

	typeShapeZero(&parent->arrayShape);
	parent->arrayLength = 0;
	parent->arraySize = 0;
	parent->arrayData = 0;

	return 0;
}

long typeNeuralArraySetImageRGB(typeNeuralArray* parent, unsigned long imagewidth, unsigned long imageheight, unsigned long* imagedata)
{
	unsigned long   I, J;
	unsigned long   A, B, C;
	unsigned long   length;
	unsigned char* color;
	typeNeuralUnit* buffer;

	if ((parent == 0) || (imagedata == 0))
		return 0;

	if (typeShapeCompareValue(&parent->arrayShape, imagewidth, imageheight, 3) == 0)
		return 0;

	color = (unsigned char*)imagedata;
	buffer = (typeNeuralUnit*)parent->arrayData;
	length = imagewidth * imageheight;
	A = length * 0;
	B = length * 1;
	C = length * 2;

	J = 0;
	for (I = 0; I < length; I++)
	{
		buffer[A + I] = (typeNeuralUnit)color[J + 0] / 255;
		buffer[B + I] = (typeNeuralUnit)color[J + 1] / 255;
		buffer[C + I] = (typeNeuralUnit)color[J + 2] / 255;

		J += 4;
	}

	return 1;
}

double neural_maxout(double a)
{
	if (a < 0)
		return 0;

	if (a > 1)
		return 1;

	return a;
}
long typeNeuralArrayGetImageRGB(typeNeuralArray* parent, unsigned long imagewidth, unsigned long imageheight, unsigned long* imagedata)
{
	unsigned long   I, J;
	unsigned long   A, B, C;
	unsigned long   length;
	unsigned char* color;
	typeNeuralUnit* buffer;

	if ((parent == 0) || (imagedata == 0))
		return 0;

	if (typeShapeCompareValue(&parent->arrayShape, imagewidth, imageheight, 3) == 0)
		return 0;

	color = (unsigned char*)imagedata;
	buffer = (typeNeuralUnit*)parent->arrayData;
	length = imagewidth * imageheight;
	A = length * 0;
	B = length * 1;
	C = length * 2;

	J = 0;
	for (I = 0; I < length; I++)
	{
		color[J + 0] = (unsigned char)(neural_maxout(buffer[A + I]) * 255);
		color[J + 1] = (unsigned char)(neural_maxout(buffer[B + I]) * 255);
		color[J + 2] = (unsigned char)(neural_maxout(buffer[C + I]) * 255);

		J += 4;
	}

	return 1;
}

long typeNeuralArrayCopy(typeNeuralArray* parent, typeNeuralArray* source)
{
	unsigned long   I;

	if ((parent == 0) || (source == 0))
		return 0;

	if (parent->arrayLength != source->arrayLength)
		return 0;

	for (I = 0; I < parent->arrayLength; I++)
	{
		parent->arrayData[I] = source->arrayData[I];
	}

	return 1;
}

long typeNeuralArrayFillZero(typeNeuralArray* parent)
{
	unsigned long   I;

	if (parent == 0)
		return 0;

	for (I = 0; I < parent->arrayLength; I++)
	{
		parent->arrayData[I] = 0;
	}

	return 1;
}

long typeNeuralArrayFillOne(typeNeuralArray* parent)
{
	unsigned long   I;

	if (parent == 0)
		return 0;

	for (I = 0; I < parent->arrayLength; I++)
	{
		parent->arrayData[I] = 1;
	}

	return 1;
}
long typeNeuralArrayFillRandom(typeNeuralArray* parent)
{
	unsigned long   I;

	if (parent == 0)
		return 0;

	type_randomseed();

	for (I = 0; I < parent->arrayLength; I++)
	{
		parent->arrayData[I] = 1 - (type_random() + type_random());
	}

	return 1;
}

long typeNeuralArrayAdd(typeNeuralArray* parent, typeNeuralArray* value1, typeNeuralArray* value2)
{
	unsigned long   I;

	if ((parent == 0) || (value1 == 0) || (value2 == 0))
		return 0;

	if ((parent->arrayLength != value1->arrayLength) ||
		(parent->arrayLength != value2->arrayLength))
		return 0;

	for (I = 0; I < parent->arrayLength; I++)
	{
		parent->arrayData[I] = value1->arrayData[I] + value2->arrayData[I];
	}

	return 1;
}

long typeNeuralArraySubtract(typeNeuralArray* parent, typeNeuralArray* value1, typeNeuralArray* value2)
{
	unsigned long   I;

	if ((parent == 0) || (value1 == 0) || (value2 == 0))
		return 0;

	if ((parent->arrayLength != value1->arrayLength) ||
		(parent->arrayLength != value2->arrayLength))
		return 0;

	for (I = 0; I < parent->arrayLength; I++)
	{
		parent->arrayData[I] = value1->arrayData[I] - value2->arrayData[I];
	}

	return 1;
}

long typeNeuralArrayMultiply(typeNeuralArray* parent, typeNeuralArray* value1, typeNeuralArray* value2)
{
	unsigned long   I;

	if ((parent == 0) || (value1 == 0) || (value2 == 0))
		return 0;

	if ((parent->arrayLength != value1->arrayLength) ||
		(parent->arrayLength != value2->arrayLength))
		return 0;

	for (I = 0; I < parent->arrayLength; I++)
	{
		parent->arrayData[I] = value1->arrayData[I] * value2->arrayData[I];
	}

	return 1;
}

long typeNeuralArrayDivide(typeNeuralArray* parent, typeNeuralArray* value1, typeNeuralArray* value2)
{
	unsigned long   I;

	if ((parent == 0) || (value1 == 0) || (value2 == 0))
		return 0;

	if ((parent->arrayLength != value1->arrayLength) ||
		(parent->arrayLength != value2->arrayLength))
		return 0;

	for (I = 0; I < parent->arrayLength; I++)
	{
		if (value2->arrayData[I] != 0)
			parent->arrayData[I] = value1->arrayData[I] * value2->arrayData[I];
		else
			parent->arrayData[I] = 0;
	}

	return 1;
}

static void kernelArrayIndexMapCopy2D(
	typeNeuralUnit* result,
	typeNeuralUnit* source,
	unsigned long   indexwidth,
	unsigned long   indexheight,
	unsigned long* indexmap,
	unsigned long resultindex,
	unsigned long resultlength,
	unsigned long sourceindex,
	unsigned long sourcelength)
{
	typeNeuralUnit sum;
	unsigned long  I, J, K;
	unsigned long  counter;
	unsigned long  index;

	for (I = 0; I < indexwidth; I++)
	{
		sum = 0;
		counter = 0;
		K = I;

		for (J = 0; J < indexheight; J++)
		{
			index = indexmap[K];
			if ((index != TYPE_INVALID_INDEX) &&
				(index < sourcelength))
			{
				sum += source[sourceindex + index];
				counter++;
			}

			K += indexwidth;
		}

		if (counter > 1)
		{
			sum /= counter;
		}

		result[resultindex + I] = sum;
	}
}

long typeNeuralArrayIndexMapCopy2D(typeNeuralArray* parent, typeNeuralArray* source, typeIndexMap* indexmap)
{
	unsigned long   I;
	unsigned long   parentlength;
	unsigned long   sourcelength;

	if ((parent == 0) || (source == 0) || (indexmap == 0))
		return 0;

	parentlength = (parent->arrayShape.shapeWidth * parent->arrayShape.shapeHeight);
	sourcelength = (source->arrayShape.shapeWidth * source->arrayShape.shapeHeight);

	if (indexmap->indexWidth != parentlength)
		return 0;

	if (parent->arrayShape.shapeDepth != source->arrayShape.shapeDepth)
		return 0;

	for (I = 0; I < parent->arrayShape.shapeDepth; I++)
	{
		kernelArrayIndexMapCopy2D(
			parent->arrayData,
			source->arrayData,
			indexmap->indexWidth,
			indexmap->indexHeight,
			indexmap->indexData,
			parentlength * I,
			parentlength,
			sourcelength * I,
			sourcelength);
	}

	return 1;
}

static void kernelArrayIndexMapPooling2DForward(
	typeNeuralUnit* result,
	typeNeuralUnit* source,
	unsigned long   indexwidth,
	unsigned long   indexheight,
	unsigned long* indexmap,
	unsigned long resultindex,
	unsigned long resultlength,
	unsigned long sourceindex,
	unsigned long sourcelength)
{
	typeNeuralUnit sum;
	unsigned long  I, J, K;
	unsigned long  index;

	for (I = 0; I < indexwidth; I++)
	{
		sum = 0;
		K = I;

		for (J = 0; J < indexheight; J++)
		{
			index = indexmap[K];
			if ((index != TYPE_INVALID_INDEX) &&
				(index < sourcelength))
			{
				if (type_abs(source[sourceindex + index]) > type_abs(sum))
					sum = source[sourceindex + index];
			}

			K += indexwidth;
		}

		result[resultindex + I] = sum;
	}
}

long typeNeuralArrayIndexMapPooling2DForward(typeNeuralArray* parent, typeNeuralArray* source, typeIndexMap* indexmap)
{
	unsigned long   I;
	unsigned long   parentlength;
	unsigned long   sourcelength;

	if ((parent == 0) || (source == 0) || (indexmap == 0))
		return 0;

	parentlength = (parent->arrayShape.shapeWidth * parent->arrayShape.shapeHeight);
	sourcelength = (source->arrayShape.shapeWidth * source->arrayShape.shapeHeight);

	if (indexmap->indexWidth != parentlength)
		return 0;

	if (parent->arrayShape.shapeDepth != source->arrayShape.shapeDepth)
		return 0;

	for (I = 0; I < parent->arrayShape.shapeDepth; I++)
	{
		kernelArrayIndexMapPooling2DForward(
			parent->arrayData,
			source->arrayData,
			indexmap->indexWidth,
			indexmap->indexHeight,
			indexmap->indexData,
			parentlength * I,
			parentlength,
			sourcelength * I,
			sourcelength);
	}

	return 1;
}

static void kernelArrayIndexMapPooling2DReverse(
	typeNeuralUnit* result,
	typeNeuralUnit* source,
	typeNeuralUnit* outputs,
	unsigned long   indexwidth,
	unsigned long   indexheight,
	unsigned long* indexmap,
	unsigned long resultindex,
	unsigned long resultlength,
	unsigned long sourceindex,
	unsigned long sourcelength)
{
	typeNeuralUnit sum;
	unsigned long  I, J, K;
	unsigned long  counter;
	unsigned long  index;

	for (I = 0; I < indexwidth; I++)
	{
		sum = 0;
		counter = 0;
		K = I;

		for (J = 0; J < indexheight; J++)
		{
			index = indexmap[K];
			if ((index != TYPE_INVALID_INDEX) &&
				(index < sourcelength))
			{
				if (type_abs(outputs[sourceindex + index]) > type_abs(sum))
				{
					sum = outputs[sourceindex + index];
					counter = J;
				}
			}

			K += indexwidth;
		}

		K = I;
		for (J = 0; J < indexheight; J++)
		{
			index = indexmap[K];
			if ((index != TYPE_INVALID_INDEX) &&
				(index < sourcelength))
			{
				if (counter == J)
					source[sourceindex + index] = result[resultindex + I];
				else
					source[sourceindex + index] = 0;
			}

			K += indexwidth;
		}
	}
}

long typeNeuralArrayIndexMapPooling2DReverse(typeNeuralArray* parent, typeNeuralArray* source, typeNeuralArray* outputs, typeIndexMap* indexmap)
{
	unsigned long   I;
	unsigned long   parentlength;
	unsigned long   sourcelength;

	if ((parent == 0) || (source == 0) || (indexmap == 0))
		return 0;

	parentlength = (parent->arrayShape.shapeWidth * parent->arrayShape.shapeHeight);
	sourcelength = (source->arrayShape.shapeWidth * source->arrayShape.shapeHeight);

	if (indexmap->indexWidth != parentlength)
		return 0;

	if (parent->arrayShape.shapeDepth != source->arrayShape.shapeDepth)
		return 0;

	if (typeShapeCompare(&source->arrayShape, &outputs->arrayShape))
		return 0;

	for (I = 0; I < parent->arrayShape.shapeDepth; I++)
	{
		kernelArrayIndexMapPooling2DReverse(
			parent->arrayData,
			source->arrayData,
			outputs->arrayData,
			indexmap->indexWidth,
			indexmap->indexHeight,
			indexmap->indexData,
			parentlength * I,
			parentlength,
			sourcelength * I,
			sourcelength);
	}

	return 1;
}

static typeNeuralUnit indexMapConvolution2DValue(
	typeNeuralUnit* source,
	typeNeuralUnit* filter,
	unsigned long   indexwidth,
	unsigned long   indexheight,
	unsigned long* indexmap,
	unsigned long  indexoffset,
	unsigned long  sourceindex,
	unsigned long  sourcelength,
	unsigned long  filterindex,
	unsigned long  filterlength)
{
	unsigned long  I, J;
	unsigned long  index;
	typeNeuralUnit sum;

	sum = 0;
	J = indexoffset;
	for (I = 0; I < indexheight; I++)
	{
		index = indexmap[J];
		if ((index != TYPE_INVALID_INDEX) &&
			(index < sourcelength))
		{
			sum += source[sourceindex + index] * filter[filterindex + I];
		}

		J += indexwidth;
	}

	return sum;
}

long typeNeuralArrayConvolution2DForward(typeNeuralArray* result, typeNeuralArray* source, typeNeuralArray* filter, typeIndexMap* indexmap)
{
	unsigned long   I, J, K;
	unsigned long   resultlength;
	unsigned long   resultcount;
	unsigned long   sourcelength;
	unsigned long   sourcecount;
	unsigned long   filterlength;
	unsigned long   filtercount;
	typeNeuralUnit  sum;

	if ((result == 0) || (source == 0) || (filter == 0))
		return 0;

	resultlength = (result->arrayShape.shapeWidth * result->arrayShape.shapeHeight);
	resultcount = result->arrayShape.shapeDepth;
	sourcelength = (source->arrayShape.shapeWidth * source->arrayShape.shapeHeight);
	sourcecount = source->arrayShape.shapeDepth;;
	filterlength = (filter->arrayShape.shapeWidth * filter->arrayShape.shapeHeight);
	filtercount = filter->arrayShape.shapeDepth;;

	if (filtercount != (resultcount * sourcecount))
		return 0;

	if ((indexmap->indexWidth != resultlength) ||
		(indexmap->indexHeight != filterlength))
		return 0;

	for (J = 0; J < resultcount; J++)
	{
		for (K = 0; K < resultlength; K++)
		{
			sum = 0;
			for (I = 0; I < sourcecount; I++)
			{
				sum += indexMapConvolution2DValue(
					source->arrayData,
					filter->arrayData,
					indexmap->indexWidth,
					indexmap->indexHeight,
					indexmap->indexData,
					K,
					sourcelength * I,
					sourcelength,
					filterlength * (J * sourcecount + I),
					filterlength);
			}
			
			result->arrayData[J * resultlength + K] = sum;
		}

	}

	return 1;
}

long typeNeuralArrayConvolution2DReverse(typeNeuralArray* result, typeNeuralArray* source, typeNeuralArray* filter, typeIndexMap* indexmap)
{
	unsigned long   I, J, K;
	unsigned long   resultlength;
	unsigned long   resultcount;
	unsigned long   sourcelength;
	unsigned long   sourcecount;
	unsigned long   filterlength;
	unsigned long   filtercount;
	typeNeuralUnit  sum;

	if ((result == 0) || (source == 0) || (filter == 0))
		return 0;

	resultlength = (result->arrayShape.shapeWidth * result->arrayShape.shapeHeight);
	resultcount = result->arrayShape.shapeDepth;
	sourcelength = (source->arrayShape.shapeWidth * source->arrayShape.shapeHeight);
	sourcecount = source->arrayShape.shapeDepth;;
	filterlength = (filter->arrayShape.shapeWidth * filter->arrayShape.shapeHeight);
	filtercount = filter->arrayShape.shapeDepth;;

	if (filtercount != (resultcount * sourcecount))
		return 0;

	if ((indexmap->indexWidth != resultlength) ||
		(indexmap->indexHeight != filterlength))
		return 0;

	for (I = 0; I < sourcecount; I++)
	{
		for (K = 0; K < sourcelength; K++)
		{
			sum = 0;
			for (J = 0; J < resultcount; J++)
			{
				sum += indexMapConvolution2DValue(
					result->arrayData,
					filter->arrayData,
					indexmap->indexWidth,
					indexmap->indexHeight,
					indexmap->indexData,
					K,
					resultlength * I,
					resultlength,
					filterlength * (J * sourcecount + I),
					filterlength);
			}

			source->arrayData[I * sourcelength + K] = sum;
		}
	}

	return 1;
}

long typeNeuralArrayMatrixForward(typeNeuralArray* result, typeNeuralArray* source, typeNeuralArray* weights)
{
	unsigned long   I, J, K;
	typeNeuralUnit  sum;

	if ((result == 0) || (source == 0) || (weights == 0))
		return 0;

	if (weights->arrayLength != (result->arrayLength * source->arrayLength))
		return 0;

	for (J = 0; J < result->arrayLength; J++)
	{
		sum = 0;
		K = J * source->arrayLength;
		for (I = 0; I < source->arrayLength; I++)
		{
			sum += source->arrayData[I] * weights->arrayData[K + I];
		}

		result->arrayData[J] = sum;
	}

	return 1;
}

long typeNeuralArrayMatrixReverse(typeNeuralArray* result, typeNeuralArray* source, typeNeuralArray* outputs, typeNeuralArray* weights)
{
	unsigned long   I, J, K;
	typeNeuralUnit  sum;

	if ((result == 0) || (source == 0) || (weights == 0))
		return 0;

	if (weights->arrayLength != (result->arrayLength * source->arrayLength))
		return 0;

	for (I = 0; I < source->arrayLength; I++)
	{
		sum = 0;
		K = I;
		for (J = 0; J < result->arrayLength; J++)
		{
			sum += result->arrayData[J] * weights->arrayData[K];

			K += source->arrayLength;
		}

		source->arrayData[I] = sum;
	}

	return 1;
}

long typeNeuralArrayUpdateMomentum(typeNeuralArray* weights, typeNeuralArray* vectors, typeNeuralArray* deltas, typeNeuralUnit learningrate, typeNeuralUnit momentum)
{
	unsigned long  I;

	if ((weights == 0) || (vectors == 0) || (deltas == 0))
		return 0;

	if ((weights->arrayLength != vectors->arrayLength) ||
		(weights->arrayLength != deltas->arrayLength))
		return 0;

	for (I = 0; I < weights->arrayLength; I++)
	{
		vectors->arrayData[I] = (vectors->arrayData[I] * momentum) + (1 - momentum) * (learningrate * deltas->arrayData[I]);

		weights->arrayData[I] = weights->arrayData[I] + vectors->arrayData[I];
	}

	return 1;
}

long typeNeuralArrayDestroy(typeNeuralArray* parent)
{
	if (parent == 0)
		return 0;

	free(parent->arrayData);

	parent->arrayLength = 0;
	parent->arraySize = 0;
	parent->arrayData = 0;

	return 1;
}

#endif // TYPENEURALARRAY_H

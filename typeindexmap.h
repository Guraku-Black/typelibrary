#ifndef TYPEINDEXMAP_H
#define TYPEINDEXMAP_H

#include <stdlib.h>
#include "typemath.h"
#include "typeshape.h"

typedef struct
{
	unsigned long  indexWidth;
	unsigned long  indexHeight;
	unsigned long  indexLength;
	unsigned long  indexSize;
	unsigned long* indexData;
} typeIndexMap;

long typeIndexMapCreate(typeIndexMap* parent, unsigned long width, unsigned long height);
long typeIndexMapCreateFlip(typeIndexMap* parent, typeShape* result, typeShape* source, long A, long B, long C);
long typeIndexMapCreateShift(typeIndexMap* parent, typeShape* result, typeShape* source, long A, long B, long C);
long typeIndexMapCreateZoom(typeIndexMap* parent, typeShape* result, typeShape* source, double A, double B, double C);
long typeIndexMapCreateScale3D(typeIndexMap* parent, typeShape* result, typeShape* source);
long typeIndexMapCreateScale2D(typeIndexMap* parent, typeShape* result, typeShape* source);
long typeIndexMapCreateRotate2D(typeIndexMap* parent, typeShape* result, typeShape* source, double angle);
long typeIndexMapCreateConvolution2D(typeIndexMap* parent, typeShape* result, typeShape* source, unsigned long filterwidth, unsigned filterheight);
long typeIndexMapCreateConvolution2DTranspose(typeIndexMap* parent, typeShape* result, typeShape* source, unsigned long filterwidth, unsigned filterheight);
long typeIndexMapCreateConvolution2DAtrous(typeIndexMap* parent, typeShape* result, typeShape* source, unsigned long filterwidth, unsigned filterheight, long lambda);
long typeIndexMapAddRow(typeIndexMap* parent);
long typeIndexMapSetValue(typeIndexMap* parent, unsigned long index, unsigned long level, unsigned long value);
long typeIndexMapAddValue(typeIndexMap* parent, unsigned long index, unsigned long value);
long typeIndexMapClear(typeIndexMap* parent);
long typeIndexMapDestroy(typeIndexMap* parent);

long typeIndexMapCreate(typeIndexMap* parent, unsigned long width, unsigned long height)
{
	unsigned long  length;
	unsigned long  datalength;
	unsigned long* data;

	if (parent == 0)
		return 0;

	length = width * height;
	datalength = length * sizeof(unsigned long);
	data = (unsigned long*)malloc(datalength);

	if (data)
	{
		parent->indexWidth = width;
		parent->indexHeight = height;
		parent->indexLength = length;
		parent->indexSize = datalength;
		parent->indexData = data;

		typeIndexMapClear(parent);

		return 1;
	}

	parent->indexWidth = 0;
	parent->indexHeight = 0;
	parent->indexLength = 0;
	parent->indexSize = 0;
	parent->indexData = 0;

	return 0;
}

long typeIndexMapCreateFlip(typeIndexMap* parent, typeShape* result, typeShape* source, long A, long B, long C)
{
	unsigned long  resultindex;
	unsigned long  sourceindex;
	unsigned long  I, J, K;
	unsigned long  X, Y, Z;

	if (typeIndexMapCreate(parent, typeShapeGetLength(result), 1))
	{
		for (K = 0; K < result->shapeDepth; K++)
			for (J = 0; J < result->shapeHeight; J++)
				for (I = 0; I < result->shapeWidth; I++)
				{
					resultindex = typeShapeGetIndex(result, I, J, K);

					X = A ? (result->shapeWidth - I - 1) : I;
					Y = B ? (result->shapeHeight - J - 1) : J;
					Z = C ? (result->shapeDepth - K - 1) : K;

					sourceindex = typeShapeGetIndex(source, X, Y, Z);

					typeIndexMapAddValue(parent, resultindex, sourceindex);
				}

		return 1;
	}

	return 0;
}

long typeIndexMapCreateShift(typeIndexMap* parent, typeShape* result, typeShape* source, long A, long B, long C)
{
	unsigned long  resultindex;
	unsigned long  sourceindex;
	unsigned long  I, J, K;
	unsigned long  X, Y, Z;

	if (typeIndexMapCreate(parent, typeShapeGetLength(result), 1))
	{
		for (K = 0; K < result->shapeDepth; K++)
			for (J = 0; J < result->shapeHeight; J++)
				for (I = 0; I < result->shapeWidth; I++)
				{
					resultindex = typeShapeGetIndex(result, I, J, K);

					X = (long)I - A;
					Y = (long)J - B;
					Z = (long)K - C;

					sourceindex = typeShapeGetIndex(source, X, Y, Z);

					typeIndexMapAddValue(parent, resultindex, sourceindex);
				}

		return 1;
	}

	return 0;
}

long typeIndexMapCreateZoom(typeIndexMap* parent, typeShape* result, typeShape* source, double A, double B, double C)
{
	unsigned long  resultindex;
	unsigned long  sourceindex;
	unsigned long  I, J, K;
	long           X, Y, Z;
	double         XC, YC, ZC;

	if (typeIndexMapCreate(parent, typeShapeGetLength(result), 1))
	{
		XC = (double)result->shapeWidth / 2;
		YC = (double)result->shapeHeight / 2;
		ZC = (double)result->shapeDepth / 2;

		for (K = 0; K < result->shapeDepth; K++)
			for (J = 0; J < result->shapeHeight; J++)
				for (I = 0; I < result->shapeWidth; I++)
				{
					resultindex = typeShapeGetIndex(result, I, J, K);

					X = (long)((I - XC) * A + XC);
					Y = (long)((J - YC) * B + YC);
					Z = (long)((I - ZC) * C + ZC);

					sourceindex = typeShapeGetIndex(source, X, Y, Z);

					typeIndexMapAddValue(parent, resultindex, sourceindex);
				}

		return 1;
	}

	return 0;
}

long typeIndexMapCreateScale2D(typeIndexMap* parent, typeShape* result, typeShape* source)
{
	unsigned long  resultindex;
	unsigned long  sourceindex;
	unsigned long  I, J;
	unsigned long  XM, YM;
	unsigned long  XR, YR;
	unsigned long  XS, YS;

	if (typeIndexMapCreate(parent, (result->shapeWidth * result->shapeHeight), 1))
	{
		XM = (unsigned long)type_max(source->shapeWidth, result->shapeWidth);
		YM = (unsigned long)type_max(source->shapeHeight, result->shapeHeight);

		for (J = 0; J < YM; J++)
			for (I = 0; I < XM; I++)
			{
				XR = (long)(I * (double)result->shapeWidth / XM);
				YR = (long)(J * (double)result->shapeHeight / YM);

				resultindex = typeShapeGetIndex(result, XR, YR, 0);

				XS = (long)(I * (double)source->shapeWidth / XM);
				YS = (long)(J * (double)source->shapeHeight / YM);

				sourceindex = typeShapeGetIndex(source, XS, YS, 0);

				typeIndexMapAddValue(parent, resultindex, sourceindex);
			}

		return 1;
	}

	return 0;
}

long typeIndexMapCreateScale3D(typeIndexMap* parent, typeShape* result, typeShape* source)
{
	unsigned long  resultindex;
	unsigned long  sourceindex;
	unsigned long  I, J, K;
	unsigned long  XM, YM, ZM;
	unsigned long  XR, YR, ZR;
	unsigned long  XS, YS, ZS;

	if (typeIndexMapCreate(parent, typeShapeGetLength(result), 1))
	{
		XM = (unsigned long)type_max(source->shapeWidth, result->shapeWidth);
		YM = (unsigned long)type_max(source->shapeHeight, result->shapeHeight);
		ZM = (unsigned long)type_max(source->shapeDepth, result->shapeDepth);

		for (K = 0; K < ZM; K++)
			for (J = 0; J < YM; J++)
				for (I = 0; I < XM; I++)
				{
					XR = (long)(I * (double)result->shapeWidth / XM);
					YR = (long)(J * (double)result->shapeHeight / YM);
					ZR = (long)(K * (double)result->shapeDepth / ZM);

					resultindex = typeShapeGetIndex(result, XR, YR, ZR);

					XS = (long)(I * (double)source->shapeWidth / XM);
					YS = (long)(J * (double)source->shapeHeight / YM);
					ZS = (long)(K * (double)source->shapeDepth / ZM);

					sourceindex = typeShapeGetIndex(source, XS, YS, ZS);

					typeIndexMapAddValue(parent, resultindex, sourceindex);
				}

		return 1;
	}

	return 0;
}

long typeIndexMapCreateRotate2D(typeIndexMap* parent, typeShape* result, typeShape* source, double angle)
{
	unsigned long  resultindex;
	unsigned long  sourceindex;
	unsigned long  I, J;
	long           X, Y;
	double         XC, YC;
	double         XT, YT;
	double         xsin, xcos;

	if (typeIndexMapCreate(parent, typeShapeGetLength(result), 1))
	{
		XC = (double)result->shapeWidth / 2;
		YC = (double)result->shapeHeight / 2;

		xsin = type_sin(angle);
		xcos = type_cos(angle);

		for (J = 0; J < result->shapeHeight; J++)
			for (I = 0; I < result->shapeWidth; I++)
			{
				resultindex = typeShapeGetIndex(result, I, J, 0);

				XT = I - XC;
				YT = J - YC;

				X = (long)((XT * xcos + YT * xsin) + XC);
				Y = (long)((YT * xcos - XT * xsin) + YC);

				sourceindex = typeShapeGetIndex(source, X, Y, 0);

				typeIndexMapAddValue(parent, resultindex, sourceindex);
			}

		return 1;
	}

	return 0;
}

long typeIndexMapCreateConvolution2D(typeIndexMap* parent, typeShape* result, typeShape* source, unsigned long filterwidth, unsigned filterheight)
{
	unsigned long  resultindex;
	unsigned long  sourceindex;
	unsigned long  I, J;
	unsigned long   X, Y;
	long           XT, YT;
	long           XC, YC;

	if (typeIndexMapCreate(parent, typeShapeGetLength(result), 1))
	{
		XC = filterwidth / 2;
		YC = filterheight / 2;

		for (J = 0; J < result->shapeHeight; J++)
			for (I = 0; I < result->shapeWidth; I++)
			{
				resultindex = typeShapeGetIndex(result, I, J, 0);

				XT = I - XC;
				YT = J - YC;

				for (Y = 0; Y < filterheight; Y++)
					for (X = 0; X < filterwidth; X++)
					{
						sourceindex = typeShapeGetIndex(source, XT + X, YT + Y, 0);

						typeIndexMapAddValue(parent, resultindex, sourceindex);
					}
			}

		return 1;
	}

	return 0;
}

long typeIndexMapCreateConvolution2DTranspose(typeIndexMap* parent, typeShape* result, typeShape* source, unsigned long filterwidth, unsigned filterheight)
{
	unsigned long  resultindex;
	unsigned long  sourceindex;
	unsigned long  I, J;
	long           X, Y;
	long           XT, YT;
	long           XC, YC;

	if (typeIndexMapCreate(parent, typeShapeGetLength(result), 1))
	{
		XC = filterwidth / 2;
		YC = filterheight / 2;

		for (J = 0; J < result->shapeHeight; J++)
			for (I = 0; I < result->shapeWidth; I++)
			{
				resultindex = typeShapeGetIndex(result, I, J, 0);

				XT = I - XC;
				YT = J - YC;

				for (Y = (filterheight - 1); Y >= 0; Y--)
					for (X = (filterwidth - 1); X >= 0; X--)
					{
						sourceindex = typeShapeGetIndex(source, XT + X, YT + Y, 0);

						typeIndexMapAddValue(parent, resultindex, sourceindex);
					}
			}

		return 1;
	}

	return 0;
}

long typeIndexMapCreateConvolution2DAtrous(typeIndexMap* parent, typeShape* result, typeShape* source, unsigned long filterwidth, unsigned filterheight, long lambda)
{
	unsigned long  resultindex;
	unsigned long  sourceindex;
	unsigned long  I, J;
	unsigned long   X, Y;
	long           XT, YT;
	long           XC, YC;

	if (typeIndexMapCreate(parent, typeShapeGetLength(result), 1))
	{
		XC = filterwidth / 2;
		YC = filterheight / 2;

		for (J = 0; J < result->shapeHeight; J++)
			for (I = 0; I < result->shapeWidth; I++)
			{
				resultindex = typeShapeGetIndex(result, I, J, 0);

				XT = I - (XC * lambda);
				YT = J - (YC * lambda);

				for (Y = 0; Y < filterheight; Y++)
					for (X = 0; X < filterwidth; X++)
					{
						sourceindex = typeShapeGetIndex(source, XT + (X * lambda), YT + (Y * lambda), 0);

						typeIndexMapAddValue(parent, resultindex, sourceindex);
					}
			}

		return 1;
	}

	return 0;
}

long typeIndexMapAddRow(typeIndexMap* parent)
{
	unsigned long  I, length;
	unsigned long  datalength;
	unsigned long* data;

	if (parent == 0)
		return 0;

	length = parent->indexWidth * (parent->indexHeight + 1);
	datalength = length * sizeof(unsigned long);
	data = (unsigned long*)malloc(datalength);

	if (data)
	{
		for (I = 0; I < length; I++)
		{
			if (I < parent->indexLength)
				data[I] = parent->indexData[I];
			else
				data[I] = TYPE_INVALID_INDEX;
		}

		free(parent->indexData);

		parent->indexHeight += 1;
		parent->indexLength = length;
		parent->indexSize = datalength;
		parent->indexData = data;

		return 1;
	}

	return 0;
}

long typeIndexMapSetValue(typeIndexMap* parent, unsigned long index, unsigned long level, unsigned long value)
{
	unsigned long   I;

	if (parent == 0)
		return 0;

	if ((index >= parent->indexWidth) ||
		(level >= parent->indexHeight))
		return 0;

	I = level * parent->indexWidth + index;

	parent->indexData[I] = value;

	return 1;
}

long typeIndexMapAddValue(typeIndexMap* parent, unsigned long index, unsigned long value)
{
	unsigned long   I, J;

	if (parent == 0)
		return 0;

	if (index >= parent->indexWidth)
		return 0;

	J = index;
	for (I = 0; I < parent->indexHeight; I++)
	{
		if (parent->indexData[J] == TYPE_INVALID_INDEX)
		{
			parent->indexData[J] = value;

			return 1;
		}

		J += parent->indexWidth;
	}

	if (typeIndexMapAddRow(parent) == 0)
		return 0;

	J = index;
	for (I = 0; I < parent->indexHeight; I++)
	{
		if (parent->indexData[J] == TYPE_INVALID_INDEX)
		{
			parent->indexData[J] = value;

			return 1;
		}

		J += parent->indexWidth;
	}

	return 0;
}

long typeIndexMapClear(typeIndexMap* parent)
{
	unsigned long     I;

	if (parent == 0)
		return 0;

	for (I = 0; I < parent->indexLength; I++)
	{
		parent->indexData[I] = TYPE_INVALID_INDEX;
	}

	return 1;
}

long typeIndexMapDestroy(typeIndexMap* parent)
{
	if (parent == 0)
		return 0;

	free(parent->indexData);

	parent->indexWidth = 0;
	parent->indexHeight = 0;
	parent->indexLength = 0;
	parent->indexSize = 0;
	parent->indexData = 0;

	return 1;
}


#endif // TYPEINDEXMAP_H

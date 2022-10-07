#ifndef TYPEINTEGERARRAY_H
#define TYPEINTEGERARRAY_H

#include "typedynamicarray.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct
{
	unsigned long     arrayLength;
	typeDynamicArray  arrayData;
} typeIntegerArray;

long typeIntegerArrayCreate(typeIntegerArray* parent);
long typeIntegerArrayCreateSequence(typeIntegerArray* parent, unsigned long length);
long typeIntegerArrayShuffle(typeIntegerArray* parent);
long typeIntegerArrayGetIndex(typeIntegerArray* parent, unsigned long index, unsigned long* result);
long typeIntegerArrayDestroy(typeIntegerArray* parent);

const auto INTEGER_PAGE   =  sizeof(unsigned long) * 100;

long typeIntegerArrayCreate(typeIntegerArray* parent)
{
	if (parent == 0)
		return 0;

	parent->arrayLength = 0;

	if (typeDynamicArrayCreate(&parent->arrayData, INTEGER_PAGE))
	{
		return 1;
	}

	return 0;
}

long typeIntegerArrayCreateSequence(typeIntegerArray* parent, unsigned long length)
{
	unsigned long  datalength;
	unsigned long* data;
	unsigned long  I;

	if (parent == 0)
		return 0;

	if (typeIntegerArrayCreate(parent))
	{
		datalength = sizeof(unsigned long) * length;

		if (datalength > INTEGER_PAGE)
			datalength = INTEGER_PAGE;

		if (typeDynamicArrayResize(&parent->arrayData, datalength, INTEGER_PAGE))
		{
			parent->arrayLength = length;

			if (typeDynamicArrayGetDataAddress(&parent->arrayData, (void**)&data))
			{
				srand((unsigned long)time(0));

				for (I = 0; I < parent->arrayLength; I++)
					data[I] = I;

				return 1;
			}
		}
	}

	return 0;
}

long typeIntegerArrayShuffle(typeIntegerArray* parent)
{
	unsigned long* data;
	unsigned long  I, J, K;

	if (parent == 0)
		return 0;

	if (typeDynamicArrayGetDataAddress(&parent->arrayData, (void**)&data))
	{
		for (I = 0; I < parent->arrayLength; I++)
		{
			J = rand() % parent->arrayLength;

			K = data[J];
			data[J] = data[I];
			data[I] = K;
		}
	}

	return 1;
}

long typeIntegerArrayGetIndex(typeIntegerArray* parent, unsigned long index, unsigned long* result)
{
	unsigned long* data;

	if (parent == 0)
		return 0;

	if (index >= parent->arrayLength)
		return 0;

	if (typeDynamicArrayGetDataAddress(&parent->arrayData, (void**)&data))
	{
		*result = data[index];

		return 1;
	}

	return 0;
}

long typeIntegerArrayDestroy(typeIntegerArray* parent)
{
	if (parent == 0)
		return 0;

	if (typeDynamicArrayDestroy(&parent->arrayData))
	{
		parent->arrayLength = 0;
	}

	return 1;
}

#endif // TYPESEQUENCE_H

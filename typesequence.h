#ifndef TYPESEQUENCE_H
#define TYPESEQUENCE_H

#include <time.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct
{
	unsigned long   sequenceLength;
	unsigned long* sequenceData;
} typeSequence;

long typeSequenceCreate(typeSequence* parent, unsigned long length);
long typeSequenceShuffle(typeSequence* parent);
long typeSequenceDestroy(typeSequence* parent);

long typeSequenceCreate(typeSequence* parent, unsigned long length)
{
	unsigned long* data;
	unsigned long  I;

	if (parent == 0)
		return 0;

	data = (unsigned long*)malloc(length * sizeof(unsigned long));

	if (data)
	{
		srand((unsigned long)time(0));

		parent->sequenceData = data;
		parent->sequenceLength = length;

		for (I = 0; I < length; I++)
			data[I] = I;

		return 1;
	}

	parent->sequenceData = 0;
	parent->sequenceLength = 0;

	return 0;
}

long typeSequenceShuffle(typeSequence* parent)
{
	unsigned long  I, J, K;

	if (parent == 0)
		return 0;

	for (I = 0; I < parent->sequenceLength; I++)
	{
		J = rand() % parent->sequenceLength;

		K = parent->sequenceData[J];
		parent->sequenceData[J] = parent->sequenceData[I];
		parent->sequenceData[I] = K;
	}

	return 1;
}

long typeSequenceDestroy(typeSequence* parent)
{
	if (parent == 0)
		return 0;

	free(parent->sequenceData);

	parent->sequenceData = 0;
	parent->sequenceLength = 0;

	return 1;
}

#endif // TYPESEQUENCE_H

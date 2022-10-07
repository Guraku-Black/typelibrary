#ifndef TYPESTRING_H
#define TYPESTRING_H

#include "typeascii.h"
#include "typedynamicarray.h"

typedef struct
{
	unsigned long     stringLength;
	typeDynamicArray  stringData;
} typeString;

long typeStringCreate(typeString* parent);
long typeStringReset(typeString* parent);
long typeStringAddCharacter(typeString* parent, typeChar value);
long typeStringCompareCharString(typeString* parent, const char* buffer, unsigned long buffersize);
long typeStringDestroy(typeString* parent);

const auto STRING_PAGE = sizeof(typeChar) * 10;

long typeStringCreate(typeString* parent)
{
	if (parent == 0)
		return 0;

	parent->stringLength = 0;
	if (typeDynamicArrayCreate(&parent->stringData, STRING_PAGE))
	{
		return 1;
	}

	return 0;
}

long typeStringReset(typeString* parent)
{
	if (parent == 0)
		return 0;

	parent->stringLength = 0;

	return 1;
}

long typeStringAddCharacter(typeString* parent, typeChar value)
{
	typeChar*      data;
	unsigned long  datasize;

	if (parent == 0)
		return 0;

	datasize = (parent->stringLength + 1) * sizeof(typeChar);

	if (typeDynamicArrayResize(&parent->stringData, datasize, STRING_PAGE))
	{
		if (typeDynamicArrayGetDataAddress(&parent->stringData, (void**)&data))
		{
			data[parent->stringLength] = value;
			
			parent->stringLength++;

			return 1;
		}
	}

	return 0;
}

long typeStringCompareCharString(typeString* parent, const char* buffer, unsigned long buffersize)
{
	typeChar*      data;
	unsigned long  I;

	if (parent == 0)
		return 0;

	if (buffersize != parent->stringLength)
		return 0;

	if (typeDynamicArrayGetDataAddress(&parent->stringData, (void**)&data))
	{
		for (I = 0; I < parent->stringLength; I++)
		{
			if (typeAsciiToLowercase(data[I]) != typeAsciiToLowercase(buffer[I]))
				return 0;
		}

		return 1;
	}

	return 0;
}

long typeStringDestroy(typeString* parent)
{
	if (parent == 0)
		return 0;

	typeDynamicArrayDestroy(&parent->stringData);

	parent->stringLength = 0;

	return 1;
}


#endif // TYPESTRING_H

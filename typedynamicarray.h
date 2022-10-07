#ifndef TYPEDYNAMICARRAY_H
#define TYPEDYNAMICARRAY_H

typedef struct
{
	unsigned long  arrayCapacity;
	unsigned long  arrayLength;
	unsigned char* arrayData;
} typeDynamicArray;

long typeDynamicArrayCreate(typeDynamicArray* parent, unsigned long pagesize);
long typeDynamicArrayResize(typeDynamicArray* parent, unsigned long length, unsigned long pagesize);
long typeDynamicArrayGetDataAddress(typeDynamicArray* parent, void** result);
long typeDynamicArrayDestroy(typeDynamicArray* parent);

long typeDynamicArrayCreate(typeDynamicArray* parent, unsigned long pagesize)
{
	unsigned char* data;

	if (parent == 0)
		return 0;

	data = (unsigned char*)malloc(pagesize);

	if (data)
	{
		parent->arrayCapacity = pagesize;
		parent->arrayLength = 0;
		parent->arrayData = data;

		return 1;
	}

	parent->arrayCapacity = 0;
	parent->arrayLength = 0;
	parent->arrayData = 0;

	return 0;
}

long typeDynamicArrayResize(typeDynamicArray* parent, unsigned long length, unsigned long pagesize)
{
	unsigned long  datalength;
	unsigned char* data;

	if (parent == 0)
		return 0;

	if (parent->arrayCapacity < length)
	{
		datalength = (unsigned long)(length / pagesize + 1) * pagesize;
		data = (unsigned char*)realloc(parent->arrayData, datalength);

		if (data == 0)
			return 0;

		parent->arrayCapacity = datalength;
		parent->arrayData = data;
	}

	if (parent->arrayCapacity >= length)
	{
		parent->arrayLength = length;

		return 1;
	}

	return 0;
}

long typeDynamicArrayGetDataAddress(typeDynamicArray* parent, void** result)
{
	if ((parent == 0) || (result == 0))
		return 0;

	if (parent->arrayData == 0)
		return 0;

	*result = parent->arrayData;

	return 1;
}

long typeDynamicArrayDestroy(typeDynamicArray* parent)
{
	if (parent == 0)
		return 0;

	free(parent->arrayData);

	parent->arrayCapacity = 0;
	parent->arrayLength = 0;
	parent->arrayData = 0;

	return 0;

}

#endif // TYPEDYNAMICARRAY_H

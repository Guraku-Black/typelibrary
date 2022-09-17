#ifndef TYPEIMAGELIST_H
#define TYPEIMAGELIST_H

#include "typeimage.h"
#include "typefilebmp.h"
#include "typedynamicarray.h"

typedef struct
{
	unsigned long    listLength;
	typeDynamicArray listData;
} typeImageList;

long typeImageListCreate(typeImageList* parent);
long typeImageListGetImage(typeImageList* parent, unsigned long index, typeImage** result);
long typeImageListAddDirectory(typeImageList* parent, LPCWSTR pathname, unsigned limit);
long typeImageListDestroy(typeImageList* parent);

const int    imagelist_page = sizeof(typeImage) * 100;

long typeImageListCreate(typeImageList* parent)
{
	if (parent == 0)
		return 0;

	if (typeDynamicArrayCreate(&parent->listData, imagelist_page))
	{
		parent->listLength = 0;

		return 1;
	}

	parent->listLength = 0;

	return 0;
}

static long find_GetFullPath(const WCHAR* pathname, WCHAR* filename, char* result, unsigned long resultlength)
{
	unsigned long I, J, namestart;

	I = J = 0;
	namestart = 0;
	while (I < resultlength)
	{
		result[I] = (char)pathname[I];

		if (pathname[I] == 0)
			break;
		if (pathname[I] == 0x2F) //47   0x2F  | 00101111 | / | Slash or divide
			namestart = I + 1;

		I++;
	}
	I = namestart;
	while (I < resultlength)
	{
		result[I] = (char)filename[J];

		if (filename[J] == 0)
			break;
		I++;
		J++;
	}

	return I;
}

long typeImageListGetImage(typeImageList* parent, unsigned long index, typeImage** result)
{
	typeImage* images;

	if ((parent == 0) || (result == 0))
		return 0;

	if (index >= parent->listLength)
		return 0;

	images = (typeImage*)parent->listData.arrayData;

	*result = &images[index];

	return 1;
}

long typeImageListAddDirectory(typeImageList* parent, LPCWSTR pathname, unsigned limit)
{
	WIN32_FIND_DATA findData;
	HANDLE          findHandle;
	typeImage* images;
	char* buffer;
	unsigned long   I;

	if (parent == 0)
		return 0;

	findHandle = FindFirstFile(pathname, &findData);

	if (findHandle != INVALID_HANDLE_VALUE)
	{
		I = 0;
		buffer = (char*)malloc(4096);

		do
		{
			find_GetFullPath(pathname, findData.cFileName, buffer, 4096);

			if (typeDynamicArrayResize(&parent->listData, sizeof(typeImage) * (I + 1), imagelist_page))
			{
				images = (typeImage*)parent->listData.arrayData;

				typeFileBmpLoad(&images[I], buffer);

				I++;

				if (I >= limit)
					break;
			}
		} while (FindNextFile(findHandle, &findData));

		parent->listLength += I;

		free(buffer);

		FindClose(findHandle);
	}

	return 1;
}

long typeImageListDestroy(typeImageList* parent)
{
	typeImage* images;
	unsigned long   I;

	if (parent == 0)
		return 0;

	images = (typeImage*)parent->listData.arrayData;

	if (images)
	{
		for (I = 0; I < parent->listLength; I++)
		{
			typeImageDestroy(&images[I]);
		}
	}

	if (typeDynamicArrayDestroy(&parent->listData))
	{
		parent->listLength = 0;

		return 1;
	}

	return 0;
}

#endif // TYPEIMAGELIST_H

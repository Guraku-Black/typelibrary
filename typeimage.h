#ifndef TYPEIMAGE_H
#define TYPEIMAGE_H

#include "typeindexmap.h"

typedef struct
{
	unsigned long  imageWidth;
	unsigned long  imageHeight;
	unsigned long  imageLength;
	unsigned long  imageSize;
	unsigned long* imageData;
} typeImage;

long typeImageCreate(typeImage* parent, unsigned long width, unsigned long height);
long typeImageIndexMapCopy(typeImage* parent, typeImage* source, typeIndexMap* indexmap);
long typeImageIndexMapConvolution(typeImage* parent, typeImage* source, double* filter, unsigned long filterwidth, unsigned long filterheight, typeIndexMap* indexmap);
long typeImageDestroy(typeImage* parent);

long typeImageCreate(typeImage* parent, unsigned long width, unsigned long height)
{
	void* data;
	unsigned long   length;
	unsigned long   datalength;

	if (parent == 0)
		return 0;

	length = width * height;
	datalength = length * sizeof(unsigned long);
	data = malloc(datalength);

	if (data)
	{
		parent->imageWidth = width;
		parent->imageHeight = height;
		parent->imageLength = length;
		parent->imageSize = datalength;
		parent->imageData = (unsigned long*)data;

		return 1;
	}

	parent->imageWidth = 0;
	parent->imageHeight = 0;
	parent->imageLength = 0;
	parent->imageSize = 0;
	parent->imageData = 0;

	return 0;
}

long typeImageIndexMapCopy(typeImage* parent, typeImage* source, typeIndexMap* indexmap)
{
	unsigned long  I, J, K;
	unsigned long  r, g, b;
	unsigned long  counter;
	unsigned long  index;
	unsigned char* color;

	if ((parent == 0) || (source == 0) || (indexmap == 0))
		return 0;

	if (parent->imageLength != indexmap->indexWidth)
		return 0;

	for (I = 0; I < indexmap->indexWidth; I++)
	{
		r = g = b = 0;
		counter = 0;
		K = I;

		for (J = 0; J < indexmap->indexHeight; J++)
		{
			index = indexmap->indexData[K];
			if ((index != TYPE_INVALID_INDEX) &&
				(index < source->imageLength))
			{
				color = (unsigned char*)&source->imageData[index];
				r += color[0];
				g += color[1];
				b += color[2];

				counter++;
			}

			K += indexmap->indexWidth;
		}

		if (counter > 1)
		{
			r /= counter;
			g /= counter;
			b /= counter;
		}

		color = (unsigned char*)&parent->imageData[I];
		color[0] = (unsigned char)r;
		color[1] = (unsigned char)g;
		color[2] = (unsigned char)b;
	}

	return 1;
}

double image_maxout(double a)
{
	if (a < 0)
		return 0;

	if (a > 255)
		return 255;

	return a;
}

long typeImageIndexMapConvolution(typeImage* parent, typeImage* source, double* filter, unsigned long filterwidth, unsigned long filterheight, typeIndexMap* indexmap)
{
	unsigned long  I, J, K;
	unsigned long  index;
	unsigned char* color;
	double         r, g, b;
	double         counter;

	if ((parent == 0) || (source == 0) || (indexmap == 0))
		return 0;

	if (parent->imageLength != indexmap->indexWidth)
		return 0;

	if ((filterwidth * filterheight) != indexmap->indexHeight)
		return 0;

	for (I = 0; I < indexmap->indexWidth; I++)
	{
		r = g = b = 0;
		counter = 0;
		K = I;

		for (J = 0; J < indexmap->indexHeight; J++)
		{
			index = indexmap->indexData[K];
			if ((index != TYPE_INVALID_INDEX) &&
				(index < source->imageLength))
			{
				color = (unsigned char*)&source->imageData[index];
				r += color[0] * filter[J];
				g += color[1] * filter[J];
				b += color[2] * filter[J];

				counter += filter[J];
			}

			K += indexmap->indexWidth;
		}

		if (counter != 0)
		{
			r = r / counter;
			g = g / counter;
			b = b / counter;
		}

		color = (unsigned char*)&parent->imageData[I];
		color[0] = (unsigned char)image_maxout(r);
		color[1] = (unsigned char)image_maxout(g);
		color[2] = (unsigned char)image_maxout(b);
	}

	return 1;
}
long typeImageDestroy(typeImage* parent)
{
	if (parent == 0)
		return 0;

	free(parent->imageData);

	parent->imageWidth = 0;
	parent->imageHeight = 0;
	parent->imageLength = 0;
	parent->imageSize = 0;
	parent->imageData = 0;

	return 1;
}

#endif // TYPEIMAGE_H
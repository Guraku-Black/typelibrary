#ifndef TYPEFILEBMP_H
#define TYPEFILEBMP_H

#include "typeimage.h"

#include <stdio.h>
#include <math.h>

static unsigned char readByte(FILE* fp)
{
	return fgetc(fp);
}

static unsigned short readWord(FILE* fp)
{
	return (fgetc(fp) | (fgetc(fp) << 8));
}

static unsigned long readDWord(FILE* fp)
{
	return (fgetc(fp) | (fgetc(fp) << 8) | (fgetc(fp) << 16) | (fgetc(fp) << 24));
}

long typeFileBmpLoad(typeImage* parent, char* filename)
{
	BITMAPFILEHEADER bfHeader{};
	BITMAPINFOHEADER bfInfo{};
	FILE* fp;

	if (fopen_s(&fp, filename, "rb") != 0)
		return 0;

	while (1)
	{
		bfHeader.bfType = readWord(fp);
		bfHeader.bfSize = readDWord(fp);
		bfHeader.bfReserved1 = readWord(fp);
		bfHeader.bfReserved2 = readWord(fp);
		bfHeader.bfOffBits = readDWord(fp);

		if (bfHeader.bfType != 0x4D42)
			break;

		bfInfo.biSize = readDWord(fp);
		bfInfo.biWidth = readDWord(fp);
		bfInfo.biHeight = readDWord(fp);
		bfInfo.biPlanes = readWord(fp);
		bfInfo.biBitCount = readWord(fp);
		bfInfo.biCompression = readDWord(fp);
		bfInfo.biSizeImage = readDWord(fp);
		bfInfo.biXPelsPerMeter = readDWord(fp);
		bfInfo.biYPelsPerMeter = readDWord(fp);
		bfInfo.biClrUsed = readDWord(fp);
		bfInfo.biClrImportant = readDWord(fp);

		if (bfInfo.biCompression != 0)
			break;

		long bytesPerPixel = ((long)bfInfo.biBitCount) / 8;
		long paddedRowSize = (long)ceil((float)(bfInfo.biWidth * bytesPerPixel) / 4) * 4;

		long            I, J, K;
		unsigned char   r, g, b;
		unsigned char* buffer;

		if (typeImageCreate(parent, bfInfo.biWidth, bfInfo.biHeight))
		{
			buffer = (unsigned char*)parent->imageData;

			K = 0;
			r = g = b = 0;
			for (J = 0; J < bfInfo.biHeight; J++)
			{
				fseek(fp, bfHeader.bfOffBits + ((bfInfo.biHeight - J - 1) * paddedRowSize), SEEK_SET);

				for (I = 0; I < bfInfo.biWidth; I++)
				{
					if (bytesPerPixel == 1)
					{
						r = g = b = readByte(fp);
					}
					else if (bytesPerPixel == 3)
					{
						r = readByte(fp);
						g = readByte(fp);
						b = readByte(fp);
					}
					else if (bytesPerPixel == 4)
					{
						r = readByte(fp);
						g = readByte(fp);
						b = readByte(fp);
						readByte(fp);
					}

					buffer[K + 0] = r;
					buffer[K + 1] = g;
					buffer[K + 2] = b;
					buffer[K + 3] = 0;

					K += 4;
				}
			}

		}

		fclose(fp);

		return 1;
	}

	fclose(fp);

	return 0;
}

#endif // TYPEFILEBMP_H

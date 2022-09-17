#ifndef TYPEBITMAP_H
#define TYPEBITMAP_H

#include "typewindow.h"

typedef struct
{
	HBITMAP     bitmapHandle;
	DWORD       bitmapWidth;
	DWORD       bitmapHeight;
	VOID* bitmapBits;
} typeBitmap;

long typeBitmapCreate(typeBitmap* parent, unsigned long width, unsigned long height);
long typeBitmapShow(typeBitmap* parent, typeWindow* window, long x, long y);
long typeBitmapDrawPixel(typeBitmap* parent, long x, long y, unsigned long color);
long typeBitmapDrawImage(typeBitmap* parent, long x, long y, unsigned long imagewidth, unsigned long imageheight, unsigned long* imagedata);
long typeBitmapDestroy(typeBitmap* parent);

long typeBitmapCreate(typeBitmap* parent, unsigned long width, unsigned long height)
{
	BITMAPINFO   bmpinfo;
	VOID* bits = 0;

	if (parent == 0)
		return 0;

	ZeroMemory(&bmpinfo, sizeof(BITMAPINFO));
	bmpinfo.bmiHeader.biSize = sizeof(BITMAPINFO);
	bmpinfo.bmiHeader.biWidth = width;
	bmpinfo.bmiHeader.biHeight = -(long)height;
	bmpinfo.bmiHeader.biPlanes = 1;
	bmpinfo.bmiHeader.biBitCount = 32;
	bmpinfo.bmiHeader.biCompression = BI_RGB;
	bmpinfo.bmiHeader.biSizeImage = 0;
	bmpinfo.bmiHeader.biXPelsPerMeter = 0;
	bmpinfo.bmiHeader.biYPelsPerMeter = 0;
	bmpinfo.bmiHeader.biClrUsed = 0;
	bmpinfo.bmiHeader.biClrImportant = 0;

	HBITMAP handle = CreateDIBSection(
		0,
		&bmpinfo,
		DIB_RGB_COLORS,
		&bits,
		NULL,
		NULL);

	if (handle)
	{
		parent->bitmapHandle = handle;
		parent->bitmapWidth = width;
		parent->bitmapHeight = height;
		parent->bitmapBits = bits;

		return 1;
	}

	parent->bitmapHandle = 0;
	parent->bitmapWidth = 0;
	parent->bitmapHeight = 0;
	parent->bitmapBits = 0;

	return 0;
}

long typeBitmapShow(typeBitmap* parent, typeWindow* window, long x, long y)
{
	HDC     hdc;
	HDC     hdcMem;

	if ((parent == 0) || (window == 0))
		return 0;

	hdc = GetDC(window->windowHandle);
	hdcMem = CreateCompatibleDC(hdc);

	SelectObject(hdcMem, parent->bitmapHandle);

	BitBlt(hdc, x, y, parent->bitmapWidth, parent->bitmapHeight, hdcMem, 0, 0, SRCCOPY);

	ReleaseDC(window->windowHandle, hdc);
	DeleteObject(hdcMem);

	return 1;
}

long typeBitmapDrawPixel(typeBitmap* parent, long x, long y, unsigned long color)
{
	unsigned long* data;

	if (parent == 0)
		return 0;

	if ((x < 0) || (x >= (long)parent->bitmapWidth) ||
		(y < 0) || (y >= (long)parent->bitmapHeight))
		return 0;

	data = (unsigned long*)parent->bitmapBits;
	data[y * parent->bitmapWidth + x] = color;

	return 1;
}

long typeBitmapDrawImage(typeBitmap* parent, long x, long y, unsigned long imagewidth, unsigned long imageheight, unsigned long* imagedata)
{
	long           cx, cy;
	long           dx, dy;
	unsigned long  dataoffset;
	unsigned long  imageoffset;
	unsigned long* source;
	unsigned long* data;

	if ((parent == 0) || (imagedata == 0))
		return 0;

	data = (unsigned long*)parent->bitmapBits;
	source = (unsigned long*)imagedata;

	for (cy = 0; cy < (long)imageheight; cy++)
	{
		dy = cy + y;
		if (dy < 0)
			continue;
		if (dy >= (long)parent->bitmapHeight)
			break;

		dataoffset = dy * parent->bitmapWidth;
		imageoffset = cy * imagewidth;
		for (cx = 0; cx < (long)imagewidth; cx++)
		{
			dx = cx + x;
			if (dx < 0)
				continue;
			if (dx >= (long)parent->bitmapWidth)
				break;

			data[dataoffset + dx] = source[imageoffset + cx];
		}
	}

	return 1;
}

long typeBitmapDestroy(typeBitmap* parent)
{
	if (parent == 0)
		return 0;

	DeleteObject(parent->bitmapHandle);

	parent->bitmapHandle = 0;
	parent->bitmapWidth = 0;
	parent->bitmapHeight = 0;
	parent->bitmapBits = 0;

	return 1;
}

#endif // TYPEBITMAP_H
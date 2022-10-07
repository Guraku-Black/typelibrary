#ifndef TYPEFILESTREAM_H
#define TYPEFILESTREAM_H

#include <stdio.h>

typedef struct
{
	FILE* fp;
} typeFileStream;

long typeFileStreamOpenNew(typeFileStream* parent, char* filename);
long typeFileStreamOpenExisting(typeFileStream* parent, char* filename);
long typeFileStreamReadByte(typeFileStream* parent, unsigned long* result);
long typeFileStreamReadWord(typeFileStream* parent, unsigned long* result);
long typeFileStreamReadDWord(typeFileStream* parent, unsigned long* result);
long typeFileStreamReadUTF8(typeFileStream* parent, unsigned long* result);
long typeFileStreamReadUTF8(typeFileStream* parent, unsigned long value);
long typeFileStreamClose(typeFileStream* parent);

long typeFileStreamOpenNew(typeFileStream* parent, char* filename)
{
	if (parent == 0)
		return 0;

	if (fopen_s(&parent->fp, filename, "w") == 0)
		return 1;

	return 0;
}

long typeFileStreamOpenExisting(typeFileStream* parent, char* filename)
{
	if (parent == 0)
		return 0;

	if (fopen_s(&parent->fp, filename, "r") == 0)
		return 1;

	return 0;
}

long typeFileStreamWriteByte(typeFileStream* parent, unsigned long value)
{
	int     ch;

	if (parent == 0)
		return 0;
	
	ch = (int)value;
	
	if (fputc(ch, parent->fp) > 0)
		return 1;

	return 0;
}

long typeFileStreamReadByte(typeFileStream* parent, unsigned long* result)
{
	int      ch;

	if ((parent == 0) || (result == 0))
		return 0;
	
	ch = fgetc(parent->fp);

	if (ch == EOF)
		return 0;

	*result = (unsigned long)ch;

	return 1;
}

long typeFileStreamReadWord(typeFileStream* parent, unsigned long* result)
{
	unsigned long   ch1, ch2;

	if ((parent == 0) || (result == 0))
		return 0;

	if (typeFileStreamReadByte(parent, &ch1) &&
		typeFileStreamReadByte(parent, &ch2))
	{
		*result = (ch1 | (ch2 << 8));

		return 1;
	}

	return 0;
}

long typeFileStreamReadDWord(typeFileStream* parent, unsigned long* result)
{
	unsigned long   ch1, ch2, ch3, ch4;

	if ((parent == 0) || (result == 0))
		return 0;

	if (typeFileStreamReadByte(parent, &ch1) &&
		typeFileStreamReadByte(parent, &ch2) &&
		typeFileStreamReadByte(parent, &ch3) &&
		typeFileStreamReadByte(parent, &ch4))
	{
		*result = ((ch1) | (ch2 << 8) | (ch3 << 16) | (ch3 << 24));

		return 1;
	}

	return 0;
}

long typeFileStreamReadUTF8(typeFileStream* parent, unsigned long* result)
{
	unsigned long   ch1, ch2, ch3, ch4, ch5, ch6;

	if ((parent == 0) || (result == 0))
		return 0;

	while (typeFileStreamReadByte(parent, &ch1))
	{
		if (ch1 < 0x80)                  // 0x80 = 10000000
		{
			*result = ch1;

			return 1;
		}
		if ((ch1 & 0xC0) == 0x80)        // 0xC0 = 11000000 0x80 = 10000000 Error = 0x10xxxxxx
		{
			continue;
		}
		if ((ch1 & 0xE0) == 0xC0)        // 0xE0 = 11100000 0xC0 = 11000000 2 Bytes = 0x110xxxxx
		{
			typeFileStreamReadByte(parent, &ch2);

			if ((ch2 & 0xC0) != 0x80)    // 0xC0 = 11000000   0x80 = 10000000
				continue;

			*result = 
				((ch1 & 0x1F) << 6) |	 // 0x1F = 00011111
				((ch2 & 0x3F) << 0);     // 0x3F = 00111111

			return 1;
		}
		if ((ch1 & 0xF0) == 0xE0)        // 0xF0 = 11110000 0xE0 = 11100000 3 Bytes = 0x1110xxxx
		{
			typeFileStreamReadByte(parent, &ch2);
			typeFileStreamReadByte(parent, &ch3);

			if ((ch2 & 0xC0) != 0x80)    // 0xC0 = 11000000     0x80 = 10000000
				continue;
			if ((ch3 & 0xC0) != 0x80)    // 0xC0 = 11000000     0x80 = 10000000
				continue;

			*result =
				((ch1 & 0x0F) << 12) |   // 0x0F = 00001111
				((ch2 & 0x3F) << 6) |	 // 0x3F = 00111111
				((ch3 & 0x3F) << 0);     // 0x3F = 00111111

			return 1;
		}
		if ((ch1 & 0xF8) == 0xF0)        // 0xF8 = 11111000 0xF0 = 11110000 4 Bytes = 0x11110xxx
		{
			typeFileStreamReadByte(parent, &ch2);
			typeFileStreamReadByte(parent, &ch3);
			typeFileStreamReadByte(parent, &ch4);

			if ((ch2 & 0xC0) != 0x80)    // 0xC0 = 11000000     0x80 = 10000000
				continue;
			if ((ch3 & 0xC0) != 0x80)    // 0xC0 = 11000000     0x80 = 10000000
				continue;
			if ((ch4 & 0xC0) != 0x80)    // 0xC0 = 11000000     0x80 = 10000000
				continue;

			*result = 
				((ch1 & 0x07) << 18) |   // 0x07 = 00000111
				((ch2 & 0x3F) << 12) |	 // 0x3F = 00111111
				((ch3 & 0x3F) << 6) |	 // 0x3F = 00111111
				((ch4 & 0x3F) << 0);     // 0x3F = 00111111

			return 1;
		}
		if ((ch1 & 0xFC) == 0xF8)        // 0xFC = 11111100 0xF8 = 11111000 5 Bytes = 0x111110xx
		{
			typeFileStreamReadByte(parent, &ch2);
			typeFileStreamReadByte(parent, &ch3);
			typeFileStreamReadByte(parent, &ch4);
			typeFileStreamReadByte(parent, &ch5);

			if ((ch2 & 0xC0) != 0x80)    // 0xC0 = 11000000     0x80 = 10000000
				continue;
			if ((ch3 & 0xC0) != 0x80)    // 0xC0 = 11000000     0x80 = 10000000
				continue;
			if ((ch4 & 0xC0) != 0x80)    // 0xC0 = 11000000     0x80 = 10000000
				continue;
			if ((ch5 & 0xC0) != 0x80)    // 0xC0 = 11000000     0x80 = 10000000
				continue;

			*result =
				((ch1 & 0x03) << 24) |   // 0x03 = 00000011
				((ch2 & 0x3F) << 18) |	 // 0x3F = 00111111
				((ch3 & 0x3F) << 12) |	 // 0x3F = 00111111
				((ch4 & 0x3F) << 6) |	 // 0x3F = 00111111
				((ch5 & 0x3F) << 0);     // 0x3F = 00111111

			return 1;
		}
		if ((ch1 & 0xFE) == 0xFC)        // 0xFE = 11111110 0xFC = 11111100 6 Bytes = 0x1111110x
		{
			typeFileStreamReadByte(parent, &ch2);
			typeFileStreamReadByte(parent, &ch3);
			typeFileStreamReadByte(parent, &ch4);
			typeFileStreamReadByte(parent, &ch5);
			typeFileStreamReadByte(parent, &ch6);

			if ((ch2 & 0xC0) != 0x80)    // 0xC0 = 11000000     0x80 = 10000000
				continue;
			if ((ch3 & 0xC0) != 0x80)    // 0xC0 = 11000000     0x80 = 10000000
				continue;
			if ((ch4 & 0xC0) != 0x80)    // 0xC0 = 11000000     0x80 = 10000000
				continue;
			if ((ch5 & 0xC0) != 0x80)    // 0xC0 = 11000000     0x80 = 10000000
				continue;
			if ((ch6 & 0xC0) != 0x80)    // 0xC0 = 11000000     0x80 = 10000000
				continue;

			*result = 
				((ch1 & 0x01) << 30) |   // 0x01 = 00000001
				((ch2 & 0x3F) << 24) |	 // 0x3F = 00111111
				((ch3 & 0x3F) << 18) |	 // 0x3F = 00111111
				((ch4 & 0x3F) << 12) |	 // 0x3F = 00111111
				((ch5 & 0x3F) << 6) |	 // 0x3F = 00111111
				((ch6 & 0x3F) << 0);     // 0x3F = 00111111

			return 1;
		}
	}

	return 0;
}

long typeFileStreamReadUTF8(typeFileStream* parent, unsigned long value)
{
	unsigned long ch1, ch2;
	long          counter;

	if (parent == 0)
		return 0;

	if (value < 0x80)
	{
		ch1 = value & 0x7F;                           // 0x7F = 01111111  1 Byte
		counter = 0;
	}
	else if (value < 0x800)
	{
		ch1 = 0xC0 | ((value >> 6) & 0x1F);           // 0xC0 = 11000000  0x1F = 00011111   2 Bytes
		counter = 1;
	}
	else if (value < 0x10000)
	{
		ch1 = 0xE0 | ((value >> 12) & 0x0F);           // 0xE0 = 11100000  0x0F = 00001111  3  Bytes
		counter = 2;
	}
	else if (value < 0x200000)
	{
		ch1 = 0xF0 | ((value >> 18) & 0x07);           // 0xF0 = 11110000  0x07 = 00000111   4  Bytes
		counter = 3;
	}
	else if (value < 0x4000000)
	{
		ch1 = 0xF8 | ((value >> 24) & 0x03);           // 0xF8 = 11111000  0x03 = 00000011   5  Bytes
		counter = 4;
	}
	else 
	{
		ch1 = 0xFC | ((value >> 30) & 0x01);           // 0xFC = 11111100  0x01 = 00000001   6  Bytes
		counter = 5;
	}

	typeFileStreamWriteByte(parent, ch1);
	while (counter > 0)
	{
		counter--;
		ch2 = 0x80 | ((value >> (counter * 6)) & 0x3F);  //0x80 = 10000000  0x3F = 00111111  

	    typeFileStreamWriteByte(parent, ch2);
	}

	return 1;
}

long typeFileStreamClose(typeFileStream* parent)
{
	if (parent == 0)
		return 0;

	fclose(parent->fp);

	parent->fp = 0;

	return 1;
}

#endif // TYPEFILESTREAM_H

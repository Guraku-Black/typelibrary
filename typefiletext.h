#ifndef TYPEFILETEXT_H
#define TYPEFILETEXT_H

#include "typeascii.h"
#include "typestring.h"
#include "typefilestream.h"

typedef struct
{
	typeFileStream      currentSource;
	typeChar            currentCharacter;
	typeString          currentString;
	double              currentNumber;
} typeFileText;

long typeFileTextOpenNew(typeFileText* parent, char* filename)
{
	if (parent == 0)
		return 0;

	if (typeFileStreamOpenNew(&parent->currentSource, filename))
	{
		parent->currentCharacter = 0;
		typeStringCreate(&parent->currentString);

		return 1;
	}

	return 0;
}

long typeFileTextOpenExisting(typeFileText* parent, char* filename)
{
	if (parent == 0)
		return 0;

	if (typeFileStreamOpenExisting(&parent->currentSource, filename))
	{
		typeStringCreate(&parent->currentString);
		parent->currentCharacter = 0;

		return 1;
	}

	return 0;
}

long typeFileTextReadCharacter(typeFileText* parent, typeChar* result)
{
	unsigned long      ch;

	if (parent == 0)
		return 0;

	if (typeFileStreamReadUTF8(&parent->currentSource, &ch))
	{
		*result = (typeChar)ch;
		parent->currentCharacter = (typeChar)ch;

		return 1;
	}

	parent->currentCharacter = 0;

	return 0;
}

long typeFileTextPeekCharacter(typeFileText* parent)
{
	if (parent == 0)
		return 0;

	if (parent->currentCharacter != 0)
	{
		return 1;
	}

	return 0;
}

long typeFileTextAdvanceCharacter(typeFileText* parent)
{
	if (parent == 0)
		return 0;

	if (typeFileTextReadCharacter(parent, &parent->currentCharacter))
	{
		if (parent->currentCharacter != 0)
			return 1;
	}

	parent->currentCharacter = 0;

	return 0;
}

long typeFileTextReadVisibleCharacter(typeFileText* parent)
{
	while (typeFileTextAdvanceCharacter(parent))
	{
		if (typeAsciiIsVisible(parent->currentCharacter))
		{
			return 1;
		}
	}

	return 0;
}

long typeFileTextReadString(typeFileText* parent)
{
	if (parent == 0)
		return 0;

	if (typeAsciiIsAlpha(parent->currentCharacter))
	{
		typeStringReset(&parent->currentString);

		while (typeAsciiIsAlphaNum(parent->currentCharacter))
		{
			typeStringAddCharacter(&parent->currentString, parent->currentCharacter);

			typeFileTextAdvanceCharacter(parent);
		}

		return 1;
	}

	return 0;
}

long typeFileTextReadLine(typeFileText* parent)
{
	if (parent == 0)
		return 0;

	typeStringReset(&parent->currentString);

	while (typeFileTextPeekCharacter(parent))
	{
			if (parent->currentCharacter == 10) // ASCII 10 ' LineFeed
			{
				typeFileTextAdvanceCharacter(parent);

				return 1;
			}

			if (parent->currentCharacter == 13) // ASCII 13 ' Cariage Return
			{
				typeFileTextAdvanceCharacter(parent);

				if (parent->currentCharacter == 10) // ASCII 10 ' LineFeed
				{
					typeFileTextAdvanceCharacter(parent);

				}
				
				return 1;
			}

			typeStringAddCharacter(&parent->currentString, parent->currentCharacter);

			typeFileTextAdvanceCharacter(parent);
	}

	return 0;
}

long typeFileTextReadSingleQuote(typeFileText* parent)
{
	if (parent == 0)
		return 0;

	if (parent->currentCharacter == 39)    // ASCII 39 ' Single Quote
	{
		typeStringReset(&parent->currentString);

		typeFileTextAdvanceCharacter(parent);

		while (typeFileTextPeekCharacter(parent))
		{
			if (parent->currentCharacter == 39) // ASCII 39 ' Single Quote
			{
				typeFileTextAdvanceCharacter(parent);

				return 1;
			}
			if (typeAsciiIsPrintable(parent->currentCharacter))
			{
				typeStringAddCharacter(&parent->currentString, parent->currentCharacter);

				typeFileTextAdvanceCharacter(parent);

				continue;
			}

			return 0;
		}
	}

	return 0;
}

long typeFileTextReadDoubleQuote(typeFileText* parent)
{
	if (parent == 0)
		return 0;

	if (parent->currentCharacter == 34)    // ASCII 34 " Double Quote
	{
		typeStringReset(&parent->currentString);

		typeFileTextAdvanceCharacter(parent);

		while (typeFileTextPeekCharacter(parent))
		{
			if (parent->currentCharacter == 34) // ASCII 34 " Double Quote
			{
				typeFileTextAdvanceCharacter(parent);

				return 1;
			}
			if (typeAsciiIsPrintable(parent->currentCharacter))
			{
				typeStringAddCharacter(&parent->currentString, parent->currentCharacter);

				typeFileTextAdvanceCharacter(parent);

				continue;
			}

			return 0;
		}
	}

	return 0;
}

long typeFileTextReadNumber(typeFileText* parent)
{
	double     sign = 1;
	double     value = 0;
	double     decimal = 1;
	double     e_sign = 1;
	double     e_value = 0;

	if (parent == 0)
		return 0;

	if (parent->currentCharacter == '+')
	{
		typeFileTextAdvanceCharacter(parent);
	}
	if (parent->currentCharacter == '-')
	{
		sign = -1;
		typeFileTextAdvanceCharacter(parent);
	}

	if ((parent->currentCharacter == 'n') || (parent->currentCharacter == 'N'))
	{
		if (typeFileTextReadString(parent))
		{
			if (typeStringCompareCharString(&parent->currentString, "nan", 3))
			{
				parent->currentNumber = 0;

				return 1;
			}
		}
	}

	if ((parent->currentCharacter == 'i') || (parent->currentCharacter == 'I'))
	{
		if (typeFileTextReadString(parent))
		{
			if (typeStringCompareCharString(&parent->currentString, "inf", 3))
			{
				parent->currentNumber = 0;

				return 1;
			}
		}
	}

	while (typeFileTextPeekCharacter(parent) && typeAsciiIsDigit(parent->currentCharacter))
	{
		value = value * 10 + (parent->currentCharacter - '0');

		typeFileTextAdvanceCharacter(parent);
	}

	if (parent->currentCharacter == '.')
	{
		typeFileTextAdvanceCharacter(parent);

		while (typeFileTextPeekCharacter(parent) && typeAsciiIsDigit(parent->currentCharacter))
		{
			value = value * 10 + (parent->currentCharacter - '0');
			decimal = decimal * 10;

			typeFileTextAdvanceCharacter(parent);
		}
	}

	if ((parent->currentCharacter == 'e') || (parent->currentCharacter == 'E'))
	{
		typeFileTextAdvanceCharacter(parent);

		if (value == 0)
			value = 1;

		if (parent->currentCharacter == '+')
		{
			typeFileTextAdvanceCharacter(parent);
		}

		if (parent->currentCharacter == '-')
		{
			e_sign = -1;
			typeFileTextAdvanceCharacter(parent);
		}

		while (typeFileTextPeekCharacter(parent) && typeAsciiIsDigit(parent->currentCharacter))
		{
			e_value = e_value * 10 + (parent->currentCharacter - '0');

			typeFileTextAdvanceCharacter(parent);
		}

		e_value = e_value * e_sign;
	}

	parent->currentNumber = (value * sign / decimal) * ((e_value == 0) ? 1 : pow(10, e_value));
}

long typeFileTextClose(typeFileText* parent)
{
	if (parent == 0)
		return 0;

	typeFileStreamClose(&parent->currentSource);

	return 1;
}

#endif // TYPEFILETEXT_H

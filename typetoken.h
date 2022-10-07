#ifndef TYPETOKEN_H
#define TYPETOKEN_H

#include <stdlib.h>
#include <string.h>

typedef struct
{
	unsigned long    tokenType;
	unsigned long    tokenDataSize;
	unsigned long    tokenChildCount;
	void*            tokenParent;
	void*            tokenFirstChild;
	void*            tokenLastChild;
	void*            tokenNext;
} typeToken;

long typeTokenInit(typeToken* parent)
{
	if (parent == 0)
		return 0;

	memset(parent, 0, sizeof(typeToken));

	return 1;
}

long typeTokenIsChild(typeToken* parent, typeToken* value)
{   
	typeToken* working;

	if ((parent == 0) || (value == 0))
		return 0;

	working = (typeToken*)parent->tokenFirstChild;

	while (working)
	{
		if (working == value)
			return 1;

		working = (typeToken*)working->tokenNext;
	}

	return 0;
}

long typeTokenInsertFirst(typeToken* parent, typeToken* value)
{
	if ((parent == 0) || (value == 0))
		return 0;

	if (parent == value)
		return 0;

	if (typeTokenIsChild(parent, value))
		return 0;

	parent->tokenChildCount++;
	value->tokenParent = parent;
	if (parent->tokenFirstChild == 0)
	{
		parent->tokenFirstChild = value;
		parent->tokenLastChild = value;
	}
	else
	{
		value->tokenNext = parent->tokenFirstChild;
		parent->tokenFirstChild = value;
	}

	return 1;
}

long typeTokenInsertLast(typeToken* parent, typeToken* value)
{
	typeToken* working;

	if ((parent == 0) || (value == 0))
		return 0;

	if (parent == value)
		return 0;

	if (typeTokenIsChild(parent, value))
		return 0;

	parent->tokenChildCount++;
	value->tokenParent = parent;
	if (parent->tokenFirstChild == 0)
	{
		parent->tokenFirstChild = value;
		parent->tokenLastChild = value;
	}
	else
	{
		working = (typeToken*)parent->tokenLastChild;
		working->tokenNext = value;
		parent->tokenLastChild = value;
	}

	return 1;
}

long typeTokenInsertBefore(typeToken* parent, typeToken* selector, typeToken* value)
{
	typeToken* working;
	typeToken* previous;

	if ((parent == 0) || (selector == 0)|| (value == 0))
		return 0;

	if (parent == value)
		return 0;

	if (typeTokenIsChild(parent, value))
		return 0;

	working = (typeToken*)parent->tokenFirstChild;
	previous = 0;

	while (working)
	{
		if (working == selector)
		{
			if (previous == 0)
			{
				typeTokenInsertFirst(parent, value);
			}
			else
			{
				parent->tokenChildCount++;
				value->tokenParent = parent;
				value->tokenNext = previous->tokenNext;
				previous->tokenNext = value;
			}

			return 1;
		}

		previous = working;
		working = (typeToken*)working->tokenNext;
	}

	return 0;
}

long typeTokenInsertAfter(typeToken* parent, typeToken* selector, typeToken* value)
{
	typeToken* working;

	if ((parent == 0) || (selector == 0) || (value == 0))
		return 0;

	if (parent == value)
		return 0;

	if (typeTokenIsChild(parent, value))
		return 0;

	working = (typeToken*)parent->tokenFirstChild;

	while (working)
	{
		if (working == selector)
		{
			if (working == parent->tokenLastChild)
			{
				typeTokenInsertLast(parent, value);
			}
			else
			{
				parent->tokenChildCount++;
				value->tokenParent = parent;
				value->tokenNext = working->tokenNext;
				working->tokenNext = value;
			}

			return 1;
		}

		working = (typeToken*)working->tokenNext;
	}

	return 0;
}

long typeTokenInsertAt(typeToken* parent, unsigned long index, typeToken* value)
{
	typeToken* working;
	typeToken* previous;
	unsigned long   counter;

	if ((parent == 0) || (value == 0))
		return 0;

	if (parent == value)
		return 0;

	if (typeTokenIsChild(parent, value))
		return 0;

	working = (typeToken*)parent->tokenFirstChild;
	previous = 0;
	counter = 0;

	while (working)
	{
		if (counter == index)
		{
			if (previous == 0)
			{
				typeTokenInsertFirst(parent, value);
			}
			else
			{
				parent->tokenChildCount++;
				value->tokenParent = parent;
				value->tokenNext = previous->tokenNext;
				previous->tokenNext = value;
			}

			return 1;
		}

		counter++;
		previous = working;
		working = (typeToken*)working->tokenNext;
	}

	return 1;
}

#endif // TYPETOKEN_H
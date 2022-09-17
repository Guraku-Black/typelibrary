#ifndef TYPESHAPE_H
#define TYPESHAPE_H

const unsigned long TYPE_INVALID_INDEX  = 0xFFFFFFFF;

typedef struct
{
	unsigned long     shapeWidth;
	unsigned long     shapeHeight;
	unsigned long     shapeDepth;
} typeShape;

void typeShapeZero(typeShape* parent)
{
	parent->shapeWidth = 0;
	parent->shapeHeight = 0;
	parent->shapeDepth = 0;
}

void typeShapeAssign(typeShape* parent, unsigned long X, unsigned long Y, unsigned long Z)
{
	parent->shapeWidth = X;
	parent->shapeHeight = Y;
	parent->shapeDepth = Z;
}

long typeShapeCompare(typeShape* parent, typeShape* target)
{
	if ((parent->shapeWidth != target->shapeWidth) ||
		(parent->shapeHeight != target->shapeHeight) ||
		(parent->shapeDepth != target->shapeDepth))
		return 0;

	return 1;
}

long typeShapeCompareValue(typeShape* parent, unsigned long X, unsigned long Y, unsigned long Z)
{
	if ((parent->shapeWidth != X) ||
		(parent->shapeHeight != Y) ||
		(parent->shapeDepth != Z))
		return 0;

	return 1;
}

long typeShapeIsInside(typeShape* parent, long X, long Y, long Z)
{
	if ((X < 0) || (X >= (long)parent->shapeWidth) ||
		(Y < 0) || (Y >= (long)parent->shapeHeight) ||
		(Z < 0) || (Z >= (long)parent->shapeDepth))
		return 0;

	return 1;
}

unsigned long typeShapeGetLength(typeShape* parent)
{
	return (parent->shapeWidth * parent->shapeHeight * parent->shapeDepth);
}

unsigned long typeShapeGetIndex(typeShape* parent, long X, long Y, long Z)
{
	if (typeShapeIsInside(parent, X, Y, Z))
		return ((parent->shapeWidth * parent->shapeHeight) * Z) + (Y * parent->shapeWidth + X);

	return TYPE_INVALID_INDEX;
}

#endif // TYPESHAPE_H

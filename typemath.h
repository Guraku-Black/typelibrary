#ifndef TYPEMATH_H
#define TYPEMATH_H

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

constexpr auto TYPE_PI = 3.141592653589793238462643383279502884197169399375105820974944;

void type_randomseed()
{
	srand((unsigned long)time(0));
}

double type_random()
{
	return (double)rand() / (RAND_MAX + 1);
}

double type_sgn(double a)
{
	if (a > 0)
		return 1;
	if (a < 0)
		return -1;

	return 0;
}

double type_abs(double a)
{
	if (a < 0)
		return 0 - a;

	return a;
}

double type_min(double a, double b)
{
	if (a > b)
		return b;

	return a;
}

double type_max(double a, double b)
{
	if (a < b)
		return b;

	return a;
}

double type_sin(double a)
{
	return sin(a / 180.0 * TYPE_PI);
}

double type_cos(double a)
{
	return cos(a / 180.0 * TYPE_PI);
}

double reluFunction(double a)
{
	if (a > 0)
		return a;

	return a * 0.01;
}

double reluDerivative(double a)
{
	if (a < 0)
		return 0.01;

	return 1;
}

double sigmoidFunction(double a)
{
	return 1 / (1 + exp(-a));
}

double sigmoidDerivative(double a)
{
	return (sigmoidFunction(a) * (1 - sigmoidFunction(a)));
}

double tanhFunction(double a)
{
	return tanh(a);
}

double tanhDerivative(double a)
{
	return 1 - (tanh(a) * tanh(a));
}

#endif // TYPEMATH_H

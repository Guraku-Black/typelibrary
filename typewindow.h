#ifndef TYPEWINDOW_H
#define TYPEWINDOW_H

#include <windows.h>

typedef struct
{
	HWND    windowHandle;
	BOOL    Running;
} typeWindow;

long typeWindowCreate(typeWindow* parent, HINSTANCE hInstance, WNDPROC windowProc);
long typeWindowGetEvent(typeWindow* parent);
long typeWindowPeekEvent(typeWindow* parent);
long typeWindowTextOut(typeWindow* parent, long x, long y, LPCWSTR text, DWORD length);
long typeWindowDestroy(typeWindow* parent);

const WCHAR  windowClass[] = L"windowClass";
const WCHAR  windowTitle[] = L"windowTitle";

long typeWindowCreate(typeWindow* parent, HINSTANCE hInstance, WNDPROC windowProc)
{
	WNDCLASSEXW wcex;

	if (parent == 0)
		return 0;

	wcex.cbSize = sizeof(WNDCLASSEX);
	wcex.style = CS_HREDRAW | CS_VREDRAW;
	wcex.lpfnWndProc = windowProc;
	wcex.cbClsExtra = 0;
	wcex.cbWndExtra = 0;
	wcex.hInstance = hInstance;
	wcex.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wcex.hCursor = LoadCursor(NULL, IDC_ARROW);
	wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
	wcex.lpszMenuName = NULL;
	wcex.lpszClassName = windowClass;
	wcex.hIconSm = LoadIcon(NULL, IDI_APPLICATION);

	if (!RegisterClassExW(&wcex))
	{
		return 0;
	}

	HWND hwnd = CreateWindowW(
		windowClass,
		windowTitle,
		WS_OVERLAPPEDWINDOW | WS_VISIBLE,
		CW_USEDEFAULT,
		CW_USEDEFAULT,
		CW_USEDEFAULT,
		CW_USEDEFAULT,
		NULL,
		NULL,
		hInstance,
		NULL);

	if (!hwnd)
	{
		return 0;
	}

	parent->windowHandle = hwnd;
	parent->Running = 1;

	return 1;
}

long typeWindowGetEvent(typeWindow* parent)
{
	MSG msg;

	if (parent == 0)
		return 0;

	if (GetMessage(&msg, NULL, 0, 0))
	{
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}

	return parent->Running;
}

long typeWindowPeekEvent(typeWindow* parent)
{
	MSG msg;

	if (parent == 0)
		return 0;

	if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
	{
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}

	return parent->Running;
}

long typeWindowTextOut(typeWindow* parent, long x, long y, LPCWSTR text, DWORD length)
{
	HDC     hdc;

	hdc = GetDC(parent->windowHandle);

	TextOut(hdc, x, y, text, length);

	ReleaseDC(parent->windowHandle, hdc);

	return 1;
}

long typeWindowDestroy(typeWindow* parent)
{
	if (parent == 0)
		return 0;

	DestroyWindow(parent->windowHandle);

	parent->windowHandle = 0;
	parent->Running = 0;

	return 1;
}

#endif // TYPEWINDOW_H

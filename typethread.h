#ifndef TYPETHREAD_H
#define TYPETHREAD_H

#include "typewindow.h"

typedef struct
{
	HANDLE     threadHandle;
	DWORD      threadID;
} typeThread;

long typeThreadCreate(typeThread* parent, LPTHREAD_START_ROUTINE threadProc);
long typeThreadDestroy(typeThread* parent);

DWORD WINAPI typeThreadProc(LPVOID lpParam)
{
	return 0;
}

long typeThreadCreate(typeThread* parent, LPTHREAD_START_ROUTINE threadProc)
{
	HANDLE  handle;
	DWORD   ID;

	if (parent == 0)
		return 0;

	handle = CreateThread(
		NULL,
		0,
		threadProc,
		0,
		0,
		&ID);

	if (handle == NULL)
		return 0;

	parent->threadHandle = handle;
	parent->threadID = ID;

	return 1;
}

long typeThreadDestroy(typeThread* parent)
{
	if (parent == 0)
		return 0;

	CloseHandle(parent->threadHandle);

	parent->threadHandle = 0;
	parent->threadID = 0;

	return 1;
}

#endif // TYPETHREAD_H

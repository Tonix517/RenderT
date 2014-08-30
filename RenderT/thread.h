#ifndef THREAD_H
#define THREAD_H

#include "windows.h"

void start_thread();
void end_thread();

//	Bloom
void start_bloom_thread();
void thread_bloom(void *);

//	Tone-mapping
void start_tone_m_thread();
void thread_tone_m(void *);

void thread_startComputing(void*);

#endif
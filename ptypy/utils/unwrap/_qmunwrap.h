/*
Quantised-level quality-map phase unwrapping.

Author: Pierre Thibault
Date: First version sometimes around 2010.
*/

#ifndef PTYPY_QMUNWRAP_H
#define PTYPY_QMUNWRAP_H

// Return codes of qmunwrap_unwrap().
#define QMUNWRAP_OK 0
#define QMUNWRAP_EINVAL 1
#define QMUNWRAP_ENOMEM 2

// Quality map (squared wrapped gradient). qmap must be zero-filled by the caller.
void qmunwrap_qualitymap(double* phase, int N0, int N1, double* qmap);

// Quantize the array a into N bins.
void qmunwrap_quantize(double* a, int size, int N, int* aout);

// Quality-guided unwrapping of phase into aout, both of shape (N0, N1).
int qmunwrap_unwrap(double* phase, int N0, int N1, int num_levels,
                    int start0, int start1, double* aout);

#endif

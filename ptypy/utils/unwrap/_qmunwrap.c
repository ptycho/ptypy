/*
Quantised-level quality-map phase unwrapping.

This implementation is meant to accelerate the quality map approach
through a binning process of the quality map. This eliminates the need for sorting
at the price of a (low-risk) non-sequential unwrapping.

Author: Pierre Thibault
Date: First version sometimes around 2010.
*/

#include <math.h>
#include <stdlib.h>
#include <limits.h>

#include "_qmunwrap.h"

// Pixel states during the flood fill.
#define FREE 0
#define QUEUED 1
#define DONE 2


void qmunwrap_qualitymap(double* phase, int N0, int N1, double* qmap)
{
    /*
    Basic quality test for the wrapped phase.
    Input:
         - phase: input phase array
         - N0, N1: dimensions of the phase array
    Output:
         - qmap: quality map array (squared gradient)
    */
    int i,j;
    double pi = 3.141592653589793;
    double d0, d1;

    // Loop over all pixels except the last row and column
    for(i=0; i<N0-1; i++)
        for(j=0; j<N1-1; j++)
        {
            d0 = phase[(i+1)*N1 + j] - phase[i*N1 + j];
            d0 -= 2.*pi*round(d0/(2.*pi));
            qmap[i*N1 + j] += d0*d0;
            qmap[(i+1)*N1 + j] += d0*d0;
            d1 = phase[i*N1 + (j+1)] - phase[i*N1 + j];
            d1 -= 2.*pi*round(d1/(2.*pi));
            qmap[i*N1 + j] += d1*d1;
            qmap[i*N1 + (j+1)] += d1*d1;
        }

    // Last column
    for(i=0; i<N0-1; i++)
    {
        j = N1-1;
        d0 = phase[(i+1)*N1 + j] - phase[i*N1 + j];
        d0 -= 2.*pi*round(d0/(2.*pi));
        qmap[i*N1 + j] += d0*d0;
        qmap[(i+1)*N1 + j] += d0*d0;
    }

    // Last row
    for(j=0; j<N1-1; j++)
    {
        i = N0-1;
        d1 = phase[i*N1 + (j+1)] - phase[i*N1 + j];
        d1 -= 2.*pi*round(d1/(2.*pi));
        qmap[i*N1 + j] += d1*d1;
        qmap[i*N1 + (j+1)] += d1*d1;
    }

}


void qmunwrap_quantize(double* a, int size, int N, int* aout)
{
    /*
    Quantize the array a into N bins.
    */
    int i;
    double amin, amax, dbin;

    amin = a[0];
    amax = a[0];
    for(i=1; i<size; i++)
    {
        if(a[i] < amin)
            amin = a[i];
        if(a[i] > amax)
            amax = a[i];
    }
    dbin = (amax - amin)/N;

    // Constant array (or NaNs around): everything goes into the first bin.
    if(!(dbin > 0.))
    {
        for(i=0; i<size; i++)
            aout[i] = 0;
        return;
    }

    for(i=0; i<size; i++)
    {
        aout[i] = (int)((a[i] - amin)/dbin);
        if(aout[i] < 0)
            aout[i] = 0;
        else if(aout[i] >= N)
            aout[i] = N-1;
    }
}


/*
Queue of pixels waiting to be unwrapped, with one slice per quality level.
Each entry is a pair: "to" is the pixel to unwrap, "from" an already unwrapped
neighbour it is unwrapped against. A pixel is queued at most once and always
lands in bin qbin[to], so slice b never needs room for more than the number of
pixels falling in bin b. offsets[] slices a single Nt-sized array accordingly.
*/
typedef struct
{
    int N1;
    int Nt;
    int* qbin;
    int* mask;
    int* offsets;
    int* nbins;
    int* bins0;
    int* bins1;
} queue;


static void push(queue* que, int from, int to)
{
    int b;

    if (que->mask[to] != FREE)
        return;
    b = que->qbin[to];
    que->bins0[que->offsets[b] + que->nbins[b]] = from;
    que->bins1[que->offsets[b] + que->nbins[b]] = to;
    que->nbins[b] += 1;
    que->mask[to] = QUEUED;
}


static void push_neighbours(queue* que, int p)
{
    int col = p % que->N1;

    if (col + 1 < que->N1)
        push(que, p, p + 1);            // east
    if (col > 0)
        push(que, p, p - 1);            // west
    if (p + que->N1 < que->Nt)
        push(que, p, p + que->N1);      // south
    if (p - que->N1 >= 0)
        push(que, p, p - que->N1);      // north
}


int qmunwrap_unwrap(double* phase, int N0, int N1, int num_levels, int start0, int start1, double* aout)
{

    /*
    Phase unwrapping using binned quantized phase gradient levels.
    Input:
         - phase: the phase array to unwrap from the interval [0, 2 pi).
         - N0, N1: dimensions of the phase array
         - num_levels: number of bins the gradients are stored into.
         - start0, start1: coordinate of the starting point in the unwrapping routine
    Output:
         - aout: unwrapped phase array
    Returns QMUNWRAP_OK, QMUNWRAP_EINVAL or QMUNWRAP_ENOMEM.
    Behaviour is not expected to be much different for num_levels > 20 or so.
    */

    int ok, k, i;
    int Nl, Nt;
    int p0,pp0,pp1,ibin;
    int status = QMUNWRAP_ENOMEM;
    double pi = 3.141592653589793;
    double a_jump;
    queue que;

    if (N0 < 1 || N1 < 1 || num_levels < 1)
        return QMUNWRAP_EINVAL;
    if (N0 > INT_MAX/N1)
        return QMUNWRAP_EINVAL;
    if (start0 < 0 || start0 >= N0 || start1 < 0 || start1 >= N1)
        return QMUNWRAP_EINVAL;

    ok = 1;
    Nl = num_levels;
    Nt = N0*N1;

    // Quality map
    double* qmap = (double*)calloc(Nt, sizeof(double));
    // Quantized quality map
    int* qbin = (int*)malloc(Nt * sizeof(int));
    // This mask keeps track of which pixels are FREE, QUEUED or DONE
    int* mask = (int*)calloc(Nt, sizeof(int));
    // Number of elements currently in each bin
    int* nbins = (int*)calloc(Nl, sizeof(int));
    // Where each bin's slice of bins0/bins1 starts
    int* offsets = (int*)calloc(Nl, sizeof(int));
    // Which pixel pair belongs to which bin
    int* bins0 = (int*)malloc(Nt * sizeof(int));
    int* bins1 = (int*)malloc(Nt * sizeof(int));

    if (!qmap || !qbin || !mask || !nbins || !offsets || !bins0 || !bins1)
        goto done;

    // generate quantized quality map
    qmunwrap_qualitymap(phase, N0, N1, qmap);
    qmunwrap_quantize(qmap, Nt, Nl, qbin);

    // Bin histogram, turned into the per-level slice offsets
    for (i = 0; i < Nt; i++)
        offsets[qbin[i]] += 1;
    for (i = 0, ibin = 0; ibin < Nl; ibin++)
    {
        int count = offsets[ibin];
        offsets[ibin] = i;
        i += count;
    }

    // Copy initial phase values
    for (i = 0; i < Nt; i++)
    {
        aout[i] = phase[i];
    }

    que.N1 = N1;
    que.Nt = Nt;
    que.qbin = qbin;
    que.mask = mask;
    que.offsets = offsets;
    que.nbins = nbins;
    que.bins0 = bins0;
    que.bins1 = bins1;

    // seed
    p0 = N1*start0 + start1;
    mask[p0] = DONE;
    k = 1;

    // Take care of the first neighbors.
    push_neighbours(&que, p0);

    while(k<Nt)
    {
        ok = 1;
        // loop over bins, always starting from the highest quality
        for (ibin = 0; ibin < Nl; ibin++)
        {
            if (ok && (nbins[ibin] > 0))
            {
                // This "pops" one of the elements
                nbins[ibin]-=1;

                // unwrap from pp0 to pp1
                pp0 = bins0[offsets[ibin] + nbins[ibin]];
                pp1 = bins1[offsets[ibin] + nbins[ibin]];
                a_jump = (aout[pp0]-aout[pp1]);

                // This is where unwrapping happens
                aout[pp1] += 2.*pi*round(a_jump/(2*pi));
                mask[pp1] = DONE;
                k+=1;

                // add neigbors
                push_neighbours(&que, pp1);

                // start over
                ok = 0;
            }
        }
        if (ok) {
            break;
        }
    }
    status = QMUNWRAP_OK;

done:
    free(qmap);
    free(qbin);
    free(mask);
    free(offsets);
    free(bins0);
    free(bins1);
    free(nbins);
    return status;
}
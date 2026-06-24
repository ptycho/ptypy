/*
Quantised-level quality-map phase unwrapping.

This implementation is meant to accelerate the quality map approach
through a binning process of the quality map. This eliminates the need for sorting
at the price of a (low-risk) non-sequential unwrapping.

Author: Pierre Thibault
Date: First version sometimes around 2010.
*/

#include "math.h"
#include "stdlib.h"
#include "stdio.h"

static inline double dround(double x) {
    return (x >= 0.0) ? floor(x + 0.5) : ceil(x - 0.5);
}


void qualitymap(double* phase, int N0, int N1, double* qmap)
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
            d0 = fmod(d0 + pi, 2.*pi) - pi;
            qmap[i*N1 + j] += d0*d0;
            qmap[(i+1)*N1 + j] += d0*d0;
            d1 = phase[i*N1 + (j+1)] - phase[i*N1 + j];
            d1 = fmod(d1 + pi, 2.*pi) - pi;
            qmap[i*N1 + j] += d1*d1;
            qmap[i*N1 + (j+1)] += d1*d1;
        }

    // Last row
    for(i==0; i<N0-1; i++)
    {
        j = N1-1;
        d0 = phase[(i+1)*N1 + j] - phase[i*N1 + j];
        d0 = fmod(d0 + pi, 2.*pi) - pi;
        qmap[i*N1 + j] += d0*d0;
        qmap[(i+1)*N1 + j] += d0*d0;
    }

    // Last column
    for(j==0; j<N1-1; j++)
    {
        i = N0-1;
        d1 = phase[i*N1 + (j+1)] - phase[i*N1 + j];
        d1 = fmod(d1 + pi, 2.*pi) - pi;
        qmap[i*N1 + j] += d1*d1;
        qmap[i*N1 + (j+1)] += d1*d1;
    }

}


void quantize(double* a, int size, int N, int* aout)
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

    for(i=0; i<size; i++)
    {
        aout[i] = (int)((a[i] - amin)/dbin);
        if(aout[i] == N)
            aout[i] = N-1;
    }
}


void unwrap(double* phase, int N0, int N1, int num_levels, int start0, int start1, double* aout)
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
    Because of memory consumption, num_levels should not be too high. Behaviour is not expected to be much different for num_levels > 20 or so.
    */

    int ok, k, percent_done, last_percent = -1;
    int Nl, Nt;
    int p,p0,q,pp0,pp1,ibin;
    double pi = 3.141592653589793;
    double a_jump;

    ok = 1;
    Nl = num_levels;
    Nt = N0*N1;

    // Quality map
    double* qmap = (double*)calloc(Nt, sizeof(double));
    // Quantized quality map
    int* qbin = (int*)malloc(Nt * sizeof(int));
    // This mask will keep track of which pixels are unwrapped
    int* mask = (int*)calloc(Nt, sizeof(int));
    // Number of elements in each bin
    int* nbins = (int*)calloc(Nl, sizeof(int));
    // Which pixel belongs to which bin
    int* bins0 = (int*)malloc(Nl * Nt * sizeof(int));
    int* bins1 = (int*)malloc(Nl * Nt * sizeof(int));
    // These two arrays are much too large, maybe there is a way to prove that no one quantized bin can contain more than a given number of elements? For now, memory is allocated as if all pixels falling into the same bin is a possibility.
    

    // generate quantized quality map
    qualitymap(phase, N0, N1, qmap);
    quantize(qmap, Nt, Nl, qbin);

    // Copy initial phase values
    for (int i = 0; i < Nt; i++)
    {
        aout[i] = phase[i];
    }

    // seed
    p0 = N1*start0 + start1;
    mask[p0] = 1;
    k = 1;
 
    // Take care of the first neighbors.
    // east
    p = p0 + 1; 
    if ((p % N1) != 0)
    {
        q = qbin[p];
        bins0[q*Nt + nbins[q]] = p0;
        bins1[q*Nt + nbins[q]] = p;
        nbins[q]+=1;
    }
    // west
    p = p0 - 1;
    if (((p0 % N1) != 0) && (mask[p] == 0))
    {
        q = qbin[p];
        bins0[q*Nt + nbins[q]] = p0;
        bins1[q*Nt + nbins[q]] = p;
        nbins[q]+=1;
    }

    // south
    p = p0 + N1;
    if ((p < Nt) && (mask[p] == 0))
    {
        q = qbin[p];
        bins0[q*Nt + nbins[q]] = p0;
        bins1[q*Nt + nbins[q]] = p;
        nbins[q]+=1;
    }
    // north
    p = p0 - N1;
    if ((p > 0) && (mask[p] == 0))
    {
        q = qbin[p];
        bins0[q*Nt + nbins[q]] = p0;
        bins1[q*Nt + nbins[q]] = p;
        nbins[q]+=1;
    }

    while(k<Nt)
    {
        ok = 1;
        // printf("Unwrapping pixel %d/%d\r", k, Nt);
        // fflush(stdout);
        // loop over bins, always starting from the highest quality
        for (ibin = 0; ibin < Nl; ibin++)
        {
            if (ok && (nbins[ibin] > 0))
            {
                // This "pops" one of the elements
                nbins[ibin]-=1;

                // unwrap from pp0 to pp1
                pp0 = bins0[ibin*Nt + nbins[ibin]];
                pp1 = bins1[ibin*Nt + nbins[ibin]];

                if (mask[pp1])
                {
                    // This pixel was already unwrapped, let's move on
                    ok = 0; // set ok to 0 to restart the loop from the highest quality
                    continue;
                }
                a_jump = (aout[pp0]-aout[pp1]);

                // This is where unwrapping happens
                aout[pp1] += 2.*pi*dround(a_jump/(2*pi));
                mask[pp1] = 1;
                k+=1;

                // add neigbors
                p0 = pp1;

                // east
                p = p0 + 1;
                if ((p % N1) != 0 && (mask[p]==0))
                {
                    q = qbin[p];
                    bins0[q*Nt + nbins[q]] = p0;
                    bins1[q*Nt + nbins[q]] = p;
                    nbins[q]+=1;
                }
                // west
                p = p0 - 1;
                if ((p0 % N1) != 0 && (mask[p]==0))
                {
                    q = qbin[p];
                    bins0[q*Nt + nbins[q]] = p0;
                    bins1[q*Nt + nbins[q]] = p;
                    nbins[q]+=1;
                }
                // south
                p = p0 + N1;
                if (p < Nt && (mask[p]==0))
                {
                    q = qbin[p];
                    bins0[q*Nt + nbins[q]] = p0;
                    bins1[q*Nt + nbins[q]] = p;
                    nbins[q]+=1;
                }
                // north
                p = p0 - N1;
                if (p > 0 && (mask[p]==0))
                {
                    q = qbin[p];
                    bins0[q*Nt + nbins[q]] = p0;
                    bins1[q*Nt + nbins[q]] = p;
                    nbins[q]+=1;
                }
                // start over
                ok = 0;
            }
        }
        /*
        percent_done = (k * 100) / Nt;
        if (percent_done != last_percent) {
            printf("%d%% completed\n", percent_done);
            last_percent = percent_done;
        }
        */
        if (ok) {
            break;
        }
    }
    free(qmap);
    free(qbin);
    free(mask);
    free(bins0);
    free(bins1);
    free(nbins);
}
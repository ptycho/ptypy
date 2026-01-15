/** Core implementation of the filtered FFT using cuFFT + callbacks.
 * Note, this is a .cu file actually. The .cpp is used to make it work with cppimport
 *
 * The FilteredFFTImpl class is implemented as a template, 
 * to allow a specific implementation with compile-time constants
 * as the offset modulo operations can by optimised a lot if the
 * compiler knows the modulo operands.
 *
 * Therefore this code assumes that the following pre-processor
 * definitions are defined during compilation - otherwise it will 
 * generate an FFT with 128x128 arrays:
 * 
 * - MY_FFT_ROWS
 * - MY_FFT_COLUMNS
 *
 * Also note that this implementation only works on Linux 64bit,
 * as this is a limitation of cuFFT.
 * On other platforms, an implementation with separately implemented
 * filters should be used, e.g. with pycuda and scikit-cuda.
 *
 */

#include "errors.h"
#include "filtered_fft.h"

#include <cufft.h>
#include <cufftXt.h>
#include <cuda.h>
#include <cmath>
#include <cassert>

template <int ROWS, int COLUMNS, bool SYMMETRIC, bool IS_FORWARD>
class FilteredFFTImpl : public FilteredFFT {
public:
    
    /** Sets up the plan on init.
     *
     * @param batches Number of batches
     * @param prefilt Device pointer to prefilter array (can be NULL)
     * @param postfilt Device pointer to postfilter array (can be NULL)
     * @param stream Stream to use for the GPU
     */
    FilteredFFTImpl(int batches, 
                complex<float>* prefilt, complex<float>* postfilt,
                cudaStream_t stream) :
        batches_(batches),
        prefilt_(prefilt),
        postfilt_(postfilt),
        stream_(stream)
    {
        setupPlan();
    }

    int getBatches() const override { return batches_; }
    int getRows() const override { return ROWS; }
    int getColumns() const override { return COLUMNS; }
    bool isForward() const override { return IS_FORWARD; }
    void setStream(cudaStream_t stream) override {
        stream_ = stream;
        cudaCheck(cufftSetStream(plan_, stream));
    };
    virtual cudaStream_t getStream() const {
        return stream_;
    }

    /// Run the FFT (forward) - can be in-place
    void fft(complex<float>* input, complex<float>* output) override
    {
        if (SYMMETRIC && !IS_FORWARD) {
            throw std::runtime_error("Calling FFT on a reverse-initialised instance");
        } else {
            cudaCheck(cufftExecC2C(plan_,
                reinterpret_cast<cufftComplex*>(input),
                reinterpret_cast<cufftComplex*>(output),
                CUFFT_FORWARD
            ));
        }
    }

    /// Run the IFFT (reverse) - can be in-place
    void ifft(complex<float>* input, complex<float>* output) override
    {
        if (SYMMETRIC && IS_FORWARD) {
            throw std::runtime_error("Calling IFFT on a forward-initialised instance");
        } else {
            cudaCheck(cufftExecC2C(plan_,
                reinterpret_cast<cufftComplex*>(input),
                reinterpret_cast<cufftComplex*>(output),
                CUFFT_INVERSE
            ));
        }
    }

    ~FilteredFFTImpl()
    {
        cufftDestroy(plan_);
    }


    ///////// Different variants of the callback device functions /////

    /// load with prefilter
    __device__ static cufftComplex CB_prefilt(
        void *dataIn,
        size_t offset,
        void* callerInf,
        void* sharedPtr
    ) {
        auto inData = reinterpret_cast<complex<float>*>(dataIn);
        auto filter = reinterpret_cast<complex<float>*>(callerInf);
        auto v = inData[offset];
        // Note:
        // Modulo with powers of 2 are replaced by bit operations by the compiler,
        // which are much faster.
        // If non-powers of 2 are needed, it might be possible to work out the 
        // per-array filter offset from the threadIdx / blockidx fields.
        v *= filter[offset % (ROWS*COLUMNS)];  
        return {v.real(), v.imag()};
    }

    /// store with postfilter + scaling
    __device__ static void CB_postfilt(
        void* dataOut,
        size_t offset,
        cufftComplex element,
        void* callerInf,
        void* sharedPtr
    ) {
        auto outData = reinterpret_cast<complex<float>*>(dataOut);
        auto filter = reinterpret_cast<complex<float>*>(callerInf);
        auto v = complex<float>(element.x, element.y);
        if (!SYMMETRIC && !IS_FORWARD) {
            v *= filter[offset % (ROWS*COLUMNS)] / (ROWS*COLUMNS);
        }
        else if (IS_FORWARD && !SYMMETRIC) {
            v *= filter[offset % (ROWS*COLUMNS)];
        }
        else {
            float fact;
            if (ROWS == COLUMNS) {
                fact = ROWS;
            } else {
                fact = sqrt(float(ROWS*COLUMNS));
            }
            v *= filter[offset % (ROWS*COLUMNS)] / fact;
        } 
        outData[offset] = v;
    }

    /// store with scaling only (no postfilter)
    __device__ static void CB_postfilt_scaleonly(
        void* dataOut,
        size_t offset,
        cufftComplex element,
        void* callerInf,
        void* sharedPtr
    ) {
        auto outData = reinterpret_cast<complex<float>*>(dataOut);
        auto v = complex<float>(element.x, element.y);
        if (!SYMMETRIC && !IS_FORWARD) {  
            v /= ROWS * COLUMNS;
        } else {
            float fact;
            if (ROWS == COLUMNS) {
                fact = ROWS;
            } else {
                fact = sqrt(float(ROWS*COLUMNS));
            }
            v /= fact;
        }
        outData[offset] = v;
    }


private:
    /// the core of the plan setup
    void setupPlan();

    int batches_;               ///< number of batchs
    cufftHandle plan_;          ///< cuFFT plan handle
    complex<float>* prefilt_;   ///< prefilter pointer
    complex<float>* postfilt_;  ///< postfilter pointer
    cudaStream_t stream_;       ///< stream to operate on
};


/// Device-globals to keep function pointers
/// These need to be set on the device, copied to host,
/// and then passed to the cuFFT plan.

__device__ cufftCallbackLoadC d_loadCallbackPtr; 
__device__ cufftCallbackStoreC d_storeCallbackPtr;

/// small kernel to set the load callback device function pointer
template <int ROWS, int COLUMNS, bool SYMMETRIC, bool IS_FORWARD>
__global__ void setLoadDevFunPtr()
{
    d_loadCallbackPtr = FilteredFFTImpl<ROWS,COLUMNS,SYMMETRIC,IS_FORWARD>::CB_prefilt;
}

/// small kernel to set the store callback device function pointer
template <int ROWS, int COLUMNS, bool SYMMETRIC, bool IS_FORWARD>
__global__ void setStoreDevFunPtr()
{
    d_storeCallbackPtr = FilteredFFTImpl<ROWS,COLUMNS,SYMMETRIC,IS_FORWARD>::CB_postfilt;
}

/// small kernel to set the store callback device function pointer for scale only
template <int ROWS, int COLUMNS, bool SYMMETRIC, bool IS_FORWARD>
__global__ void setStoreScaleDevFunPtr()
{
    d_storeCallbackPtr = FilteredFFTImpl<ROWS,COLUMNS,SYMMETRIC,IS_FORWARD>::CB_postfilt_scaleonly;
}


/// setup the plan
template <int ROWS, int COLUMNS, bool SYMMETRIC, bool IS_FORWARD>
void FilteredFFTImpl<ROWS,COLUMNS,SYMMETRIC,IS_FORWARD>::setupPlan() {
    // basic plan setup
    cudaCheck(cufftCreate(&plan_));
    int dims[] = {ROWS, COLUMNS};
    size_t workSize;
    cudaCheck(cufftMakePlanMany(
        plan_, 2, dims, 0, 0, 0, 0, 0, 0, CUFFT_C2C, batches_, &workSize
    ));
    cudaCheck(cufftSetStream(plan_, stream_));

    /*
    std::cout << "Created plan for " << ROWS << "x" << COLUMNS
              << ", for " << batches_ << " with scratch memory of "
              << double(workSize) / 1024.0 / 1024.0 << "MB"
              << std::endl;
    */

    // pre-filter
    if (prefilt_)   // no need to set load callback if we're not prefiltering
    {
        setLoadDevFunPtr<ROWS,COLUMNS,SYMMETRIC,IS_FORWARD><<<1,1>>>();
        cufftCallbackLoadC h_loadCallbackPtr;
        cudaCheck(cudaMemcpyFromSymbol(&h_loadCallbackPtr, d_loadCallbackPtr, sizeof(h_loadCallbackPtr)));
        cudaCheck(cufftXtSetCallback(plan_, 
            (void**)&h_loadCallbackPtr, 
            CUFFT_CB_LD_COMPLEX, 
            (void**)&prefilt_));    
    }

    // post-filter
    if (!(IS_FORWARD && !SYMMETRIC) || postfilt_)  // we scale in postCall, so also needed if not postfiltering
    {
        cufftCallbackStoreC h_storeCallbackPtr;
        if (postfilt_) {
            setStoreDevFunPtr<ROWS,COLUMNS,SYMMETRIC,IS_FORWARD><<<1,1>>>();
        } 
        else {
            setStoreScaleDevFunPtr<ROWS,COLUMNS,SYMMETRIC,IS_FORWARD><<<1,1>>>();
        }
        cudaCheck(cudaMemcpyFromSymbol(&h_storeCallbackPtr, d_storeCallbackPtr, sizeof(h_storeCallbackPtr)));
        cudaCheck(cufftXtSetCallback(plan_, 
            (void**)&h_storeCallbackPtr, 
            CUFFT_CB_ST_COMPLEX, 
            (void**)&postfilt_));
    }
}

template <bool SYMMETRIC, bool FORWARD>
static FilteredFFT* make(int batches, int rows, int cols, complex<float>* prefilt, complex<float>* postfilt, 
  cudaStream_t stream)
{
    // we only support rows / colums are equal and powers of 2, from 16x16 to 512x512
    if (rows != cols) 
      throw std::runtime_error("Only equal numbers of rows and columns are supported");
    switch (rows)
    {
        case 16: return new FilteredFFTImpl<16, 16, SYMMETRIC, FORWARD>(batches, prefilt, postfilt, stream);
        case 32: return new FilteredFFTImpl<32, 32, SYMMETRIC, FORWARD>(batches, prefilt, postfilt, stream);
        case 64: return new FilteredFFTImpl<64, 64, SYMMETRIC, FORWARD>(batches, prefilt, postfilt, stream);
        case 128: return new FilteredFFTImpl<128, 128, SYMMETRIC, FORWARD>(batches, prefilt, postfilt, stream);
        case 256: return new FilteredFFTImpl<256, 256, SYMMETRIC, FORWARD>(batches, prefilt, postfilt, stream);
        case 512: return new FilteredFFTImpl<512, 512, SYMMETRIC, FORWARD>(batches, prefilt, postfilt, stream);
        case 1024: return new FilteredFFTImpl<1024, 1024, SYMMETRIC, FORWARD>(batches, prefilt, postfilt, stream);
        case 2048: return new FilteredFFTImpl<2048, 2048, SYMMETRIC, FORWARD>(batches, prefilt, postfilt, stream);
        default: throw std::runtime_error("Only powers of 2 from 16 to 2048 are supported");
    }
}

//////////// Factory Functions for Python

// Note: This will instantiate templates for 8 powers of 2, with 4 combinations of forward/reverse, symmetric/not,
// i.e. 32 different FFTs into the binary. Compile time might be quite long, but we intend to do this once
// during installation

FilteredFFT* make_filtered(
  int batches, 
  int rows, int cols,
  bool symmetricScaling,
  bool isForward,
  complex<float>* prefilt, complex<float>* postfilt, 
  cudaStream_t stream)
{
    if (symmetricScaling)
    {
        if (isForward) {
            return make<true, true>(batches, rows, cols, prefilt, postfilt, stream);
        } else {
            return make<true, false>(batches, rows, cols, prefilt, postfilt, stream);
        }
    }
    else
    {
        if (isForward) {
            return make<false, true>(batches, rows, cols, prefilt, postfilt, stream);
        } else {
            return make<false, false>(batches, rows, cols, prefilt, postfilt, stream);
        }
    }
    
}

void destroy_filtered(FilteredFFT* fft)
{
    delete fft;
}
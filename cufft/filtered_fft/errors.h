#pragma once
#include <string>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cufft.h>


inline std::string getloc(const char* func, const char* file, int line)
{
    return std::string(file) + ":" + std::to_string(line) + ": in function "
        + func;
}

inline void handleCudaCheck(cudaError_t res, const char* func, const char *file, int line) 
{
    if (res != 0) {
        throw std::runtime_error(std::string("CUDA Error: ") 
            + getloc(func, file, line) + ": " + cudaGetErrorString(res));
    }
}

inline void handleCudaCheck(cufftResult res, const char* func, const char* file, int line) {
    if (res) {
        const char* errstr;
        switch (res) {
            case CUFFT_INVALID_PLAN:
                errstr = "cuFFT was passed an invalid plan handle";
                break;
            case CUFFT_ALLOC_FAILED:
                errstr = "cuFFT failed to allocate GPU or CPU memory";
                break;
            case CUFFT_INVALID_TYPE:
                errstr = "No longer used";
                break;
            case CUFFT_INVALID_VALUE:
                errstr = "User specified an invalid pointer or parameter";
                break;
            case CUFFT_INTERNAL_ERROR:
                errstr = "Driver or internal cuFFT library error";
                break;
            case CUFFT_EXEC_FAILED:
                errstr = "Failed to execute an FFT on the GPU";
                break;
            case CUFFT_SETUP_FAILED:
                errstr = "The cuFFT library failed to initialize";
                break;
            case CUFFT_INVALID_SIZE:
                errstr = "User specified an invalid transform size";
                break;
            case CUFFT_UNALIGNED_DATA:
                errstr = " No longer used";
                break;
            case CUFFT_INCOMPLETE_PARAMETER_LIST:
                errstr = " Missing parameters in call ";
                break;
            case CUFFT_INVALID_DEVICE:
                errstr = "Execution of a plan was on different GPU than plan creation";
                break;
            case CUFFT_PARSE_ERROR:
                errstr = "Internal plan database error";
                break;
            case CUFFT_NO_WORKSPACE:
                errstr = "No workspace has been provided prior to plan execution";
                break;
            case CUFFT_NOT_IMPLEMENTED:
                errstr =
                    "Function does not implement functionality for parameters given.";
                break;
            case CUFFT_LICENSE_ERROR:
                errstr = "Used in previous versions.";
                break;
            case CUFFT_NOT_SUPPORTED:
                errstr = "Operation is not supported for parameters given.";
                break;
            default:
                errstr = "Unknown";
        }

        throw std::runtime_error(std::string("CuFFT Error: ") + 
            getloc(func, file, line) +
            ": " + errstr);
    }
}

#define cudaCheck(e) \
    handleCudaCheck(e, __FUNCTION__, __FILE__, __LINE__)

#include <pybind11/pybind11.h>
#include "filtered_fft.h"


/** Wrapper class to expose to Python, taking size_t instead of all
 * the pointers, which should contain the addresses of the data.
 * For gpuarrays, the gpuarray.gpudata member can be cast to int
 * in Python and it will be the raw pointer address.
 * cudaStreams can be cast to int as well.
 * (this saves us a lot of type definition / conversion code with pybind11)
 *
 */
class FilteredFFTPython
{
public:
    FilteredFFTPython(int batches, int rows, int columns, bool symmetric, 
        bool is_forward,
        std::size_t prefilt_ptr,
        std::size_t postfilt_ptr,
        std::size_t stream) 
    {
        fft_ = make_filtered(
            batches, 
            rows, columns,
            symmetric,
            is_forward,
            reinterpret_cast<complex<float>*>(prefilt_ptr),
            reinterpret_cast<complex<float>*>(postfilt_ptr),
            reinterpret_cast<cudaStream_t>(stream)
        );
    }

    int getBatches() const { return fft_->getBatches(); }
    int getRows() const { return fft_->getRows(); }
    int getColumns() const { return fft_->getColumns(); }
    bool isForward() const { return fft_->isForward(); }
    void setStream(std::size_t stream) {
        fft_->setStream(reinterpret_cast<cudaStream_t>(stream));
    }
    std::size_t getStream() const {
        return reinterpret_cast<std::size_t>(fft_->getStream());
    }

    void fft(std::size_t in_ptr, std::size_t out_ptr)
    {
        fft_->fft(
            reinterpret_cast<complex<float>*>(in_ptr), 
            reinterpret_cast<complex<float>*>(out_ptr)
        );
    }

    void ifft(std::size_t in_ptr, std::size_t out_ptr)
    {
        fft_->ifft(
            reinterpret_cast<complex<float>*>(in_ptr), 
            reinterpret_cast<complex<float>*>(out_ptr)
        );
    }

    ~FilteredFFTPython() {
        delete fft_;
    }

private:
    FilteredFFT* fft_;
};


/////////////// Pybind11 Export Definition ///////////////

namespace py = pybind11;


PYBIND11_MODULE(filtered_cufft, m) {
    m.doc() = "Filtered FFT for PtyPy";

    py::class_<FilteredFFTPython>(m, "FilteredFFT", py::module_local())
        .def(py::init<int, int, int, bool, bool, std::size_t, std::size_t,std::size_t>(),
             py::arg("batches"), 
             py::arg("rows"),
             py::arg("columns"),
             py::arg("symmetricScaling"), 
             py::arg("is_forward"),
             py::arg("prefilt"), 
             py::arg("postfilt"), 
             py::arg("stream")
        )
        .def("fft", &FilteredFFTPython::fft, 
             py::arg("input_ptr"), 
             py::arg("output_ptr")
        )
        .def("ifft", &FilteredFFTPython::ifft, 
             py::arg("input_ptr"), 
             py::arg("output_ptr")
        )
        .def_property_readonly("batches", &FilteredFFTPython::getBatches)
        .def_property_readonly("rows", &FilteredFFTPython::getRows)
        .def_property_readonly("columns", &FilteredFFTPython::getColumns)
        .def_property_readonly("is_forward", &FilteredFFTPython::isForward)
        .def_property("queue", &FilteredFFTPython::getStream, &FilteredFFTPython::setStream);
}


from libcpp.string cimport string

cdef extern void difference_map_realspace_constraint_c(
    const float* fobj_and_probe,
    const float* fexit_wave,
    float alpha,
    int i,
    int m, 
    int n,
    float* fout
) except +

cdef extern void scan_and_multiply_c(
    const float* fprobe,
    int i_probe,
    int m_probe,
    int n_probe,
    const float* fobj,
    int i_obj,
    int m_obj,
    int n_obj,
    const int* addr_info,
    int addr_len,
    int batch_size,
    int m, 
    int n,
    float* fout
) except +

cdef extern void sqrt_abs_c(
    const float* indata, 
    float* outdata, 
    int m, 
    int n
) except +

cdef extern void log_likelihood_c(
    const float* probe_obj,
    const unsigned char* mask,
    const float* Idata,
    const float* prefilter,
    const float* postfilter,
    const int* addr_info,
    float* out,
    int i,
    int m,
    int n,
    int addr_i,
    int Idata_i
) except +

cdef extern void farfield_propagator_c(
    const float* indata, 
    const float* prefilter,
    const float* postfilter,
    float* out,
    int b,
    int m,
    int n,
    int isForward
) except +

cdef extern void sum_to_buffer_c(const float *in1, int in1_0, int in1_1,
                                int in1_2, float *out, int out_0, int out_1,
                                int out_2, const int *in_addr, int in_addr_0,
                                const int *out_addr, int out_addr_0,
                                int isComplex) except +
cdef extern void sum_to_buffer_stride_c(const float* in1, int in1_0, int in1_1, int in1_2,
        float* out, int out_0, int out_1, int out_2, const int* addr_info, int addr_info_0,
        int isComplex) except +

cdef extern void abs2_c(
    const float* indata,
    float* out,
    int n,
    int isComplex
) except +

cdef extern void abs2d_c(
    const double* indata,
    double* out,
    int n,
    int isComplex
) except +

cdef extern void far_field_error_c(
    const float* current,
    const float* measured,
    const unsigned char* mask,
    float* out,
    int i, int m, int n
) except +

cdef extern void realspace_error_c(
    const float* difference,
    const int* ea_first_column,
    const int* da_first_column,
    int addr_len,
    float* out,
    int i, int m, int n,
    int outlen
) except +

cdef extern void get_difference_c(
    const int* addr_info,
    float alpha,
    const float* fbackpropagated_solution,
    const float* err_fmag,
    const float* fexit_wave,
    float pbound,
    const float* fprobe_obj,
    float* fout,
    int i, int m, int n,
    int usePbound
) except +

cdef extern void renormalise_fourier_magnitudes_c(
    const float* f_f,
    const float* af,
    const float* fmag,
    const unsigned char* mask,
    const float* err_fmag,
    const int* addr_info,
    float pbound,
    float* f_out,
    int M, int N, int A, int B,
    int usePbound
) except +

cdef extern void difference_map_fourier_constraint_c(const unsigned char *mask,
                                                    const float *Idata,
                                                    const float *f_obj,
                                                    const float *f_probe,
                                                    float *f_exit_wave,
                                                    const int *addr_info,
                                                    const float *f_prefilter,
                                                    const float *f_postfilter,
                                                    float pbound,
                                                    float alpha,
                                                    int do_LL_error,
                                                    int do_realspace_error,
                                                    int doPbound,
                                                    int M,
                                                    int N,
                                                    int A,
                                                    int B,
                                                    int C,
                                                    int D,
                                                    int ob_modes,
                                                    int pr_modes,
                                                    float *errors
) except +

cdef extern void norm2_c(const float* data, float* out, int size, int isComplex) except +
cdef extern void mass_center_c(const float* data, int i, int m, int n, float* output) except +
cdef extern void clip_complex_magnitudes_to_range_c(float* f_data, int n, float clip_min, float clip_max) except +
cdef extern void extract_array_from_exit_wave_c(
    const float* f_exit_wave, 
    int A,
    int B,
    int C,
    const int* exit_addr,     
    const float* f_array_to_be_extracted, 
    int D,
    int E,
    int F,
    const int* extract_addr,       
    float* f_array_to_be_updated,  
    int G,
    int H,
    int I,
    const int* update_addr,  
    const float* weights,    
    const float* f_cfact           
) except +
cdef extern void interpolated_shift_c(const float* f_in, float* f_out, int items, int rows, int columns, float offsetRow, float offsetCol, int doLinear) except+
cdef extern int get_num_gpus_c() except +
cdef extern int get_gpu_compute_capability_c(int dev) except +
cdef extern void select_gpu_device_c(int dev) except +
cdef extern int get_gpu_memory_mb_c(int dev) except +
cdef extern string get_gpu_name_c(int dev) except +
cdef extern void reset_function_cache_c() except +
cdef extern void center_probe_c(float* f_probe, float center_tolerance, int i, int m, int n) except +
cdef extern float difference_map_update_probe_c(
    const float* f_obj,
    const float* probe_weights,        
    float* f_probe,
    const float* f_exit_wave,    
    const int* addr_info,        
    const float* f_cfact_probe,        
    const float* f_probe_support, 
    int A, int B, int C,
    int D, int E, int F,
    int G, int H, int I
) except +
cdef extern void difference_map_update_object_c(
    float* f_obj,                 
    const float* object_weights,  
    const float* f_probe,         
    const float* f_exit_wave,     
    const int* addr_info,         
    const float* f_cfact_object,  
    float ob_smooth_std,          
    float clip_min,               
    float clip_max,               
    int doSmoothing,              
    int doClipping,               
    int A, int B, int C, int D,
    int E, int F, int G, int H, int I) except +
cdef extern void difference_map_overlap_constraint_c(
    const int* addr_info,         
    const float* f_cfact_object,  
    const float* f_cfact_probe,   
    const float* f_exit_wave,     
    float* f_obj,                 
    const float* object_weights,  
    float* f_probe,               
    const float* f_probe_support, 
    const float* probe_weights,   
    float obj_smooth_std,
    float clip_min,
    float clip_max,
    float probe_center_tol,
    float overlap_converge_factor,
    int max_iterations,
    int doUpdateObjectFirst,
    int doUpdateProbe,
    int doSmoothing,
    int doClipping,
    int doCentering,
    int A, int B, int C, int D,
    int E, int F, int G, int H, int I) except +


cdef extern void complex_gaussian_filter_c(const float* f_input,
                                          float* f_output,
                                          const float* mfs,
                                          int ndims,
                                          const int* shape) except +
cdef extern void difference_map_iterator_c(
    const float* diffraction,  
    float* f_obj,              
    const float* object_weights,
    const float* f_cfact_object,
    const unsigned char* mask,  
    float* f_probe,             
    const float* f_cfact_probe, 
    const float* f_probe_support,
    const float* probe_weights,  
    float* f_exit_wave,          
    const int* addr_info,        
    const float* f_pre_fft,      
    const float* f_post_fft,     
    float* errors,               
    float pbound,
    int overlap_max_iterations,
    int doUpdateObjectFirst,
    float obj_smooth_std,
    float overlap_converge_factor,
    float probe_center_tol,
    int probe_update_start,
    float alpha,
    float clip_min,
    float clip_max,
    int do_LL_error,
    int do_realspace_error,
    int num_iterations,
    int A,
    int B,
    int C,
    int D,
    int E,
    int F,
    int G,
    int H,
    int I,
    int N,
    int doSmoothing,
    int doClipping,
    int doCentering,
    int doPbound) except+
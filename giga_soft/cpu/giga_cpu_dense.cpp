/*!
 * (C) 2025 Airbus copyright all rights reserved
 * \author Roland Brochard (roland.brochard@airbus.com)
 * \date 15/01/2025
 *
 * Baseline CPU implementation of the GIGA API
 *
 */

#include "giga_cpu.h"
#include "utils.h"

template<GIGA_data_type i_GT, GIGA_data_type o_GT, GIGA_data_type k_GT>
GIGA_error _dense_impl(const GIGA_dense_t *params, const GIGA_tensor_t *in, GIGA_tensor_t *out)
{
    typedef typename GIGA_C_Type<i_GT>::CType i_T;
    typedef typename GIGA_C_Type<o_GT>::CType o_T;
    typedef typename GIGA_C_Type<k_GT>::CType k_T;

    typedef typename GIGA_Compute_Type<o_GT>::CType c_T;

    if(in->nb_dims > 2) RETURN_ERROR(GIGA_Inconsistent_Number_Of_Dimensions);

    int nb_batch = 1;
    if(in->nb_dims == 2)
    {
        if(in->dims[0] != out->dims[0]) RETURN_ERROR(GIGA_Inconsistent_Tensor_Sizes);

        nb_batch = in->dims[0];
    }

    const uint32_t nb_out_elts = out->dims[1];
    const uint32_t nb_in_elts = in->dims[1];

    const GIGA_tensor_t * __restrict__ kernel = params->kernel;
    if(kernel->nb_dims != 2) RETURN_ERROR(GIGA_Incorrect_Parameter);
    //For the kernel, the dimensions are always Co, Ci;

    //check tensors dimensions relative to the kernel
    if(kernel->dims[0] != nb_out_elts)  RETURN_ERROR(GIGA_Inconsistent_Tensor_Sizes);
    if(kernel->dims[1] != nb_in_elts)   RETURN_ERROR(GIGA_Inconsistent_Tensor_Sizes);

    if(params->bias != NULL)
    {
        if(!check_tensor_exists(params->bias)) RETURN_ERROR(GIGA_Incorrect_Parameter);
        if(kernel->type != params->bias->type) RETURN_ERROR(GIGA_Incorrect_Parameter);
        if(params->bias->nb_dims != 1) RETURN_ERROR(GIGA_Incorrect_Parameter);
        if(params->bias->dims[0] != nb_out_elts) RETURN_ERROR(GIGA_Inconsistent_Tensor_Sizes);
    }

    const int out_shift = int(out->fp_shift) - (int(in->fp_shift) + int(params->kernel->fp_shift));
    const int bias_reshift = params->bias ? -int(params->bias->fp_shift) + (int(in->fp_shift) + int(params->kernel->fp_shift)) : 0;

    const uint32_t batch_end = nb_batch;
    const k_T * const bias_ptr = params->bias ? get_cptr<k_T>(params->bias) : nullptr;

    const uint32_t out_stride0 = out->strides[0] / sizeof(o_T);
    const uint32_t out_stride1 = out->strides[1] / sizeof(o_T);

    const uint32_t in_stride0 = in->strides[0] / sizeof(i_T);
    const uint32_t in_stride1 = in->strides[1] / sizeof(i_T);

    const uint32_t kernel_stride0 = kernel->strides[0] / sizeof(k_T);
    const uint32_t kernel_stride1 = kernel->strides[1] / sizeof(k_T);

#ifdef ENABLE_OPTIMIZATION
    // Assume kernel_stride1 == 1
    // Assume out_stride1 == 1
    // Assume in_stride1 == 1
    //Not fancy at all matrix multiplication algorithm
#pragma omp parallel
    for(uint32_t batch = 0; batch < batch_end ; ++batch)
    {
        o_T * const out_ptr0 = get_ptr<o_T>(out) + batch * out_stride0;
        const i_T * const in_ptr0 = get_cptr<i_T>(in) + batch * in_stride0;
#pragma omp for
        for(uint32_t out_i = 0 ; out_i < nb_out_elts ; ++out_i)
        {
            c_T acc = c_T(0);
            if(bias_ptr)
                acc = shift(bias_ptr[out_i], bias_reshift);

            o_T * const out_ptr1 = out_ptr0 + out_i;

            const k_T * k_ptr = get_cptr<k_T>(kernel) + out_i * kernel_stride0;
            const i_T * in_ptr1 = in_ptr0;

            for(uint32_t in_i = 0; in_i < nb_in_elts; ++in_i, ++k_ptr, ++in_ptr1)
                acc += c_T(*k_ptr) * c_T(*in_ptr1);

            if(params->b_ReLU)
                *out_ptr1 = acc > 0 ? o_T(shift(acc, out_shift)) : o_T(0);
            else
                *out_ptr1 = o_T(shift(acc, out_shift));
        }
    }
#else
    //Not fancy at all matrix multiplication algorithm
    for(uint32_t batch = 0; batch < batch_end ; ++batch)
    {
        for(uint32_t out_i = 0 ; out_i < nb_out_elts ; ++out_i)
        {
            c_T acc = c_T(0);
            if(bias_ptr)
                acc = shift(bias_ptr[out_i], bias_reshift);

            const uint32_t out_offset = batch * out_stride0 + out_i * out_stride1;
            o_T * const out_ptr = get_ptr<o_T>(out) + out_offset;

            *out_ptr = 0;
            for(uint32_t in_i = 0; in_i < nb_in_elts; ++in_i)
            {
                const uint32_t in_offset = batch * in_stride0 + in_i * in_stride1;
                const i_T * const in_ptr = get_cptr<i_T>(in) + in_offset;

                const uint32_t k_offset = out_i * kernel_stride0 + in_i * kernel_stride1;
                const k_T * const k_ptr = get_cptr<k_T>(kernel) + k_offset;

                acc += c_T(*k_ptr) * c_T(*in_ptr);
            }

            if(params->b_ReLU)
                *out_ptr = acc > 0 ? o_T(shift(acc, out_shift)) : o_T(0);
            else
                *out_ptr = o_T(shift(acc, out_shift));
        }
    }
#endif

    return GIGA_Success;
}

GIGA_error giga_dense_(const GIGA_dense_t *params, const GIGA_tensor_t *in, GIGA_tensor_t *out, const char *file, int line)
{
    if (!check_tensor_exists(in) || !check_tensor_exists(out) || !check_tensor_exists(params->kernel))
        RETURN_ERROR(GIGA_Unknown_tensor);

    GIGA_error ret;
#ifdef ENABLE_OPTIMIZATION
    GIGA_CALL_TEMPLATED_FUNC_ON_3_TENSORS_SIGNED_KERNELS(_dense_impl, in->type, out->type, params->kernel->type, params, in, out)
#else
    GIGA_CALL_TEMPLATED_FUNC_ON_3_TENSORS(_dense_impl, in->type, out->type, params->kernel->type, params, in, out)
#endif
    RETURN_ERROR(ret);
}

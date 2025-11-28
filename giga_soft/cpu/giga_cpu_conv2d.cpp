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

/*Compilation options to define the operational domain of the implementation*/
#define MAX_CONV_STRIDE 2
#define MAX_DILATION 1
#define KERNEL_SIZE 3

template<GIGA_data_type i_GT, GIGA_data_type o_GT, GIGA_data_type k_GT>
GIGA_error _conv2d_impl(const GIGA_conv2d_t *params, const GIGA_tensor_t *in, GIGA_tensor_t *out)
{
    typedef typename GIGA_C_Type<i_GT>::CType i_T;
    typedef typename GIGA_C_Type<o_GT>::CType o_T;
    typedef typename GIGA_C_Type<k_GT>::CType k_T;

    typedef typename GIGA_Compute_Type<o_GT>::CType c_T;

    if(in->nb_dims != out->nb_dims) RETURN_ERROR(GIGA_Inconsistent_Tensor_Sizes);
    if(in->nb_dims < 2)             RETURN_ERROR(GIGA_Inconsistent_Number_Of_Dimensions);
    uint32_t nb_batch = 1;

    uint32_t batch_stride_in = 0;
    uint32_t batch_stride_out = 0;
    if(in->nb_dims == 4)
    {
        //batch dimension
        if(in->dims[0] != out->dims[0])
            RETURN_ERROR(GIGA_Inconsistent_Tensor_Sizes);

        nb_batch = in->dims[0];
        batch_stride_in = in->strides[0];
        batch_stride_out = out->strides[0];
    }

    const uint32_t nb_out_channels = out->nb_dims == 2 ? 1 : out->dims[out->nb_dims - 3];
    const uint32_t nb_in_channels = in->nb_dims == 2 ? 1 : in->dims[in->nb_dims - 3];

    const GIGA_tensor_t * __restrict__ kernel = params->kernel;
    if(kernel->nb_dims != 4)
        RETURN_ERROR(GIGA_Incorrect_Parameter);
    //For the kernel, the dimensions are always Co, Ci, H, W;

    //check tensor dimensions relative to the kernel
    if(kernel->dims[0] != nb_out_channels)  RETURN_ERROR(GIGA_Inconsistent_Tensor_Sizes);
    if(kernel->dims[1] != nb_in_channels)   RETURN_ERROR(GIGA_Inconsistent_Tensor_Sizes);
    if(kernel->dims[2] != KERNEL_SIZE)      RETURN_ERROR(GIGA_Inconsistent_Tensor_Sizes);
    if(kernel->dims[3] != KERNEL_SIZE)      RETURN_ERROR(GIGA_Inconsistent_Tensor_Sizes);

    if(params->stride[0] > 2 || params->stride[0] < 1) RETURN_ERROR(GIGA_Incorrect_Parameter);
    if(params->stride[1] > 2 || params->stride[1] < 1) RETURN_ERROR(GIGA_Incorrect_Parameter);

    if(params->dilation[0] != 1 || params->dilation[1] != 1)    RETURN_ERROR(GIGA_Incorrect_Parameter);

    uint32_t bias_dimension = 0;

    if(params->bias != NULL)
    {
        bias_dimension = params->bias->nb_dims - 1;
        if(!check_tensor_exists(params->bias)) RETURN_ERROR(GIGA_Incorrect_Parameter);

        if(kernel->type != params->bias->type) RETURN_ERROR(GIGA_Incorrect_Parameter);

        if(!(params->bias->nb_dims == 1 || (params->bias->nb_dims == 2 && params->bias->dims[0] == 1)))
            RETURN_ERROR(GIGA_Incorrect_Parameter);

        if(params->bias->dims[bias_dimension] != nb_out_channels) RETURN_ERROR(GIGA_Inconsistent_Tensor_Sizes);
    }

    //Check H,W dimensions depending on the padding parameter
    //The width dimension always immediately follows the height dimension
    const uint32_t H_dim_in = in->nb_dims - 2;
    const uint32_t W_dim_in = H_dim_in + 1;

    const uint32_t H_dim_out = out->nb_dims - 2;
    const uint32_t W_dim_out = H_dim_out + 1;

    //check dimensions
    if(out->dims[H_dim_out] != (in->dims[H_dim_in] + params->padding[0][0] + params->padding[0][1] - (KERNEL_SIZE - 1) - 1) / params->stride[0] + 1 )
        RETURN_ERROR(GIGA_Inconsistent_Tensor_Sizes);
    if(out->dims[W_dim_out] != (in->dims[W_dim_in] + params->padding[1][0] + params->padding[1][1] - (KERNEL_SIZE - 1) - 1) / params->stride[1] + 1 )
        RETURN_ERROR(GIGA_Inconsistent_Tensor_Sizes);

    const uint32_t in_stride_C = (in->nb_dims == 2) ? 1 : in->strides[in->nb_dims - 3] / sizeof(i_T);
    const uint32_t out_stride_C = (out->nb_dims == 2) ? 1 : out->strides[out->nb_dims - 3] / sizeof(o_T);

    const int out_shift = int(out->fp_shift) - (int(in->fp_shift) + int(params->kernel->fp_shift)) ;
    const int bias_reshift = (params->bias != nullptr) ? -int(params->bias->fp_shift) + (int(in->fp_shift) + int(params->kernel->fp_shift)) : 0;

    const uint32_t bias_stride = params->bias ? params->bias->strides[bias_dimension] / sizeof(k_T) : 0;

    const uint32_t out_y_end = out->dims[H_dim_out];
    const uint32_t out_x_end = out->dims[W_dim_out];

    const uint32_t out_stride_B = batch_stride_out / sizeof(o_T);
    const uint32_t out_stride_H = out->strides[H_dim_out] / sizeof(o_T);
    const uint32_t out_stride_W = out->strides[W_dim_out] / sizeof(o_T);

    const uint32_t in_stride_B = batch_stride_in / sizeof(i_T);
    const uint32_t in_stride_H = in->strides[H_dim_in] / sizeof(i_T);
    const uint32_t in_stride_W = in->strides[W_dim_in] / sizeof(i_T);

    const uint32_t kernel_stride0 = kernel->strides[0] / sizeof(k_T);
    const uint32_t kernel_stride1 = kernel->strides[1] / sizeof(k_T);
    const uint32_t kernel_stride2 = kernel->strides[2] / sizeof(k_T);
    const uint32_t kernel_stride3 = kernel->strides[3] / sizeof(k_T);

    const uint32_t stride0 = params->stride[0];
    const uint32_t stride1 = params->stride[1];

    const uint32_t H = in->dims[H_dim_in];
    const uint32_t W = in->dims[W_dim_in];

    //Actually perform convolution
    const uint32_t batch_end = nb_batch;

#ifdef ENABLE_OPTIMIZATION
    const int32_t padding_y = params->padding[0][0];
    const int32_t padding_x = params->padding[1][0];

    // Assume out_stride_W == 1
    // Assume in_stride_W == 1
    // Assume kernel_stride3 == 1
#pragma omp parallel
    for (uint32_t batch = 0 ; batch < batch_end ; ++batch)
    {
        const i_T * const in_ptr0 = get_cptr<i_T>(in) + batch * in_stride_B;
        const k_T * bias_ptr = params->bias ? get_cptr<k_T>(params->bias) : nullptr;
        o_T * const out_ptr0 = get_ptr<o_T>(out) + batch * out_stride_B;
        for (uint32_t out_ch = 0; out_ch < nb_out_channels ; ++out_ch)
        {
            const k_T * const k_ptr0 = get_cptr<k_T>(kernel) + out_ch * kernel_stride0;
            o_T * const out_ptr1 = out_ptr0 + out_ch * out_stride_C;
            c_T bias = c_T(0);
            if(bias_ptr)
            {
                bias = *bias_ptr;
                bias_ptr += bias_stride;
            }

#pragma omp for
            for (uint32_t out_y = 0 ; out_y < out_y_end ; ++out_y)
            {
                o_T * const out_ptr2 = out_ptr1 + out_y * out_stride_H;
                const uint32_t in_y_offset0 = out_y * stride0 - padding_y;
                for (uint32_t out_x = 0 ; out_x < out_x_end ; ++out_x)
                {
                    const int32_t in_x_offset0 = out_x * stride1 - padding_x;
                    const i_T * const in_ptr1 = in_ptr0 + in_x_offset0;

                    o_T * const out_ptr3 = out_ptr2 + out_x;
                    c_T acc = 0;

                    const k_T * k_ptr = k_ptr0;
                    for (uint32_t c_in = 0 ; c_in < nb_in_channels; ++c_in)
                    {
                        const i_T * const in_ptr2 = in_ptr1 + c_in * in_stride_C;
                        for(uint32_t ker_y = 0; ker_y < KERNEL_SIZE ; ++ker_y)
                        {
                            const uint32_t in_y_offset1 = in_y_offset0 + ker_y;
                            if (in_y_offset1 >= H)
                            {
                                k_ptr += kernel_stride2;
                                continue;
                            }
                            const i_T * in_ptr3 = in_ptr2 + in_y_offset1 * in_stride_H;
                            for(uint32_t ker_x = 0 ; ker_x < KERNEL_SIZE ; ++ker_x, ++k_ptr, ++in_ptr3)
                            {
                                /*Boundary checking */
                                const uint32_t in_x_offset1 = in_x_offset0 + ker_x;
                                if(in_x_offset1 >= W)
                                    continue;

                                acc += c_T(*k_ptr) * c_T(*in_ptr3);
                            }
                        }
                    }

                    acc += shift(bias, bias_reshift);
                    if(params->b_ReLU)
                        *out_ptr3 = acc > 0 ? o_T(shift(acc, out_shift)) : o_T(0);
                    else
                        *out_ptr3 = o_T(shift(acc, out_shift));
                }
            }
        }
    }

#else       // Reference implementation
    for (uint32_t batch = 0 ; batch < batch_end ; ++batch)
    {
        for (uint32_t out_ch = 0; out_ch < nb_out_channels ; ++out_ch)
        {
            c_T bias = c_T(0);
            if(params->bias != NULL)
            {
                const k_T * const bias_ptr = get_cptr<k_T>(params->bias) + out_ch * bias_stride;
                bias = *bias_ptr;
            }

            for (uint32_t out_y = 0 ; out_y < out_y_end ; ++out_y)
            {
                for (uint32_t out_x = 0 ; out_x < out_x_end ; ++out_x)
                {
                    const uint32_t out_offset = batch * out_stride_B + out_ch * out_stride_C + out_y * out_stride_H + out_x * out_stride_W;
                    o_T * const out_ptr = get_ptr<o_T>(out) + out_offset;
                    c_T acc = 0;

                    for(uint32_t ker_y = 0; ker_y < KERNEL_SIZE ; ++ker_y)
                    {
                        const uint32_t in_y_offset = out_y * stride0 - params->padding[0][0] + ker_y;
                        if (in_y_offset >= H)
                            continue;
                        for(uint32_t ker_x = 0 ; ker_x < KERNEL_SIZE ; ++ker_x)
                        {
                            /*Boundary checking */
                            const uint32_t in_x_offset = out_x * stride1 - params->padding[1][0] + ker_x;
                            if(in_x_offset >= W)
                                continue;
                            for (uint32_t c_in = 0 ; c_in < nb_in_channels; ++c_in)
                            {

                                const uint32_t in_offset = batch * in_stride_B
                                                           + c_in * in_stride_C
                                                           + in_y_offset * in_stride_H
                                                           + in_x_offset * in_stride_W;

                                const i_T * const in_ptr = get_cptr<i_T>(in) + in_offset;

                                const uint32_t k_offset =  out_ch * kernel_stride0
                                                           + c_in * kernel_stride1
                                                           + ker_y * kernel_stride2
                                                           + ker_x * kernel_stride3;

                                const k_T * const k_ptr = get_cptr<k_T>(kernel) + k_offset;
                                acc += c_T(*k_ptr) * c_T(*in_ptr);
                            }
                        }
                    }

                    acc += shift(bias, bias_reshift);
                    if(params->b_ReLU)
                        *out_ptr = acc > 0 ? o_T(shift(acc, out_shift)) : o_T(0);
                    else
                        *out_ptr = o_T(shift(acc, out_shift));
                }
            }
        }
    }
#endif

    return GIGA_Success;
}

GIGA_error giga_conv2d_(const GIGA_conv2d_t *params, const GIGA_tensor_t *in, GIGA_tensor_t *out, const char *file, int line)
{
    if (!check_tensor_exists(in) || !check_tensor_exists(out) || !check_tensor_exists(params->kernel))
        RETURN_ERROR(GIGA_Unknown_tensor);

    GIGA_error ret;
#ifdef ENABLE_OPTIMIZATION
    GIGA_CALL_TEMPLATED_FUNC_ON_3_TENSORS_SIGNED_KERNELS(_conv2d_impl, in->type, out->type, params->kernel->type, params, in, out)
#else
    GIGA_CALL_TEMPLATED_FUNC_ON_3_TENSORS(_conv2d_impl, in->type, out->type, params->kernel->type, params, in, out)
#endif
    RETURN_ERROR(ret);
}

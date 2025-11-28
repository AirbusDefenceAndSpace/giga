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

template<GIGA_data_type i_GT>
GIGA_error _giga_upsample_impl(const GIGA_upsample_t *params, const GIGA_tensor_t *in, GIGA_tensor_t *out)
{
    typedef typename GIGA_C_Type<i_GT>::CType i_T;

    if(params->factor != 2)         RETURN_ERROR(GIGA_Incorrect_Parameter);
    if(in->nb_dims != out->nb_dims) RETURN_ERROR(GIGA_Inconsistent_Number_Of_Dimensions);

    if(in->nb_dims < 2) RETURN_ERROR(GIGA_Inconsistent_Number_Of_Dimensions);
    uint32_t nb_batch = 1;

    uint32_t batch_stride_in = 0;
    uint32_t batch_stride_out = 0;
    if(in->nb_dims == 4)
    {
        //batch dimension
        if(in->dims[0] != out->dims[0])
            RETURN_ERROR(GIGA_Inconsistent_Tensor_Sizes);

        //channels dimension
        if(in->dims[1] != out->dims[1])
            RETURN_ERROR(GIGA_Inconsistent_Tensor_Sizes);

        nb_batch = in->dims[0];
        batch_stride_in = in->strides[0];
        batch_stride_out = out->strides[0];
    }

    if(in->nb_dims == 3)
    {
        //channels dimension
        if(in->dims[0] != out->dims[0])
            RETURN_ERROR(GIGA_Inconsistent_Tensor_Sizes);
    }

    const uint32_t nb_channels = out->nb_dims == 2 ? 1 : out->dims[out->nb_dims - 3];
    const uint32_t chan_stride_in = in->nb_dims == 2 ? 1 : in->strides[in->nb_dims - 3];
    const uint32_t chan_stride_out = out->nb_dims == 2 ? 1 : out->strides[out->nb_dims - 3];

    //Check H,W dimensions depending on the padding parameter
    //The width dimension always immediately follows the height dimension
    const uint32_t H_dim = in->nb_dims - 2;
    const uint32_t W_dim = H_dim + 1;

    const uint32_t out_y_end = out->dims[H_dim];
    const uint32_t out_x_end = out->dims[W_dim];

    const uint32_t out_stride_B = batch_stride_out / sizeof(i_T);
    const uint32_t out_stride_C = chan_stride_out / sizeof(i_T);
    const uint32_t out_stride_H = out->strides[H_dim] / sizeof(i_T);
    const uint32_t out_stride_W = out->strides[W_dim] / sizeof(i_T);

    const uint32_t in_stride_B = batch_stride_in / sizeof(i_T);
    const uint32_t in_stride_C = chan_stride_in / sizeof(i_T);
    const uint32_t in_stride_H = in->strides[H_dim] / sizeof(i_T);
    const uint32_t in_stride_W = in->strides[W_dim] / sizeof(i_T);

    //check dimensions
    if(out->dims[H_dim] != in->dims[H_dim] * 2) RETURN_ERROR(GIGA_Inconsistent_Tensor_Sizes);
    if(out->dims[W_dim] != in->dims[W_dim] * 2) RETURN_ERROR(GIGA_Inconsistent_Tensor_Sizes);

    //Actually perform upsampling
    const uint32_t batch_end = nb_batch;
#ifdef ENABLE_OPTIMIZATION
    const uint32_t in_y_end = in->dims[H_dim];
    const uint32_t in_x_end = in->dims[W_dim];
    const uint32_t out_stride_H2 = out_stride_H * 2;

    // Assume out_stride_W == 1
    // Assume in_stride_W == 1
#pragma omp parallel
    for (uint32_t batch = 0 ; batch < batch_end ; ++batch)
    {
        i_T * const out_ptr0 = get_ptr<i_T>(out) + batch * out_stride_B;
        const i_T * const in_ptr0 = get_cptr<i_T>(in) + batch * in_stride_B;
        for (uint32_t channel = 0; channel < nb_channels; ++channel)
        {
            i_T * const out_ptr1 = out_ptr0 + channel * out_stride_C;
            const i_T * const in_ptr1 = in_ptr0 + channel * in_stride_C;
#pragma omp for
            for (uint32_t in_y = 0 ; in_y < in_y_end ; ++in_y)
            {
                i_T * out0_ptr2 = out_ptr1 + in_y * out_stride_H2;
                i_T * out1_ptr2 = out0_ptr2 + out_stride_H;
                const i_T * in_ptr2 = in_ptr1 + in_y * in_stride_H;
                for (uint32_t in_x = 0 ; in_x < in_x_end ; ++in_x, out0_ptr2 += 2, out1_ptr2 += 2, ++in_ptr2)
                {
                    const i_T v = *in_ptr2;
                    out0_ptr2[0] = v;
                    out0_ptr2[1] = v;
                    out1_ptr2[0] = v;
                    out1_ptr2[1] = v;
                }
            }
        }
    }
#else
    for (uint32_t batch = 0 ; batch < batch_end ; ++batch)
    {
        for (uint32_t channel = 0; channel < nb_channels; ++channel)
        {
            for (uint32_t out_y = 0 ; out_y < out_y_end ; ++out_y)
            {
                for (uint32_t out_x = 0 ; out_x < out_x_end ; ++out_x)
                {
                    const uint32_t out_offset = batch * out_stride_B + channel * out_stride_C + out_y * out_stride_H + out_x * out_stride_W;

                    i_T * const out_ptr = get_ptr<i_T>(out) + out_offset;

                    const uint32_t in_offset = batch * in_stride_B
                                               + channel * in_stride_C
                                               + (out_y / 2) * in_stride_H
                                               + (out_x / 2) * in_stride_W;

                    const i_T * const in_ptr = get_cptr<i_T>(in) + in_offset;

                    *out_ptr = i_T(*in_ptr);
                }
            }
        }
    }
#endif
    return GIGA_Success;
}

GIGA_error giga_upsample_(const GIGA_upsample_t *params, const GIGA_tensor_t *in, GIGA_tensor_t *out, const char *file, int line)
{
    if (!check_tensor_exists(in) || !check_tensor_exists(out))
        RETURN_ERROR(GIGA_Unknown_tensor);

    GIGA_error ret;
    GIGA_CALL_TEMPLATED_FUNC_ON_TENSOR(_giga_upsample_impl, in->type, params, in, out)

    RETURN_ERROR(ret);
}

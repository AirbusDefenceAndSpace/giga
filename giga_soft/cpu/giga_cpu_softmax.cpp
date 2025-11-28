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
#include <cmath>
#include <algorithm>
#include <vector>

template<GIGA_data_type i_GT, GIGA_data_type o_GT>
GIGA_error _softmax_impl(const GIGA_softmax_t * params, const GIGA_tensor_t *in, GIGA_tensor_t *out)
{
    typedef typename GIGA_C_Type<i_GT>::CType i_T;
    typedef typename GIGA_C_Type<o_GT>::CType o_T;

    typedef typename GIGA_Compute_Type<o_GT>::CType c_T;

    if(in->nb_dims != out->nb_dims)
        RETURN_ERROR(GIGA_Inconsistent_Number_Of_Dimensions);

    for(unsigned int i = 0; i < in->nb_dims; i++)
    {
        if(in->dims[i] != out->dims[i])
            RETURN_ERROR(GIGA_Inconsistent_Tensor_Sizes);
    }

    //Didn't find an elegant way without making a disjunction of cases
    if(in->nb_dims == 1)
    {
        //Find the maximum element to ensure numerical stability
        i_T max_value = *((i_T*)((char*)((Tensor_data_t*)in->data)->data_start));
        const i_T * in_ptr = get_cptr<i_T>(in);
        const uint32_t in_stride0 = in->strides[0] / sizeof(i_T);
        const uint32_t out_stride0 = out->strides[0] / sizeof(o_T);
        const uint32_t in_i1_end = (int)in->dims[1];
        for(uint32_t in_i1 = 0; in_i1 < in_i1_end; ++in_i1, in_ptr += in_stride0)
            max_value = std::max(max_value, *in_ptr);

        float sum = 0.f;
        std::vector<float> accs(in->dims[0]);
        const uint32_t in_i0_end = (int)in->dims[0];
        in_ptr = get_cptr<i_T>(in);
        for(uint32_t in_i0 = 0; in_i0 < in_i0_end ; ++in_i0, in_ptr += in_stride0)
        {
            const float value = std::exp(float(*in_ptr) - max_value);
            accs[in_i0] = value;
            sum += value;
        }

        const uint32_t out_i_end = out->dims[0];
        const float _sum = 1.f / sum;
        o_T *out_ptr = get_ptr<o_T>(out);
        for(uint32_t out_i = 0; out_i < out_i_end; ++out_i, out_ptr += out_stride0)
            *out_ptr = o_T(accs[out_i] * _sum);
    }
    else if(in->nb_dims == 2)
    {
        std::vector<float> accs(in->dims[1]);
        //The first dimension is the batch dimension. If it's not you're doing it wrong.
        const uint32_t batch_end = in->dims[0];
        const uint32_t in_i_end = in->dims[1];
        const uint32_t out_i_end = out->dims[1];
        const uint32_t in_stride0 = in->strides[0] / sizeof(i_T);
        const uint32_t in_stride1 = in->strides[1] / sizeof(i_T);
        const uint32_t out_stride0 = out->strides[0] / sizeof(o_T);
        const uint32_t out_stride1 = out->strides[1] / sizeof(o_T);
        const i_T * const in_ptr0 = get_cptr<i_T>(in);
        o_T * const out_ptr0 = get_ptr<o_T>(out);
        for(uint32_t batch = 0; batch < batch_end; ++batch)
        {
            const uint32_t in_offset0 = batch * in_stride0;
            const uint32_t out_offset0 = batch * out_stride0;
            //Find the maximum element to ensure numerical stability
            float max_value = in_ptr0[in_offset0];
            for(uint32_t in_i = 1; in_i < in_i_end ; ++in_i)
            {
                const uint32_t in_offset1 = in_offset0 + in_i * in_stride1;
                max_value = std::max<float>(max_value, in_ptr0[in_offset1]);
            }

            float sum = 0.f;
            for(uint32_t in_i = 0; in_i < in_i_end; ++in_i)
            {
                const uint32_t in_offset1 = in_offset0 + in_i * in_stride1;
                const float value = std::exp(float(in_ptr0[in_offset1]) - max_value);
                accs[in_i] = value;
                sum += value;
            }

            const float _sum = 1.f / sum;
            for(uint32_t out_i = 0; out_i < out_i_end; ++out_i)
            {
                const uint32_t out_offset1 = out_offset0 + out_i * out_stride1;
                out_ptr0[out_offset1] = o_T(accs[out_i] * _sum);
            }
        }
    }
    else // nb_dims >= 3
    {
        //Typically the softmax will happen along the channels dimension for an image
        const uint32_t nb_elements = in->nb_dims == 3 ? in->dims[2] : in->dims[3]*in->dims[2];

        std::vector<float> accs(in->dims[1]);
        const uint32_t batch_end = in->dims[0];
        const uint32_t in_i_end = in->dims[1];
        const uint32_t out_i_end = out->dims[1];
        const uint32_t in_stride0 = in->strides[0] / sizeof(i_T);
        const uint32_t in_stride1 = in->strides[1] / sizeof(i_T);
        const uint32_t in_strideL = in->strides[in->nb_dims-1] / sizeof(i_T);
        const uint32_t out_stride0 = out->strides[0] / sizeof(o_T);
        const uint32_t out_stride1 = out->strides[1] / sizeof(o_T);
        const uint32_t out_strideL = out->strides[out->nb_dims-1] / sizeof(o_T);
        const i_T * const in_ptr0 = get_cptr<i_T>(in);
        o_T * const out_ptr0 = get_ptr<o_T>(out);
        for(int32_t batch = 0; batch < batch_end; ++batch)
        {
            const uint32_t in_offset0 = batch * in_stride0;
            const uint32_t out_offset0 = batch * out_stride0;
            for(uint32_t elt_i = 0; elt_i < nb_elements; ++elt_i)
            {
                const uint32_t in_offset1 = in_offset0 + elt_i * in_strideL;
                const uint32_t out_offset1 = out_offset0 + elt_i * out_strideL;
                //Find the maximum element to ensure numerical stability
                float max_value = in_ptr0[in_offset1];
                for(uint32_t in_i = 1 ; in_i < in_i_end; ++in_i)
                {
                    const uint32_t in_offset2 = in_offset1 + in_i * in_stride1;
                    max_value = std::max<float>(max_value, in_ptr0[in_offset2]);
                }

                float sum = 0.f;
                for(uint32_t in_i = 0; in_i < in_i_end; ++in_i)
                {
                    const uint32_t in_offset2 = in_offset1 + in_i * in_stride1;
                    const float value = std::exp(float(in_ptr0[in_offset2]) - max_value);
                    accs[in_i] = value;
                    sum += value;
                }

                //Divide by the sum to get the final softmax
                const float _sum = 1.f / sum;
                for(uint32_t out_i = 0; out_i < out_i_end; ++out_i)
                {
                    const uint32_t out_offset2 = out_offset1 + out_i * out_stride1;
                    out_ptr0[out_offset2] = o_T(accs[out_i] * _sum);
                }
            }
        }
    }

    return GIGA_Success;
}

GIGA_error giga_softmax_(const GIGA_softmax_t * params, const GIGA_tensor_t *in, GIGA_tensor_t *out, const char *file, int line)
{
    if (!check_tensor_exists(in) || !check_tensor_exists(out))
        RETURN_ERROR(GIGA_Unknown_tensor);

    GIGA_error ret;
#ifdef ENABLE_OPTIMIZATION
    GIGA_CALL_TEMPLATED_FUNC_ON_2_TENSORS_SAME_TYPE(_softmax_impl, in->type, out->type, params, in, out)
#else
    GIGA_CALL_TEMPLATED_FUNC_ON_2_TENSORS(_softmax_impl, in->type, out->type, params, in, out)
#endif
    RETURN_ERROR(ret);
}

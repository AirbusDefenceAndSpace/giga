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

template<GIGA_data_type a_GT, GIGA_data_type b_GT,  GIGA_data_type o_GT>
GIGA_error _add_impl(const GIGA_add_t * params, const GIGA_tensor_t *a, const GIGA_tensor_t *b, GIGA_tensor_t *out)
{
    typedef typename GIGA_C_Type<a_GT>::CType a_T;
    typedef typename GIGA_C_Type<b_GT>::CType b_T;
    typedef typename GIGA_C_Type<o_GT>::CType o_T;

    typedef typename GIGA_Compute_Type<o_GT>::CType c_T;

    //Computation of shifts in the fixed-point case. To simplify, before adding, the values are transformed to have the representation of the output
    const int ashift = int(out->fp_shift) - int(a->fp_shift);
    const int bshift = int(out->fp_shift) - int(b->fp_shift);

    if(a->nb_dims != b->nb_dims || a->nb_dims != out->nb_dims)
        RETURN_ERROR(GIGA_Inconsistent_Tensor_Sizes);

    unsigned int nb_elements = 1;
    for(unsigned int i = 0; i < a->nb_dims; i++)
    {
        if(a->dims[i] != b->dims[i] || a->dims[i] != out->dims[i])
            RETURN_ERROR(GIGA_Inconsistent_Tensor_Sizes);

        nb_elements *= out->dims[i];
    }

    const int elt_i_end = nb_elements;
    o_T * out_ptr = get_ptr<o_T>(out);
    const a_T * a_ptr = get_cptr<a_T>(a);
    const b_T * b_ptr = get_cptr<b_T>(b);
    for(int elt_i = 0; elt_i < elt_i_end; ++elt_i, ++out_ptr, ++a_ptr, ++b_ptr)
    {
        *out_ptr = o_T(shift(c_T(*a_ptr), ashift) + shift(c_T(*b_ptr), bshift));
    }

    return GIGA_Success;
}

GIGA_error giga_add_(const GIGA_add_t * params, const GIGA_tensor_t *a, const GIGA_tensor_t *b, GIGA_tensor_t *out, const char *file, int line)
{
    if (!check_tensor_exists(a) || !check_tensor_exists(b) || !check_tensor_exists(out))
        RETURN_ERROR(GIGA_Unknown_tensor);

    GIGA_error ret;
#ifdef ENABLE_OPTIMIZATION
    GIGA_CALL_TEMPLATED_FUNC_ON_3_TENSORS_SAME_TYPE(_add_impl, a->type, b->type, out->type, params, a, b, out)
#else
    GIGA_CALL_TEMPLATED_FUNC_ON_3_TENSORS(_add_impl, a->type, b->type, out->type, params, a, b, out)
#endif
    RETURN_ERROR(ret);
}

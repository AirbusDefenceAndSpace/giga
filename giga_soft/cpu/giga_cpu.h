/*!
 * (C) 2025 Airbus copyright all rights reserved
 * \author Lucas Marti (lucas.marti@airbus.com)
 * \author Roland Brochard (roland.brochard@airbus.com)
 * \date 3/04/2023
 *
 * Baseline CPU implementation of the GIGA API
 *
 */

#ifndef GIGA_CPU_H_dc5903c6ada2890c9551b4dbdc3b203b
#define GIGA_CPU_H_dc5903c6ada2890c9551b4dbdc3b203b

#include <cstdint>
#include <giga/giga.h>
#include <giga/float16.h>
#include <stdexcept>

extern const bool giga_cpu_use_exceptions;

#define RETURN_ERROR(ERROR)\
    do\
    {\
        if ((ERROR) != GIGA_Success && giga_cpu_use_exceptions)\
            throw std::runtime_error(giga_str_error(ERROR));\
        return (ERROR);\
    } while(false)

//Structure to produce the underlying C type for a GIGA data type
template<GIGA_data_type giga_data_type> struct GIGA_C_Type  {};

#define MAP2CType(GIGA_Type, C_Type)\
template<> struct GIGA_C_Type<GIGA_Type>\
{\
    typedef C_Type CType;\
}

MAP2CType(GIGA_Float16, half);
MAP2CType(GIGA_Float32, float);
MAP2CType(GIGA_SFixed4, int8_t);
MAP2CType(GIGA_SFixed8, int8_t);
MAP2CType(GIGA_SFixed16, int16_t);
MAP2CType(GIGA_UFixed4, uint8_t);
MAP2CType(GIGA_UFixed8, uint8_t);
MAP2CType(GIGA_UFixed16, uint16_t);

#undef MAP2CType

bool is_float(GIGA_data_type type);
bool is_signed(GIGA_data_type type);

//Structure to produce the underlying C compute type for a GIGA data type
template<GIGA_data_type giga_data_type>
struct GIGA_Compute_Type
{
    typedef typename GIGA_C_Type<giga_data_type>::CType CType;
};

#define MAP2CType(GIGA_Type, C_Type)\
template<> struct GIGA_Compute_Type<GIGA_Type>\
{\
    typedef C_Type CType;\
}

MAP2CType(GIGA_Float16, float);
MAP2CType(GIGA_SFixed8, int_fast32_t);
MAP2CType(GIGA_UFixed8, int_fast32_t);
MAP2CType(GIGA_SFixed16, int_fast32_t);
MAP2CType(GIGA_UFixed16, int_fast32_t);

#undef MAP2CType

/* Internal structure to store internal tensor information and the pointer to the actual tensor data*/
struct Tensor_data_t
{
    bool is_allocated = false;
    bool is_mapped = false;
    uint64_t id = 0;
    uint64_t memory_zone_id = 0;
    void* data_ptr = nullptr;       // Pointer to the beginning of the buffer (parent buffer is any parent)
    void* data_start = nullptr;     // Pointer to the beginning of this tensor (data_start == data_ptr if no parent)
    uint64_t view_of = 0;
};

template<class T>
inline T * get_ptr(GIGA_tensor_t * const tensor)
{
    return (T*)((const Tensor_data_t*)tensor->data)->data_start;
}

template<class T>
inline const T * get_cptr(const GIGA_tensor_t * const tensor)
{
    return (const T*)((const Tensor_data_t*)tensor->data)->data_start;
}

bool check_tensor_exists(const GIGA_tensor_t *tensor);

size_t element_size_in_bits(GIGA_data_type data_type);

#endif // GIGA_CPU_H_dc5903c6ada2890c9551b4dbdc3b203b

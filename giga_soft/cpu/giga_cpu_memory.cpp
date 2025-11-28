/*!
/*!
 * (C) 2025 Airbus copyright all rights reserved
 * \author Lucas Marti (lucas.marti@airbus.com)
 * \author Roland Brochard (roland.brochard@airbus.com)
 * \date 15/01/2025
 *
 * Baseline CPU implementation of the GIGA API
 *
 */

#include "giga_cpu.h"

#include <new>
#include <vector>
#include <iostream>
#include <sstream>
#include <unistd.h>
#include <cstring>
#ifndef GIGA_CPU_MEMORY
#include <cstdlib>
#endif
#include "utils.h"

class ignore
{
public:
    template<class T>
    const ignore &operator<<(const T &) const { return *this; }
};

class MemoryPool
{
public :
    MemoryPool() :
        nb_tensors(0) {}

    size_t size() const {   return m_data.size();   }

    uint8_t *ptr()      {   return m_data.data();   }

public:
    uint64_t nb_tensors;
    std::vector<uint8_t> m_data;
};

namespace
{
    static uint64_t current_tensor_id = 1;
    static std::vector<MemoryPool> * s_memory_zones = nullptr;

    std::vector<MemoryPool> &GetMemoryZoneCollection()
    {
        if (s_memory_zones == nullptr)
        {
            s_memory_zones = new std::vector<MemoryPool>();

            // Expected format is a list of ';' separated sizes expressed in bytes, KB (K suffix), MB (M suffix) or GB (G suffix)
            const char *_GIGA_CPU_MEMORY = nullptr;
            // Allow overriding this with a define
#ifdef GIGA_CPU_MEMORY
            _GIGA_CPU_MEMORY = GIGA_CPU_MEMORY;
#else
            _GIGA_CPU_MEMORY = getenv("GIGA_CPU_MEMORY");
#endif
            if (!_GIGA_CPU_MEMORY)
                _GIGA_CPU_MEMORY = "128M";      // Default is a single pool of 128MB

            int nb_zones = 1;
            for(const char *ptr = _GIGA_CPU_MEMORY ; *ptr ; ++ptr)
                nb_zones += *ptr == ';';
            s_memory_zones->reserve(nb_zones);

            size_t zone_size = 0;
            for(const char *ptr = _GIGA_CPU_MEMORY ; *ptr ; ++ptr)
            {
                if (*ptr == ';')
                {
                    s_memory_zones->emplace_back();
                    s_memory_zones->back().m_data.resize(zone_size);
                    zone_size = 0;
                }
                else if (*ptr == 'G')
                    zone_size *= 1024 * 1024 * 1024;
                else if (*ptr == 'M')
                    zone_size *= 1024 * 1024;
                else if (*ptr == 'K')
                    zone_size *= 1024;
                else if (*ptr >= '0' && *ptr <= '9')
                    zone_size = zone_size * 10 + (*ptr - '0');
            }
            if (zone_size > 0 || s_memory_zones->size() < nb_zones)
            {
                s_memory_zones->emplace_back();
                s_memory_zones->back().m_data.resize(zone_size);
            }

            std::stringstream out;
            out << s_memory_zones->size() << " memory pools:" << std::endl;
            for(const auto &pool : *s_memory_zones)
                out << "    " << pool.size() << " bytes" << std::endl;
            out << std::endl;
            const std::string &buf = out.str();
            ignore() << write(0, buf.data(), buf.size());
        }
        return *s_memory_zones;
    }

    void on_init() __attribute__((constructor));

    void on_init()
    {
        // Make sure this is allocated on load
        GetMemoryZoneCollection();
    }
}

GIGA_error giga_allocate_tensor_(GIGA_tensor_t *tensor, const GIGA_allocate_t *params, const char *file, int line)
{
    if(tensor->nb_dims > 4 || tensor->nb_dims < 1) return GIGA_Inconsistent_Number_Of_Dimensions;

    auto &memory_zones = GetMemoryZoneCollection();
    if (params->memory_zone_id >= memory_zones.size())
        RETURN_ERROR(GIGA_Out_Of_Device_Memory);

    MemoryPool &memory_pool = memory_zones[params->memory_zone_id];

    const size_t element_size = element_size_in_bits(tensor->type) / 8;

    // Row major
    tensor->strides[tensor->nb_dims - 1] = element_size;
    for(int32_t i = int32_t(tensor->nb_dims) - 2 ; i >= 0 ; --i)
        tensor->strides[i] = tensor->strides[i + 1] * tensor->dims[i + 1];

    try
    {
        tensor->data = new Tensor_data_t();
        Tensor_data_t * typed_data = (Tensor_data_t *)(tensor->data);
        const size_t buffer_size = tensor->strides[0] * tensor->dims[0];

        if (params->offset + buffer_size > memory_pool.size())
            RETURN_ERROR(GIGA_Out_Of_Device_Memory);

        typed_data->data_ptr = memory_pool.ptr();
        typed_data->data_start = (char*)typed_data->data_ptr + params->offset;
        typed_data->memory_zone_id = params->memory_zone_id;
        typed_data->is_allocated = true;
        typed_data->id = current_tensor_id++;

        memory_pool.nb_tensors++;
    }
    catch(const std::bad_alloc &e)
    {
        std::cerr << e.what() << std::endl;
        RETURN_ERROR(GIGA_Bad_Alloc);
    }

    return GIGA_Success;
}

GIGA_error giga_map_tensor_(GIGA_tensor_t *tensor, void **ptr, GIGA_memory_flag flags, const char *file, int line)
{
    if (!check_tensor_exists(tensor))
        RETURN_ERROR(GIGA_Unknown_tensor);

    if (flags != GIGA_Memory_Discard && flags != GIGA_Memory_Sync)
        RETURN_ERROR(GIGA_Incorrect_Parameter);

    Tensor_data_t * __restrict__ data_ptr = ((Tensor_data_t*)tensor->data);
    *ptr = (void*)data_ptr->data_start;
    data_ptr->is_mapped = true;
    return GIGA_Success;
}

GIGA_error giga_unmap_tensor_(GIGA_tensor_t *tensor, void *ptr, GIGA_memory_flag flags, const char *file, int line)
{
    if (!check_tensor_exists(tensor))
        RETURN_ERROR(GIGA_Unknown_tensor);

    if (flags != GIGA_Memory_Discard && flags != GIGA_Memory_Sync)
        RETURN_ERROR(GIGA_Incorrect_Parameter);

    ((Tensor_data_t*)tensor->data)->is_mapped = false;
    return GIGA_Success;
}

GIGA_error giga_release_tensor_(GIGA_tensor_t *tensor, const char *file, int line)
{
    if (!check_tensor_exists(tensor))
        RETURN_ERROR(GIGA_Unknown_tensor);

    auto & memory_zones = GetMemoryZoneCollection();

    Tensor_data_t* data_ptr = ((Tensor_data_t*)tensor->data);
    if(data_ptr->is_allocated == false)
        RETURN_ERROR(GIGA_Unknown_tensor);

    if(data_ptr->view_of != 0)
    {
        delete data_ptr;
        data_ptr = nullptr;
        return GIGA_Success;
    }

    auto zone_id = data_ptr->memory_zone_id;
    if (zone_id >= memory_zones.size())
        RETURN_ERROR(GIGA_Unknown_tensor);

    if (memory_zones[zone_id].nb_tensors > 1)
    {
        --memory_zones[zone_id].nb_tensors;
        delete data_ptr;
        data_ptr = nullptr;
        return GIGA_Success;
    }

    delete data_ptr;
    data_ptr = nullptr;
    return GIGA_Success;
}

GIGA_error giga_reshape_(const GIGA_reshape_t * params, const GIGA_tensor_t *in, GIGA_tensor_t *out, const char *file, int line)
{
    if (!check_tensor_exists(in) || !check_tensor_exists(out))
        RETURN_ERROR(GIGA_Unknown_tensor);

    if(in->type != out->type)           RETURN_ERROR(GIGA_Inconsistent_Tensor_Types);
    if(in->fp_shift != out->fp_shift)   RETURN_ERROR(GIGA_Inconsistent_Tensor_Types);

    unsigned int total_size_in = 1U;
    for(unsigned int i = 0; i < in->nb_dims; i++)
        total_size_in *= in->dims[i];

    unsigned int total_size_out = 1U;
    for(unsigned int i = 0; i < out->nb_dims; i++)
        total_size_out *= out->dims[i];

    if(total_size_in != total_size_out)
        RETURN_ERROR(GIGA_Inconsistent_Tensor_Sizes);

    //TODO, be careful the out tensor has not been allocated
    ((Tensor_data_t*) out->data)->data_ptr = ((Tensor_data_t*) in->data)->data_ptr;
    ((Tensor_data_t*) out->data)->data_start = ((Tensor_data_t*) in->data)->data_start;
    return GIGA_Success;
}

GIGA_error giga_view_(const GIGA_view_t *params, const GIGA_tensor_t *in, GIGA_tensor_t *out, const char *file, int line)
{
    if (!check_tensor_exists(in))
        RETURN_ERROR(GIGA_Unknown_tensor);

    if (in->type != out->type)          RETURN_ERROR(GIGA_Inconsistent_Tensor_Types);
    if (in->fp_shift != out->fp_shift)  RETURN_ERROR(GIGA_Inconsistent_Tensor_Types);
    if (in->nb_dims != out->nb_dims)    RETURN_ERROR(GIGA_Inconsistent_Number_Of_Dimensions);

    Tensor_data_t * data_in = (Tensor_data_t*)in->data;
    out->data = new Tensor_data_t;
    Tensor_data_t * data_out = (Tensor_data_t*) out->data;
    data_out->id = current_tensor_id++;
    data_out->memory_zone_id = data_in->memory_zone_id;

    auto & memory_zones = GetMemoryZoneCollection();
    memory_zones[data_out->memory_zone_id].nb_tensors++;

    //TODO need more checks

    data_out->data_ptr = data_in->data_ptr;
    data_out->data_start = (uint8_t*)data_in->data_start;
    for(uint32_t dim = 0; dim < in->nb_dims; ++dim)
    {
        out->strides[dim] = in->strides[dim];
        data_out->data_start = (uint8_t*)data_out->data_start + params->offset[dim] * in->strides[dim];
    }

    data_out->is_allocated = true;
    data_out->view_of = data_in->id;

    return GIGA_Success;
}

// from float
template<class T>   inline T cast_to(float x, int32_t fp_shift, float f);

template<>  inline float    cast_to<   float>(float x, int32_t fp_shift, float f)   {   return x;       }
template<>  inline half     cast_to<    half>(float x, int32_t fp_shift, float f)   {   return x;       }
template<>  inline uint8_t  cast_to< uint8_t>(float x, int32_t fp_shift, float f)   {   return x * f;   }
template<>  inline uint16_t cast_to<uint16_t>(float x, int32_t fp_shift, float f)   {   return x * f;   }
template<>  inline int8_t   cast_to<  int8_t>(float x, int32_t fp_shift, float f)   {   return x * f;   }
template<>  inline int16_t  cast_to< int16_t>(float x, int32_t fp_shift, float f)   {   return x * f;   }

// from half
template<class T>   inline T cast_to(half x, int32_t fp_shift, float f);

template<>  inline float    cast_to<   float>(half x, int32_t fp_shift, float f)    {   return x;       }
template<>  inline half     cast_to<    half>(half x, int32_t fp_shift, float f)    {   return x;       }
template<>  inline uint8_t  cast_to< uint8_t>(half x, int32_t fp_shift, float f)    {   return x * f;   }
template<>  inline uint16_t cast_to<uint16_t>(half x, int32_t fp_shift, float f)    {   return x * f;   }
template<>  inline int8_t   cast_to<  int8_t>(half x, int32_t fp_shift, float f)    {   return x * f;   }
template<>  inline int16_t  cast_to< int16_t>(half x, int32_t fp_shift, float f)    {   return x * f;   }

// from uint8
template<class T>   inline T cast_to(uint8_t x, int32_t fp_shift, float f);

template<>  inline float    cast_to<   float>(uint8_t x, int32_t fp_shift, float f) {   return x * f;   }
template<>  inline half     cast_to<    half>(uint8_t x, int32_t fp_shift, float f) {   return x * f;   }
template<>  inline uint8_t  cast_to< uint8_t>(uint8_t x, int32_t fp_shift, float f) {   return shift<uint32_t>(x, fp_shift);    }
template<>  inline uint16_t cast_to<uint16_t>(uint8_t x, int32_t fp_shift, float f) {   return shift<uint32_t>(x, fp_shift);    }
template<>  inline int8_t   cast_to<  int8_t>(uint8_t x, int32_t fp_shift, float f) {   return shift< int32_t>(x, fp_shift);    }
template<>  inline int16_t  cast_to< int16_t>(uint8_t x, int32_t fp_shift, float f) {   return shift< int32_t>(x, fp_shift);    }

// from uint16
template<class T>   inline T cast_to(uint16_t x, int32_t fp_shift, float f);

template<>  inline float    cast_to<   float>(uint16_t x, int32_t fp_shift, float f)    {   return x * f;   }
template<>  inline half     cast_to<    half>(uint16_t x, int32_t fp_shift, float f)    {   return x * f;   }
template<>  inline uint8_t  cast_to< uint8_t>(uint16_t x, int32_t fp_shift, float f)    {   return shift<uint32_t>(x, fp_shift);    }
template<>  inline uint16_t cast_to<uint16_t>(uint16_t x, int32_t fp_shift, float f)    {   return shift<uint32_t>(x, fp_shift);    }
template<>  inline int8_t   cast_to<  int8_t>(uint16_t x, int32_t fp_shift, float f)    {   return shift<int32_t>(x, fp_shift);     }
template<>  inline int16_t  cast_to< int16_t>(uint16_t x, int32_t fp_shift, float f)    {   return shift<int32_t>(x, fp_shift);     }

// from int8
template<class T>   inline T cast_to(int8_t x, int32_t fp_shift, float f);

template<>  inline float    cast_to<   float>(int8_t x, int32_t fp_shift, float f)  {   return x * f;   }
template<>  inline half     cast_to<    half>(int8_t x, int32_t fp_shift, float f)  {   return x * f;   }
template<>  inline uint8_t  cast_to< uint8_t>(int8_t x, int32_t fp_shift, float f)  {   return shift<uint32_t>(x, fp_shift);    }
template<>  inline uint16_t cast_to<uint16_t>(int8_t x, int32_t fp_shift, float f)  {   return shift<uint32_t>(x, fp_shift);    }
template<>  inline int8_t   cast_to<  int8_t>(int8_t x, int32_t fp_shift, float f)  {   return shift<int32_t>(x, fp_shift);     }
template<>  inline int16_t  cast_to< int16_t>(int8_t x, int32_t fp_shift, float f)  {   return shift<int32_t>(x, fp_shift);     }

// from int16
template<class T>   inline T cast_to(int16_t x, int32_t fp_shift, float f);

template<>  inline float    cast_to<   float>(int16_t x, int32_t fp_shift, float f) {   return x * f;   }
template<>  inline half     cast_to<    half>(int16_t x, int32_t fp_shift, float f) {   return x * f;   }
template<>  inline uint8_t  cast_to< uint8_t>(int16_t x, int32_t fp_shift, float f) {   return shift<uint32_t>(x, fp_shift);    }
template<>  inline uint16_t cast_to<uint16_t>(int16_t x, int32_t fp_shift, float f) {   return shift<uint32_t>(x, fp_shift);    }
template<>  inline int8_t   cast_to<  int8_t>(int16_t x, int32_t fp_shift, float f) {   return shift<int32_t>(x, fp_shift);     }
template<>  inline int16_t  cast_to< int16_t>(int16_t x, int32_t fp_shift, float f) {   return shift<int32_t>(x, fp_shift);     }

GIGA_error giga_copy_to_tensor_(const void *user_ptr, GIGA_data_type source_type, uint32_t fp_shift, GIGA_tensor_t *tensor, const char *file, int line)
{
#define FALLTHROUGH

    GIGA_error error;
    uint32_t dims[4] = {1, 1, 1, 1};
    uint32_t strides[4] = {0, 0, 0, 0};
\
    switch(tensor->nb_dims)
    {
    case 4:
        dims[3] = tensor->dims[3];
        strides[3] = tensor->strides[3];
        FALLTHROUGH
    case 3:
        dims[2] = tensor->dims[2];
        strides[2] = tensor->strides[2];
        FALLTHROUGH
    case 2:
        dims[1] = tensor->dims[1];
        strides[1] = tensor->strides[1];
        FALLTHROUGH
    case 1:
        dims[0] = tensor->dims[0];
        strides[0] = tensor->strides[0];
        break;
    }

    const bool b_tensor_is_float = tensor->type == GIGA_Float32 || tensor->type == GIGA_Float16;

    const auto &impl_for_types = [&](const auto *src, auto *dst)
    {
        typedef typename std::remove_reference<decltype(*dst)>::type T;
        const int delta_fp_shift = (b_tensor_is_float ? 0 : tensor->fp_shift) - int(fp_shift);
        const float f = b_tensor_is_float ? 1.f / (1 << -delta_fp_shift) : float(1 << delta_fp_shift);
        const size_t tensor_size = dims[0] * dims[1] * dims[2] * dims[3];
        for(size_t i = 0 ; i < tensor_size ; ++i)
            dst[i] = cast_to<T>(src[i], delta_fp_shift, f);
    };

    if (source_type == tensor->type && fp_shift == tensor->fp_shift)    // Simple copy
    {
        const size_t tensor_size = element_size_in_bits(tensor->type) / 8 * dims[0] * dims[1] * dims[2] * dims[3];
        memcpy(get_ptr<uint8_t>(tensor), user_ptr, tensor_size);
        return GIGA_Success;
    }

    switch(source_type)
    {
#define IMPL_TYPE(ENUM, TYPE)\
    case ENUM:\
        switch(tensor->type)\
        {\
        case GIGA_Float32:  impl_for_types((const TYPE*)user_ptr, get_ptr<float>(tensor));     break;\
        case GIGA_Float16:  impl_for_types((const TYPE*)user_ptr, get_ptr<half>(tensor));      break;\
        case GIGA_UFixed8:  impl_for_types((const TYPE*)user_ptr, get_ptr<uint8_t>(tensor));   break;\
        case GIGA_UFixed16: impl_for_types((const TYPE*)user_ptr, get_ptr<uint16_t>(tensor));  break;\
        case GIGA_SFixed8:  impl_for_types((const TYPE*)user_ptr, get_ptr< int8_t>(tensor));   break;\
        case GIGA_SFixed16: impl_for_types((const TYPE*)user_ptr, get_ptr< int16_t>(tensor));  break;\
        default:\
            RETURN_ERROR(GIGA_Unimplemented_Type);\
        }\
        break

    IMPL_TYPE(GIGA_Float32, float);
    IMPL_TYPE(GIGA_Float16, half);
    IMPL_TYPE(GIGA_UFixed8, uint8_t);
    IMPL_TYPE(GIGA_UFixed16, uint16_t);
    IMPL_TYPE(GIGA_SFixed8, int8_t);
    IMPL_TYPE(GIGA_SFixed16, int16_t);

#undef IMPL_TYPE
    default:
        RETURN_ERROR(GIGA_Unimplemented_Type);
    }

    return GIGA_Success;
}

GIGA_error giga_copy_from_tensor_(void *user_ptr, GIGA_data_type target_type, uint32_t fp_shift, const GIGA_tensor_t *tensor, const char *file, int line)
{
    GIGA_error error;
    uint32_t dims[4] = {1, 1, 1, 1};
    uint32_t strides[4] = {0, 0, 0, 0};
\
    switch(tensor->nb_dims)
    {
    case 4:
        dims[3] = tensor->dims[3];
        strides[3] = tensor->strides[3];
        FALLTHROUGH
    case 3:
        dims[2] = tensor->dims[2];
        strides[2] = tensor->strides[2];
        FALLTHROUGH
    case 2:
        dims[1] = tensor->dims[1];
        strides[1] = tensor->strides[1];
        FALLTHROUGH
    case 1:
        dims[0] = tensor->dims[0];
        strides[0] = tensor->strides[0];
        break;
    }

    const bool b_target_is_float = target_type == GIGA_Float32 || target_type == GIGA_Float16;

    const auto &impl_for_types = [&](auto *dst, const auto *src)
    {
        typedef typename std::remove_reference<decltype(*dst)>::type T;
        const int delta_fp_shift = (b_target_is_float ? 0 : int(fp_shift)) - int(tensor->fp_shift);
        const float f = b_target_is_float ? 1.f / (1 << -delta_fp_shift) : float(1 << delta_fp_shift);
        const size_t tensor_size = dims[0] * dims[1] * dims[2] * dims[3];
        for(size_t i = 0 ; i < tensor_size ; ++i)
            dst[i] = cast_to<T>(src[i], delta_fp_shift, f);
    };

    if (target_type == tensor->type && fp_shift == tensor->fp_shift)    // Simple copy
    {
        const size_t tensor_size = element_size_in_bits(tensor->type) / 8 * dims[0] * dims[1] * dims[2] * dims[3];
        memcpy(user_ptr, get_cptr<uint8_t>(tensor), tensor_size);
        return GIGA_Success;
    }

    switch(target_type)
    {
#define IMPL_TYPE(ENUM, TYPE)\
    case ENUM:\
        switch(tensor->type)\
        {\
        case GIGA_Float32:  impl_for_types((TYPE*)user_ptr, get_cptr<float>(tensor));       break;\
        case GIGA_Float16:  impl_for_types((TYPE*)user_ptr, get_cptr<half>(tensor));        break;\
        case GIGA_UFixed8:  impl_for_types((TYPE*)user_ptr, get_cptr<uint8_t>(tensor));     break;\
        case GIGA_UFixed16: impl_for_types((TYPE*)user_ptr, get_cptr<uint16_t>(tensor));    break;\
        case GIGA_SFixed8:  impl_for_types((TYPE*)user_ptr, get_cptr< int8_t>(tensor));     break;\
        case GIGA_SFixed16: impl_for_types((TYPE*)user_ptr, get_cptr< int16_t>(tensor));    break;\
        default:\
            RETURN_ERROR(GIGA_Unimplemented_Type);\
        }\
        break

    IMPL_TYPE(GIGA_Float32, float);
    IMPL_TYPE(GIGA_Float16, half);
    IMPL_TYPE(GIGA_UFixed8, uint8_t);
    IMPL_TYPE(GIGA_UFixed16, uint16_t);
    IMPL_TYPE(GIGA_SFixed8, int8_t);
    IMPL_TYPE(GIGA_SFixed16, int16_t);

#undef IMPL_TYPE

    default:
        RETURN_ERROR(GIGA_Unimplemented_Type);
    }

    return GIGA_Success;
}

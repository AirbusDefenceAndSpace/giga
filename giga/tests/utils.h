/*!
 * (C) 2025 Airbus copyright all rights reserved
 * \date 16/01/2025
 */

#ifndef UTILS_H
#define UTILS_H

#include <giga/giga.h>
#include <giga/float16.h>
#include <cstring>
#include <cmath>
#include <iostream>
#include <vector>
#include <sys/time.h>
#include <cstdlib>
#include <sstream>

inline size_t usec_timer()
{
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1000000UL + tv.tv_usec;
}

inline bool is_float(GIGA_data_type type)
{
    switch(type)
    {
    case GIGA_Float16:
    case GIGA_Float32:
        return true;
    default:
        return false;
    }
}

inline bool is_signed(GIGA_data_type type)
{
    switch(type)
    {
    case GIGA_Float16:
    case GIGA_Float32:
    case GIGA_SFixed4:
    case GIGA_SFixed8:
    case GIGA_SFixed16:
        return true;
    default:
        return false;
    }
}

inline size_t tensor_elements_count(const GIGA_tensor_t *tensor)
{
    size_t size = 1;
    for(uint32_t i = 0 ; i < tensor->nb_dims ; ++i)
        size *= tensor->dims[i];
    return size;
}

inline size_t element_size_in_bits(const GIGA_tensor_t *tensor)
{
    switch(tensor->type)
    {
    case GIGA_SFixed4:
    case GIGA_UFixed4:
        return 4;

    case GIGA_SFixed8:
    case GIGA_UFixed8:
        return 8;

    case GIGA_SFixed16:
    case GIGA_UFixed16:
    case GIGA_Float16:
        return 16;

    case GIGA_Float32:
        return 32;
    }
    return 0;
}

inline size_t tensor_size_in_bytes(const GIGA_tensor_t *tensor)
{
    return tensor_elements_count(tensor) * element_size_in_bits(tensor) >> 3;
}

inline size_t align_address(size_t addr, size_t alignment)
{
    const size_t mask = alignment - 1;
    return (addr + mask) & ~mask;
}

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

template<typename ElementType>
GIGA_error __impl_fill_contiguous_tensor_with_random_data(GIGA_tensor_t &tensor, const float lower, const float upper)
{
    uint32_t dims[4] = {1, 1, 1, 1};
    uint32_t strides[4] = {0, 0, 0, 0};

    for(int64_t dim = tensor.nb_dims - 1; dim >= 0; dim--)
    {
        dims[dim] = tensor.dims[dim];
        strides[dim] = tensor.strides[dim];
    }

    ElementType * write_ptr = nullptr;
    GIGA_error error = giga_map_tensor(&tensor, (void**)&write_ptr, GIGA_Memory_Discard);
    if(error != GIGA_Success)
    {
        std::cerr << "Error: could not map tensor! (" << giga_str_error(error) << ")" << std::endl;
        return error;
    }

    const uint32_t size = tensor_elements_count(&tensor);

    uint32_t seed = rand();
    const auto &rnd = [&]()
    {
        return float(rand_r(&seed)) / float(RAND_MAX) * (upper - lower) + lower;
    };

    const float shift_factor = tensor.fp_shift ? (1UL << tensor.fp_shift) : 1.f;

    for(uint32_t i = 0 ; i < size ; ++i)
        write_ptr[i] = static_cast<ElementType>(rnd() * shift_factor);

    error = giga_unmap_tensor(&tensor, (void*)write_ptr, GIGA_Memory_Sync);
    if(error != GIGA_Success)
    {
        std::cerr << "Error: could not unmap tensor! (" << giga_str_error(error) << ")" << std::endl;
        return error;
    }

    return GIGA_Success;
}

GIGA_error fill_contiguous_tensor_with_random_data(GIGA_tensor_t &tensor, const float lower, const float upper)
{
    switch(tensor.type)
    {
#define IMPL_CASE(T)    case T:  return __impl_fill_contiguous_tensor_with_random_data<GIGA_C_Type<T>::CType>(tensor, lower, upper)
    IMPL_CASE(GIGA_Float16);
    IMPL_CASE(GIGA_Float32);
    IMPL_CASE(GIGA_SFixed8);
    IMPL_CASE(GIGA_SFixed16);
    IMPL_CASE(GIGA_UFixed8);
    IMPL_CASE(GIGA_UFixed16);
    default:
        return GIGA_Unimplemented_Type;
    }
#undef IMPL_CASE
}

template<typename inputType, typename TensorType = inputType>
GIGA_error __impl_fill_4d_tensor(const inputType * const data, GIGA_tensor_t &tensor)
{
    uint32_t dims[4] = {1, 1, 1, 1};
    uint32_t strides[4] = {0, 0, 0, 0};

    for(uint32_t dim = 0 ; dim < tensor.nb_dims ; ++dim)
    {
        dims[dim] = tensor.dims[dim];
        strides[dim] = tensor.strides[dim];
    }

    TensorType * write_ptr = nullptr;
    GIGA_error error = giga_map_tensor(&tensor, (void**)&write_ptr, GIGA_Memory_Discard);
    if(error != GIGA_Success)
    {
        std::cerr << "Error: could not map tensor! (" << giga_str_error(error) << ")" << std::endl;
        return error;
    }

    const inputType * data_ptr = data;
    for(uint32_t dim0_i = 0; dim0_i < dims[0]; ++dim0_i)
        for(uint32_t dim1_i = 0; dim1_i < dims[1]; ++dim1_i)
            for(uint32_t dim2_i = 0; dim2_i < dims[2]; ++dim2_i)
                for(uint32_t dim3_i = 0; dim3_i < dims[3]; ++dim3_i)
                {
                    if(tensor.fp_shift)
                    {
                        write_ptr[(dim0_i * strides[0] + dim1_i * strides[1] + dim2_i * strides[2] + dim3_i * strides[3])/sizeof(TensorType)] =
                            static_cast<TensorType>((*(data_ptr++)) * (1UL << tensor.fp_shift));
                    }
                    else
                    {
                        write_ptr[(dim0_i * strides[0] + dim1_i * strides[1] + dim2_i * strides[2] + dim3_i * strides[3])/sizeof(TensorType)] =
                            static_cast<TensorType>(*(data_ptr++));
                    }
                }

    error = giga_unmap_tensor(&tensor, (void*)write_ptr, GIGA_Memory_Sync);
    if(error != GIGA_Success)
    {
        std::cerr << "Error: could not unmap tensor! (" << giga_str_error(error) << ")" << std::endl;
        return error;
    }

    return GIGA_Success;
}

template<typename inputType>
GIGA_error fill_4d_tensor(const inputType * const data, GIGA_tensor_t &tensor)
{
    switch(tensor.type)
    {
#define IMPL_CASE(T)    case T:  return __impl_fill_4d_tensor<inputType, GIGA_C_Type<T>::CType>(data, tensor)
    IMPL_CASE(GIGA_Float16);
    IMPL_CASE(GIGA_Float32);
    IMPL_CASE(GIGA_SFixed8);
    IMPL_CASE(GIGA_SFixed16);
    IMPL_CASE(GIGA_UFixed8);
    IMPL_CASE(GIGA_UFixed16);
    default:
        return GIGA_Unimplemented_Type;
    }
#undef IMPL_CASE
}

template<GIGA_data_type GT>
bool _compare_tensors_impl(GIGA_tensor_t *t1, GIGA_tensor_t *t2, const double epsilon = 0)
{
    typedef typename GIGA_C_Type<GT>::CType T;

    if(epsilon < 0)
        return false;

    if (is_float(t1->type) != is_float(t2->type))
        return false;

    if(t1->nb_dims != t2->nb_dims)
        return false;

    for(unsigned int dim_i = 0; dim_i < t1->nb_dims; dim_i++)
    {
        if(t1->dims[dim_i] != t2->dims[dim_i])
        {
            return false;
        }
    }

    uint32_t dims[4] = {1, 1, 1, 1};
    uint32_t strides1[4] = {0, 0, 0, 0};
    uint32_t strides2[4] = {0, 0, 0, 0};

    for(uint32_t dim = 0; dim < t1->nb_dims; dim++)
    {
        dims[dim] = t1->dims[dim];
        strides1[dim] = t1->strides[dim];
        strides2[dim] = t2->strides[dim];
    }

    T *t1_ptr, *t2_ptr = nullptr;

    GIGA_error error;
    if ((error = giga_map_tensor(t1, (void**)&t1_ptr, GIGA_Memory_Sync)) != GIGA_Success)
    {
        std::cerr << "Error mapping tensor t1: " << giga_str_error(error) << std::endl;
        return false;
    }
    if ((error = giga_map_tensor(t2, (void**)&t2_ptr, GIGA_Memory_Sync)) != GIGA_Success)
    {
        std::cerr << "Error mapping tensor t2: " << giga_str_error(error) << std::endl;
        return false;
    }

    if(!t1->fp_shift && !t2->fp_shift)
    {
        for(uint32_t dim0_i = 0; dim0_i < dims[0]; ++dim0_i)
            for(uint32_t dim1_i = 0; dim1_i < dims[1]; ++dim1_i)
                for(uint32_t dim2_i = 0; dim2_i < dims[2]; ++dim2_i)
                    for(uint32_t dim3_i = 0; dim3_i < dims[3]; ++dim3_i)
                    {
                        const T val1 = t1_ptr[(dim0_i * strides1[0] + dim1_i * strides1[1] + dim2_i * strides1[2] + dim3_i * strides1[3])/sizeof(T)];
                        const T val2 = t2_ptr[(dim0_i * strides2[0] + dim1_i * strides2[1] + dim2_i * strides2[2] + dim3_i * strides2[3])/sizeof(T)];

                        if(epsilon == 0)
                        {
                            if(val2 != val1)
                                return false;
                        }
                        else
                        {
                            if(std::abs(val2 - val1) > epsilon)
                                return false;
                        }
                    }
    }
    else
    {
        for(uint32_t dim0_i = 0; dim0_i < dims[0]; ++dim0_i)
            for(uint32_t dim1_i = 0; dim1_i < dims[1]; ++dim1_i)
                for(uint32_t dim2_i = 0; dim2_i < dims[2]; ++dim2_i)
                    for(uint32_t dim3_i = 0; dim3_i < dims[3]; ++dim3_i)
                    {
                        const double val1 = double(t1_ptr[(dim0_i * strides1[0] + dim1_i * strides1[1] + dim2_i * strides1[2] + dim3_i * strides1[3])/sizeof(T)]) / (1 << t1->fp_shift);
                        const double val2 = double(t2_ptr[(dim0_i * strides2[0] + dim1_i * strides2[1] + dim2_i * strides2[2] + dim3_i * strides2[3])/sizeof(T)]) / (1 << t2->fp_shift);

                        if(epsilon == 0)
                        {
                            if(val2 != val1)
                                return false;
                        }
                        else
                        {
                            if(std::abs(val2 - val1) > epsilon)
                                return false;
                        }
                    }
    }

    giga_unmap_tensor(t1, (void*)t1_ptr, GIGA_Memory_Discard);
    giga_unmap_tensor(t2, (void*)t2_ptr, GIGA_Memory_Discard);
    return true;
}

#define GIGA_TYPE_TEMPLATED_CASE(func, type, tensor, ...) \
case type:\
    ret = func <type>(tensor, ##__VA_ARGS__);\
    break;

#define GIGA_CALL_TEMPLATED_FUNC_ON_TENSOR(func, tensor, ...)\
switch(tensor ->type)\
    {\
            GIGA_TYPE_TEMPLATED_CASE(func, GIGA_Float16, tensor, ##__VA_ARGS__ ) \
            GIGA_TYPE_TEMPLATED_CASE(func, GIGA_Float32, tensor, ##__VA_ARGS__ ) \
            GIGA_TYPE_TEMPLATED_CASE(func, GIGA_SFixed4, tensor, ##__VA_ARGS__ ) \
            GIGA_TYPE_TEMPLATED_CASE(func, GIGA_SFixed8, tensor, ##__VA_ARGS__ ) \
            GIGA_TYPE_TEMPLATED_CASE(func, GIGA_SFixed16, tensor, ##__VA_ARGS__ ) \
            GIGA_TYPE_TEMPLATED_CASE(func, GIGA_UFixed4, tensor, ##__VA_ARGS__ ) \
            GIGA_TYPE_TEMPLATED_CASE(func, GIGA_UFixed8, tensor, ##__VA_ARGS__ ) \
            GIGA_TYPE_TEMPLATED_CASE(func, GIGA_UFixed16, tensor, ##__VA_ARGS__ ) \
            default:\
            ret = GIGA_Unimplemented_Type;\
    }


inline bool compare_tensors(GIGA_tensor_t *t1, GIGA_tensor_t *t2, double epsilon = 0)
{
    bool ret;
    GIGA_CALL_TEMPLATED_FUNC_ON_TENSOR(_compare_tensors_impl,t1, t2, epsilon);

    return ret;
}

template<typename T>
bool __impl_print_tensor(std::ostream &out, GIGA_tensor_t &tensor, const std::string &name)
{
    uint32_t dims[4] = {1, 1, 1, 1};
    uint32_t strides[4] = {0, 0, 0, 0};

    std::vector<std::string> separators(4);
    for(uint32_t dim = 0; dim < tensor.nb_dims; dim++)
    {
        dims[dim] = tensor.dims[dim];
        strides[dim] = tensor.strides[dim];
    }

    switch(tensor.nb_dims)
    {
    case 4:
        separators = {"----\n", "****\n", "\n", ";\t"};
        break;
    case 3:
        separators = {"****\n", "\n", ";\t", ""};
        break;
    case 2:
        separators = {"\n", ";\t", "", ""};
        break;
    case 1:
        separators = {";\t", "", "", ""};
        break;
    }

    out << name << ":\n";
    T *t_ptr = nullptr;
    GIGA_error error = giga_map_tensor(&tensor, (void**)&t_ptr, GIGA_Memory_Sync);
    if (!t_ptr)
    {
        std::cerr << "Error: could not map tensor! (" << giga_str_error(error) << ")" << std::endl;
        return false;
    }
    for(unsigned int dim0_i = 0; dim0_i < dims[0]; dim0_i++)
    {
        for(unsigned int dim1_i = 0; dim1_i < dims[1]; dim1_i++)
        {
            for(unsigned int dim2_i = 0; dim2_i < dims[2]; dim2_i++)
            {
                for(unsigned int dim3_i = 0; dim3_i < dims[3]; dim3_i++)
                {
                    out << double(t_ptr[(dim0_i * strides[0] + dim1_i * strides[1] + dim2_i * strides[2] + dim3_i * strides[3])/sizeof(T)]) / (1 << tensor.fp_shift) << separators[3];
                }
                out << separators[2];
            }
            out << separators[1];
        }
        out << separators[0];
    }

    out << std::endl;
    giga_unmap_tensor(&tensor, (void*)t_ptr, GIGA_Memory_Discard);

    return true;
}

inline bool print_tensor(std::ostream &out, GIGA_tensor_t &tensor, const std::string &name)
{
    switch(tensor.type)
    {
#define IMPL_CASE(T)    case T:  return __impl_print_tensor<GIGA_C_Type<T>::CType>(out, tensor, name)
    IMPL_CASE(GIGA_Float16);
    IMPL_CASE(GIGA_Float32);
    IMPL_CASE(GIGA_SFixed8);
    IMPL_CASE(GIGA_SFixed16);
    IMPL_CASE(GIGA_UFixed8);
    IMPL_CASE(GIGA_UFixed16);
    default:
        return GIGA_Unimplemented_Type;
    }
#undef IMPL_CASE
}

/*!
 * \brief Small class to automatically print a message when leaving a scope
 * This is meant to be used in case of error triggering an early termination,
 * the message is disabled/replaced when everything goes through normal code path.
 */
class ScopedMessage
{
public:
    ScopedMessage(const std::string &msg = std::string())
    {
        this->msg << msg;
    }

    ~ScopedMessage()
    {
        if (!msg.str().empty())
            std::cout << msg.str() << std::endl;
    }

    void clear()
    {
        msg.str(std::string());
    }

    void replaceMessage(const std::string &msg)
    {
        clear();
        this->msg << msg;
    }

    operator std::ostream &()   {   return msg; }

    template<class T>
    ScopedMessage &operator<<(const T &v)
    {
        msg << v;
        return *this;
    }

    template<class T>
    ScopedMessage &operator<<(T &v)
    {
        msg << v;
        return *this;
    }

    std::string message() const
    {
        return msg.str();
    }

private:
    std::stringstream msg;
};

#endif // UTILS_H

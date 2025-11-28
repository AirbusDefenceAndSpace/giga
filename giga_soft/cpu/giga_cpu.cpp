/*!
 * (C) 2025 Airbus copyright all rights reserved
 * \author Lucas Marti (lucas.marti@airbus.com)
 * \author Roland Brochard (roland.brochard@airbus.com)
 * \date 3/04/2023
 *
 * Baseline CPU implementation of the GIGA API
 *
 */

#include "giga_cpu.h"
#include <cstdlib>
#include <cstring>

const bool giga_cpu_use_exceptions = getenv("GIGA_CPU_USE_EXCEPTION") ? strcmp(getenv("GIGA_CPU_USE_EXCEPTION"), "1") == 0 : false;

bool is_float(GIGA_data_type type)
{
    return static_cast<int>(type) <= 1;
}

bool is_signed(GIGA_data_type type)
{
    return static_cast<int>(type) <= 4;
}

size_t element_size_in_bits(GIGA_data_type data_type)
{
    switch(data_type)
    {
    case GIGA_SFixed4:
    case GIGA_UFixed4:
        return 4;

    case GIGA_SFixed8:
    case GIGA_UFixed8:
        return 8;

    case GIGA_Float16:
    case GIGA_SFixed16:
    case GIGA_UFixed16:
        return 16;

    case GIGA_Float32:
        return 32;
    }
    return 0;
}

bool check_tensor_exists(const GIGA_tensor_t *tensor)
{
    if (tensor->data == nullptr)
        return false;

    return ((Tensor_data_t*)tensor->data)->is_allocated;
}

uint32_t giga_get_default_device_id(GIGA_error *err)
{
    *err = GIGA_Success;
    return 0;
}

GIGA_error giga_list_devices(uint32_t *device_ids, uint32_t *nb_devices)
{
    device_ids[0] = 0;
    *nb_devices = 1;
    return GIGA_Success;
}

GIGA_error giga_initialize_device(uint32_t device_id)
{
    return GIGA_Success;
}

GIGA_error giga_callback_(uint32_t device_id, void (*callback)(void *user_ptr), void *user_ptr, const char *file, int line)
{
    (*callback)(user_ptr);
    return GIGA_Success;
}

GIGA_error giga_wait_for_completion()
{
    return GIGA_Success;
}

GIGA_error giga_flush(uint32_t device_id)
{
    return GIGA_Success;
}

const char *giga_str_error(GIGA_error err)
{
    switch(err)
    {
#define IMPL_CASE(X) case GIGA_##X: return #X
    IMPL_CASE(Success);
    IMPL_CASE(Unknown_Error);
    IMPL_CASE(Incorrect_Parameter);
    IMPL_CASE(Out_Of_Host_Memory);
    IMPL_CASE(Out_Of_Device_Memory);
    IMPL_CASE(Inconsistent_Tensor_Sizes);
    IMPL_CASE(Inconsistent_Number_Of_Dimensions);
    IMPL_CASE(Unimplemented_Type);
    IMPL_CASE(Unknown_tensor);
    IMPL_CASE(Inconsistent_Tensor_Types);
    IMPL_CASE(Bad_Alloc);
    IMPL_CASE(Device_Not_Initialized);
    IMPL_CASE(Bad_Memory_Alignment);
    IMPL_CASE(Not_Implemented);
    IMPL_CASE(Device_Error);
    IMPL_CASE(Inconsistent_Device);
    IMPL_CASE(Process_Mapped_Tensor);
    IMPL_CASE(Memory_Alignement_Error);
    IMPL_CASE(Memory_Layout_Error);
    default:
        return "Unknown error code";
    }
}

/*!
 * (C) 2025 Airbus copyright all rights reserved
 *
 */
/*!
 * \file
 * \author Roland Brochard (roland.brochard@airbus.com)
 * \date 16/01/2025
 *
 * Stub implementation of GIGA to allow linking programs without the need for a real backend.
 * Typical use: link unit tests.
 *
 */

#include "giga.h"
#include <stdexcept>

extern "C"
{
    #define STUB(ret, name, ...)\
    ret name(__VA_ARGS__)\
    {\
        throw std::runtime_error("Function " #name " not implemented!");\
    }

    STUB(const char *, giga_str_error, GIGA_error err)
    STUB(uint32_t, giga_get_default_device_id, GIGA_error *err)
    STUB(GIGA_error, giga_list_devices, uint32_t *device_ids, uint32_t *nb_devices);
    STUB(GIGA_error, giga_initialize_device, uint32_t device_id);
    STUB(GIGA_error, giga_allocate_tensor_, GIGA_tensor_t *tensor, const GIGA_allocate_t *params, const char *file, int line);
    STUB(GIGA_error, giga_map_tensor_, GIGA_tensor_t *tensor, void **ptr, GIGA_memory_flag flags, const char *file, int line);
    STUB(GIGA_error, giga_unmap_tensor_, GIGA_tensor_t *tensor, void *ptr, GIGA_memory_flag flags, const char *file, int line);
    STUB(GIGA_error, giga_release_tensor_, GIGA_tensor_t *tensor, const char *file, int line);
    STUB(GIGA_error, giga_conv2d_, const GIGA_conv2d_t *params, const GIGA_tensor_t *in, GIGA_tensor_t *out, const char *file, int line);
    STUB(GIGA_error, giga_dense_, const GIGA_dense_t *params, const GIGA_tensor_t *in, GIGA_tensor_t *out, const char *file, int line);
    STUB(GIGA_error, giga_reshape_, const GIGA_reshape_t * params, const GIGA_tensor_t *in, GIGA_tensor_t *out, const char *file, int line);
    STUB(GIGA_error, giga_softmax_, const GIGA_softmax_t * params, const GIGA_tensor_t *in, GIGA_tensor_t *out, const char *file, int line);
    STUB(GIGA_error, giga_add_, const GIGA_add_t * params, const GIGA_tensor_t *a, const GIGA_tensor_t *b, GIGA_tensor_t *out, const char *file, int line);
    STUB(GIGA_error, giga_upsample_, const GIGA_upsample_t *params, const GIGA_tensor_t *in, GIGA_tensor_t *out, const char *file, int line);
    STUB(GIGA_error, giga_view_, const GIGA_view_t *params, const GIGA_tensor_t *in, GIGA_tensor_t *out, const char *file, int line);
    STUB(GIGA_error, giga_callback_, uint32_t device_id, void (*callback)(void *user_ptr), void *user_ptr, const char *file, int line);
    STUB(GIGA_error, giga_wait_for_completion);
    STUB(GIGA_error, giga_flush, uint32_t device_id);
    STUB(GIGA_error, giga_register_error_callback, void (*callback)(void *user_ptr, GIGA_error err, const char *file, int line), void *user_ptr);
    STUB(GIGA_error, giga_copy_to_tensor_, const void *user_ptr, GIGA_data_type source_type, uint32_t fp_shift, GIGA_tensor_t *tensor, const char *file, int line);
    STUB(GIGA_error, giga_copy_from_tensor_, void *user_ptr, GIGA_data_type target_type, uint32_t fp_shift, const GIGA_tensor_t *tensor, const char *file, int line);
}

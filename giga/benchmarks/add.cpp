/*!
 * (C) 2025 Airbus copyright all rights reserved
 * \date 17/01/2025
 */
#include <giga/giga.h>
#include "../tests/utils.h"
#include <cmath>

GIGA_error addition_benchmark(GIGA_data_type GT, int nb_runs, uint8_t a_shift = 0, uint8_t b_shift = 0, uint8_t out_shift = 0)
{
    ScopedMessage on_error_message(std::string("Error on Add ")
                                   + giga_data_type_str(GT)
                                   + ", a_shift " + std::to_string(int(a_shift))
                                   + ", b_shift " + std::to_string(int(b_shift))
                                   + ", out_shift " + std::to_string(int(out_shift)));
    std::cout << "Add " << giga_data_type_str(GT) << ", a_shift " << int(a_shift) << ", b_shift " << int(b_shift) << ", out_shift " << int(out_shift) << " : " << std::flush;

    GIGA_error error;
    uint32_t device_id = giga_get_default_device_id(&error);

    if(error != GIGA_Success)
    {
        std::cerr << "Error getting default device" << std::endl;
        return error;
    }

    if((error = giga_initialize_device(device_id)) != GIGA_Success)
    {
        std::cerr << "Error initializing device" << std::endl;
        return error;
    }

    size_t offset = 0;

    GIGA_tensor_t a;
    a.nb_dims = 4;
    a.dims[0] = 1;
    a.dims[1] = 2;
    a.dims[2] = 1024;
    a.dims[3] = 1024;
    a.device_id = device_id;
    a.type = GT;
    a.data = NULL;
    a.fp_shift = a_shift;

    GIGA_allocate_t a_params;
    a_params.memory_zone_id = 0;
    a_params.offset = offset;
    offset += tensor_size_in_bytes(&a);
    if((error = giga_allocate_tensor(&a, &a_params)) != GIGA_Success)
    {
        std::cerr << "Error allocating tensor a" << std::endl;
        return error;
    }

    fill_contiguous_tensor_with_random_data(a, 0.f, 1.f);

    GIGA_tensor_t b = a;
    b.data = NULL;
    b.fp_shift = b_shift,

    b.device_id = device_id;
    b.type = GT;

    GIGA_allocate_t b_params;
    b_params.memory_zone_id = 0;
    b_params.offset = offset;
    offset += tensor_size_in_bytes(&b);
    if((error = giga_allocate_tensor(&b, &b_params) ) != GIGA_Success)
    {
        std::cerr << "Error allocating tensor b" << std::endl;
        return error;
    }

    fill_contiguous_tensor_with_random_data(b, 0.f, 1.f);

    GIGA_tensor_t out = a;
    out.device_id = device_id;
    out.type = GT;
    out.data = NULL;
    out.fp_shift = out_shift;

    GIGA_allocate_t out_params;
    out_params.memory_zone_id = 0;
    out_params.offset = offset;
    offset += tensor_size_in_bytes(&out);
    if((error = giga_allocate_tensor(&out, &out_params) ) != GIGA_Success)
    {
        std::cerr << "Error allocating tensor out" << std::endl;
        return error;
    }

    const size_t start = usec_timer();
    for(int it = 0 ; it < nb_runs ; ++it)
    {
        GIGA_add_t add_params;
        if((error = giga_add(&add_params, &a, &b, &out) ) != GIGA_Success)
        {
            if (error == GIGA_Unimplemented_Type)
            {
                std::cout << "Type not implemented" << std::endl;
                on_error_message.clear();
                return GIGA_Success;
            }
            if (error == GIGA_Not_Implemented)
            {
                std::cout << "Function not implemented" << std::endl;
                on_error_message.clear();
                return GIGA_Success;
            }
            std::cerr << "Error performing add on a and b to out" << std::endl;
            return error;
        }
    }
    if ((error = giga_flush(device_id)) != GIGA_Success)
    {
        std::cerr << "Error flushing device" << std::endl;
        return error;
    }
    if ((error = giga_wait_for_completion()) != GIGA_Success)
    {
        std::cerr << "Error waiting for completion" << std::endl;
        return error;
    }

    const size_t end = usec_timer();
    std::cout << (end - start) / nb_runs << "µs per call" << std::endl;

    //Clean up
    if((error = giga_release_tensor(&a) ) != GIGA_Success)
    {
        std::cerr << "Error releasing tensor a" << std::endl;
        return error;
    }

    if((error = giga_release_tensor(&b) ) != GIGA_Success)
    {
        std::cerr << "Error releasing tensor b" << std::endl;
        return error;
    }

    if((error = giga_release_tensor(&out) ) != GIGA_Success)
    {
        std::cerr << "Error releasing tensor out" << std::endl;
        return error;
    }

    on_error_message.clear();

    return GIGA_Success;
}

int main()
{
    GIGA_error error = GIGA_Success;

#define EARLY_ABORT() if (error != GIGA_Not_Implemented)    throw std::runtime_error("Error")

    const int nb_runs = 10;

    try
    {
        if((error = addition_benchmark(GIGA_Float32, nb_runs)) != GIGA_Success)
            EARLY_ABORT();
        if((error = addition_benchmark(GIGA_Float16, nb_runs)) != GIGA_Success)
            EARLY_ABORT();
        if((error = addition_benchmark(GIGA_SFixed8, nb_runs, 4, 4, 4)) != GIGA_Success)
            EARLY_ABORT();
        if((error = addition_benchmark(GIGA_SFixed16, nb_runs, 4, 4, 4)) != GIGA_Success)
            EARLY_ABORT();
        if((error = addition_benchmark(GIGA_UFixed8, nb_runs, 4, 4, 4)) != GIGA_Success)
            EARLY_ABORT();
        if((error = addition_benchmark(GIGA_UFixed16, nb_runs, 4, 4, 4)) != GIGA_Success)
            EARLY_ABORT();
    }
    catch(const std::exception &e)
    {
        if (error != GIGA_Success)
            std::cerr << "Error: " << giga_str_error(error) << std::endl;
        else
        {
            std::cerr << "Exception caught: " << e.what() << std::endl;
            error = GIGA_Unknown_Error;
        }
    }

    return error;
}

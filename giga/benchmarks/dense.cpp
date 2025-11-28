/*!
 * (C) 2025 Airbus copyright all rights reserved
 * \date 16/01/2025
 */
#include <giga/giga.h>
#include "../tests/utils.h"

GIGA_error dense_benchmark(GIGA_data_type i_GT, GIGA_data_type o_GT, GIGA_data_type k_GT, int nb_runs, uint8_t in_shift = 0, uint8_t ker_shift = 0, uint8_t out_shift = 0)
{
    ScopedMessage on_error_message(std::string("Error on ")
                                   + "Dense, in " + giga_data_type_str(i_GT)
                                   + ", out " + giga_data_type_str(o_GT)
                                   + ", params " + giga_data_type_str(k_GT)
                                   + ", in_shift " + std::to_string(int(in_shift))
                                   + ", ker_shift " + std::to_string(int(ker_shift))
                                   + ", out_shift " + std::to_string(int(out_shift)));

    std::cout << "Dense, in " << giga_data_type_str(i_GT)
              << ", out " << giga_data_type_str(o_GT)
              << ", params " << giga_data_type_str(k_GT)
              << ", in_shift " << int(in_shift)
              << ", ker_shift " << int(ker_shift)
              << ", out_shift " << int(out_shift) << " : " << std::flush;
    GIGA_error error;
    uint32_t device_id = giga_get_default_device_id(&error);

    if(error != GIGA_Success)
    {
        std::cerr << "Error getting default device id" << std::endl;
        return error;
    }

    if((error = giga_initialize_device(device_id)) != GIGA_Success)
    {
        std::cerr << "Error initializing device" << std::endl;
        return error;
    }

    size_t offset = 0;

    GIGA_tensor_t in;
    in.nb_dims = 2;
    in.dims[0] = 16;
    in.dims[1] = 1024;
    in.device_id = device_id;
    in.type = i_GT;
    in.fp_shift = 0;

    GIGA_allocate_t in_params;
    in_params.memory_zone_id = 0;
    in_params.offset = offset;
    offset += tensor_size_in_bytes(&in);
    if((error = giga_allocate_tensor(&in, &in_params)) != GIGA_Success)
    {
        std::cerr << "Error allocating tensor in" << std::endl;
        return error;
    }

    fill_contiguous_tensor_with_random_data(in, -1.f, 1.f);

    GIGA_tensor_t out;
    out.nb_dims = 2;
    out.dims[0] = 16;
    out.dims[1] = 1024;
    out.device_id = device_id;
    out.type = o_GT;
    out.fp_shift = 0;

    GIGA_allocate_t out_params;
    out_params.memory_zone_id = 0;
    out_params.offset = offset;
    offset += tensor_size_in_bytes(&out);
    if((error = giga_allocate_tensor(&out, &out_params)) != GIGA_Success)
    {
        std::cerr << "Error allocating tensor out" << std::endl;
        return error;
    }

    GIGA_tensor_t ker;
    ker.nb_dims = 2;
    ker.dims[0] = 1024;
    ker.dims[1] = 1024;
    ker.device_id = device_id;
    ker.type = k_GT;
    ker.fp_shift = 0;

    GIGA_allocate_t ker_params;
    ker_params.memory_zone_id = 0;
    ker_params.offset = offset;
    offset += tensor_size_in_bytes(&ker);
    if((error = giga_allocate_tensor(&ker, &ker_params)) != GIGA_Success)
    {
        std::cerr << "Error allocating tensor ker" << std::endl;
        return error;
    }

    fill_contiguous_tensor_with_random_data(ker, -1.f, 1.f);

    GIGA_dense_t params;
    params.kernel = &ker;
    params.b_ReLU = false;
    params.bias = NULL;

    const size_t start = usec_timer();
    for(int it = 0 ; it < nb_runs ; ++it)
    {
        if((error = giga_dense(&params, &in, &out)) != GIGA_Success)
        {
            if (error == GIGA_Not_Implemented)
            {
                std::cout << "Function not implemented!" << std::endl;
                on_error_message.clear();
                return GIGA_Success;
            }
            if (error == GIGA_Unimplemented_Type)
            {
                std::cout << "Type not implemented!" << std::endl;
                on_error_message.clear();
                return GIGA_Success;
            }
            std::cerr << "Error performing giga_dense" << std::endl;
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

    if((error = giga_release_tensor(&in)) != GIGA_Success)
    {
        std::cerr << "Error releasing tensor in" << std::endl;
        return error;
    }

    if((error = giga_release_tensor(&out)) != GIGA_Success)
    {
        std::cerr << "Error releasing tensor out" << std::endl;
        return error;
    }

    if((error = giga_release_tensor(&ker)) != GIGA_Success)
    {
        std::cerr << "Error releasing tensor ker" << std::endl;
        return error;
    }

    on_error_message.clear();

    return GIGA_Success;
}

int main()
{
    GIGA_error error = GIGA_Success;

    const int nb_runs = 100;

#define EARLY_ABORT() throw std::runtime_error("Error")

    try
    {
        if((error = dense_benchmark(GIGA_Float32, GIGA_Float32, GIGA_Float32, nb_runs)) != GIGA_Success)
            EARLY_ABORT();
        if((error = dense_benchmark(GIGA_Float16, GIGA_Float16, GIGA_Float16, nb_runs)) != GIGA_Success)
            EARLY_ABORT();

        if((error = dense_benchmark(GIGA_SFixed8, GIGA_SFixed8, GIGA_SFixed8, nb_runs, 4, 4, 4)) != GIGA_Success)
            EARLY_ABORT();
        if((error = dense_benchmark(GIGA_SFixed16, GIGA_SFixed16, GIGA_SFixed16, nb_runs, 4, 4, 4)) != GIGA_Success)
            EARLY_ABORT();

        if((error = dense_benchmark(GIGA_SFixed16, GIGA_SFixed8, GIGA_SFixed8, nb_runs, 4, 4, 4)) != GIGA_Success)
            EARLY_ABORT();
        if((error = dense_benchmark(GIGA_SFixed16, GIGA_SFixed8, GIGA_SFixed16, nb_runs, 4, 4, 4)) != GIGA_Success)
            EARLY_ABORT();

        if((error = dense_benchmark(GIGA_UFixed8, GIGA_UFixed8, GIGA_UFixed8, nb_runs, 4, 4, 4)) != GIGA_Success)
            EARLY_ABORT();
        if((error = dense_benchmark(GIGA_UFixed16, GIGA_UFixed16, GIGA_UFixed16, nb_runs, 4, 4, 4)) != GIGA_Success)
            EARLY_ABORT();

        if((error = dense_benchmark(GIGA_SFixed16, GIGA_UFixed8, GIGA_SFixed8, nb_runs, 4, 4, 4)) != GIGA_Success)
            EARLY_ABORT();
        if((error = dense_benchmark(GIGA_UFixed16, GIGA_SFixed8, GIGA_SFixed16, nb_runs, 4, 4, 4)) != GIGA_Success)
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

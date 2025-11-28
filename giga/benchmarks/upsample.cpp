/*!
 * (C) 2025 Airbus copyright all rights reserved
 * \date 17/01/2025
 */
#include <giga/giga.h>
#include "../tests/utils.h"

GIGA_error upsample_benchmark(GIGA_data_type GT, const int nb_runs)
{
    ScopedMessage on_error_message(std::string("Error on ")
                                   + "Upsample " + giga_data_type_str(GT));
    std::cout << "Upsample " << giga_data_type_str(GT) << " : " << std::flush;

    GIGA_error error;
    uint32_t device_id = giga_get_default_device_id(&error);

    if(error != GIGA_Success)
    {
        std::cerr << "Error getting default device id" << std::endl;
        return error;
    }

    error = giga_initialize_device(device_id);
    if(error != GIGA_Success)
    {
        std::cerr << "Error initializing device" << std::endl;
        return error;
    }

    size_t offset = 0;

    GIGA_tensor_t tensor;
    tensor.nb_dims = 4;
    tensor.dims[0] = 1;
    tensor.dims[1] = 8;
    tensor.dims[2] = 512;
    tensor.dims[3] = 512;
    tensor.device_id = device_id;
    tensor.type = GT;
    tensor.fp_shift = 0;

    GIGA_allocate_t tensor_params;
    tensor_params.memory_zone_id = 0;
    tensor_params.offset = offset;
    offset += tensor_size_in_bytes(&tensor);
    error = giga_allocate_tensor(&tensor, &tensor_params);
    if(error != GIGA_Success)
    {
        std::cerr << "Error allocating tensor tensor" << std::endl;
        return error;
    }

    fill_contiguous_tensor_with_random_data(tensor, -1.f, 1.f);

    GIGA_tensor_t upsampled = tensor;
    upsampled.dims[2] *= 2;
    upsampled.dims[3] *= 2;
    upsampled.device_id = device_id;
    upsampled.type = GT;
    upsampled.fp_shift = 0;

    GIGA_allocate_t upsampled_params;
    upsampled_params.memory_zone_id = 0;
    upsampled_params.offset = offset;
    offset += tensor_size_in_bytes(&upsampled);

    error = giga_allocate_tensor(&upsampled, &upsampled_params);
    if(error != GIGA_Success)
    {
        std::cerr << "Error allocating tensor upsampled" << std::endl;
        return error;
    }

    GIGA_upsample_t upsample_params;
    upsample_params.factor = 2;

    const size_t start = usec_timer();
    for(int it = 0 ; it < nb_runs ; ++it)
    {
        error = giga_upsample(&upsample_params, &tensor, &upsampled);
        if(error != GIGA_Success)
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
            std::cerr << "Error performing giga_upsample" << std::endl;
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
    std::cout << double(end - start) / nb_runs << "µs per call" << std::endl;

    error = giga_release_tensor(&upsampled);
    if(error != GIGA_Success)
    {
        std::cerr << "Error releasing tensor upsampled" << std::endl;
        return error;
    }

    error = giga_release_tensor(&tensor);
    if(error != GIGA_Success)
    {
        std::cerr << "Error releasing tensor tensor" << std::endl;
        return error;
    }

    on_error_message.clear();

    return GIGA_Success;
}

int main()
{
    GIGA_error error = GIGA_Success;

    const int nb_runs = 10;

#define EARLY_ABORT() throw std::runtime_error("Error")

    try
    {
        if((error = upsample_benchmark(GIGA_Float32, nb_runs)) != GIGA_Success)
            EARLY_ABORT();
        if((error = upsample_benchmark(GIGA_Float16, nb_runs)) != GIGA_Success)
            EARLY_ABORT();
        if((error = upsample_benchmark(GIGA_SFixed8, nb_runs)) != GIGA_Success)
            EARLY_ABORT();
        if((error = upsample_benchmark(GIGA_SFixed16, nb_runs)) != GIGA_Success)
            EARLY_ABORT();
        if((error = upsample_benchmark(GIGA_UFixed8, nb_runs)) != GIGA_Success)
            EARLY_ABORT();
        if((error = upsample_benchmark(GIGA_UFixed16, nb_runs)) != GIGA_Success)
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

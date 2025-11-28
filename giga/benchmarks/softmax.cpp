/*!
 * (C) 2025 Airbus copyright all rights reserved
 * \date 17/01/2025
 */
#include <giga/giga.h>

#include "../tests/utils.h"

GIGA_error softmax_benchmark(GIGA_data_type i_GT, GIGA_data_type o_GT, int nb_runs, uint8_t in_shift = 0, uint8_t out_shift = 0)
{
    ScopedMessage on_error_message(std::string("Error on ")
                                   + "Softmax, in " + giga_data_type_str(i_GT)
                                   + ", out " + giga_data_type_str(o_GT));

    std::cout << "Softmax, in " << giga_data_type_str(i_GT)
              << ", out " << giga_data_type_str(o_GT) << " : " << std::flush;

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
    tensor.nb_dims = 2;
    tensor.dims[0] = 1;
    tensor.dims[1] = 1024;
    tensor.device_id = device_id;
    tensor.data = NULL;
    tensor.type = i_GT;
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

    GIGA_tensor_t softmaxed = tensor;
    softmaxed.device_id = device_id;
    softmaxed.data = NULL;
    softmaxed.type = o_GT;
    softmaxed.fp_shift = 0;

    GIGA_allocate_t softmaxed_params;
    softmaxed_params.memory_zone_id = 0;
    softmaxed_params.offset = offset;
    offset += tensor_size_in_bytes(&softmaxed);
    error = giga_allocate_tensor(&softmaxed, &softmaxed_params);
    if(error != GIGA_Success)
    {
        std::cerr << "Error allocating tensor softmaxed" << std::endl;
        return error;
    }

    const size_t start = usec_timer();
    for(int it = 0 ; it < nb_runs ; ++it)
    {
        GIGA_softmax_t softmax_params;
        error = giga_softmax(&softmax_params, &tensor, &softmaxed);
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
            std::cerr << "Error performing giga_softmax" << std::endl;
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

    error = giga_release_tensor(&tensor);
    if(error != GIGA_Success)
    {
        std::cerr << "Error releasing tensor tensor" << std::endl;
        return error;
    }

    error = giga_release_tensor(&softmaxed);
    if(error != GIGA_Success)
    {
        std::cerr << "Error releasing tensor softmaxed" << std::endl;
        return error;
    }

    on_error_message.clear();

    return GIGA_Success;
}

int main()
{
    GIGA_error error = GIGA_Success;

    const int nb_runs = 10000;

#define EARLY_ABORT() throw std::runtime_error("Error")

    try
    {
        if((error = softmax_benchmark(GIGA_Float32, GIGA_Float32, nb_runs)) != GIGA_Success)
            EARLY_ABORT();
        if((error = softmax_benchmark(GIGA_Float16, GIGA_Float16, nb_runs)) != GIGA_Success)
            EARLY_ABORT();
        if((error = softmax_benchmark(GIGA_SFixed8, GIGA_SFixed8, nb_runs, 4, 4)) != GIGA_Success)
            EARLY_ABORT();
        if((error = softmax_benchmark(GIGA_SFixed16, GIGA_SFixed16, nb_runs, 4, 4)) != GIGA_Success)
            EARLY_ABORT();
        if((error = softmax_benchmark(GIGA_UFixed8, GIGA_UFixed8, nb_runs, 4, 4)) != GIGA_Success)
            EARLY_ABORT();
        if((error = softmax_benchmark(GIGA_UFixed16, GIGA_UFixed16, nb_runs, 4, 4)) != GIGA_Success)
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

/*!
 * (C) 2025 Airbus copyright all rights reserved
 * \date 16/01/2025
 */

#include <giga/giga.h>

#include "utils.h"

GIGA_error softmax_test(GIGA_data_type i_GT, GIGA_data_type o_GT)
{
    ScopedMessage msg;

    msg << "Softmax, in " << giga_data_type_str(i_GT)
        << ", out " << giga_data_type_str(o_GT) << "\n";

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
    tensor.dims[1] = 3;
    tensor.dims[2] = 5;
    tensor.dims[3] = 5;
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

    float data[3*5*5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                         -1.0f, -2.0f, -3.0f, -4.0f, -5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,

                         -1.0f, -2.0f, -3.0f, -4.0f, -5.0f,
                         -1.0f, -2.0f, -3.0f, -4.0f, -5.0f,
                         -1.0f, -2.0f, -3.0f, -4.0f, -5.0f,
                         -1.0f, -2.0f, -3.0f, -4.0f, -5.0f,
                         -1.0f, -2.0f, -3.0f, -4.0f, -5.0f,

                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                         -1.0f, -2.0f, -3.0f, -4.0f, -5.0f,
                         -11.0f,-22.0f, -33.0f, -44.0f, -55.0f,
                         10.0f, 20.0f, 30.0f, 40.0f, 50.0f};

    fill_4d_tensor(data, tensor);

    GIGA_tensor_t softmaxed;
    softmaxed.nb_dims = 4;
    softmaxed.dims[0] = 1;
    softmaxed.dims[1] = 3;
    softmaxed.dims[2] = 5;
    softmaxed.dims[3] = 5;
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

    // Fill output with garbage to make sure we don't test an unwritten tensor
    fill_contiguous_tensor_with_random_data(softmaxed, 0.f, 255.f);

    GIGA_softmax_t softmax_params;
    error = giga_softmax(&softmax_params, &tensor, &softmaxed);
    if(error != GIGA_Success)
    {
        if (error == GIGA_Unimplemented_Type)
        {
            std::cout << "Type not implemented!" << std::endl;
            msg.clear();
            return GIGA_Success;
        }
        std::cerr << "Error performing giga_softmax" << std::endl;
        return error;
    }

    print_tensor(msg, softmaxed, "giga_conv2d output");

    GIGA_tensor_t result;
    result.nb_dims = 4;
    result.dims[0] = 1;
    result.dims[1] = 3;
    result.dims[2] = 5;
    result.dims[3] = 5;
    result.device_id = device_id;
    result.type = o_GT;
    result.data = NULL;
    result.fp_shift = 0;

    float data_result[3*5*5] = {4.6831e-01f, 4.9546e-01f, 4.9938e-01f, 4.9992e-01f, 4.9999e-01f,
                                4.2232e-01f, 4.6831e-01f, 4.8786e-01f, 4.9546e-01f, 4.9832e-01f,
                                3.3333e-01f, 3.3333e-01f, 3.3333e-01f, 3.3333e-01f, 3.3333e-01f,
                                8.8079e-01f, 9.8201e-01f, 9.9753e-01f, 9.9966e-01f, 9.9995e-01f,
                                1.2339e-04f, 1.5230e-08f, 1.8795e-12f, 2.3195e-16f, 2.8625e-20f,

                                6.3379e-02f, 9.0747e-03f, 1.2378e-03f, 1.6770e-04f, 2.2699e-05f,
                                1.5536e-01f, 6.3379e-02f, 2.4289e-02f, 9.0747e-03f, 3.3577e-03f,
                                3.3333e-01f, 3.3333e-01f, 3.3333e-01f, 3.3333e-01f, 3.3333e-01f,
                                1.1920e-01f, 1.7986e-02f, 2.4726e-03f, 3.3535e-04f, 4.5398e-05f,
                                1.6699e-05f, 2.7895e-10f, 4.6589e-15f, 7.7811e-20f, 1.2996e-24f,

                                4.6831e-01f, 4.9546e-01f, 4.9938e-01f, 4.9992e-01f, 4.9999e-01f,
                                4.2232e-01f, 4.6831e-01f, 4.8786e-01f, 4.9546e-01f, 4.9832e-01f,
                                3.3333e-01f, 3.3333e-01f, 3.3333e-01f, 3.3333e-01f, 3.3333e-01f,
                                5.4118e-06f, 3.7072e-11f, 2.3138e-16f, 1.4247e-21f, 8.7561e-27f,
                                9.9986e-01f, 1.0000e+00f, 1.0000e+00f, 1.0000e+00f, 1.0000e+00};

    GIGA_allocate_t result_params;
    result_params.memory_zone_id = 0;
    result_params.offset = offset;
    offset += tensor_size_in_bytes(&result);
    error = giga_allocate_tensor(&result, &result_params);
    if(error != GIGA_Success)
    {
        std::cerr << "Error allocating tensor result" << std::endl;
        return error;
    }

    fill_4d_tensor(data_result, result);
    print_tensor(msg, result, "expected output");

    if(!compare_tensors(&softmaxed, &result, 0.01))
    {
        std::cerr << "Error comparing tensors softmaxed and result" << std::endl;
        return GIGA_Unknown_Error;
    }

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

    error = giga_release_tensor(&result);
    if(error != GIGA_Success)
    {
        std::cerr << "Error releasing tensor result" << std::endl;
        return error;
    }

    const std::string line = msg.message();
    msg.replaceMessage(line.substr(0, line.find('\n')));

    return GIGA_Success;
}

int main()
{
    GIGA_error error = GIGA_Success;

#define EARLY_ABORT() throw std::runtime_error("Error")

    try
    {
        if((error = softmax_test(GIGA_Float32, GIGA_Float32)) != GIGA_Success)
            EARLY_ABORT();
        if((error = softmax_test(GIGA_Float16, GIGA_Float16)) != GIGA_Success)
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

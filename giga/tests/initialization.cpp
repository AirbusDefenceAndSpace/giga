/*!
 * (C) 2025 Airbus copyright all rights reserved
 * \date 16/01/2025
 */

#include <giga/giga.h>
#include <stdexcept>
#include <iostream>

int main()
{
    GIGA_error error = GIGA_Success;

#define EARLY_ABORT() throw std::runtime_error("Error")

    try
    {
        uint32_t device_id = giga_get_default_device_id(&error);

        if(error != GIGA_Success)
        {
            std::cerr << "Error getting default device id" << std::endl;
            EARLY_ABORT();
        }

        uint32_t device_ids[8];
        uint32_t nb_devices = 0;
        if((error = giga_list_devices(device_ids, &nb_devices)) != GIGA_Success)
        {
            std::cerr << "Error listing devices" << std::endl;
            EARLY_ABORT();
        }

       if((error = giga_initialize_device(device_id)) != GIGA_Success)
       {
           std::cerr << "Error initializing device" << std::endl;
           EARLY_ABORT();
       }
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

/*!
 * (C) 2025 Airbus copyright all rights reserved
 * \date 16/01/2025
 */

#include <giga/float16.h>
#include <iostream>

int main()
{
    half h1(-0.005);
    half h2(0.5);

    half result = h1*h2;
    std::cout << result << std::endl;

    result = h1+h2;
    std::cout << result << std::endl;

    h1+=h2;
    std::cout << h1 << std::endl;

    return 0;
}

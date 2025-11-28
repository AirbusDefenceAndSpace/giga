/*!
 * (C) 2025 Airbus copyright all rights reserved
 * \author Roland Brochard (roland.brochard@airbus.com)
 * \date 15/01/2025
 *
 * Baseline CPU implementation of the GIGA API
 *
 */

#ifndef GIGA_CPU_UTILS_H_9c68eb97c15ee38145df875a89b2ccbd
#define GIGA_CPU_UTILS_H_9c68eb97c15ee38145df875a89b2ccbd

#include <giga/float16.h>

template<typename T>
inline T shift(T value, int shift)
{
    if (shift > 0)
    {
        if (value < 0)
            return -( (-value) << shift); //Beware of undefined behaviours of bitshift with negative numbers
        else
            return value << shift;
    }
    else
        return value >> -shift;
}

template<>
inline float shift<float>(float value, int shift)
{
    return value;
}

template<>
inline half shift<half>(half value, int shift)
{
    return value;
}

// Simple case with a single type
#define GIGA_TYPE_TEMPLATED_CASE(func, type, ...) \
case type:\
    ret = func <type>(__VA_ARGS__);\
    break;

#define GIGA_CALL_TEMPLATED_FUNC_ON_TENSOR(func, type, ...)\
switch(type)\
{\
GIGA_TYPE_TEMPLATED_CASE(func, GIGA_Float16, __VA_ARGS__ ) \
GIGA_TYPE_TEMPLATED_CASE(func, GIGA_Float32, __VA_ARGS__ ) \
GIGA_TYPE_TEMPLATED_CASE(func, GIGA_SFixed4, __VA_ARGS__ ) \
GIGA_TYPE_TEMPLATED_CASE(func, GIGA_SFixed8, __VA_ARGS__ ) \
GIGA_TYPE_TEMPLATED_CASE(func, GIGA_SFixed16, __VA_ARGS__ ) \
GIGA_TYPE_TEMPLATED_CASE(func, GIGA_UFixed4, __VA_ARGS__ ) \
GIGA_TYPE_TEMPLATED_CASE(func, GIGA_UFixed8, __VA_ARGS__ ) \
GIGA_TYPE_TEMPLATED_CASE(func, GIGA_UFixed16, __VA_ARGS__ ) \
default:\
    ret = GIGA_Unimplemented_Type;\
}

// More complex case with 2 types
#define GIGA_TYPE_TEMPLATED_CASE_2T_TYPE2(func, type1, type2, ...) \
case type2:\
    ret = func <type1, type2>(__VA_ARGS__);\
    break;

#define GIGA_TYPE_TEMPLATED_CASE_2T_TYPE1(func, type1, type2, ...)\
case type1:\
    switch(type2)\
    {\
    GIGA_TYPE_TEMPLATED_CASE_2T_TYPE2(func, type1, GIGA_Float16, __VA_ARGS__ ) \
    GIGA_TYPE_TEMPLATED_CASE_2T_TYPE2(func, type1, GIGA_Float32, __VA_ARGS__ ) \
    GIGA_TYPE_TEMPLATED_CASE_2T_TYPE2(func, type1, GIGA_SFixed4, __VA_ARGS__ ) \
    GIGA_TYPE_TEMPLATED_CASE_2T_TYPE2(func, type1, GIGA_SFixed8, __VA_ARGS__ ) \
    GIGA_TYPE_TEMPLATED_CASE_2T_TYPE2(func, type1, GIGA_SFixed16, __VA_ARGS__ ) \
    GIGA_TYPE_TEMPLATED_CASE_2T_TYPE2(func, type1, GIGA_UFixed4, __VA_ARGS__ ) \
    GIGA_TYPE_TEMPLATED_CASE_2T_TYPE2(func, type1, GIGA_UFixed8, __VA_ARGS__ ) \
    GIGA_TYPE_TEMPLATED_CASE_2T_TYPE2(func, type1, GIGA_UFixed16, __VA_ARGS__ ) \
    default:\
        ret = GIGA_Unimplemented_Type;\
    }\
    break;

#define GIGA_CALL_TEMPLATED_FUNC_ON_2_TENSORS(func, type1, type2, ...)\
switch(type1)\
{\
GIGA_TYPE_TEMPLATED_CASE_2T_TYPE1(func, GIGA_Float16, type2, __VA_ARGS__ ) \
GIGA_TYPE_TEMPLATED_CASE_2T_TYPE1(func, GIGA_Float32, type2, __VA_ARGS__ ) \
GIGA_TYPE_TEMPLATED_CASE_2T_TYPE1(func, GIGA_SFixed4, type2, __VA_ARGS__ ) \
GIGA_TYPE_TEMPLATED_CASE_2T_TYPE1(func, GIGA_SFixed8, type2, __VA_ARGS__ ) \
GIGA_TYPE_TEMPLATED_CASE_2T_TYPE1(func, GIGA_SFixed16, type2, __VA_ARGS__ ) \
GIGA_TYPE_TEMPLATED_CASE_2T_TYPE1(func, GIGA_UFixed4, type2, __VA_ARGS__ ) \
GIGA_TYPE_TEMPLATED_CASE_2T_TYPE1(func, GIGA_UFixed8, type2, __VA_ARGS__ ) \
GIGA_TYPE_TEMPLATED_CASE_2T_TYPE1(func, GIGA_UFixed16, type2, __VA_ARGS__ ) \
default:\
    ret = GIGA_Unimplemented_Type;\
}

// Tedious case with 3 types
#define GIGA_TYPE_TEMPLATED_CASE_3T_TYPE3(func, type1, type2, type3, ...) \
case type3:\
    ret = func <type1, type2, type3>(__VA_ARGS__);\
    break;

#define GIGA_TYPE_TEMPLATED_CASE_3T_TYPE2(func, type1, type2, type3, ...)\
case type2:\
    switch(type3)\
    {\
    GIGA_TYPE_TEMPLATED_CASE_3T_TYPE3(func, type1, type2, GIGA_Float16, __VA_ARGS__ ) \
    GIGA_TYPE_TEMPLATED_CASE_3T_TYPE3(func, type1, type2, GIGA_Float32, __VA_ARGS__ ) \
    GIGA_TYPE_TEMPLATED_CASE_3T_TYPE3(func, type1, type2, GIGA_SFixed4, __VA_ARGS__ ) \
    GIGA_TYPE_TEMPLATED_CASE_3T_TYPE3(func, type1, type2, GIGA_SFixed8, __VA_ARGS__ ) \
    GIGA_TYPE_TEMPLATED_CASE_3T_TYPE3(func, type1, type2, GIGA_SFixed16, __VA_ARGS__ ) \
    GIGA_TYPE_TEMPLATED_CASE_3T_TYPE3(func, type1, type2, GIGA_UFixed4, __VA_ARGS__ ) \
    GIGA_TYPE_TEMPLATED_CASE_3T_TYPE3(func, type1, type2, GIGA_UFixed8, __VA_ARGS__ ) \
    GIGA_TYPE_TEMPLATED_CASE_3T_TYPE3(func, type1, type2, GIGA_UFixed16, __VA_ARGS__ ) \
    default:\
        ret = GIGA_Unimplemented_Type;\
    }\
    break;

#define GIGA_TYPE_TEMPLATED_CASE_3T_TYPE1(func, type1, type2, type3, ...)\
case type1:\
    switch(type2)\
    {\
    GIGA_TYPE_TEMPLATED_CASE_3T_TYPE2(func, type1, GIGA_Float16, type3, __VA_ARGS__ ) \
    GIGA_TYPE_TEMPLATED_CASE_3T_TYPE2(func, type1, GIGA_Float32, type3, __VA_ARGS__ ) \
    GIGA_TYPE_TEMPLATED_CASE_3T_TYPE2(func, type1, GIGA_SFixed4, type3, __VA_ARGS__ ) \
    GIGA_TYPE_TEMPLATED_CASE_3T_TYPE2(func, type1, GIGA_SFixed8, type3, __VA_ARGS__ ) \
    GIGA_TYPE_TEMPLATED_CASE_3T_TYPE2(func, type1, GIGA_SFixed16, type3, __VA_ARGS__ ) \
    GIGA_TYPE_TEMPLATED_CASE_3T_TYPE2(func, type1, GIGA_UFixed4, type3, __VA_ARGS__ ) \
    GIGA_TYPE_TEMPLATED_CASE_3T_TYPE2(func, type1, GIGA_UFixed8, type3, __VA_ARGS__ ) \
    GIGA_TYPE_TEMPLATED_CASE_3T_TYPE2(func, type1, GIGA_UFixed16, type3, __VA_ARGS__ ) \
    default:\
        ret = GIGA_Unimplemented_Type;\
    }\
    break;

#define GIGA_CALL_TEMPLATED_FUNC_ON_3_TENSORS(func, type1, type2, type3, ...)\
switch(type1)\
{\
GIGA_TYPE_TEMPLATED_CASE_3T_TYPE1(func, GIGA_Float16, type2, type3, __VA_ARGS__ ) \
GIGA_TYPE_TEMPLATED_CASE_3T_TYPE1(func, GIGA_Float32, type2, type3, __VA_ARGS__ ) \
GIGA_TYPE_TEMPLATED_CASE_3T_TYPE1(func, GIGA_SFixed4, type2, type3, __VA_ARGS__ ) \
GIGA_TYPE_TEMPLATED_CASE_3T_TYPE1(func, GIGA_SFixed8, type2, type3, __VA_ARGS__ ) \
GIGA_TYPE_TEMPLATED_CASE_3T_TYPE1(func, GIGA_SFixed16, type2, type3, __VA_ARGS__ ) \
GIGA_TYPE_TEMPLATED_CASE_3T_TYPE1(func, GIGA_UFixed4, type2, type3, __VA_ARGS__ ) \
GIGA_TYPE_TEMPLATED_CASE_3T_TYPE1(func, GIGA_UFixed8, type2, type3, __VA_ARGS__ ) \
GIGA_TYPE_TEMPLATED_CASE_3T_TYPE1(func, GIGA_UFixed16, type2, type3, __VA_ARGS__ ) \
default:\
    ret = GIGA_Unimplemented_Type;\
}

// Versions with all types equal
#define GIGA_TYPE_TEMPLATED_CASE_2T_ST(func, type, ...) \
case type:\
    ret = func <type, type>(__VA_ARGS__);\
    break;

#define GIGA_CALL_TEMPLATED_FUNC_ON_2_TENSORS_SAME_TYPE(func, type1, type2, ...)\
if (type1 != type2)\
    ret = GIGA_Unimplemented_Type;\
else\
    switch(type1)\
    {\
    GIGA_TYPE_TEMPLATED_CASE_2T_ST(func, GIGA_Float16, __VA_ARGS__ ) \
    GIGA_TYPE_TEMPLATED_CASE_2T_ST(func, GIGA_Float32, __VA_ARGS__ ) \
    GIGA_TYPE_TEMPLATED_CASE_2T_ST(func, GIGA_SFixed8, __VA_ARGS__ ) \
    GIGA_TYPE_TEMPLATED_CASE_2T_ST(func, GIGA_SFixed16, __VA_ARGS__ ) \
    GIGA_TYPE_TEMPLATED_CASE_2T_ST(func, GIGA_UFixed8, __VA_ARGS__ ) \
    GIGA_TYPE_TEMPLATED_CASE_2T_ST(func, GIGA_UFixed16, __VA_ARGS__ ) \
    default:\
        ret = GIGA_Unimplemented_Type;\
    }

#define TYPES3(T0,T1,T2)    (((T0) << 16) | ((T1) << 8) | (T2))

#define GIGA_TYPE_TEMPLATED_CASE_3T_ST(func, type, ...) \
case type:\
    ret = func <type, type, type>(__VA_ARGS__);\
    break;

#define GIGA_CALL_TEMPLATED_FUNC_ON_3_TENSORS_SAME_TYPE(func, type1, type2, type3, ...)\
if (type1 != type2 || type1 != type3)\
    ret = GIGA_Unimplemented_Type;\
else\
    switch(type1)\
    {\
    GIGA_TYPE_TEMPLATED_CASE_3T_ST(func, GIGA_Float16, __VA_ARGS__ ) \
    GIGA_TYPE_TEMPLATED_CASE_3T_ST(func, GIGA_Float32, __VA_ARGS__ ) \
    GIGA_TYPE_TEMPLATED_CASE_3T_ST(func, GIGA_SFixed8, __VA_ARGS__ ) \
    GIGA_TYPE_TEMPLATED_CASE_3T_ST(func, GIGA_SFixed16, __VA_ARGS__ ) \
    GIGA_TYPE_TEMPLATED_CASE_3T_ST(func, GIGA_UFixed8, __VA_ARGS__ ) \
    GIGA_TYPE_TEMPLATED_CASE_3T_ST(func, GIGA_UFixed16, __VA_ARGS__ ) \
    default:\
        ret = GIGA_Unimplemented_Type;\
    }

#define GIGA_TYPE_TEMPLATED_CASE_3T_SK(func, type1, type2, type3, ...) \
case TYPES3(type1, type2, type3):\
    ret = func <type1, type2, type3>(__VA_ARGS__);\
    break;

#define GIGA_CALL_TEMPLATED_FUNC_ON_3_TENSORS_SIGNED_KERNELS(func, type1, type2, type3, ...)\
switch(TYPES3(type1, type2, type3))\
{\
GIGA_TYPE_TEMPLATED_CASE_3T_SK(func, GIGA_Float16, GIGA_Float16, GIGA_Float16, __VA_ARGS__ ) \
GIGA_TYPE_TEMPLATED_CASE_3T_SK(func, GIGA_Float32, GIGA_Float32, GIGA_Float32, __VA_ARGS__ ) \
GIGA_TYPE_TEMPLATED_CASE_3T_SK(func, GIGA_SFixed8, GIGA_SFixed8, GIGA_SFixed8, __VA_ARGS__ ) \
GIGA_TYPE_TEMPLATED_CASE_3T_SK(func, GIGA_SFixed16, GIGA_SFixed16, GIGA_SFixed16, __VA_ARGS__ ) \
GIGA_TYPE_TEMPLATED_CASE_3T_SK(func, GIGA_UFixed8, GIGA_UFixed8, GIGA_UFixed8, __VA_ARGS__ ) \
GIGA_TYPE_TEMPLATED_CASE_3T_SK(func, GIGA_UFixed16, GIGA_UFixed16, GIGA_UFixed16, __VA_ARGS__ ) \
GIGA_TYPE_TEMPLATED_CASE_3T_SK(func, GIGA_UFixed8, GIGA_UFixed8, GIGA_SFixed8, __VA_ARGS__ ) \
GIGA_TYPE_TEMPLATED_CASE_3T_SK(func, GIGA_UFixed16, GIGA_UFixed16, GIGA_SFixed16, __VA_ARGS__ ) \
GIGA_TYPE_TEMPLATED_CASE_3T_SK(func, GIGA_UFixed8, GIGA_SFixed8, GIGA_SFixed8, __VA_ARGS__ ) \
GIGA_TYPE_TEMPLATED_CASE_3T_SK(func, GIGA_UFixed16, GIGA_SFixed16, GIGA_SFixed16, __VA_ARGS__ ) \
default:\
    ret = GIGA_Unimplemented_Type;\
}

#endif

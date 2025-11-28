/*!
 * (C) 2025 Airbus copyright all rights reserved
 * Created on Wed Jan 15 2025
 *
 */
#ifndef GIGA_FLOAT16_H_e76aa9209c0fc37cb75514cdaa50beff
#define GIGA_FLOAT16_H_e76aa9209c0fc37cb75514cdaa50beff

#include <stdint.h>
#ifdef __cplusplus
#include <type_traits>
#endif

/*! \brief This class implements a half precision floatting point type
 *
 * This is used for storage only! Only conversions to and from float are supported (in C++)!
 */
typedef struct half
{
#ifdef __cplusplus
    inline half()  {}
    inline half(const half &v) : data(v.data)  {}
    inline half(const float &v);

    inline half &operator=(const half &v);

    inline operator float() const;
#endif

    // Unsigned type is mandatory in order to avoid sign aware shifts!!
    uint16_t data;
} half;

#ifdef __cplusplus
half::half(const float &v)
{
    union
    {
        float f;
        uint32_t i;
    } u;
    u.f = v;

    int32_t e = int32_t((u.i >> 23) & 0xFFU);
    if (e == 0)     // 0 or denorm
    {
        data = 0;
        return;
    }
    e -= 127;
    if (e < -14)        // Flush to 0
    {
        data = 0;
        return;
    }
    else if (e > 15)    // Flush to infinity
        e = 31;
    else
        e += 15;
    // Sign
    data = (u.i >> 16) & 0x8000U;
    if (e != 31)
    {
        const uint32_t m = (u.i  >> 13) & 0x3FFU;     // Keep only 10bits
        data |= m;
    }
    data |= (uint32_t(e) << 10);
}

half &half::operator=(const half &v)
{
    data = v.data;
    return *this;
}

half::operator float() const
{
    if (data == 0)
        return 0.f;

    union
    {
        float f;
        uint32_t i;
    } u;
    const uint32_t v = data;
    // Sign
    u.i = (v & 0x8000U) << 16;
    uint32_t e = ((v >> 10) & 0x1FU) + uint32_t(127 - 15);
    if (e == 31 + (127 - 15))   // Infinity
    {
        e = 0xFF;
        u.i |= e << 23;
    }
    else
    {
        u.i |= e << 23;
        const uint32_t m = v & 0x3FFU;
        u.i |= m << 13;
    }

    return u.f;
}

template<typename T, std::enable_if_t<std::is_integral<T>::value>>
half operator*(const T& ui, const half& h)
{
    return ui*float(h);
}

template<typename T, std::enable_if_t<std::is_integral<T>::value>>
half operator*(const half& h, const T& ui)
{
    return ui*h;
}

inline half& operator+=(half& h1, const half& h2)
{
    h1 = float(h1)+float(h2);
    return h1;
}

template<typename T, std::enable_if_t<std::is_integral<T>::value>>
half operator+(const T& h1, const half& h2)
{
    return h1+float(h2);
}

template<typename T, std::enable_if_t<std::is_integral<T>::value>>
half operator+(const half& h1, const T& h2)
{
    return h2+h1;
}
#endif

#endif

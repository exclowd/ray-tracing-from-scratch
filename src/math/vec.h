#ifndef vec_H
#define vec_H

#include <array>
#include <cmath>
#include <iostream>

template <typename T, size_t N>
struct vec {
   public:
    vec() : m_data{0, 0, 0} {}
    vec(T e0, T e1, T e2) : m_data{e0, e1, e2} {}

    T x() const { return m_data[0]; }
    T y() const { return m_data[1]; }
    T z() const { return m_data[2]; }

    vec operator-() const { return vec(-m_data[0], -m_data[1], -m_data[2]); }
    T operator[](int i) const { return m_data[i]; }
    T& operator[](int i) { return m_data[i]; }

    vec& operator+=(const vec<T>& v) {
        for (int i = 0; i < N; ++i) {
            m_data[i] += v.m_data[i];
        }
        return *this;
    }

    vec& operator-=(const vec<T>& v) {
        for (int i = 0; i < N; ++i) {
            m_data[i] -= v.m_data[i];
        }
        return *this;
    }

    vec& operator*=(const T t) {
        for (int i = 0; i < N; ++i) {
            m_data[i] *= t;
        }
        return *this;
    }

    vec& operator/=(const T t) {
        return *this *= 1 / t;
    }

    double length() const {
        return std::sqrt(length_squared());
    }

    T length_squared() const {
        T a{};
        for (int i = 0; i < N; i++) {
            a += m_data[i] * m_data[i];
        }
        return a;
    }

   public:
    std::array<T, N> m_data;
};

// Type aliases for vec
using point3 = vec<double>;  // 3D point
using color = vec<uint8_t>;  // RGB color

#endif
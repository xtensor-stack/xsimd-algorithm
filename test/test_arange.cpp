/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
 * Martin Renou                                                             *
 * Copyright (c) QuantStack                                                 *
 * Copyright (c) Serge Guelton                                              *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include "xsimd_algorithm/stl/arange.hpp"

#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE

#include "doctest/doctest.h"

#include <cstddef>
#include <vector>

#if XSIMD_WITH_NEON && !XSIMD_WITH_NEON64
#define ARANGE_TYPES int, float
#else
#define ARANGE_TYPES int, long, float, double
#endif

template <typename Type>
struct arange_test
{
    using vector = std::vector<Type>;
    static constexpr std::size_t simd_size = xsimd::batch<Type>::size;

    static vector fill_expected(Type start, size_t size, Type step = Type { 1 })
    {
        vector result(size);

        for (size_t i = 0; i < size; ++i)
        {
            result[i] = start + (i * step);
        }

        return result;
    }

    void test_arange_aligned(Type init_value, size_t size, Type step) const
    {
        vector c(size);
        xsimd::arange(c.begin(), c.end(), init_value, step);
        const vector expected = fill_expected(init_value, size, step);

        CHECK(std::equal(expected.begin(), expected.end(), c.begin()));
        CHECK(expected.size() == c.size());
    }

    void test_arange_misaligned(Type init_value, size_t size, Type step) const
    {
        const size_t missalignment = 1;
        vector c(size + missalignment * 2);
        auto first = c.begin() + missalignment;
        auto last = first + size;

        xsimd::arange(first, last, init_value, step);

        const vector expected = fill_expected(init_value, size, step);
        CHECK(std::equal(expected.begin(), expected.end(), first));
        CHECK(expected.size() == static_cast<size_t>(std::distance(first, last)));
    }
};

TEST_CASE_TEMPLATE("arange test", T, ARANGE_TYPES)
{
    using Test = arange_test<T>;

    const std::array test_sizes {
        size_t { 0 },
        size_t { 1 },
        Test::simd_size - 1,
        Test::simd_size,
        Test ::simd_size + 1,
    };

    const std::array init_values {
        T { -1 },
        T { 0 },
        T { 1 },
    };

    const std::array steps { -2, -1, 0, 1, 2 };

    Test test;

    size_t sub_case_id = 0;
    for (const auto size : test_sizes)
    {
        for (const auto init_value : init_values)
        {
            for (const auto step : steps)
            {
                const std::string aligned_subcase_name = "Aligned" + std::to_string(sub_case_id++) + ", size: " + std::to_string(size) + ", init_value: " + std::to_string(init_value) + ", step: " + std::to_string(step);
                SUBCASE(aligned_subcase_name.c_str())
                {
                    test.test_arange_aligned(init_value, size, step);
                }
                const std::string misaligned_subcase_name = "Misaligned" + std::to_string(sub_case_id++) + ", size: " + std::to_string(size) + ", init_value: " + std::to_string(init_value) + ", step: " + std::to_string(step);
                SUBCASE(misaligned_subcase_name.c_str())
                {
                    test.test_arange_misaligned(init_value, size, step);
                }
            }
        }
    }
}

#endif

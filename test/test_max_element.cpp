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

#include "xsimd_algorithm/stl/max_element.hpp"

#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE

#include "doctest/doctest.h"

#include <cstddef>
#include <vector>

#if XSIMD_WITH_NEON && !XSIMD_WITH_NEON64
#define MAX_ELEMENT_TYPES int, float
#else
#define MAX_ELEMENT_TYPES int, long, float, double
#endif

template <typename Type>
struct max_element_test
{
    using vector = std::vector<Type>;
    static constexpr std::size_t simd_size = xsimd::batch<Type>::size;

    static vector fill_data(size_t size, Type max, size_t position)
    {
        vector result(size,Type(0));
        
        if (position < size)
        {
            result[position] = max;
        }

        return result;
    }

    void test_max_element_aligned( size_t size, size_t position) const
    {
        const auto maximum = size -1;
        const vector data = fill_data(size, maximum, position);
        const auto result = xsimd::max_element(data.begin(), data.end());
        const auto expected = std::max_element(data.begin(), data.end());
        CHECK(result == expected);
    }

    void test_max_element_misaligned(size_t size, size_t position) const
    {
        const auto maximum = size -1;
        const size_t missalignment = 1;
        const vector data = fill_data(size + missalignment * 2, maximum, position);
        auto first = data.begin() + missalignment;
        auto last = first + size;

        const auto result = xsimd::max_element(first, last);
        const auto expected = std::max_element(first, last);

        CHECK(result == expected);
    }
};

TEST_CASE_TEMPLATE("max_element test", T, MAX_ELEMENT_TYPES)
{
    using Test = max_element_test<T>;

    const std::array test_sizes {
        size_t { 0 },
        size_t { 1 },
        Test::simd_size - 1,
        Test::simd_size,
        Test ::simd_size + 1,
        Test ::simd_size *4,   
    };

    const std::array positions {
        size_t { 0 },
        size_t { 1 },
        Test::simd_size / 4,
        Test::simd_size / 2,
        Test::simd_size - 1,
    };

    Test test;

    size_t sub_case_id = 0;
    for (const auto size : test_sizes)
    {
        for (const auto position : positions)
        {
            const std::string aligned_subcase_name = "Aligned" + std::to_string(sub_case_id++) + ", size: " + std::to_string(size) + ", position: " + std::to_string(position);
            SUBCASE(aligned_subcase_name.c_str())
            {
                test.test_max_element_aligned(size, position);
            }
            const std::string misaligned_subcase_name = "Misaligned" + std::to_string(sub_case_id++) + ", size: " + std::to_string(size) + ", position: " + std::to_string(position);
            SUBCASE(misaligned_subcase_name.c_str())
            {
                test.test_max_element_misaligned(size, position);
            }
        }
    }
}
#endif
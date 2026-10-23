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

#ifndef XSIMD_ALGORITHMS_FIND_IF_NOT_HPP
#define XSIMD_ALGORITHMS_FIND_IF_NOT_HPP

#include "xsimd/xsimd.hpp"
#include "xsimd_algorithm/stl/find_if.hpp"

#include <iostream>
#include <iterator>

namespace xsimd
{
   template <class Arch = default_arch, class ContiguousIterator, class Predicate>
    ContiguousIterator find_if_not(ContiguousIterator first, ContiguousIterator last, Predicate&& p) noexcept
    {
        return find_if<Arch>(first, last, [p](const auto& x) { return !p(x); });
    }
}

#endif // XSIMD_ALGORITHMS_FIND_IF_NOT_HPP
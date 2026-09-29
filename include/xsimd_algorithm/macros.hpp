/****************************************************************************
 * Copyright (c) xsimd-algorithm contributors                               *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XSIMD_ALGORITHM_MACRO_HPP
#define XSIMD_ALGORITHM_MACRO_HPP

#if defined(_MSC_VER) && !defined(__clang__)
#define XSIMD_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define XSIMD_RESTRICT __restrict__
#else
#define XSIMD_RESTRICT
#endif

#endif

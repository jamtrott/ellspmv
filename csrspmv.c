/*
 * Benchmark program for CSR SpMV
 *
 * Copyright (C) 2025 James D. Trotter
 *
 * This program is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see
 * <https://www.gnu.org/licenses/>.
 *
 * Benchmarking program for sparse matrix-vector multiplication (SpMV)
 * with matrices in CSR format.
 *
 * Authors:
 *  James D. Trotter <james@simula.no>
 *  Sergej Breiter <breiter@nm.ifi.lmu.de>
 *
 *
 * History:
 *
 *  1.10 - 2025-06-03:
 *
 *   - Fix a performancce regression in v1.8 (e78e060e0), where he
 *     measured performance decreased especially with many threads due
 *     to an additional OpenMP barrier that was added.  The issue is
 *     fixed by stopping the timer after the first barrier, thus only
 *     timing the SpMV kernel
 *
 *  1.9 - 2025-02-06:
 *
 *   - place output vector and row pointers in sector 1 together with
 *     matrix nonzeros and column indices when using cache
 *     partitioning on A64FX.
 *
 *  1.8 - 2023-05-30:
 *
 *   - add option for performing warmup iterations
 *
 *  1.7 - 2023-05-13:
 *
 *   - add options for tuning L1 and L2 hardware prefetching on A64FX.
 *
 *   - add an option for sorting nonzeros within each row by column
 *
 *  1.6 - 2023-05-04:
 *
 *   - enable A64FX sector cache before the loop that repeatedly
 *     performs SpMV.
 *
 *  1.5 - 2023-04-15:
 *
 *   - add support for performance monitoring using PAPI
 *
 *   - minor fixes to A64FX sector cache configuration
 *
 *  1.4 - 2023-04-01:
 *
 *   - add options for specifying the number of rows/columns per thread
 *
 *   - fix bug in converting symmetric matrices to CSR
 *
 *  1.3 - 2023-03-30:
 *
 *   - allow symmetric matrices by adding two nonzeros to the CSR
 *     representation for every off-diagonal nonzero
 *
 *   - add an option for precomputing first and final rows for each
 *     thread for the load-balanced SpMV kernel (csrgemvnz)
 *
 *  1.2 - 2023-03-14:
 *
 *   - add a missing barrier for the CSR SpMV with balanced nonzeros
 *     to ensure that the execution time and performance is accurately
 *     displayed.
 *
 *  1.1 - 2023-03-02:
 *
 *   - add option for reading gzip-compressed Matrix Market files
 *
 *   - add compile-time option for selecting 32- or 64-bit integers
 *     for row/column offsets
 *
 *   - allow matrices and vectors with integer and pattern fields by
 *     converting integers to doubles or by setting the double
 *     precision floating-point matrix/vector values equal to one.
 *
 *   - align allocations to page boundaries if HAVE_ALIGNED_ALLOC is
 *     defined.
 *
 *   - add a CSR SpMV kernel where nonzeros are partitioned evenly
 *     among threads
 *
 *  1.0 — 2023-02-22:
 *
 *   - initial version based on ellspmv.
 */

#ifdef HAVE_PAPI
#include "papi_util.h"
#include <papi.h>
#endif

#include <errno.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef HAVE_LIBZ
#include <zlib.h>
#endif

#if defined(__ARM_FEATURE_SVE) && !defined(ARM_NOSVE)
#include <arm_sve.h>
#endif

#include <unistd.h>

#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <locale.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef IDXTYPEWIDTH
typedef int idx_t;
#define PRIdx "d"
#define IDX_T_MIN INT_MIN
#define IDX_T_MAX INT_MAX
#define parse_idx_t parse_int
#elif IDXTYPEWIDTH == 32
typedef int32_t idx_t;
#define PRIdx PRId32
#define IDX_T_MIN INT32_MIN
#define IDX_T_MAX INT32_MAX
#define parse_idx_t parse_int32_t
#elif IDXTYPEWIDTH == 64
typedef int64_t idx_t;
#define PRIdx PRId64
#define IDX_T_MIN INT64_MIN
#define IDX_T_MAX INT64_MAX
#define parse_idx_t parse_int64_t
#endif

#ifdef USE_A64FX_SECTOR_CACHE
#ifndef A64FX_SECTOR_CACHE_L2_WAYS
#define A64FX_SECTOR_CACHE_L2_WAYS 4
#endif
#endif

/*
 * setting model-specific registers to control prefetching on A64FX
 */
#if defined(__FCC_version__)
#define print_bits(f, x)                                                \
    do {                                                                \
        typeof(x) a__     = (x);                                        \
        char     *p__     = (char *)&a__ + sizeof(x) - 1;               \
        size_t    bytes__ = sizeof(x);                                  \
        printf(#x ": ");                                                \
        while (bytes__--) {                                             \
            char bits__ = 8;                                            \
            while (bits__--)                                            \
                fputc(*p__ & (1ULL << bits__) ? '1' : '0', (f));        \
            p__--;                                                      \
        }                                                               \
    } while (0)

static const size_t BITS_PER_LONG_LONG = CHAR_BIT * sizeof(long long);

#define ULL(x) x##ULL

#define GENMASK_ULL(h, l)                                                                \
    (((~ULL(0)) - (ULL(1) << (l)) + 1) & (~ULL(0) >> (BITS_PER_LONG_LONG - 1 - (h))))

#define ARM_READ_MRS(val_u64, reg_name)                                 \
    __asm__ volatile("mrs %[" #val_u64 "]," #reg_name : [val_u64] "=r"(val_u64) : :);

#define ARM_WRITE_MSR(val_u64, reg_name)                                \
    __asm__ volatile("msr " #reg_name ",%[" #val_u64 "]" : : [val_u64] "r"(val_u64) :);

// IMP_PF_CTRL_EL1 : S3_0_C11_C4_0
// set bit 62-63 makes IMP_PF_STREAM_DETECT_CTRL_EL0 accessible from EL0

#define A64FX_ENABLE_PF_CTRL()                                          \
    {                                                                   \
        uint64_t val = GENMASK_ULL(63, 62);                             \
        ARM_WRITE_MSR(val, S3_0_C11_C4_0);                              \
    }

#define A64FX_DISABLE_PF_CTRL()                                         \
    {                                                                   \
        uint64_t val = 0ULL;                                            \
        ARM_WRITE_MSR(val, S3_0_C11_C4_0);                              \
    }

// IMP_PF_STREAM_DETECT_CTRL_EL0: S3_3_C11_C4_0
//
// [63] V RW
// 1: A value of IMP_PF_STREAM_DETECT_CTRL_EL0 is valid
// 0: It operates by a set value of Default.
//
// [27:24] L1_DISTRW
// The distance of the hardware prefetch to L1 cache is specified.
// The prefetch distance for L1 is calculated as (L1_DIST * 256B).
// When L1_DIST = 0, the hardware prefetch for L1 operates by the
// default distance.
//
// [19:16] L2_DISTRW
// The distance of the hardware prefetch to L2 cache is specified.
// The prefetch distance for L2 is calculated as (L2_DIST * 1KB).
// When L2_DIST = 0, the hardware prefetch for L2 operates by the
// default distance.

// read the old value of the register into @val_u64 and
// set the l1 prefetch distance to @dist
#define A64FX_SET_PF_DST_L1(val_u64, dist)                                               \
    {                                                                   \
        ARM_READ_MRS(val_u64, S3_3_C11_C4_0);                           \
        uint64_t distu64 = dist;                                        \
        uint64_t val     = val_u64;                                     \
        uint64_t mask    = ~GENMASK_ULL(27, 24);                        \
        val              = val & mask;                                  \
        val              = val | (distu64 << 24) | (1ULL << 63);        \
        ARM_WRITE_MSR(val, S3_3_C11_C4_0);                              \
    }

// read the old value of the register into @val_u64 and
// set the l2 prefetch distance to @dist
#define A64FX_SET_PF_DST_L2(val_u64, dist)                              \
    {                                                                   \
        ARM_READ_MRS(val_u64, S3_3_C11_C4_0);                           \
        uint64_t distu64 = dist;                                        \
        uint64_t val     = val_u64;                                     \
        uint64_t mask    = ~GENMASK_ULL(19, 16);                        \
        val              = val & mask;                                  \
        val              = val | (distu64 << 16) | (1ULL << 63);        \
        ARM_WRITE_MSR(val, S3_3_C11_C4_0);                              \
    }

// reads the register value
#define A64FX_READ_PF_DST(val_u64)                                      \
    {                                                                   \
        ARM_READ_MRS(val_u64, S3_3_C11_C4_0);                           \
    }

// sets the register value
#define A64FX_WRITE_PF_DST(regval)                                      \
    {                                                                   \
        uint64_t val = regval;                                          \
        ARM_WRITE_MSR(val, S3_3_C11_C4_0);                              \
    }
#endif

const char * program_name = "csrspmv";
const char * program_version = "1.10";
const char * program_copyright =
    "Copyright (C) 2025 James D. Trotter";
const char * program_license =
    "License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>\n"
    "This is free software: you are free to change and redistribute it.\n"
    "There is NO WARRANTY, to the extent permitted by law.";
const char * program_invocation_name;
const char * program_invocation_short_name;

enum partition
{
    partition_rows,
    partition_nonzeros,
};

/**
 * ‘program_options’ contains data to related program options.
 */
struct program_options
{
    char * Apath;
    char * xpath;
    char * ypath;
#ifdef HAVE_LIBZ
    int gzip;
#endif
    bool separate_diagonal;
    bool sort_rows;
    enum partition partition;
    bool precompute_partition;
    int rows_per_thread_size;
    idx_t * rows_per_thread;
    int columns_per_thread_size;
    idx_t * columns_per_thread;
    int repeat;
    int warmup;
    int verbose;
    int quiet;
#ifdef HAVE_PAPI
    char * papi_event_file;
    int papi_event_format;
    bool papi_event_per_thread;
    bool papi_event_summary;
#endif
    int l1pfdst;
    int l2pfdst;
};

/**
 * ‘program_options_init()’ configures the default program options.
 */
static int program_options_init(
    struct program_options * args)
{
    args->Apath = NULL;
    args->xpath = NULL;
    args->ypath = NULL;
#ifdef HAVE_LIBZ
    args->gzip = 0;
#endif
    args->separate_diagonal = false;
    args->sort_rows = false;
    args->partition = partition_rows;
    args->precompute_partition = false;
    args->rows_per_thread_size = 0;
    args->rows_per_thread = NULL;
    args->columns_per_thread_size = 0;
    args->columns_per_thread = NULL;
    args->repeat = 1;
    args->warmup = 0;
    args->quiet = 0;
    args->verbose = 0;
#ifdef HAVE_PAPI
    args->papi_event_file = NULL;
    args->papi_event_format = 0;
    args->papi_event_per_thread = false;
    args->papi_event_summary = false;
#endif
    args->l1pfdst = -1;
    args->l2pfdst = -1;
    return 0;
}

/**
 * ‘program_options_free()’ frees memory and other resources
 * associated with parsing program options.
 */
static void program_options_free(
    struct program_options * args)
{
#ifdef HAVE_PAPI
    if (args->papi_event_file) free(args->papi_event_file);
#endif
    if (args->columns_per_thread) free(args->columns_per_thread);
    if (args->rows_per_thread) free(args->rows_per_thread);
    if (args->ypath) free(args->ypath);
    if (args->xpath) free(args->xpath);
    if (args->Apath) free(args->Apath);
}

/**
 * ‘program_options_print_usage()’ prints a usage text.
 */
static void program_options_print_usage(
    FILE * f)
{
    fprintf(f, "Usage: %s [OPTION..] A [x] [y]\n", program_name);
}

/**
 * ‘program_options_print_help()’ prints a help text.
 */
static void program_options_print_help(
    FILE * f)
{
    program_options_print_usage(f);
    fprintf(f, "\n");
    fprintf(f, " Multiply a matrix by a vector.\n");
    fprintf(f, "\n");
    fprintf(f, " The operation performed is ‘y := A*x + y’, where\n");
    fprintf(f, " ‘A’ is a matrix, and ‘x’ and ‘y’ are vectors.\n");
    fprintf(f, "\n");
    fprintf(f, " Positional arguments are:\n");
    fprintf(f, "  A        path to Matrix Market file for the matrix A\n");
    fprintf(f, "  x        optional path to Matrix Market file for the vector x\n");
    fprintf(f, "  y        optional path for to Matrix Market file for the vector y\n");
    fprintf(f, "\n");
    fprintf(f, " Other options are:\n");
#ifdef HAVE_LIBZ
    fprintf(f, "  -z, --gzip, --gunzip, --ungzip    filter files through gzip\n");
#endif
    fprintf(f, "  --separate-diagonal       store diagonal nonzeros separately\n");
    fprintf(f, "  --sort-rows               sort nonzeros by column within each row\n");
#ifdef _OPENMP
    fprintf(f, "  --partition-rows          partition rows evenly among threads (default)\n");
    fprintf(f, "  --partition-nonzeros      partition nonzeros evenly among threads\n");
    fprintf(f, "  --precompute-partition    perform per-thread partitioning once as a precomputation\n");
    fprintf(f, "  --rows-per-thread=N..     comma-separated list of number of rows assigned to threads\n");
    fprintf(f, "  --columns-per-thread=N..  comma-separated list of number of columns assigned to threads\n");
#endif
    fprintf(f, "  --repeat=N                repeat matrix-vector multiplication N times\n");
    fprintf(f, "  --warmup=N                perform N additional warmup iterations\n");
    fprintf(f, "  -q, --quiet               do not print Matrix Market output\n");
    fprintf(f, "  -v, --verbose             be more verbose\n");
    fprintf(f, "\n");
#ifdef HAVE_PAPI
    fprintf(f, " Options for performance monitoring (PAPI) are:\n");
    fprintf(f, "  --papi-event-file=FILE    file describing which events to monitor\n");
    fprintf(f, "  --papi-event-format=FMT   output format for events: plain or csv. [plain]\n");
    fprintf(f, "  --papi-event-per-thread   display events per thread\n");
    fprintf(f, "  --papi-event-summary      display summary of performance monitoring\n");
    fprintf(f, "\n");
#endif
#if defined(__FCC_version__)
    fprintf(f, " Options for A64FX hardware prefetching are:\n");
    fprintf(f, "  --l1-prefetch-distance=N  set L1 prefetch distance to 256*N bytes for N=1,...,15\n");
    fprintf(f, "  --l2-prefetch-distance=N  set L2 prefetch distance to N KiB for N=1,...,15\n");
    fprintf(f, "\n");
#endif
    fprintf(f, "  -h, --help                display this help and exit\n");
    fprintf(f, "  --version                 display version information and exit\n");
    fprintf(f, "\n");
    fprintf(f, "Report bugs to: <james@simula.no>\n");
}

/**
 * ‘program_options_print_version()’ prints version information.
 */
static void program_options_print_version(
    FILE * f)
{
    fprintf(f, "%s %s\n", program_name, program_version);
    fprintf(f, "row/column offsets: %ld-bit\n", sizeof(idx_t)*CHAR_BIT);
#ifdef HAVE_ALIGNED_ALLOC
    fprintf(f, "page-aligned allocations: yes (page size: %ld)\n", sysconf(_SC_PAGESIZE));
#else
    fprintf(f, "page-aligned allocations: no\n");
#endif
#ifdef _OPENMP
    fprintf(f, "OpenMP: yes (%d)\n", _OPENMP);
#else
    fprintf(f, "OpenMP: no\n");
#endif
#ifdef HAVE_LIBZ
    fprintf(f, "zlib: yes ("ZLIB_VERSION")\n");
#else
    fprintf(f, "zlib: no\n");
#endif
#ifdef HAVE_PAPI
    fprintf(f, "PAPI: yes (%d.%d.%d.%d)\n", PAPI_VERSION_MAJOR(PAPI_VERSION), PAPI_VERSION_MINOR(PAPI_VERSION), PAPI_VERSION_REVISION(PAPI_VERSION), PAPI_VERSION_INCREMENT(PAPI_VERSION));
#else
    fprintf(f, "PAPI: no\n");
#endif
#if defined(__FCC_version__)
    fprintf(f, "Fujitsu compiler version: %s\n", __FCC_version__);
#if defined(USE_A64FX_SECTOR_CACHE)
    fprintf(f, "Fujitsu A64FX sector cache: yes (L1 ways: ");
#ifndef A64FX_SECTOR_CACHE_L1_WAYS
    fprintf(f, "disabled");
#else
    fprintf(f, "%d", A64FX_SECTOR_CACHE_L1_WAYS);
#endif
    fprintf(f, ", L2 ways: %d)\n", A64FX_SECTOR_CACHE_L2_WAYS);
#else
    fprintf(f, "Fujitsu A64FX sector cache: no\n");
#endif
#endif
#if defined(__ARM_FEATURE_SVE) && !defined(ARM_NOSVE)
    fprintf(f, "Arm Aarch64 SVE: yes\n");
#else
    fprintf(f, "Arm Aarch64 SVE: no\n");
#endif
    fprintf(f, "\n");
    fprintf(f, "%s\n", program_copyright);
    fprintf(f, "%s\n", program_license);
}

/**
 * ‘parse_long_long_int()’ parses a string to produce a number that
 * may be represented with the type ‘long long int’.
 */
static int parse_long_long_int(
    const char * s,
    char ** outendptr,
    int base,
    long long int * out_number,
    int64_t * bytes_read)
{
    errno = 0;
    char * endptr;
    long long int number = strtoll(s, &endptr, base);
    if ((errno == ERANGE && (number == LLONG_MAX || number == LLONG_MIN)) ||
        (errno != 0 && number == 0))
        return errno;
    if (outendptr) *outendptr = endptr;
    if (bytes_read) *bytes_read += endptr - s;
    *out_number = number;
    return 0;
}

/**
 * ‘parse_int()’ parses a string to produce a number that may be
 * represented as an integer.
 *
 * The number is parsed using ‘strtoll()’, following the conventions
 * documented in the man page for that function.  In addition, some
 * further error checking is performed to ensure that the number is
 * parsed correctly.  The parsed number is stored in ‘x’.
 *
 * If ‘endptr’ is not ‘NULL’, the address stored in ‘endptr’ points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * On success, ‘0’ is returned. Otherwise, if the input contained
 * invalid characters, ‘EINVAL’ is returned. If the resulting number
 * cannot be represented as a signed integer, ‘ERANGE’ is returned.
 */
int parse_int(
    int * x,
    const char * s,
    char ** endptr,
    int64_t * bytes_read)
{
    long long int y;
    int err = parse_long_long_int(s, endptr, 10, &y, bytes_read);
    if (err) return err;
    if (y < INT_MIN || y > INT_MAX) return ERANGE;
    *x = y;
    return 0;
}

/**
 * ‘parse_int32_t()’ parses a string to produce a number that may be
 * represented as a signed, 32-bit integer.
 *
 * The number is parsed using ‘strtoll()’, following the conventions
 * documented in the man page for that function.  In addition, some
 * further error checking is performed to ensure that the number is
 * parsed correctly.  The parsed number is stored in ‘x’.
 *
 * If ‘endptr’ is not ‘NULL’, the address stored in ‘endptr’ points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * On success, ‘0’ is returned. Otherwise, if the input contained
 * invalid characters, ‘EINVAL’ is returned. If the resulting number
 * cannot be represented as a signed integer, ‘ERANGE’ is returned.
 */
int parse_int32_t(
    int32_t * x,
    const char * s,
    char ** endptr,
    int64_t * bytes_read)
{
    long long int y;
    int err = parse_long_long_int(s, endptr, 10, &y, bytes_read);
    if (err) return err;
    if (y < INT32_MIN || y > INT32_MAX) return ERANGE;
    *x = y;
    return 0;
}

/**
 * ‘parse_int64_t()’ parses a string to produce a number that may be
 * represented as a signed, 64-bit integer.
 *
 * The number is parsed using ‘strtoll()’, following the conventions
 * documented in the man page for that function.  In addition, some
 * further error checking is performed to ensure that the number is
 * parsed correctly.  The parsed number is stored in ‘x’.
 *
 * If ‘endptr’ is not ‘NULL’, the address stored in ‘endptr’ points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * On success, ‘0’ is returned. Otherwise, if the input contained
 * invalid characters, ‘EINVAL’ is returned. If the resulting number
 * cannot be represented as a signed integer, ‘ERANGE’ is returned.
 */
int parse_int64_t(
    int64_t * x,
    const char * s,
    char ** endptr,
    int64_t * bytes_read)
{
    long long int y;
    int err = parse_long_long_int(s, endptr, 10, &y, bytes_read);
    if (err) return err;
    if (y < INT64_MIN || y > INT64_MAX) return ERANGE;
    *x = y;
    return 0;
}

/**
 * ‘parse_double()’ parses a string to produce a number that may be
 * represented as ‘double’.
 *
 * The number is parsed using ‘strtod()’, following the conventions
 * documented in the man page for that function.  In addition, some
 * further error checking is performed to ensure that the number is
 * parsed correctly.  The parsed number is stored in ‘number’.
 *
 * If ‘endptr’ is not ‘NULL’, the address stored in ‘endptr’ points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * On success, ‘0’ is returned. Otherwise, if the input contained
 * invalid characters, ‘EINVAL’ is returned. If the resulting number
 * cannot be represented as a double, ‘ERANGE’ is returned.
 */
int parse_double(
    double * x,
    const char * s,
    char ** outendptr,
    int64_t * bytes_read)
{
    errno = 0;
    char * endptr;
    *x = strtod(s, &endptr);
    if ((errno == ERANGE && (*x == HUGE_VAL || *x == -HUGE_VAL)) ||
        (errno != 0 && x == 0)) { return errno; }
    if (outendptr) *outendptr = endptr;
    if (bytes_read) *bytes_read += endptr - s;
    return 0;
}

/**
 * ‘parse_program_options()’ parses program options.
 */
static int parse_program_options(
    int argc,
    char ** argv,
    struct program_options * args,
    int * nargs)
{
    int err;
    *nargs = 0;
    (*nargs)++; argv++;

    /* Set default program options. */
    err = program_options_init(args);
    if (err) return err;

    /* Parse program options. */
    int num_positional_arguments_consumed = 0;
    while (*nargs < argc) {
        if (strcmp(argv[0], "--separate-diagonal") == 0) {
            args->separate_diagonal = true;
            (*nargs)++; argv++; continue;
        }
        if (strcmp(argv[0], "--sort-rows") == 0) {
            args->sort_rows = true;
            (*nargs)++; argv++; continue;
        }

#ifdef _OPENMP
        if (strcmp(argv[0], "--partition-rows") == 0) {
            args->partition = partition_rows;
            (*nargs)++; argv++; continue;
        }
        if (strcmp(argv[0], "--partition-nonzeros") == 0) {
            args->partition = partition_nonzeros;
            (*nargs)++; argv++; continue;
        }
        if (strcmp(argv[0], "--precompute-partition") == 0) {
            args->precompute_partition = true;
            (*nargs)++; argv++; continue;
        }

        if (strstr(argv[0], "--rows-per-thread") == argv[0]) {
            int n = strlen("--rows-per-thread");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            args->rows_per_thread_size = 1;
            const char * t = s;
            while (*t != '\0') { if (*t == ',') { args->rows_per_thread_size++; } t++; }
            if (args->rows_per_thread) free(args->rows_per_thread);
            args->rows_per_thread = malloc(args->rows_per_thread_size * sizeof(idx_t));
            if (!args->rows_per_thread) return errno;
            int i = 0;
            while (true) {
                if (i >= args->rows_per_thread_size) { program_options_free(args); return EINVAL; }
                err = parse_idx_t(&args->rows_per_thread[i++], s, (char **) &t, NULL);
                if (err) { program_options_free(args); return EINVAL; }
                if (s == t) { program_options_free(args); return EINVAL; }
                else if (*t == '\0') break;
                else if (*t != ',') { program_options_free(args); return EINVAL; }
                s = t+1;
            }
            (*nargs)++; argv++; continue;
        }

        if (strstr(argv[0], "--columns-per-thread") == argv[0]) {
            int n = strlen("--columns-per-thread");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            args->columns_per_thread_size = 1;
            const char * t = s;
            while (*t != '\0') { if (*t == ',') { args->columns_per_thread_size++; } t++; }
            if (args->columns_per_thread) free(args->columns_per_thread);
            args->columns_per_thread = malloc(args->columns_per_thread_size * sizeof(idx_t));
            if (!args->columns_per_thread) return errno;
            int i = 0;
            while (true) {
                if (i >= args->columns_per_thread_size) { program_options_free(args); return EINVAL; }
                err = parse_idx_t(&args->columns_per_thread[i++], s, (char **) &t, NULL);
                if (err) { program_options_free(args); return EINVAL; }
                if (s == t) { program_options_free(args); return EINVAL; }
                else if (*t == '\0') break;
                else if (*t != ',') { program_options_free(args); return EINVAL; }
                s = t+1;
            }
            (*nargs)++; argv++; continue;
        }
#endif

        if (strstr(argv[0], "--repeat") == argv[0]) {
            int n = strlen("--repeat");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_int(&args->repeat, s, (char **) &s, NULL);
            if (err || *s != '\0') { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }
        if (strstr(argv[0], "--warmup") == argv[0]) {
            int n = strlen("--warmup");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_int(&args->warmup, s, (char **) &s, NULL);
            if (err || *s != '\0') { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }

#ifdef HAVE_LIBZ
        if (strcmp(argv[0], "-z") == 0 ||
            strcmp(argv[0], "--gzip") == 0 ||
            strcmp(argv[0], "--gunzip") == 0 ||
            strcmp(argv[0], "--ungzip") == 0)
        {
            args->gzip = 1;
            (*nargs)++; argv++; continue;
        }
#endif

        if (strcmp(argv[0], "-q") == 0 || strcmp(argv[0], "--quiet") == 0) {
            args->quiet = 1;
            (*nargs)++; argv++; continue;
        }

        if (strcmp(argv[0], "-v") == 0 || strcmp(argv[0], "--verbose") == 0) {
            args->verbose++;
            (*nargs)++; argv++; continue;
        }

#ifdef HAVE_PAPI
        if (strstr(argv[0], "--papi-event-file") == argv[0]) {
            int n = strlen("--papi-event-file");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            free(args->papi_event_file);
            args->papi_event_file = strdup(s);
            if (!args->papi_event_file) { program_options_free(args); return errno; }
            (*nargs)++; argv++; continue;
        }
        if (strstr(argv[0], "--papi-event-format") == argv[0]) {
            int n = strlen("--papi-event-format");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            if (strcmp(s, "csv") == 0) args->papi_event_format = 1;
            else if (strcmp(s, "plain") == 0) args->papi_event_format = 0;
            else { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }
        if (strcmp(argv[0], "--papi-event-per-thread") == 0) {
            args->papi_event_per_thread = true;
            (*nargs)++; argv++; continue;
        }
        if (strcmp(argv[0], "--papi-event-summary") == 0) {
            args->papi_event_summary = true;
            (*nargs)++; argv++; continue;
        }
#endif

#if defined(__FCC_version__)
        if (strstr(argv[0], "--l1-prefetch-distance") == argv[0]) {
            int n = strlen("--l1-prefetch-distance");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_int(&args->l1pfdst, s, (char **) &s, NULL);
            if (err || *s != '\0' || args->l1pfdst < 0 || args->l1pfdst > 15) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }
        if (strstr(argv[0], "--l2-prefetch-distance") == argv[0]) {
            int n = strlen("--l2-prefetch-distance");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { program_options_free(args); return EINVAL; }
            err = parse_int(&args->l2pfdst, s, (char **) &s, NULL);
            if (err || *s != '\0' || args->l2pfdst < 0 || args->l2pfdst > 15) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++; continue;
        }
#endif

        /* If requested, print program help text. */
        if (strcmp(argv[0], "-h") == 0 || strcmp(argv[0], "--help") == 0) {
            program_options_free(args);
            program_options_print_help(stdout);
            exit(EXIT_SUCCESS);
        }

        /* If requested, print program version information. */
        if (strcmp(argv[0], "--version") == 0) {
            program_options_free(args);
            program_options_print_version(stdout);
            exit(EXIT_SUCCESS);
        }

        /* Stop parsing options after '--'.  */
        if (strcmp(argv[0], "--") == 0) {
            (*nargs)++; argv++;
            break;
        }

        /*
         * Parse positional arguments.
         */
        if (num_positional_arguments_consumed == 0) {
            args->Apath = strdup(argv[0]);
            if (!args->Apath) { program_options_free(args); return errno; }
        } else if (num_positional_arguments_consumed == 1) {
            args->xpath = strdup(argv[0]);
            if (!args->xpath) { program_options_free(args); return errno; }
        } else if (num_positional_arguments_consumed == 2) {
            args->ypath = strdup(argv[0]);
            if (!args->ypath) { program_options_free(args); return errno; }
        } else { program_options_free(args); return EINVAL; }
        num_positional_arguments_consumed++;
        (*nargs)++; argv++;
    }

    if (num_positional_arguments_consumed < 1) {
        program_options_free(args);
        program_options_print_usage(stdout);
        exit(EXIT_FAILURE);
    }
    return 0;
}

/**
 * `timespec_duration()` is the duration, in seconds, elapsed between
 * two given time points.
 */
static double timespec_duration(
    struct timespec t0,
    struct timespec t1)
{
    return (t1.tv_sec - t0.tv_sec) +
        (t1.tv_nsec - t0.tv_nsec) * 1e-9;
}

enum streamtype
{
    stream_stdio,
#ifdef HAVE_LIBZ
    stream_zlib,
#endif
};

union stream
{
    FILE * f;
#ifdef HAVE_LIBZ
    gzFile gzf;
#endif
};

void stream_close(
    enum streamtype streamtype,
    union stream s)
{
    if (streamtype == stream_stdio) {
        fclose(s.f);
#ifdef HAVE_LIBZ
    } else if (streamtype == stream_zlib) {
        gzclose(s.gzf);
#endif
    }
}

/**
 * ‘freadline()’ reads a single line from a stream.
 */
static int freadline(
    char * linebuf,
    size_t line_max,
    enum streamtype streamtype,
    union stream stream)
{
    if (streamtype == stream_stdio) {
        char * s = fgets(linebuf, line_max+1, stream.f);
        if (!s && feof(stream.f)) return -1;
        else if (!s) return errno;
        int n = strlen(s);
        if (n > 0 && n == line_max && s[n-1] != '\n') return EOVERFLOW;
        return 0;
#ifdef HAVE_LIBZ
    } else if (streamtype == stream_zlib) {
        char * s = gzgets(stream.gzf, linebuf, line_max+1);
        if (!s && gzeof(stream.gzf)) return -1;
        else if (!s) return errno;
        int n = strlen(s);
        if (n > 0 && n == line_max && s[n-1] != '\n') return EOVERFLOW;
        return 0;
#endif
    } else { return EINVAL; }
}

enum mtxobject
{
    mtxmatrix,
    mtxvector,
};

enum mtxformat
{
    mtxarray,
    mtxcoordinate,
};

enum mtxfield
{
    mtxreal,
    mtxinteger,
    mtxpattern,
};

enum mtxsymmetry
{
    mtxgeneral,
    mtxsymmetric,
};

static int mtxfile_fread_header(
    enum mtxobject * object,
    enum mtxformat * format,
    enum mtxfield * field,
    enum mtxsymmetry * symmetry,
    idx_t * num_rows,
    idx_t * num_columns,
    int64_t * num_nonzeros,
    enum streamtype streamtype,
    union stream stream,
    int64_t * lines_read,
    int64_t * bytes_read)
{
    int line_max = sysconf(_SC_LINE_MAX);
    char * linebuf = malloc(line_max+1);
    if (!linebuf) return errno;

    /* read and parse header line */
    int err = freadline(linebuf, line_max, streamtype, stream);
    if (err) { free(linebuf); return err; }
    char * s = linebuf;
    char * t = s;
    if (strncmp("%%MatrixMarket ", t, strlen("%%MatrixMarket ")) == 0) {
        t += strlen("%%MatrixMarket ");
    } else { free(linebuf); return EINVAL; }
    if (bytes_read) *bytes_read += t-s;
    s = t;
    if (strncmp("matrix ", t, strlen("matrix ")) == 0) {
        t += strlen("matrix ");
        *object = mtxmatrix;
    } else if (strncmp("vector ", t, strlen("vector ")) == 0) {
        t += strlen("vector ");
        *object = mtxvector;
    } else { free(linebuf); return EINVAL; }
    if (bytes_read) *bytes_read += t-s;
    s = t;
    if (strncmp("array ", t, strlen("array ")) == 0) {
        t += strlen("array ");
        *format = mtxarray;
    } else if (strncmp("coordinate ", t, strlen("coordinate ")) == 0) {
        t += strlen("coordinate ");
        *format = mtxcoordinate;
    } else { free(linebuf); return EINVAL; }
    if (bytes_read) *bytes_read += t-s;
    s = t;
    if (strncmp("real ", t, strlen("real ")) == 0) {
        t += strlen("real ");
        *field = mtxreal;
    } else if (strncmp("integer ", t, strlen("integer ")) == 0) {
        t += strlen("integer ");
        *field = mtxinteger;
    } else if (strncmp("pattern ", t, strlen("pattern ")) == 0) {
        t += strlen("pattern ");
        *field = mtxpattern;
    } else { free(linebuf); return EINVAL; }
    if (bytes_read) *bytes_read += t-s;
    s = t;
    if (strncmp("general", t, strlen("general")) == 0) {
        t += strlen("general");
        *symmetry = mtxgeneral;
    } else if (strncmp("symmetric", t, strlen("symmetric")) == 0) {
        t += strlen("symmetric");
        *symmetry = mtxsymmetric;
    } else { free(linebuf); return EINVAL; }
    if (bytes_read) *bytes_read += t-s;
    s = t;

    /* skip lines starting with '%' */
    do {
        if (lines_read) (*lines_read)++;
        err = freadline(linebuf, line_max, streamtype, stream);
        if (err) { free(linebuf); return err; }
        s = t = linebuf;
    } while (linebuf[0] == '%');

    /* parse size line */
    if (*object == mtxmatrix && *format == mtxcoordinate) {
        err = parse_idx_t(num_rows, s, &t, bytes_read);
        if (err) { free(linebuf); return err; }
        if (s == t || *t != ' ') { free(linebuf); return EINVAL; }
        if (bytes_read) (*bytes_read)++;
        s = t+1;
        err = parse_idx_t(num_columns, s, &t, bytes_read);
        if (err) { free(linebuf); return err; }
        if (s == t || *t != ' ') { free(linebuf); return EINVAL; }
        if (bytes_read) (*bytes_read)++;
        s = t+1;
        err = parse_int64_t(num_nonzeros, s, &t, bytes_read);
        if (err) { free(linebuf); return err; }
        if (s == t) { free(linebuf); return EINVAL; }
        if (lines_read) (*lines_read)++;
    } else if (*object == mtxvector && *format == mtxarray) {
        err = parse_idx_t(num_rows, s, &t, bytes_read);
        if (err) { free(linebuf); return err; }
        if (s == t) { free(linebuf); return EINVAL; }
        if (lines_read) (*lines_read)++;
    } else { free(linebuf); return EINVAL; }
    free(linebuf);
    return 0;
}

static int mtxfile_fread_matrix_coordinate(
    enum mtxfield field,
    idx_t num_rows,
    idx_t num_columns,
    int64_t num_nonzeros,
    idx_t * rowidx,
    idx_t * colidx,
    double * a,
    enum streamtype streamtype,
    union stream stream,
    int64_t * lines_read,
    int64_t * bytes_read)
{
    int line_max = sysconf(_SC_LINE_MAX);
    char * linebuf = malloc(line_max+1);
    if (!linebuf) return errno;
    if (field == mtxreal || field == mtxinteger) {
        for (int64_t i = 0; i < num_nonzeros; i++) {
            int err = freadline(linebuf, line_max, streamtype, stream);
            if (err) { free(linebuf); return err; }
            char * s = linebuf;
            char * t = s;
            err = parse_idx_t(&rowidx[i], s, &t, bytes_read);
            if (err) { free(linebuf); return err; }
            if (s == t || *t != ' ') { free(linebuf); return EINVAL; }
            if (bytes_read) (*bytes_read)++;
            s = t+1;
            err = parse_idx_t(&colidx[i], s, &t, bytes_read);
            if (err) { free(linebuf); return err; }
            if (s == t || *t != ' ') { free(linebuf); return EINVAL; }
            if (bytes_read) (*bytes_read)++;
            s = t+1;
            err = parse_double(&a[i], s, &t, bytes_read);
            if (err) { free(linebuf); return err; }
            if (s == t) { free(linebuf); return EINVAL; }
            if (lines_read) (*lines_read)++;
        }
    } else if (field == mtxinteger) {
        for (int64_t i = 0; i < num_nonzeros; i++) {
            int err = freadline(linebuf, line_max, streamtype, stream);
            if (err) { free(linebuf); return err; }
            char * s = linebuf;
            char * t = s;
            err = parse_idx_t(&rowidx[i], s, &t, bytes_read);
            if (err) { free(linebuf); return err; }
            if (s == t || *t != ' ') { free(linebuf); return EINVAL; }
            if (bytes_read) (*bytes_read)++;
            s = t+1;
            err = parse_idx_t(&colidx[i], s, &t, bytes_read);
            if (err) { free(linebuf); return err; }
            if (s == t || *t != ' ') { free(linebuf); return EINVAL; }
            if (bytes_read) (*bytes_read)++;
            s = t+1;
            int x;
            err = parse_int(&x, s, &t, bytes_read);
            if (err) { free(linebuf); return err; }
            if (s == t) { free(linebuf); return EINVAL; }
            a[i] = x;
            if (lines_read) (*lines_read)++;
        }
    } else if (field == mtxpattern) {
        for (int64_t i = 0; i < num_nonzeros; i++) {
            int err = freadline(linebuf, line_max, streamtype, stream);
            if (err) { free(linebuf); return err; }
            char * s = linebuf;
            char * t = s;
            err = parse_idx_t(&rowidx[i], s, &t, bytes_read);
            if (err) { free(linebuf); return err; }
            if (s == t || *t != ' ') { free(linebuf); return EINVAL; }
            if (bytes_read) (*bytes_read)++;
            s = t+1;
            err = parse_idx_t(&colidx[i], s, &t, bytes_read);
            if (err) { free(linebuf); return err; }
            if (s == t) { free(linebuf); return EINVAL; }
            a[i] = 1;
            if (lines_read) (*lines_read)++;
        }
    } else { free(linebuf); return EINVAL; }
    free(linebuf);
    return 0;
}

static int mtxfile_fread_vector_array(
    enum mtxfield field,
    idx_t num_rows,
    double * x,
    enum streamtype streamtype,
    union stream stream,
    int64_t * lines_read,
    int64_t * bytes_read)
{
    int line_max = sysconf(_SC_LINE_MAX);
    char * linebuf = malloc(line_max+1);
    if (!linebuf) return errno;
    if (field == mtxreal) {
        for (idx_t i = 0; i < num_rows; i++) {
            int err = freadline(linebuf, line_max, streamtype, stream);
            if (err) { free(linebuf); return err; }
            char * s = linebuf;
            char * t = s;
            err = parse_double(&x[i], s, &t, bytes_read);
            if (err) { free(linebuf); return err; }
            if (s == t) { free(linebuf); return EINVAL; }
            if (lines_read) (*lines_read)++;
        }
    } else if (field == mtxinteger) {
        for (idx_t i = 0; i < num_rows; i++) {
            int err = freadline(linebuf, line_max, streamtype, stream);
            if (err) { free(linebuf); return err; }
            char * s = linebuf;
            char * t = s;
            int y;
            err = parse_int(&y, s, &t, bytes_read);
            if (err) { free(linebuf); return err; }
            if (s == t) { free(linebuf); return EINVAL; }
            x[i] = y;
            if (lines_read) (*lines_read)++;
        }
    } else { free(linebuf); return EINVAL; }
    free(linebuf);
    return 0;
}

static int csr_from_coo_size(
    enum mtxsymmetry symmetry,
    idx_t num_rows,
    idx_t num_columns,
    int64_t num_nonzeros,
    const idx_t * __restrict rowidx,
    const idx_t * __restrict colidx,
    const double * __restrict a,
    int64_t * __restrict rowptr,
    int64_t * __restrict csrsize,
    idx_t * __restrict rowsizemin,
    idx_t * __restrict rowsizemax,
    idx_t * __restrict diagsize,
    bool separate_diagonal,
    enum partition partition)
{
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (idx_t i = 0; i < num_rows; i++) rowptr[i] = 0;
    rowptr[num_rows] = 0;
    if (num_rows == num_columns && symmetry == mtxsymmetric && separate_diagonal) {
        for (int64_t k = 0; k < num_nonzeros; k++) {
            if (rowidx[k] != colidx[k]) { rowptr[rowidx[k]]++; rowptr[colidx[k]]++; }
        }
    } else if (num_rows == num_columns && symmetry == mtxsymmetric && !separate_diagonal) {
        for (int64_t k = 0; k < num_nonzeros; k++) {
            if (rowidx[k] != colidx[k]) { rowptr[rowidx[k]]++; rowptr[colidx[k]]++; }
            else { rowptr[rowidx[k]]++; }
        }
    } else if (num_rows == num_columns && separate_diagonal) {
        for (int64_t k = 0; k < num_nonzeros; k++) {
            if (rowidx[k] != colidx[k]) rowptr[rowidx[k]]++;
        }
    } else { for (int64_t k = 0; k < num_nonzeros; k++) rowptr[rowidx[k]]++; }
    idx_t rowmin = num_rows > 0 ? rowptr[1] : 0;
    idx_t rowmax = 0;
    for (idx_t i = 1; i <= num_rows; i++) {
        rowmin = rowmin <= rowptr[i] ? rowmin : rowptr[i];
        rowmax = rowmax >= rowptr[i] ? rowmax : rowptr[i];
        rowptr[i] += rowptr[i-1];
    }
    if (num_rows == num_columns && separate_diagonal) { rowmin++; rowmax++; }
    *rowsizemin = rowmin;
    *rowsizemax = rowmax;
    *csrsize = rowptr[num_rows];
    *diagsize = (num_rows == num_columns && separate_diagonal) ? num_rows : 0;
    return 0;
}

static int rowsort(
    idx_t num_rows,
    idx_t num_columns,
    int64_t * __restrict rowptr,
    idx_t rowsizemax,
    idx_t * __restrict colidx,
    double * __restrict a)
{
    idx_t threshold = 1 << 4;

    /* sort short rows using insertion sort */
    #pragma omp parallel for
    for (idx_t i = 0; i < num_rows; i++) {
        idx_t rowlen = rowptr[i+1]-rowptr[i];
        if (rowlen > threshold) continue;
        for (int64_t k = rowptr[i]+1; k < rowptr[i+1]; k++) {
            idx_t j = colidx[k];
            double b = a[k];
            int64_t l = k-1;
            while (l >= rowptr[i] && colidx[l] > j) {
                colidx[l+1] = colidx[l];
                a[l+1] = a[l];
                l--;
            }
            colidx[l+1] = j;
            a[l+1] = b;
        }
    }

    if (rowsizemax <= threshold) return 0;

    /* sort long rows using a hybrid sort that first uses insertion
     * sort to sort blocks of size 'threshold', and then switches to a
     * bottom-up merge sort. */
    idx_t * tmpcolidx = malloc(rowsizemax * sizeof(idx_t));
    if (!tmpcolidx) return errno;
    double * tmpa = malloc(rowsizemax * sizeof(double));
    if (!tmpa) { free(tmpcolidx); return errno; }
    #pragma omp parallel
    for (idx_t i = 0; i < num_rows; i++) {
        idx_t rowlen = rowptr[i+1]-rowptr[i];
        if (rowlen <= threshold) continue;

        /* #pragma omp single */
        /* fprintf(stderr, "i=%d, rowlen=%d\n", i, rowlen); */

        idx_t p = threshold;
        #pragma omp for
        for (idx_t q = 0; q < rowlen-1; q += p) {
            idx_t r = q+p < rowlen ? q+p : rowlen;
            for (int64_t k = q+1; k < r; k++) {
                idx_t j = colidx[rowptr[i]+k];
                double b = a[rowptr[i]+k];
                int64_t l = rowptr[i]+k-1;
                while (l >= rowptr[i]+q && colidx[l] > j) {
                    colidx[l+1] = colidx[l];
                    a[l+1] = a[l];
                    l--;
                }
                colidx[l+1] = j;
                a[l+1] = b;
            }
        }

        for (; p < rowlen; p*=2) {
            /* fprintf(stderr, "a) t=%d, i=%d, p=%d, rowlen=%d\n", omp_get_thread_num(), i, p, rowlen); */

            #pragma omp for
            for (idx_t k = 0; k < rowlen; k++) {
                tmpcolidx[k] = colidx[rowptr[i]+k];
                tmpa[k] = a[rowptr[i]+k];
            }

            #pragma omp for
            for (idx_t q = 0; q < rowlen-1; q += 2*p) {
                idx_t left = q;
                idx_t middle = q+p < rowlen ? q+p : rowlen;
                idx_t right = q+2*p < rowlen ? q+2*p : rowlen;

                /* printf("i=%d: merging [", i); */
                /* for (idx_t j = left; j < middle; j++) printf(" %d", tmpcolidx[j]); */
                /* printf("] and ["); */
                /* for (idx_t j = middle; j < right; j++) printf(" %d", tmpcolidx[j]); */
                /* printf("] -> "); */

                idx_t u = left, v = left, w = middle;
                while (v < middle && w < right) {
                    if (tmpcolidx[v] < tmpcolidx[w]) {
                        colidx[rowptr[i]+u] = tmpcolidx[v]; a[rowptr[i]+u] = tmpa[v++]; u++;
                    } else {
                        colidx[rowptr[i]+u] = tmpcolidx[w]; a[rowptr[i]+u] = tmpa[w++]; u++;
                    }
                }
                while (v < middle) {
                    colidx[rowptr[i]+u] = tmpcolidx[v]; a[rowptr[i]+u] = tmpa[v++]; u++;
                }
                while (w < right) {
                    colidx[rowptr[i]+u] = tmpcolidx[w]; a[rowptr[i]+u] = tmpa[w++]; u++;
                }

                /* printf("["); */
                /* for (idx_t j = left; j < right; j++) printf(" %d", colidx[rowptr[i]+j]); */
                /* printf("]\n"); */
            }

            /* fprintf(stderr, "b) t=%d, i=%d, p=%d, rowlen=%d\n", omp_get_thread_num(), i, p, rowlen); */
        }
    }
    free(tmpa); free(tmpcolidx);

#if 0
    /* verify that rows are sorted */
    for (idx_t i = 0; i < num_rows; i++) {
        for (int64_t k = rowptr[i]+1; k < rowptr[i+1]; k++) {
            if (colidx[k] < colidx[k-1]) return EINVAL;
        }
    }
#endif
    return 0;
}

static int csr_from_coo(
    enum mtxsymmetry symmetry,
    idx_t num_rows,
    idx_t num_columns,
    int64_t num_nonzeros,
    const idx_t * __restrict rowidx,
    const idx_t * __restrict colidx,
    const double * __restrict a,
    int64_t * __restrict rowptr,
    int64_t csrsize,
    idx_t rowsizemin,
    idx_t rowsizemax,
    idx_t * __restrict csrcolidx,
    double * __restrict csra,
    double * __restrict csrad,
    bool separate_diagonal,
    bool sort_rows,
    enum partition partition)
{
    if (num_rows == num_columns && symmetry == mtxsymmetric && separate_diagonal) {
        for (int64_t k = 0; k < num_nonzeros; k++) {
            if (rowidx[k] == colidx[k]) { csrad[rowidx[k]-1] += a[k]; }
            else {
                idx_t i = rowidx[k]-1, j = colidx[k]-1;
                csrcolidx[rowptr[i]] = j; csra[rowptr[i]] = a[k]; rowptr[i]++;
                csrcolidx[rowptr[j]] = i; csra[rowptr[j]] = a[k]; rowptr[j]++;
            }
        }
        for (idx_t i = num_rows; i > 0; i--) rowptr[i] = rowptr[i-1];
        rowptr[0] = 0;
    } else if (num_rows == num_columns && symmetry == mtxsymmetric && !separate_diagonal) {
        for (int64_t k = 0; k < num_nonzeros; k++) {
            idx_t i = rowidx[k]-1, j = colidx[k]-1;
            csrcolidx[rowptr[i]] = j; csra[rowptr[i]] = a[k]; rowptr[i]++;
            if (i != j) { csrcolidx[rowptr[j]] = i; csra[rowptr[j]] = a[k]; rowptr[j]++; }
        }
        for (idx_t i = num_rows; i > 0; i--) rowptr[i] = rowptr[i-1];
        rowptr[0] = 0;
    } else if (num_rows == num_columns && separate_diagonal) {
        for (int64_t k = 0; k < num_nonzeros; k++) {
            idx_t i = rowidx[k]-1, j = colidx[k]-1;
            if (i == j) { csrad[i] += a[k]; }
            else { csrcolidx[rowptr[i]] = j; csra[rowptr[i]] = a[k]; rowptr[i]++; }
        }
        for (idx_t i = num_rows; i > 0; i--) rowptr[i] = rowptr[i-1];
        rowptr[0] = 0;
    } else {
        /* simpler, serial version: */
        /* for (int64_t k = 0; k < num_nonzeros; k++) { */
        /*     idx_t i = rowidx[k]-1, j = colidx[k]-1; */
        /*     csrcolidx[rowptr[i]] = j; csra[rowptr[i]] = a[k]; rowptr[i]++; */
        /* } */
        /* for (idx_t i = num_rows; i > 0; i--) rowptr[i] = rowptr[i-1]; */
        /* rowptr[0] = 0; */

        int64_t * __restrict perm = malloc(num_nonzeros * sizeof(int64_t));
        if (!perm) { return errno; }
#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (int64_t k = 0; k < num_nonzeros; k++) perm[k] = 0;
        for (int64_t k = 0; k < num_nonzeros; k++) {
            idx_t i = rowidx[k]-1;
            perm[rowptr[i]++] = k;
        }
        for (idx_t i = num_rows; i > 0; i--) rowptr[i] = rowptr[i-1];
        rowptr[0] = 0;
#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (int64_t k = 0; k < num_nonzeros; k++) {
            csrcolidx[k] = colidx[perm[k]]-1;
            csra[k] = a[perm[k]];
        }
        free(perm);
    }

    /* If requested, sort nonzeros by column within each row */
    if (sort_rows) {
        int err = rowsort(
            num_rows, num_columns,
            rowptr, rowsizemax, csrcolidx, csra);
        if (err) return err;
    }
    return 0;
}

#if defined(__ARM_FEATURE_SVE) && !defined(ARM_NOSVE)
static int csrgemv(
    idx_t num_rows,
    double * __restrict y,
    idx_t num_columns,
    const double * __restrict x,
    int64_t csrsize,
    idx_t rowsizemin,
    idx_t rowsizemax,
    const int64_t * __restrict rowptr,
    const idx_t * __restrict colidx,
    const double * __restrict a)
{
#if defined(__FCC_version__) && defined(USE_A64FX_SECTOR_CACHE)
    #pragma procedure scache_isolate_assign a
#endif

#ifdef _OPENMP
    #pragma omp for simd
#endif
    for (idx_t i = 0; i < num_rows; i++) {

        // 2x Modulo Variable Expansion (MVE) accumulator vectors
        svfloat64_t acc0 = svdup_f64(0.0);
        svfloat64_t acc1 = svdup_f64(0.0);

        // 2x DP Vector Length per iteration of the inner loop
        for (int64_t k = rowptr[i]; k < rowptr[i + 1]; k += svcntw()) {

            /* L1 software prefetching */
            const int SW_PREFETCH_DIST_L1 = 16;
            svbool_t pg_prf = svwhilelt_b32((int64_t)(k + svcntw() * SW_PREFETCH_DIST_L1), rowptr[num_rows]);
            // load 16 (2 * svcntd()) colidx values ahead
            svint32_t cidxv_prf = svld1(pg_prf, &colidx[k + svcntw() * SW_PREFETCH_DIST_L1]);
            // extend low half into index vectors and prefetch (col0-col7)
            svprfd_gather_index(svunpklo(pg_prf), (void *) x, svunpklo(cidxv_prf), SV_PLDL1KEEP);
            // extend high half into index vector and prefetch (col8-col15)
            svprfd_gather_index(svunpkhi(pg_prf), (void *) x, svunpkhi(cidxv_prf), SV_PLDL1KEEP);

            svbool_t pg_row = svwhilelt_b32(k, rowptr[i+1]);

#if defined(__FCC_version__) && defined(USE_A64FX_SECTOR_CACHE)
            #pragma statement scache_isolate_assign colidx
#endif

            svint32_t cidxv = svld1(pg_row, &colidx[k]);

#if defined(__FCC_version__) && defined(USE_A64FX_SECTOR_CACHE)
            #pragma statement end_scache_isolate_assign
#endif

            // extend low half into index vector (col0-col7)
            svint64_t cidxvlo = svunpklo(cidxv);
            // extend high half into index vector (col8-col15)
            svint64_t cidxvhi = svunpkhi(cidxv);

            // extend low / high half into predicate vectors
            svbool_t pg_row_lo = svunpklo(pg_row);
            svbool_t pg_row_hi = svunpkhi(pg_row);

            // load x[col0,...,col7]
            svfloat64_t xv0 = svld1_gather_index(pg_row_lo, x, cidxvlo);
            // load x[col8,...,cold15]
            svfloat64_t xv1 = svld1_gather_index(pg_row_hi, x, cidxvhi);

            // load a[0-7]
            svfloat64_t av0 = svld1(pg_row_lo, &a[k]);
            // load a[8-15]
            svfloat64_t av1 = svld1(pg_row_hi, &a[k + svcntd()]);

            // acc0 <- acc0 + a[0-7] * x[col0,...,col7]
            acc0 = svmla_m(pg_row_lo, acc0, av0, xv0);
            // acc1 <- acc1 + a[8-15] * x[col8,...,col15]
            acc1 = svmla_m(pg_row_hi, acc1, av1, xv1);
        }

        // Add MVE vectors
        svfloat64_t acc = svadd_x(svptrue_b64(), acc0, acc1);
        // reduce a*x into y
        y[i] = svaddv(svptrue_b64(), acc);
    }
    return 0;
}
#else
static int csrgemv(
    idx_t num_rows,
    double * __restrict y,
    idx_t num_columns,
    const double * __restrict x,
    int64_t csrsize,
    idx_t rowsizemin,
    idx_t rowsizemax,
    const int64_t * __restrict rowptr,
    const idx_t * __restrict colidx,
    const double * __restrict a)
{
#if defined(__FCC_version__) && defined(USE_A64FX_SECTOR_CACHE)
    #pragma procedure scache_isolate_assign a, colidx, rowptr, y
#endif

#ifdef _OPENMP
    #pragma omp for simd
#endif
    for (idx_t i = 0; i < num_rows; i++) {
        double yi = 0;
        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
            yi += a[k] * x[colidx[k]];
        y[i] += yi;
    }
    return 0;
}
#endif

static int csrgemvsd(
    idx_t num_rows,
    double * __restrict y,
    idx_t num_columns,
    const double * __restrict x,
    int64_t csrsize,
    idx_t rowsizemin,
    idx_t rowsizemax,
    const int64_t * __restrict rowptr,
    const idx_t * __restrict colidx,
    const double * __restrict a,
    const double * __restrict ad)
{
#if defined(__FCC_version__) && defined(USE_A64FX_SECTOR_CACHE)
    #pragma procedure scache_isolate_assign a, ad, colidx, rowptr, y
#endif

#ifdef _OPENMP
    #pragma omp for
#endif
    for (idx_t i = 0; i < num_rows; i++) {
        double yi = 0;
        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
            yi += a[k] * x[colidx[k]];
        y[i] += ad[i]*x[i] + yi;
    }
    return 0;
}

static int csrgemvrp(
    idx_t num_rows,
    double * __restrict y,
    idx_t num_columns,
    const double * __restrict x,
    int64_t csrsize,
    idx_t rowsizemin,
    idx_t rowsizemax,
    const int64_t * __restrict rowptr,
    const idx_t * __restrict colidx,
    const double * __restrict a,
    idx_t diagsize,
    const double * __restrict ad,
    const idx_t * __restrict startrows,
    const idx_t * __restrict endrows)
{
#ifdef _OPENMP
    if (ad && diagsize > 0) {
        int p = omp_get_thread_num();
        for (idx_t i = startrows[p]; i < endrows[p]; i++) {
            double yi = 0;
            for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
                yi += a[k] * x[colidx[k]];
            y[i] += ad[i]*x[i] + yi;
        }
        #pragma omp barrier
    } else {
        int p = omp_get_thread_num();
        for (idx_t i = startrows[p]; i < endrows[p]; i++) {
            double yi = 0;
            for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
                yi += a[k] * x[colidx[k]];
            y[i] += yi;
        }
        #pragma omp barrier
    }
    return 0;
#else
    if (ad) {
        return csrgemvsd(
            num_rows, y, num_columns, x, csrsize,
            rowsizemin, rowsizemax, rowptr, colidx, a, ad);
    } else {
        return csrgemv(
            num_rows, y, num_columns, x, csrsize,
            rowsizemin, rowsizemax, rowptr, colidx, a);
    }
#endif
}

static int csrgemvnz(
    idx_t num_rows,
    double * __restrict y,
    idx_t num_columns,
    const double * __restrict x,
    int64_t csrsize,
    idx_t rowsizemin,
    idx_t rowsizemax,
    const int64_t * __restrict rowptr,
    const idx_t * __restrict colidx,
    const double * __restrict a,
    idx_t diagsize,
    const double * __restrict ad,
    const idx_t * __restrict startrows,
    const idx_t * __restrict endrows)
{
#ifdef _OPENMP
    int nthreads = omp_get_num_threads();
    int p = omp_get_thread_num();
    int64_t startnz = p*(csrsize+nthreads-1)/nthreads;
    int64_t endnz = (p+1)*(csrsize+nthreads-1)/nthreads;
    if (endnz > csrsize) endnz = csrsize;
    idx_t startrow = 0;
    if (startrows) { startrow = startrows[p]; }
    else { while (startrow < num_rows && startnz > rowptr[startrow+1]) startrow++; }
    idx_t endrow = startrow;
    if (endrows) { endrow = endrows[p]; }
    else { while (endrow < num_rows && endnz-1 > rowptr[endrow+1]) endrow++; }
    for (idx_t i = startrow; i < endrow; i++) y[i] = 0.0;

    if (startrow == endrow) {
        double yi = 0;
        for (int64_t k = startnz; k < endnz; k++)
            yi += a[k] * x[colidx[k]];
        #pragma omp atomic
        y[startrow] += yi;
    } else {
        {
            double yi = 0;
            for (int64_t k = startnz; k < rowptr[startrow+1]; k++)
                yi += a[k] * x[colidx[k]];
            #pragma omp atomic
            y[startrow] += yi;
        }

        for (idx_t i = startrow+1; i < endrow; i++) {
            double yi = 0;
            for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
                yi += a[k] * x[colidx[k]];
            y[i] += yi;
        }

        {
            double yi = 0;
            for (int64_t k = rowptr[endrow]; k < endnz; k++)
                yi += a[k] * x[colidx[k]];
            #pragma omp atomic
            y[endrow] += yi;
        }
    }

    if (ad && diagsize > 0) {
        #pragma omp for
        for (idx_t i = 0; i < num_rows; i++)
            y[i] += ad[i]*x[i];
    } else {
        #pragma omp barrier
    }
    return 0;
#else
    if (ad) {
        return csrgemvsd(
            num_rows, y, num_columns, x, csrsize,
            rowsizemin, rowsizemax, rowptr, colidx, a, ad);
    } else {
        return csrgemv(
            num_rows, y, num_columns, x, csrsize,
            rowsizemin, rowsizemax, rowptr, colidx, a);
    }
#endif
}

/**
 * `main()`.
 */
int main(int argc, char *argv[])
{
    int err;
    struct timespec t0, t1;
    setlocale(LC_ALL, "");

#ifdef HAVE_ALIGNED_ALLOC
    long pagesize = sysconf(_SC_PAGESIZE);
#endif

    /* Set program invocation name. */
    program_invocation_name = argv[0];
    program_invocation_short_name = (
        strrchr(program_invocation_name, '/')
        ? strrchr(program_invocation_name, '/') + 1
        : program_invocation_name);

    /* 1. Parse program options. */
    struct program_options args;
    int nargs;
    err = parse_program_options(argc, argv, &args, &nargs);
    if (err) {
        fprintf(stderr, "%s: %s %s\n", program_invocation_short_name,
                strerror(err), argv[nargs]);
        return EXIT_FAILURE;
    }

#ifdef _OPENMP
    #pragma omp parallel
    {
      /*
       * This empty parallel section is used to make the OpenMP
       * runtime output its configuration now if the environment
       * variable OMP_DISPLAY_ENV is set.
       */
    }
#endif

    /* 2. Read the matrix from a Matrix Market file. */
    if (args.verbose > 0) {
        fprintf(stderr, "mtxfile_read: ");
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    enum streamtype streamtype;
    union stream stream;
#ifdef HAVE_LIBZ
    if (!args.gzip) {
#endif
        streamtype = stream_stdio;
        if ((stream.f = fopen(args.Apath, "r")) == NULL) {
            fprintf(stderr, "%s: %s: %s\n",
                    program_invocation_short_name, args.Apath, strerror(errno));
            program_options_free(&args);
            return EXIT_FAILURE;
        }
#ifdef HAVE_LIBZ
    } else {
        streamtype = stream_zlib;
        if ((stream.gzf = gzopen(args.Apath, "r")) == NULL) {
            fprintf(stderr, "%s: %s: %s\n",
                    program_invocation_short_name, args.Apath, strerror(errno));
            program_options_free(&args);
            return EXIT_FAILURE;
        }
    }
#endif

    enum mtxobject object;
    enum mtxformat format;
    enum mtxfield field;
    enum mtxsymmetry symmetry;
    idx_t num_rows;
    idx_t num_columns;
    int64_t num_nonzeros;
    int64_t lines_read = 0;
    int64_t bytes_read = 0;
    err = mtxfile_fread_header(
        &object, &format, &field, &symmetry,
        &num_rows, &num_columns, &num_nonzeros,
        streamtype, stream, &lines_read, &bytes_read);
    if (err) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s:%"PRId64": %s\n",
                program_invocation_short_name,
                args.Apath, lines_read+1, strerror(err));
        stream_close(streamtype, stream);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
#ifdef HAVE_ALIGNED_ALLOC
    size_t rowidxsize = num_nonzeros*sizeof(idx_t);
    idx_t * rowidx = aligned_alloc(pagesize, rowidxsize + pagesize - rowidxsize % pagesize);
#else
    idx_t * rowidx = malloc(num_nonzeros * sizeof(idx_t));
#endif
    if (!rowidx) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        stream_close(streamtype, stream);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
#ifdef HAVE_ALIGNED_ALLOC
    size_t colidxsize = num_nonzeros*sizeof(idx_t);
    idx_t * colidx = aligned_alloc(pagesize, colidxsize + pagesize - colidxsize % pagesize);
#else
    idx_t * colidx = malloc(num_nonzeros * sizeof(idx_t));
#endif
    if (!colidx) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(rowidx);
        stream_close(streamtype, stream);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
#ifdef HAVE_ALIGNED_ALLOC
    size_t asize = num_nonzeros*sizeof(double);
    double * a = aligned_alloc(pagesize, asize + pagesize - asize % pagesize);
#else
    double * a = malloc(num_nonzeros * sizeof(double));
#endif
    if (!a) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(colidx); free(rowidx);
        stream_close(streamtype, stream);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    err = mtxfile_fread_matrix_coordinate(
        field, num_rows, num_columns, num_nonzeros, rowidx, colidx, a,
        streamtype, stream, &lines_read, &bytes_read);
    if (err) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s:%"PRId64": %s\n",
                program_invocation_short_name,
                args.Apath, lines_read+1, strerror(err));
        free(a); free(colidx); free(rowidx);
        stream_close(streamtype, stream);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    if (args.verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(stderr, "%'.6f seconds (%'.1f MB/s)\n",
                timespec_duration(t0, t1),
                1.0e-6 * bytes_read / timespec_duration(t0, t1));
    }
    stream_close(streamtype, stream);

    /* 3. Convert to CSR format. */
    if (args.verbose > 0) {
        fprintf(stderr, "csr_from_coo: ");
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

#ifdef HAVE_ALIGNED_ALLOC
    size_t csrrowptrsize = (num_rows+1)*sizeof(int64_t);
    int64_t * csrrowptr = aligned_alloc(pagesize, csrrowptrsize + pagesize - csrrowptrsize % pagesize);
#else
    int64_t * csrrowptr = malloc((num_rows+1) * sizeof(int64_t));
#endif
    if (!csrrowptr) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(a); free(colidx); free(rowidx);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    int64_t csrsize;
    idx_t rowsizemin, rowsizemax;
    idx_t diagsize;
    err = csr_from_coo_size(
        symmetry, num_rows, num_columns, num_nonzeros, rowidx, colidx, a,
        csrrowptr, &csrsize, &rowsizemin, &rowsizemax, &diagsize,
        args.separate_diagonal, args.partition);
    if (err) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(err));
        free(csrrowptr); free(a); free(colidx); free(rowidx);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    /* precompute per-thread partitioning of rows/columns/nonzeros */
    idx_t * startrows = NULL;
    idx_t * endrows = NULL;
    idx_t * startcolumns = NULL;
    idx_t * endcolumns = NULL;
#ifdef _OPENMP
    if (args.partition == partition_rows && args.rows_per_thread ||
        args.partition == partition_nonzeros && args.precompute_partition)
    {
        #pragma omp parallel
        #pragma omp master
        {
            int nthreads = omp_get_num_threads();
            startrows = malloc(nthreads * sizeof(idx_t));
        }
        if (!startrows) {
            if (args.verbose > 0) fprintf(stderr, "\n");
            fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
            free(csrrowptr); free(a); free(colidx); free(rowidx);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
        #pragma omp parallel
        #pragma omp master
        {
            int nthreads = omp_get_num_threads();
            endrows = malloc(nthreads * sizeof(idx_t));
        }
        if (!endrows) {
            if (args.verbose > 0) fprintf(stderr, "\n");
            fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
            free(startrows);
            free(csrrowptr); free(a); free(colidx); free(rowidx);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
    }
    if (args.partition == partition_rows && args.columns_per_thread)
    {
        #pragma omp parallel
        #pragma omp master
        {
            int nthreads = omp_get_num_threads();
            startcolumns = malloc(nthreads * sizeof(idx_t));
        }
        if (!startcolumns) {
            if (args.verbose > 0) fprintf(stderr, "\n");
            fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
            free(endrows); free(startrows);
            free(csrrowptr); free(a); free(colidx); free(rowidx);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
        #pragma omp parallel
        #pragma omp master
        {
            int nthreads = omp_get_num_threads();
            endcolumns = malloc(nthreads * sizeof(idx_t));
        }
        if (!endcolumns) {
            if (args.verbose > 0) fprintf(stderr, "\n");
            fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
            free(startcolumns); free(endrows); free(startrows);
            free(csrrowptr); free(a); free(colidx); free(rowidx);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
    }

    if (args.partition == partition_rows && args.rows_per_thread) {
        int nthreads;
        #pragma omp parallel
        #pragma omp master
        {
            nthreads = omp_get_num_threads();
            if (nthreads > 0) startrows[0] = 0;
            if (nthreads > 0) endrows[0] = args.rows_per_thread > 0 ? args.rows_per_thread[0] : 0;
            for (int p = 1; p < nthreads; p++) {
                startrows[p] = endrows[p-1];
                if (p < args.rows_per_thread_size) endrows[p] = startrows[p] + args.rows_per_thread[p];
                else endrows[p] = startrows[p];
            }
        }
        if (args.rows_per_thread_size != nthreads) {
            if (args.verbose > 0) fprintf(stderr, "\n");
            fprintf(stderr, "%s: warning: --rows-per-thread does not match the number of threads (%d)\n",
                    program_invocation_short_name, nthreads);
        }
        if (nthreads > 0 && endrows[nthreads-1] > num_rows) {
            if (args.verbose > 0) fprintf(stderr, "\n");
            fprintf(stderr, "%s: %s: the sum of --rows-per-thread (%'"PRIdx") exceeds the number of rows (%'"PRIdx")\n",
                    program_invocation_short_name, strerror(EINVAL), endrows[nthreads-1], num_rows);
            free(endrows); free(startrows);
            free(csrrowptr); free(a); free(colidx); free(rowidx);
            program_options_free(&args);
            return EXIT_FAILURE;
        } else if (nthreads > 0 && endrows[nthreads-1] < num_rows) {
            if (args.verbose > 0) fprintf(stderr, "\n");
            fprintf(stderr, "%s: warning: the sum of --rows-per-thread (%'"PRIdx") is less than the number of rows (%'"PRIdx")\n",
                    program_invocation_short_name, endrows[nthreads-1], num_rows);
        }
    } else if (args.partition == partition_nonzeros &&
               args.precompute_partition)
    {
        #pragma omp parallel
        {
            int nthreads = omp_get_num_threads();
            int p = omp_get_thread_num();
            int64_t startnz = p*(csrsize+nthreads-1)/nthreads;
            int64_t endnz = (p+1)*(csrsize+nthreads-1)/nthreads;
            if (endnz > csrsize) endnz = csrsize;
            idx_t startrow = 0;
            while (startrow < num_rows && startnz > csrrowptr[startrow+1]) startrow++;
            idx_t endrow = startrow;
            while (endrow < num_rows && endnz-1 > csrrowptr[endrow+1]) endrow++;
            startrows[p] = startrow;
            endrows[p] = endrow;
        }
    }

    if (args.partition == partition_rows && args.columns_per_thread) {
        int nthreads;
        #pragma omp parallel
        #pragma omp master
        {
            nthreads = omp_get_num_threads();
            if (nthreads > 0) startcolumns[0] = 0;
            if (nthreads > 0) endcolumns[0] = args.columns_per_thread > 0 ? args.columns_per_thread[0] : 0;
            for (int p = 1; p < nthreads; p++) {
                startcolumns[p] = endcolumns[p-1];
                if (p < args.columns_per_thread_size) endcolumns[p] = startcolumns[p] + args.columns_per_thread[p];
                else endcolumns[p] = startcolumns[p];
            }
        }
        if (args.columns_per_thread_size != nthreads) {
            if (args.verbose > 0) fprintf(stderr, "\n");
            fprintf(stderr, "%s: warning: --columns-per-thread does not match the number of threads (%d)\n",
                    program_invocation_short_name, nthreads);
        }
        if (nthreads > 0 && endcolumns[nthreads-1] > num_columns) {
            if (args.verbose > 0) fprintf(stderr, "\n");
            fprintf(stderr, "%s: %s: the sum of --columns-per-thread (%'"PRIdx") exceeds the number of columns (%'"PRIdx")\n",
                    program_invocation_short_name, strerror(EINVAL), endcolumns[nthreads-1], num_columns);
            free(endcolumns); free(startcolumns); free(endrows); free(startrows);
            free(csrrowptr); free(a); free(colidx); free(rowidx);
            program_options_free(&args);
            return EXIT_FAILURE;
        } else if (nthreads > 0 && endcolumns[nthreads-1] < num_columns) {
            if (args.verbose > 0) fprintf(stderr, "\n");
            fprintf(stderr, "%s: warning: the sum of --columns-per-thread (%'"PRIdx") is less than the number of columns (%'"PRIdx")\n",
                    program_invocation_short_name, endcolumns[nthreads-1], num_columns);
        }
    }
#endif

#ifdef HAVE_ALIGNED_ALLOC
    size_t csrcolidxsize = csrsize*sizeof(idx_t);
    idx_t * csrcolidx = aligned_alloc(pagesize, csrcolidxsize + pagesize - csrcolidxsize % pagesize);
#else
    idx_t * csrcolidx = malloc(csrsize * sizeof(idx_t));
#endif
    if (!csrcolidx) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(endcolumns); free(startcolumns); free(endrows); free(startrows);
        free(csrrowptr); free(a); free(colidx); free(rowidx);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
#ifdef _OPENMP
    if (args.partition == partition_rows && !args.rows_per_thread) {
        #pragma omp parallel for
        for (idx_t i = 0; i < num_rows; i++) {
            for (int64_t k = csrrowptr[i]; k < csrrowptr[i+1]; k++)
                csrcolidx[k] = 0;
        }
    } else if (args.partition == partition_rows) {
        #pragma omp parallel
        {
            int p = omp_get_thread_num();
            for (idx_t i = startrows[p]; i < endrows[p]; i++) {
                for (int64_t k = csrrowptr[i]; k < csrrowptr[i+1]; k++)
                    csrcolidx[k] = 0;
            }
        }
    } else if (args.partition == partition_nonzeros) {
        #pragma omp parallel for
        for (int64_t k = 0; k < csrsize; k++) csrcolidx[k] = 0;
    }
#endif
#ifdef HAVE_ALIGNED_ALLOC
    size_t csrasize = csrsize*sizeof(double);
    double * csra = aligned_alloc(pagesize, csrasize + pagesize - csrasize % pagesize);
#else
    double * csra = malloc(csrsize * sizeof(double));
#endif
    if (!csra) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(csrcolidx);
        free(endcolumns); free(startcolumns); free(endrows); free(startrows);
        free(csrrowptr); free(a); free(colidx); free(rowidx);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
#ifdef HAVE_ALIGNED_ALLOC
    size_t csradsize = diagsize*sizeof(double);
    double * csrad = aligned_alloc(pagesize, csradsize + pagesize - csradsize % pagesize);
#else
    double * csrad = malloc(diagsize * sizeof(double));
#endif
    if (!csrad) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(csra); free(csrcolidx);
        free(endcolumns); free(startcolumns); free(endrows); free(startrows);
        free(csrrowptr); free(a); free(colidx); free(rowidx);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
#ifdef _OPENMP
    if (args.partition == partition_rows && !args.rows_per_thread) {
        #pragma omp parallel for
        for (idx_t i = 0; i < num_rows; i++) {
            for (int64_t k = csrrowptr[i]; k < csrrowptr[i+1]; k++)
                csra[k] = 0;
        }
        if (diagsize > 0) {
            #pragma omp parallel for
            for (idx_t i = 0; i < num_rows; i++) csrad[i] = 0;
        }
    } else if (args.partition == partition_rows) {
        #pragma omp parallel
        {
            int p = omp_get_thread_num();
            for (idx_t i = startrows[p]; i < endrows[p]; i++) {
                for (int64_t k = csrrowptr[i]; k < csrrowptr[i+1]; k++)
                    csra[k] = 0;
            }
            if (diagsize > 0) {
                for (idx_t i = startrows[p]; i < endrows[p]; i++) csrad[i] = 0;
            }
        }
    } else if (args.partition == partition_nonzeros) {
        #pragma omp parallel for
        for (int64_t k = 0; k < csrsize; k++) csra[k] = 0;
        if (diagsize > 0) {
            #pragma omp parallel for
            for (idx_t i = 0; i < num_rows; i++) csrad[i] = 0;
        }
    }
#endif
    err = csr_from_coo(
        symmetry, num_rows, num_columns, num_nonzeros, rowidx, colidx, a,
        csrrowptr, csrsize, rowsizemin, rowsizemax, csrcolidx, csra, csrad,
        args.separate_diagonal, args.sort_rows, args.partition);
    if (err) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(err));
        free(csrad); free(csra); free(csrcolidx);
        free(endcolumns); free(startcolumns); free(endrows); free(startrows);
        free(csrrowptr); free(a); free(colidx); free(rowidx);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    free(a); free(colidx); free(rowidx);

    if (args.verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(stderr, "%'.6f seconds, %'"PRIdx" rows, %'"PRIdx" columns, %'"PRId64" nonzeros"
                ", %'"PRIdx" to %'"PRIdx" nonzeros per row",
                timespec_duration(t0, t1), num_rows, num_columns, csrsize+diagsize, rowsizemin, rowsizemax);
#ifdef _OPENMP
        int nthreads;
        idx_t min_rows_per_thread = IDX_T_MAX;
        idx_t max_rows_per_thread = 0;
        int64_t min_nonzeros_per_thread = INT64_MAX;
        int64_t max_nonzeros_per_thread = 0;
        if (args.partition == partition_rows && !args.rows_per_thread) {
            #pragma omp parallel \
                reduction(min:min_rows_per_thread) reduction(max:max_rows_per_thread) \
                reduction(min:min_nonzeros_per_thread) reduction(max:max_nonzeros_per_thread)
            {
                nthreads = omp_get_num_threads();
                int p = omp_get_thread_num();
                min_rows_per_thread = max_rows_per_thread = num_rows/nthreads + (p < (num_rows % nthreads));
                int64_t num_nonzeros = 0;
                #pragma omp for
                for (int i = 0; i < num_rows; i++)
                    num_nonzeros += csrrowptr[i+1]-csrrowptr[i] + (diagsize > 0 ? 1 : 0);
                min_nonzeros_per_thread = num_nonzeros;
                max_nonzeros_per_thread = num_nonzeros;
            }
        } else if (args.partition == partition_rows) {
            #pragma omp parallel \
                reduction(min:min_rows_per_thread) reduction(max:max_rows_per_thread) \
                reduction(min:min_nonzeros_per_thread) reduction(max:max_nonzeros_per_thread)
            {
                nthreads = omp_get_num_threads();
                int p = omp_get_thread_num();
                idx_t startrow = startrows[p];
                idx_t endrow = endrows[p];
                min_rows_per_thread = max_rows_per_thread = endrow - startrow;
                int64_t num_nonzeros = 0;
                for (idx_t i = startrows[p]; i < endrows[p]; i++)
                    num_nonzeros += csrrowptr[i+1]-csrrowptr[i] + (diagsize > 0 ? 1 : 0);
                min_nonzeros_per_thread = num_nonzeros;
                max_nonzeros_per_thread = num_nonzeros;
            }
        } else if (args.partition == partition_nonzeros) {
            #pragma omp parallel \
                reduction(min:min_rows_per_thread) reduction(max:max_rows_per_thread) \
                reduction(min:min_nonzeros_per_thread) reduction(max:max_nonzeros_per_thread)
            {
                nthreads = omp_get_num_threads();
                int p = omp_get_thread_num();
                int64_t startnz = p*(csrsize+nthreads-1)/nthreads;
                int64_t endnz = (p+1)*(csrsize+nthreads-1)/nthreads;
                if (endnz > csrsize) endnz = csrsize;
                idx_t startrow = 0;
                if (startrows) { startrow = startrows[p]; }
                else { while (startrow < num_rows && startnz > csrrowptr[startrow+1]) startrow++; }
                idx_t endrow = startrow;
                if (endrows) { endrow = endrows[p]; }
                else { while (endrow < num_rows && endnz-1 > csrrowptr[endrow+1]) endrow++; }
                min_rows_per_thread = max_rows_per_thread = endrow - startrow;
                min_nonzeros_per_thread = max_nonzeros_per_thread = csrsize/nthreads + (p < (csrsize % nthreads));
            }
        }
        fprintf(stderr, ", %'d threads, %'"PRIdx" to %'"PRIdx" rows per thread, %'"PRId64" to %'"PRId64" nonzeros per thread",
                nthreads, min_rows_per_thread, max_rows_per_thread,
                min_nonzeros_per_thread, max_nonzeros_per_thread);
#endif
        fputc('\n', stderr);
    }

    /* 4. allocate vectors */
#ifdef HAVE_ALIGNED_ALLOC
    size_t xsize = num_columns*sizeof(double);
    double * x = aligned_alloc(pagesize, xsize + pagesize - xsize % pagesize);
#else
    double * x = malloc(num_columns * sizeof(double));
#endif
    if (!x) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(endcolumns); free(startcolumns); free(endrows); free(startrows);
        free(csrad); free(csra); free(csrcolidx); free(csrrowptr);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

#ifdef _OPENMP
    if (args.partition == partition_rows && args.columns_per_thread) {
        #pragma omp parallel
        {
            int p = omp_get_thread_num();
            for (idx_t i = startcolumns[p]; i < endcolumns[p]; i++) x[i] = 1.0;
            int nthreads = omp_get_num_threads();
            #pragma omp master
            for (idx_t i = endcolumns[nthreads-1]; i < num_columns; i++) x[i] = 1.0;
        }
    } else if (args.partition == partition_rows && args.rows_per_thread &&
               num_rows == num_columns)
    {
        #pragma omp parallel
        {
            int p = omp_get_thread_num();
            for (idx_t i = startrows[p]; i < endrows[p]; i++) x[i] = 1.0;
            int nthreads = omp_get_num_threads();
            #pragma omp master
            for (idx_t i = endrows[nthreads-1]; i < num_rows; i++) x[i] = 1.0;
        }
    } else {
        #pragma omp parallel for
        for (idx_t i = 0; i < num_columns; i++) x[i] = 1.0;
    }
#else
    for (idx_t i = 0; i < num_columns; i++) x[i] = 1.0;
#endif

    /* read x vector from a Matrix Market file */
    if (args.xpath) {
        if (args.verbose > 0) {
            fprintf(stderr, "mtxfile_read: ");
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        enum streamtype streamtype;
        union stream stream;
#ifdef HAVE_LIBZ
        if (!args.gzip) {
#endif
            streamtype = stream_stdio;
            if ((stream.f = fopen(args.xpath, "r")) == NULL) {
                fprintf(stderr, "%s: %s: %s\n",
                        program_invocation_short_name, args.xpath, strerror(errno));
                free(x);
                free(endcolumns); free(startcolumns); free(endrows); free(startrows);
                free(csrad); free(csra); free(csrcolidx); free(csrrowptr);
                program_options_free(&args);
                return EXIT_FAILURE;
            }
#ifdef HAVE_LIBZ
        } else {
            streamtype = stream_zlib;
            if ((stream.gzf = gzopen(args.xpath, "r")) == NULL) {
                fprintf(stderr, "%s: %s: %s\n",
                        program_invocation_short_name, args.xpath, strerror(errno));
                free(x);
                free(endcolumns); free(startcolumns); free(endrows); free(startrows);
                free(csrad); free(csra); free(csrcolidx); free(csrrowptr);
                program_options_free(&args);
                return EXIT_FAILURE;
            }
        }
#endif

        enum mtxobject object;
        enum mtxformat format;
        enum mtxfield field;
        enum mtxsymmetry symmetry;
        idx_t xnum_rows;
        idx_t xnum_columns;
        int64_t xnum_nonzeros;
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        err = mtxfile_fread_header(
            &object, &format, &field, &symmetry,
            &xnum_rows, &xnum_columns, &xnum_nonzeros,
            streamtype, stream, &lines_read, &bytes_read);
        if (err) {
            if (args.verbose > 0) fprintf(stderr, "\n");
            fprintf(stderr, "%s: %s:%"PRId64": %s\n",
                    program_invocation_short_name,
                    args.xpath, lines_read+1, strerror(err));
            stream_close(streamtype, stream);
            free(x);
            free(endcolumns); free(startcolumns); free(endrows); free(startrows);
            free(csrad); free(csra); free(csrcolidx); free(csrrowptr);
            program_options_free(&args);
            return EXIT_FAILURE;
        } else if (object != mtxvector || format != mtxarray || xnum_rows != num_columns) {
            if (args.verbose > 0) fprintf(stderr, "\n");
            fprintf(stderr, "%s: %s:%"PRId64": "
                    "expected vector in array format of size %"PRIdx"\n",
                    program_invocation_short_name,
                    args.xpath, lines_read+1, num_columns);
            stream_close(streamtype, stream);
            free(x);
            free(endcolumns); free(startcolumns); free(endrows); free(startrows);
            free(csrad); free(csra); free(csrcolidx); free(csrrowptr);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        err = mtxfile_fread_vector_array(
            field, num_rows, x, streamtype, stream, &lines_read, &bytes_read);
        if (err) {
            if (args.verbose > 0) fprintf(stderr, "\n");
            fprintf(stderr, "%s: %s:%"PRId64": %s\n",
                    program_invocation_short_name,
                    args.xpath, lines_read+1, strerror(err));
            free(x);
            free(endcolumns); free(startcolumns); free(endrows); free(startrows);
            free(csrad); free(csra); free(csrcolidx); free(csrrowptr);
            stream_close(streamtype, stream);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(stderr, "%'.6f seconds (%'.1f MB/s)\n",
                    timespec_duration(t0, t1),
                    1.0e-6 * bytes_read / timespec_duration(t0, t1));
        }
        stream_close(streamtype, stream);
    }

#ifdef HAVE_ALIGNED_ALLOC
    size_t ysize = num_rows*sizeof(double);
    double * y = aligned_alloc(pagesize, ysize + pagesize - ysize % pagesize);
#else
    double * y = malloc(num_rows * sizeof(double));
#endif
    if (!y) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(x);
        free(endcolumns); free(startcolumns); free(endrows); free(startrows);
        free(csrad); free(csra); free(csrcolidx); free(csrrowptr);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

#ifdef _OPENMP
    if (args.partition == partition_rows && !args.rows_per_thread) {
        #pragma omp parallel for
        for (idx_t i = 0; i < num_rows; i++) y[i] = 0.0;
    } else if (args.partition == partition_rows) {
        #pragma omp parallel
        {
            int p = omp_get_thread_num();
            for (idx_t i = startrows[p]; i < endrows[p]; i++) y[i] = 0.0;
            int nthreads = omp_get_num_threads();
            #pragma omp master
            for (idx_t i = endrows[nthreads-1]; i < num_rows; i++) y[i] = 0;
        }
    } else if (args.partition == partition_nonzeros) {
        #pragma omp parallel
        {
            int nthreads = omp_get_num_threads();
            int p = omp_get_thread_num();
            int64_t startnz = p*(csrsize+nthreads-1)/nthreads;
            int64_t endnz = (p+1)*(csrsize+nthreads-1)/nthreads;
            if (endnz > csrsize) endnz = csrsize;
            idx_t startrow = 0;
            if (startrows) { startrow = startrows[p]; }
            else { while (startrow < num_rows && startnz > csrrowptr[startrow+1]) startrow++; }
            idx_t endrow = startrow;
            if (endrows) { endrow = endrows[p]; }
            else { while (endrow < num_rows && endnz-1 > csrrowptr[endrow+1]) endrow++; }
            for (idx_t i = startrow; i < endrow; i++) y[i] = 0.0;
        }
    }
#else
    for (idx_t i = 0; i < num_rows; i++) y[i] = 0.0;
#endif

    /* read y vector from a Matrix Market file */
    if (args.ypath) {
        if (args.verbose > 0) {
            fprintf(stderr, "mtxfile_read: ");
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        enum streamtype streamtype;
        union stream stream;
#ifdef HAVE_LIBZ
        if (!args.gzip) {
#endif
            streamtype = stream_stdio;
            if ((stream.f = fopen(args.ypath, "r")) == NULL) {
                fprintf(stderr, "%s: %s: %s\n",
                        program_invocation_short_name, args.ypath, strerror(errno));
                free(y); free(x);
                free(endcolumns); free(startcolumns); free(endrows); free(startrows);
                free(csrad); free(csra); free(csrcolidx); free(csrrowptr);
                program_options_free(&args);
                return EXIT_FAILURE;
            }
#ifdef HAVE_LIBZ
        } else {
            streamtype = stream_zlib;
            if ((stream.gzf = gzopen(args.ypath, "r")) == NULL) {
                fprintf(stderr, "%s: %s: %s\n",
                        program_invocation_short_name, args.ypath, strerror(errno));
                free(y); free(x);
                free(endcolumns); free(startcolumns); free(endrows); free(startrows);
                free(csrad); free(csra); free(csrcolidx); free(csrrowptr);
                program_options_free(&args);
                return EXIT_FAILURE;
            }
        }
#endif

        enum mtxobject object;
        enum mtxformat format;
        enum mtxfield field;
        enum mtxsymmetry symmetry;
        idx_t ynum_rows;
        idx_t ynum_columns;
        int64_t ynum_nonzeros;
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        err = mtxfile_fread_header(
            &object, &format, &field, &symmetry,
            &ynum_rows, &ynum_columns, &ynum_nonzeros,
            streamtype, stream, &lines_read, &bytes_read);
        if (err) {
            if (args.verbose > 0) fprintf(stderr, "\n");
            fprintf(stderr, "%s: %s:%"PRId64": %s\n",
                    program_invocation_short_name,
                    args.ypath, lines_read+1, strerror(err));
            stream_close(streamtype, stream);
            free(y); free(x);
            free(endcolumns); free(startcolumns); free(endrows); free(startrows);
            free(csrad); free(csra); free(csrcolidx); free(csrrowptr);
            program_options_free(&args);
            return EXIT_FAILURE;
        } else if (object != mtxvector || format != mtxarray || ynum_rows != num_rows) {
            if (args.verbose > 0) fprintf(stderr, "\n");
            fprintf(stderr, "%s: %s:%"PRId64": "
                    "expected vector in array format of size %'"PRIdx"\n",
                    program_invocation_short_name,
                    args.ypath, lines_read+1, num_rows);
            stream_close(streamtype, stream);
            free(y); free(x);
            free(endcolumns); free(startcolumns); free(endrows); free(startrows);
            free(csrad); free(csra); free(csrcolidx); free(csrrowptr);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        err = mtxfile_fread_vector_array(
            field, num_rows, y, streamtype, stream, &lines_read, &bytes_read);
        if (err) {
            if (args.verbose > 0) fprintf(stderr, "\n");
            fprintf(stderr, "%s: %s:%"PRId64": %s\n",
                    program_invocation_short_name,
                    args.ypath, lines_read+1, strerror(err));
            free(y); free(x);
            free(endcolumns); free(startcolumns); free(endrows); free(startrows);
            free(csrad); free(csra); free(csrcolidx); free(csrrowptr);
            stream_close(streamtype, stream);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(stderr, "%'.6f seconds (%'.1f MB/s)\n",
                    timespec_duration(t0, t1),
                    1.0e-6 * bytes_read / timespec_duration(t0, t1));
        }
        stream_close(streamtype, stream);
    }

    /*
     * 5. compute the matrix-vector multiplication.
     */

    /* configure hardware performance monitoring with PAPI */
#ifdef HAVE_PAPI
    int papierr = 0;
    struct papi_util_opt papi_opt = {
        .event_file = args.papi_event_file,
        .print_csv = args.papi_event_format == 1,
        .print_threads = args.papi_event_per_thread,
        .print_summary = args.papi_event_summary,
        .print_region = 0,
        .component = 0,
        .multiplex = 0,
        .output = stderr
    };

    if (papi_opt.event_file) {
        fprintf(stderr, "[PAPI util] using event file: %s\n", papi_opt.event_file);
        err = PAPI_UTIL_setup(&papi_opt, &papierr);
        if (err) {
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name, PAPI_UTIL_strerror(err, papierr));
            free(y); free(x);
            free(endcolumns); free(startcolumns); free(endrows); free(startrows);
            free(csrad); free(csra); free(csrcolidx); free(csrrowptr);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
    }
#endif

    /* enable A64FX sector cache */
#if defined(__FCC_version__) && defined(USE_A64FX_SECTOR_CACHE)
#ifdef A64FX_SECTOR_CACHE_L1_WAYS
    #pragma statement scache_isolate_way L2=A64FX_SECTOR_CACHE_L2_WAYS L1=A64FX_SECTOR_CACHE_L1_WAYS
    if (args.verbose >= 0) fprintf(stderr, "enabling sector cache (%d L2 ways, %d L1 ways)\n", A64FX_SECTOR_CACHE_L2_WAYS, A64FX_SECTOR_CACHE_L1_WAYS);
#else
    #pragma statement scache_isolate_way L2=A64FX_SECTOR_CACHE_L2_WAYS
    if (args.verbose >= 0) fprintf(stderr, "enabling sector cache (%d L2 ways)\n", A64FX_SECTOR_CACHE_L2_WAYS);
#endif
#endif

    /* configure A64FX prefetch distance */
#if defined(__FCC_version__)
    uint64_t * a64fxpfdst;
#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
#ifdef _OPENMP
        int t = omp_get_thread_num();
        #pragma omp master
        a64fxpfdst = malloc(omp_get_num_threads() * sizeof(uint64_t));
        #pragma omp barrier
#else
        int t = 0;
        a64fxpfdst = malloc(sizeof(uint64_t));
#endif
    }
    if (!a64fxpfdst) {
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(y); free(x);
        free(endcolumns); free(startcolumns); free(endrows); free(startrows);
        free(csrad); free(csra); free(csrcolidx); free(csrrowptr);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
#ifdef _OPENMP
        int t = omp_get_thread_num();
#else
        int t = 0;
#endif

        uint64_t tmp = 0;
        if (args.l1pfdst >= 0 || args.l2pfdst >= 0) A64FX_READ_PF_DST(tmp);
        a64fxpfdst[t] = tmp;

        if (args.l1pfdst >= 0) {
#ifdef _OPENMP
#pragma omp master
#endif
            if (args.verbose >= 0) fprintf(stderr, "setting L1 prefetch distance to %d\n", args.l1pfdst);
            A64FX_SET_PF_DST_L1(tmp, args.l1pfdst);
        }
        if (args.l2pfdst >= 0) {
#ifdef _OPENMP
#pragma omp master
#endif
            if (args.verbose > 0) fprintf(stderr, "setting L2 prefetch distance to %d\n", args.l2pfdst);
            A64FX_SET_PF_DST_L2(tmp, args.l2pfdst);
        }

#pragma omp master
        if (args.verbose > 0 && (args.l1pfdst >= 0 || args.l2pfdst >= 0)) {
            A64FX_READ_PF_DST(tmp);
            fprintf(stderr, "register value: ");
            print_bits(stderr, tmp);
            fprintf(stderr, "\n");
        }
    }
#endif

    /* perform warmup iterations */
#ifdef _OPENMP
    #pragma omp parallel
#endif
    for (int repeat = 0; repeat < args.warmup; repeat++) {
#ifdef _OPENMP
        #pragma omp barrier
        #pragma omp master
#endif
        if (args.verbose > 0) {
            if (args.separate_diagonal) fprintf(stderr, "gemvsd (warmup): ");
            else fprintf(stderr, "gemv (warmup): ");
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }
#ifdef _OPENMP
        #pragma omp barrier
#endif

        int priverr = 0;
        if (args.partition == partition_rows && !args.rows_per_thread) {
            if (args.separate_diagonal) {
                priverr = csrgemvsd(
                    num_rows, y, num_columns, x, csrsize, rowsizemin, rowsizemax, csrrowptr, csrcolidx, csra, csrad);
            } else {
                priverr = csrgemv(
                    num_rows, y, num_columns, x, csrsize, rowsizemin, rowsizemax, csrrowptr, csrcolidx, csra);
            }
        } else if (args.partition == partition_rows) {
            priverr = csrgemvrp(
                num_rows, y, num_columns, x, csrsize, rowsizemin, rowsizemax, csrrowptr, csrcolidx, csra, diagsize, csrad,
                startrows, endrows);
        } else if (args.partition == partition_nonzeros) {
            priverr = csrgemvnz(
                num_rows, y, num_columns, x, csrsize, rowsizemin, rowsizemax, csrrowptr, csrcolidx, csra, diagsize, csrad,
                startrows, endrows);
        }

#ifdef _OPENMP
        #pragma omp barrier
        clock_gettime(CLOCK_MONOTONIC, &t1);
        for (int t = 0; t < omp_get_num_threads(); t++) {
            if (t == omp_get_thread_num() && !err && priverr) err = priverr;
            #pragma omp barrier
        }
#else
        if (priverr) { err = priverr; }
#endif
        if (err) break;

        int64_t num_flops = 2*(csrsize+diagsize);
        int64_t min_bytes = num_rows*sizeof(*y) + num_columns*sizeof(*x)
            + (num_rows+1)*sizeof(*csrrowptr) + csrsize*sizeof(*csrcolidx) + csrsize*sizeof(*csra) + diagsize*sizeof(*csrad);
        int64_t max_bytes = num_rows*sizeof(*y) + csrsize*sizeof(*x)
            + num_rows*sizeof(*csrrowptr) + csrsize*sizeof(*csrcolidx) + csrsize*sizeof(*csra)
            + diagsize*sizeof(*csrad) + diagsize*sizeof(*x);

#ifdef _OPENMP
        #pragma omp barrier
        #pragma omp master
#endif
        if (args.verbose > 0) {
            fprintf(stderr, "%'.6f seconds (%'.3f Gnz/s, %'.3f Gflop/s, %'.1f to %'.1f GB/s)\n",
                    timespec_duration(t0, t1),
                    (double) num_nonzeros * 1e-9 / (double) timespec_duration(t0, t1),
                    (double) num_flops * 1e-9 / (double) timespec_duration(t0, t1),
                    (double) min_bytes * 1e-9 / (double) timespec_duration(t0, t1),
                    (double) max_bytes * 1e-9 / (double) timespec_duration(t0, t1));
        }
    }

    /* enable PAPI hardware performance monitoring */
#ifdef HAVE_PAPI
    if (papi_opt.event_file) {
        if (args.verbose > 0)
            fprintf(stderr, "[PAPI util] start recording events for region \"gemv\"\n");
        err = PAPI_UTIL_start("gemv", &papierr);
        if (err) {
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name, PAPI_UTIL_strerror(err, papierr));
#if defined(__FCC_version__)
            free(a64fxpfdst);
#endif
            free(y); free(x);
            free(endcolumns); free(startcolumns); free(endrows); free(startrows);
            free(csrad); free(csra); free(csrcolidx); free(csrrowptr);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
    }
#endif

    /* perform sparse matrix-vector multiplications */
#ifdef _OPENMP
    #pragma omp parallel
#endif
    for (int repeat = 0; repeat < args.repeat; repeat++) {
#ifdef _OPENMP
        #pragma omp barrier
        #pragma omp master
#endif
        if (args.verbose > 0) {
            if (args.separate_diagonal) fprintf(stderr, "gemvsd: ");
            else fprintf(stderr, "gemv: ");
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }
#ifdef _OPENMP
        #pragma omp barrier
#endif

        int priverr = 0;
        if (args.partition == partition_rows && !args.rows_per_thread) {
            if (args.separate_diagonal) {
                priverr = csrgemvsd(
                    num_rows, y, num_columns, x, csrsize, rowsizemin, rowsizemax, csrrowptr, csrcolidx, csra, csrad);
            } else {
                priverr = csrgemv(
                    num_rows, y, num_columns, x, csrsize, rowsizemin, rowsizemax, csrrowptr, csrcolidx, csra);
            }
        } else if (args.partition == partition_rows) {
            priverr = csrgemvrp(
                num_rows, y, num_columns, x, csrsize, rowsizemin, rowsizemax, csrrowptr, csrcolidx, csra, diagsize, csrad,
                startrows, endrows);
        } else if (args.partition == partition_nonzeros) {
            priverr = csrgemvnz(
                num_rows, y, num_columns, x, csrsize, rowsizemin, rowsizemax, csrrowptr, csrcolidx, csra, diagsize, csrad,
                startrows, endrows);
        }

#ifdef _OPENMP
        #pragma omp barrier
        clock_gettime(CLOCK_MONOTONIC, &t1);
        for (int t = 0; t < omp_get_num_threads(); t++) {
            if (t == omp_get_thread_num() && !err && priverr) err = priverr;
            #pragma omp barrier
        }
#else
        if (priverr) { err = priverr; }
#endif
        if (err) break;

        int64_t num_flops = 2*(csrsize+diagsize);
        int64_t min_bytes = num_rows*sizeof(*y) + num_columns*sizeof(*x)
            + (num_rows+1)*sizeof(*csrrowptr) + csrsize*sizeof(*csrcolidx) + csrsize*sizeof(*csra) + diagsize*sizeof(*csrad);
        int64_t max_bytes = num_rows*sizeof(*y) + csrsize*sizeof(*x)
            + num_rows*sizeof(*csrrowptr) + csrsize*sizeof(*csrcolidx) + csrsize*sizeof(*csra)
            + diagsize*sizeof(*csrad) + diagsize*sizeof(*x);

#ifdef _OPENMP
        #pragma omp barrier
        #pragma omp master
#endif
        if (args.verbose > 0) {
            fprintf(stderr, "%'.6f seconds (%'.3f Gnz/s, %'.3f Gflop/s, %'.1f to %'.1f GB/s)\n",
                    timespec_duration(t0, t1),
                    (double) num_nonzeros * 1e-9 / (double) timespec_duration(t0, t1),
                    (double) num_flops * 1e-9 / (double) timespec_duration(t0, t1),
                    (double) min_bytes * 1e-9 / (double) timespec_duration(t0, t1),
                    (double) max_bytes * 1e-9 / (double) timespec_duration(t0, t1));
        }
    }

    /* reset A64FX prefetch distance configuration */
#if defined(__FCC_version__)
#ifdef _OPENMP
    #pragma omp parallel
    {
        int t = omp_get_thread_num();
        if (args.l1pfdst >= 0 || args.l2pfdst >= 0) A64FX_WRITE_PF_DST(a64fxpfdst[t]);
    }
#else
    if (args.l1pfdst >= 0 || args.l2pfdst >= 0) A64FX_WRITE_PF_DST(*a64fxpfdst);
#endif
    free(a64fxpfdst);
#endif

#if defined(__FCC_version__) && defined(USE_A64FX_SECTOR_CACHE)
    #pragma statement end_scache_isolate_way
#endif

#ifdef HAVE_PAPI
    if (papi_opt.event_file) {
        PAPI_UTIL_finish();
        PAPI_UTIL_finalize();
    }
#endif

    if (err) {
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(err));
        free(y); free(x);
        free(endcolumns); free(startcolumns); free(endrows); free(startrows);
        free(csrad); free(csra); free(csrcolidx); free(csrrowptr);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    free(x);
    free(endcolumns); free(startcolumns); free(endrows); free(startrows);
    free(csrad); free(csra); free(csrcolidx); free(csrrowptr);

    /* 6. write the result vector to a file */
    if (!args.quiet) {
        if (args.verbose > 0) {
            fprintf(stderr, "mtxfile_write:\n");
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        fprintf(stdout, "%%%%MatrixMarket vector array real general\n");
        fprintf(stdout, "%"PRIdx"\n", num_rows);
        for (idx_t i = 0; i < num_rows; i++) fprintf(stdout, "%.*g\n", DBL_DIG, y[i]);
        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(stderr, "mtxfile_write done in %'.6f seconds\n", timespec_duration(t0, t1));
        }
    }

    free(y);
    program_options_free(&args);
    return EXIT_SUCCESS;
}

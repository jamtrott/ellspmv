/*
 * Benchmark program for ELLPACK SpMV
 *
 * Copyright (C) 2023 James D. Trotter
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
 * with matrices in ELLPACK format.
 *
 * Authors:
 *  James D. Trotter <james@simula.no>
 *  Sergej Breiter <breiter@nm.ifi.lmu.de>
 *
 *
 * History:
 *
 *  1.8 - 2023-05-30:
 *
 *   - add option for performing warmup iterations
 *
 *  1.7 - 2023-05-06:
 *
 *   - add an option for sorting nonzeros within each row by column
 *
 *  1.6 - 2023-05-04:
 *
 *   - enable A64FX sector cache before the loop that repeatedly
 *     performs SpMV.
 *
 *  1.4 - 2023-04-15:
 *
 *   - add support for performance monitoring using PAPI
 *
 *   - minor fixes to A64FX sector cache configuration
 *
 *  1.3 - 2023-03-01:
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
 *  1.2 — 2023-02-10:
 *
 *   - add option for separating diagonal and off-diagonal entries
 *
 *  1.1 — 2022-10-23:
 *
 *   - add cache partitioning for A64FX
 *
 *  1.0 — 2022-05-16:
 *
 *   - initial version
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

const char * program_name = "ellspmv";
const char * program_version = "1.8";
const char * program_copyright =
    "Copyright (C) 2023 James D. Trotter";
const char * program_license =
    "License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>\n"
    "This is free software: you are free to change and redistribute it.\n"
    "There is NO WARRANTY, to the extent permitted by law.";
const char * program_invocation_name;
const char * program_invocation_short_name;

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
    fprintf(f, "  A    path to Matrix Market file for the matrix A\n");
    fprintf(f, "  x    optional path to Matrix Market file for the vector x\n");
    fprintf(f, "  y    optional path for to Matrix Market file for the vector y\n");
    fprintf(f, "\n");
    fprintf(f, " Other options are:\n");
#ifdef HAVE_LIBZ
    fprintf(f, "  -z, --gzip, --gunzip, --ungzip    filter files through gzip\n");
#endif
    fprintf(f, "  --separate-diagonal  store diagonal nonzeros separately\n");
    fprintf(f, "  --sort-rows          sort nonzeros by column within each row\n");
    fprintf(f, "  --repeat=N           repeat matrix-vector multiplication N times\n");
    fprintf(f, "  --warmup=N                perform N additional warmup iterations\n");
    fprintf(f, "  -q, --quiet          do not print Matrix Market output\n");
    fprintf(f, "  -v, --verbose        be more verbose\n");
    fprintf(f, "\n");
#ifdef HAVE_PAPI
    fprintf(f, " Options for performance monitoring (PAPI) are:\n");
    fprintf(f, "  --papi-event-file=FILE    file describing which events to monitor\n");
    fprintf(f, "  --papi-event-format=FMT   output format for events: plain or csv. [plain]\n");
    fprintf(f, "  --papi-event-per-thread   display events per thread\n");
    fprintf(f, "  --papi-event-summary      display summary of performance monitoring\n");
    fprintf(f, "\n");
#endif
    fprintf(f, "  -h, --help           display this help and exit\n");
    fprintf(f, "  --version            display version information and exit\n");
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
    fprintf(f, "Fujitsu A64FX sector cache support enabled (L1 ways: ");
#ifndef A64FX_SECTOR_CACHE_L1_WAYS
    fprintf(f, "disabled");
#else
    fprintf(f, "%d", A64FX_SECTOR_CACHE_L1_WAYS);
#endif
    fprintf(f, ", L2 ways: %d)\n", A64FX_SECTOR_CACHE_L2_WAYS);
#endif
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

        if (strcmp(argv[0], "--repeat") == 0) {
            if (argc - *nargs < 2) { program_options_free(args); return EINVAL; }
            (*nargs)++; argv++;
            err = parse_int(&args->repeat, argv[0], NULL, NULL);
            if (err) { program_options_free(args); return err; }
            (*nargs)++; argv++; continue;
        } else if (strstr(argv[0], "--repeat=") == argv[0]) {
            err = parse_int(
                &args->repeat, argv[0] + strlen("--repeat="), NULL, NULL);
            if (err) { program_options_free(args); return err; }
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

static int ell_from_coo_size(
    idx_t num_rows,
    idx_t num_columns,
    int64_t num_nonzeros,
    const idx_t * rowidx,
    const idx_t * colidx,
    const double * a,
    int64_t * rowptr,
    int64_t * ellsize,
    idx_t * rowsize,
    idx_t * diagsize,
    bool separate_diagonal)
{
    idx_t rowmax = 0;
    for (idx_t i = 0; i <= num_rows; i++) rowptr[i] = 0;
    for (int64_t k = 0; k < num_nonzeros; k++) {
        if (!separate_diagonal || rowidx[k] != colidx[k])
            rowptr[rowidx[k]]++;
    }
    for (idx_t i = 1; i <= num_rows; i++) {
        rowmax = rowmax >= rowptr[i] ? rowmax : rowptr[i];
        rowptr[i] += rowptr[i-1];
    }
    *rowsize = rowmax;
    *ellsize = num_rows * (*rowsize);
    *diagsize = num_rows < num_columns ? num_rows : num_columns;
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

static int ell_from_coo(
    idx_t num_rows,
    idx_t num_columns,
    int64_t num_nonzeros,
    const idx_t * rowidx,
    const idx_t * colidx,
    const double * a,
    int64_t * rowptr,
    int64_t ellsize,
    idx_t rowsize,
    idx_t * ellcolidx,
    double * ella,
    double * ellad,
    bool separate_diagonal,
    bool sort_rows)
{
    for (idx_t i = 0; i <= num_rows; i++) rowptr[i] = 0;
    for (int64_t k = 0; k < num_nonzeros; k++) {
        if (separate_diagonal && rowidx[k] == colidx[k]) {
            ellad[rowidx[k]-1] += a[k];
        } else {
            idx_t i = rowidx[k]-1;
            ellcolidx[i*rowsize+rowptr[i]] = colidx[k]-1;
            ella[i*rowsize+rowptr[i]] = a[k];
            rowptr[i]++;
        }
    }
#ifdef _OPENMP
    #pragma omp for
#endif
    for (idx_t i = 0; i < num_rows; i++) {
        idx_t j =  i < num_columns ? i : num_columns-1;
        for (int64_t l = rowptr[i]; l < rowsize; l++) {
            ellcolidx[i*rowsize+l] = j;
            ella[i*rowsize+l] = 0.0;
        }
    }

    /* If requested, sort nonzeros by column within each row */
    if (sort_rows) {
        int err = rowsort(
            num_rows, num_columns,
            rowptr, rowsize, ellcolidx, ella);
        if (err) return err;
    }
    return 0;
}

static int ellgemv(
    idx_t num_rows,
    double * __restrict y,
    idx_t num_columns,
    const double * __restrict x,
    int64_t ellsize,
    idx_t rowsize,
    const idx_t * __restrict colidx,
    const double * __restrict a)
{
#if defined(__FCC_version__) && defined(USE_A64FX_SECTOR_CACHE)
    #pragma procedure scache_isolate_assign a, colidx
#endif

#ifdef _OPENMP
    #pragma omp for simd
#endif
    for (idx_t i = 0; i < num_rows; i++) {
        double yi = 0;
        for (idx_t l = 0; l < rowsize; l++)
            yi += a[i*rowsize+l] * x[colidx[i*rowsize+l]];
        y[i] += yi;
    }
    return 0;
}

static int ellgemvsd(
    idx_t num_rows,
    double * __restrict y,
    idx_t num_columns,
    const double * __restrict x,
    int64_t ellsize,
    idx_t rowsize,
    const idx_t * __restrict colidx,
    const double * __restrict a,
    const double * __restrict ad)
{
#if defined(__FCC_version__) && defined(USE_A64FX_SECTOR_CACHE)
    #pragma procedure scache_isolate_assign a, ad, colidx
#endif

#ifdef _OPENMP
    #pragma omp for simd
#endif
    for (idx_t i = 0; i < num_rows; i++) {
        double yi = 0;
        for (idx_t l = 0; l < rowsize; l++)
            yi += a[i*rowsize+l] * x[colidx[i*rowsize+l]];
        y[i] += ad[i]*x[i] + yi;
    }
    return 0;
}

static int ellgemv16sd(
    idx_t num_rows,
    double * __restrict y,
    idx_t num_columns,
    const double * __restrict x,
    int64_t ellsize,
    idx_t rowsize,
    const idx_t * __restrict colidx,
    const double * __restrict a,
    const double * __restrict ad)
{
#if defined(__FCC_version__) && defined(USE_A64FX_SECTOR_CACHE)
    #pragma procedure scache_isolate_assign a, ad, colidx
#endif

    if (rowsize != 16) return EINVAL;
#ifdef _OPENMP
    #pragma omp for simd
#endif
    for (idx_t i = 0; i < num_rows; i++) {
        y[i] += ad[i]*x[i] +
            a[i*16+ 0] * x[colidx[i*16+ 0]] +
            a[i*16+ 1] * x[colidx[i*16+ 1]] +
            a[i*16+ 2] * x[colidx[i*16+ 2]] +
            a[i*16+ 3] * x[colidx[i*16+ 3]] +
            a[i*16+ 4] * x[colidx[i*16+ 4]] +
            a[i*16+ 5] * x[colidx[i*16+ 5]] +
            a[i*16+ 6] * x[colidx[i*16+ 6]] +
            a[i*16+ 7] * x[colidx[i*16+ 7]] +
            a[i*16+ 8] * x[colidx[i*16+ 8]] +
            a[i*16+ 9] * x[colidx[i*16+ 9]] +
            a[i*16+10] * x[colidx[i*16+10]] +
            a[i*16+11] * x[colidx[i*16+11]] +
            a[i*16+12] * x[colidx[i*16+12]] +
            a[i*16+13] * x[colidx[i*16+13]] +
            a[i*16+14] * x[colidx[i*16+14]] +
            a[i*16+15] * x[colidx[i*16+15]];
    }
    return 0;
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

    /* 3. Convert to ELLPACK format. */
    if (args.verbose > 0) {
        fprintf(stderr, "ell_from_coo: ");
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

#ifdef HAVE_ALIGNED_ALLOC
    size_t rowptrsize = (num_rows+1)*sizeof(int64_t);
    int64_t * rowptr = aligned_alloc(pagesize, rowptrsize + pagesize - rowptrsize % pagesize);
#else
    int64_t * rowptr = malloc((num_rows+1) * sizeof(int64_t));
#endif
    if (!rowptr) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(a); free(colidx); free(rowidx);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    int64_t ellsize;
    idx_t rowsize;
    idx_t diagsize;
    err = ell_from_coo_size(
        num_rows, num_columns, num_nonzeros, rowidx, colidx, a,
        rowptr, &ellsize, &rowsize, &diagsize,
        args.separate_diagonal);
    if (err) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(err));
        free(rowptr); free(a); free(colidx); free(rowidx);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
#ifdef HAVE_ALIGNED_ALLOC
    size_t ellcolidxsize = ellsize*sizeof(idx_t);
    idx_t * ellcolidx = aligned_alloc(pagesize, ellcolidxsize + pagesize - ellcolidxsize % pagesize);
#else
    idx_t * ellcolidx = malloc(ellsize * sizeof(idx_t));
#endif
    if (!ellcolidx) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(rowptr); free(a); free(colidx); free(rowidx);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (idx_t i = 0; i < num_rows; i++) {
        for (idx_t l = 0; l < rowsize; l++)
            ellcolidx[i*rowsize+l] = 0;
    }
#ifdef HAVE_ALIGNED_ALLOC
    size_t ellasize = ellsize*sizeof(double);
    double * ella = aligned_alloc(pagesize, ellasize + pagesize - ellasize % pagesize);
#else
    double * ella = malloc(ellsize * sizeof(double));
#endif
    if (!ella) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(ellcolidx);
        free(rowptr); free(a); free(colidx); free(rowidx);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
#ifdef HAVE_ALIGNED_ALLOC
    size_t elladsize = diagsize*sizeof(double);
    double * ellad = aligned_alloc(pagesize, elladsize + pagesize - elladsize % pagesize);
#else
    double * ellad = malloc(diagsize * sizeof(double));
#endif
    if (!ellad) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(ella); free(ellcolidx);
        free(rowptr); free(a); free(colidx); free(rowidx);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (idx_t i = 0; i < num_rows; i++) {
        ellad[i] = 0;
        for (idx_t l = 0; l < rowsize; l++)
            ella[i*rowsize+l] = 0;
    }
    err = ell_from_coo(
        num_rows, num_columns, num_nonzeros, rowidx, colidx, a,
        rowptr, ellsize, rowsize, ellcolidx, ella, ellad,
        args.sort_rows, args.separate_diagonal);
    if (err) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(err));
        free(ellad); free(ella); free(ellcolidx);
        free(rowptr); free(a); free(colidx); free(rowidx);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    free(rowptr); free(a); free(colidx); free(rowidx);

    if (args.verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(stderr, "%'.6f seconds, %'"PRIdx" rows, %'"PRId64" nonzeros, %'"PRIdx" nonzeros per row\n",
                timespec_duration(t0, t1), num_rows, ellsize + num_rows, rowsize);
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
        free(ellad); free(ella); free(ellcolidx);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (idx_t j = 0; j < num_columns; j++) x[j] = 1.0;

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
                free(x); free(ellad); free(ella); free(ellcolidx);
                program_options_free(&args);
                return EXIT_FAILURE;
            }
#ifdef HAVE_LIBZ
        } else {
            streamtype = stream_zlib;
            if ((stream.gzf = gzopen(args.xpath, "r")) == NULL) {
                fprintf(stderr, "%s: %s: %s\n",
                        program_invocation_short_name, args.xpath, strerror(errno));
                free(x); free(ellad); free(ella); free(ellcolidx);
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
            free(x); free(ellad); free(ella); free(ellcolidx);
            program_options_free(&args);
            return EXIT_FAILURE;
        } else if (object != mtxvector || format != mtxarray || xnum_rows != num_columns) {
            if (args.verbose > 0) fprintf(stderr, "\n");
            fprintf(stderr, "%s: %s:%"PRId64": "
                    "expected vector in array format of size %"PRIdx"\n",
                    program_invocation_short_name,
                    args.xpath, lines_read+1, num_columns);
            stream_close(streamtype, stream);
            free(x); free(ellad); free(ella); free(ellcolidx);
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
            free(x); free(ellad); free(ella); free(ellcolidx);
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
        free(ellad); free(ella); free(ellcolidx);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (idx_t i = 0; i < num_rows; i++) y[i] = 0.0;

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
                free(y); free(x); free(ellad); free(ella); free(ellcolidx);
                program_options_free(&args);
                return EXIT_FAILURE;
            }
#ifdef HAVE_LIBZ
        } else {
            streamtype = stream_zlib;
            if ((stream.gzf = gzopen(args.ypath, "r")) == NULL) {
                fprintf(stderr, "%s: %s: %s\n",
                        program_invocation_short_name, args.ypath, strerror(errno));
                free(y); free(x); free(ellad); free(ella); free(ellcolidx);
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
            free(y); free(x); free(ellad); free(ella); free(ellcolidx);
            program_options_free(&args);
            return EXIT_FAILURE;
        } else if (object != mtxvector || format != mtxarray || ynum_rows != num_rows) {
            if (args.verbose > 0) fprintf(stderr, "\n");
            fprintf(stderr, "%s: %s:%"PRId64": "
                    "expected vector in array format of size %"PRIdx"\n",
                    program_invocation_short_name,
                    args.ypath, lines_read+1, num_rows);
            stream_close(streamtype, stream);
            free(y); free(x); free(ellad); free(ella); free(ellcolidx);
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
            free(y); free(x); free(ellad); free(ella); free(ellcolidx);
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
            free(ellad); free(ella); free(ellcolidx);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
    }
#endif

    /* enable A64FX sector cache */
#if defined(__FCC_version__) && defined(USE_A64FX_SECTOR_CACHE)
#ifdef A64FX_SECTOR_CACHE_L1_WAYS
    #pragma statement scache_isolate_way L2=A64FX_SECTOR_CACHE_L2_WAYS L1=A64FX_SECTOR_CACHE_L1_WAYS
#else
    #pragma statement scache_isolate_way L2=A64FX_SECTOR_CACHE_L2_WAYS
#endif
#endif

    /* perform warmup iterations */
#ifdef _OPENMP
    #pragma omp parallel
#endif
    for (int repeat = 0; repeat < args.warmup; repeat++) {
        #pragma omp master
        if (args.verbose > 0) {
            if (args.separate_diagonal && rowsize == 16) fprintf(stderr, "gemv16sd (warmup): ");
            else if (args.separate_diagonal) fprintf(stderr, "gemvsd (warmup): ");
            else fprintf(stderr, "gemv (warmup): ");
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        int priverr;
        if (args.separate_diagonal && rowsize == 16) {
            priverr = ellgemv16sd(
                num_rows, y, num_columns, x, ellsize, rowsize, ellcolidx, ella, ellad);
        } else if (args.separate_diagonal) {
            priverr = ellgemvsd(
                num_rows, y, num_columns, x, ellsize, rowsize, ellcolidx, ella, ellad);
        } else {
            priverr = ellgemv(
                num_rows, y, num_columns, x, ellsize, rowsize, ellcolidx, ella);
        }

#ifdef _OPENMP
        #pragma omp barrier
        for (int t = 0; t < omp_get_num_threads(); t++) {
            if (t == omp_get_thread_num() && !err && priverr) err = priverr;
            #pragma omp barrier
        }
#else
        if (priverr) { err = priverr; }
#endif
        if (err) break;

        int64_t num_flops = 2*(ellsize+diagsize);
        int64_t min_bytes = num_rows*sizeof(*y) + num_columns*sizeof(*x)
            + ellsize*sizeof(*ellcolidx) + ellsize*sizeof(*ella) + diagsize*sizeof(*ellad);
        int64_t max_bytes = num_rows*sizeof(*y) + ellsize*sizeof(*x)
            + ellsize*sizeof(*ellcolidx) + ellsize*sizeof(*ella)
            + diagsize*sizeof(*ellad) + diagsize*sizeof(*x);

#ifdef _OPENMP
        #pragma omp barrier
        #pragma omp master
#endif
        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
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
            free(y); free(x);
            free(ellad); free(ella); free(ellcolidx);
            program_options_free(&args);
            return EXIT_FAILURE;
        }
    }
#endif

    /* perform matrix-vector multiplication */
#ifdef _OPENMP
    #pragma omp parallel
#endif
    for (int repeat = 0; repeat < args.repeat; repeat++) {
        #pragma omp master
        if (args.verbose > 0) {
            if (args.separate_diagonal && rowsize == 16) fprintf(stderr, "gemv16sd: ");
            else if (args.separate_diagonal) fprintf(stderr, "gemvsd: ");
            else fprintf(stderr, "gemv: ");
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        int priverr;
        if (args.separate_diagonal && rowsize == 16) {
            priverr = ellgemv16sd(
                num_rows, y, num_columns, x, ellsize, rowsize, ellcolidx, ella, ellad);
        } else if (args.separate_diagonal) {
            priverr = ellgemvsd(
                num_rows, y, num_columns, x, ellsize, rowsize, ellcolidx, ella, ellad);
        } else {
            priverr = ellgemv(
                num_rows, y, num_columns, x, ellsize, rowsize, ellcolidx, ella);
        }

#ifdef _OPENMP
        #pragma omp barrier
        for (int t = 0; t < omp_get_num_threads(); t++) {
            if (t == omp_get_thread_num() && !err && priverr) err = priverr;
            #pragma omp barrier
        }
#else
        if (priverr) { err = priverr; }
#endif
        if (err) break;

        int64_t num_flops = 2*(ellsize+diagsize);
        int64_t min_bytes = num_rows*sizeof(*y) + num_columns*sizeof(*x)
            + ellsize*sizeof(*ellcolidx) + ellsize*sizeof(*ella) + diagsize*sizeof(*ellad);
        int64_t max_bytes = num_rows*sizeof(*y) + ellsize*sizeof(*x)
            + ellsize*sizeof(*ellcolidx) + ellsize*sizeof(*ella)
            + diagsize*sizeof(*ellad) + diagsize*sizeof(*x);

#ifdef _OPENMP
        #pragma omp barrier
        #pragma omp master
#endif
        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(stderr, "%'.6f seconds (%'.3f Gnz/s, %'.3f Gflop/s, %'.1f to %'.1f GB/s)\n",
                    timespec_duration(t0, t1),
                    (double) num_nonzeros * 1e-9 / (double) timespec_duration(t0, t1),
                    (double) num_flops * 1e-9 / (double) timespec_duration(t0, t1),
                    (double) min_bytes * 1e-9 / (double) timespec_duration(t0, t1),
                    (double) max_bytes * 1e-9 / (double) timespec_duration(t0, t1));
        }
    }

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
        free(ellad); free(ella); free(ellcolidx);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    free(x); free(ellad); free(ella); free(ellcolidx);

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

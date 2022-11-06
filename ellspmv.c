/*
 * Benchmark program for ELLPACK SpMV
 * Copyright (C) 2022 James D. Trotter
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
 * Authors: James D. Trotter <james@simula.no>
 * Last modified: 2022-05-27
 *
 * Benchmarking program for sparse matrix-vector multiplication (SpMV)
 * with matrices in ELLPACK format.
 */

#include <errno.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "papi_util/include/papi_util.h"
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

const char * program_name = "ellspmv";
const char * program_version = "1.0";
const char * program_copyright =
    "Copyright (C) 2022 James D. Trotter";
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
    char * ypath;
    int repeat;
    int verbose;
    int quiet;
    struct papi_util_opt papi_opt;
};


/**
 * ‘program_options_init()’ configures the default program options.
 */
static int program_options_init(
    struct program_options * args)
{
    args->Apath = NULL;
    args->ypath = NULL;
    args->repeat = 1;
    args->quiet = 0;
    args->verbose = 0;

    args->papi_opt = (struct papi_util_opt) {
                            .event_file = NULL,
                            .print_csv = 0,
                            .print_threads = 1,
                            .print_summary = 1,
                            .print_region = 1,
                            .component = 0,
                            .multiplex = 0,
                            .output = stdout
                            };
    return 0;
}

/**
 * ‘program_options_free()’ frees memory and other resources
 * associated with parsing program options.
 */
static void program_options_free(
    struct program_options * args)
{
    if (args->ypath) free(args->ypath);
    if (args->Apath) free(args->Apath);
}

/**
 * ‘program_options_print_usage()’ prints a usage text.
 */
static void program_options_print_usage(
    FILE * f)
{
    fprintf(f, "Usage: %s [OPTION..] A [y] [e]\n", program_name);
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
    fprintf(f, "  A\tpath to Matrix Market file for the matrix A\n");
    fprintf(f, "  y\toptional path for writing the result vector y\n");
    fprintf(f, "  e\toptional event file for PAPI performance events\n");
    fprintf(f, "\n");
    fprintf(f, " Other options are:\n");
    fprintf(f, "  --repeat=N\t\trepeat matrix-vector multiplication N times\n");
    fprintf(f, "  -q, --quiet\t\tdo not print Matrix Market output\n");
    fprintf(f, "  -v, --verbose\t\tbe more verbose\n");
    fprintf(f, "\n");
    fprintf(f, "  -h, --help\t\tdisplay this help and exit\n");
    fprintf(f, "  --version\t\tdisplay version information and exit\n");
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

        if (strcmp(argv[0], "-q") == 0 || strcmp(argv[0], "--quiet") == 0) {
            args->quiet = true;
            (*nargs)++; argv++; continue;
        }

        if (strcmp(argv[0], "-v") == 0 || strcmp(argv[0], "--verbose") == 0) {
            args->verbose++;
            (*nargs)++; argv++; continue;
        }

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
            args->ypath = strdup(argv[0]);
            if (!args->ypath) { program_options_free(args); return errno; }
        } else if (num_positional_arguments_consumed == 2) {
            args->papi_opt.event_file = argv[0];
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

/**
 * ‘freadline()’ reads a single line from a stream.
 */
static int freadline(char * linebuf, size_t line_max, FILE * f) {
    char * s = fgets(linebuf, line_max+1, f);
    if (!s && feof(f)) return -1;
    else if (!s) return errno;
    int n = strlen(s);
    if (n > 0 && n == line_max && s[n-1] != '\n') return EOVERFLOW;
    return 0;
}

static int mtxfile_fread_header(
    int * num_rows,
    int * num_columns,
    int64_t * num_nonzeros,
    FILE * f,
    int64_t * lines_read,
    int64_t * bytes_read)
{
    int line_max = sysconf(_SC_LINE_MAX);
    char * linebuf = malloc(line_max+1);
    if (!linebuf) return errno;

    /* read and parse header line */
    int err = freadline(linebuf, line_max, f);
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
    } else { free(linebuf); return EINVAL; }
    if (bytes_read) *bytes_read += t-s;
    s = t;
    if (strncmp("coordinate ", t, strlen("coordinate ")) == 0) {
        t += strlen("coordinate ");
    } else { free(linebuf); return EINVAL; }
    if (bytes_read) *bytes_read += t-s;
    s = t;
    if (strncmp("real ", t, strlen("real ")) == 0) {
        t += strlen("real ");
    } else { free(linebuf); return EINVAL; }
    if (bytes_read) *bytes_read += t-s;
    s = t;
    if (strncmp("general", t, strlen("general")) == 0) {
        t += strlen("general");
    } else { free(linebuf); return EINVAL; }
    if (bytes_read) *bytes_read += t-s;
    s = t;

    /* skip lines starting with '%' */
    do {
        if (lines_read) (*lines_read)++;
        err = freadline(linebuf, line_max, f);
        if (err) { free(linebuf); return err; }
        s = t = linebuf;
    } while (linebuf[0] == '%');

    /* parse size line */
    err = parse_int(num_rows, s, &t, bytes_read);
    if (err) { free(linebuf); return err; }
    if (s == t || *t != ' ') { free(linebuf); return EINVAL; }
    if (bytes_read) (*bytes_read)++;
    s = t+1;
    err = parse_int(num_columns, s, &t, bytes_read);
    if (err) { free(linebuf); return err; }
    if (s == t || *t != ' ') { free(linebuf); return EINVAL; }
    if (bytes_read) (*bytes_read)++;
    s = t+1;
    err = parse_int64_t(num_nonzeros, s, &t, bytes_read);
    if (err) { free(linebuf); return err; }
    if (s == t) { free(linebuf); return EINVAL; }
    free(linebuf);
    return 0;
}

static int mtxfile_fread_data(
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    int * rowidx,
    int * colidx,
    double * a,
    FILE * f,
    int64_t * lines_read,
    int64_t * bytes_read)
{
    int line_max = sysconf(_SC_LINE_MAX);
    char * linebuf = malloc(line_max+1);
    if (!linebuf) return errno;
    for (int64_t i = 0; i < num_nonzeros; i++) {
        int err = freadline(linebuf, line_max, f);
        if (err) { free(linebuf); return err; }
        char * s = linebuf;
        char * t = s;
        err = parse_int(&rowidx[i], s, &t, bytes_read);
        if (err) { free(linebuf); return err; }
        if (s == t || *t != ' ') { free(linebuf); return EINVAL; }
        if (bytes_read) (*bytes_read)++;
        s = t+1;
        err = parse_int(&colidx[i], s, &t, bytes_read);
        if (err) { free(linebuf); return err; }
        if (s == t || *t != ' ') { free(linebuf); return EINVAL; }
        if (bytes_read) (*bytes_read)++;
        s = t+1;
        err = parse_double(&a[i], s, &t, bytes_read);
        if (err) { free(linebuf); return err; }
        if (s == t) { free(linebuf); return EINVAL; }
    }
    free(linebuf);
    return 0;
}

static int ell_from_coo_size(
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const double * a,
    int64_t * rowptr,
    int64_t * ellsize,
    int * rowsize,
    int * diagsize)
{
    int rowmax = 0;
    for (int i = 0; i <= num_rows; i++) rowptr[i] = 0;
    for (int64_t k = 0; k < num_nonzeros; k++) {
        if (rowidx[k] != colidx[k])
            rowptr[rowidx[k]]++;
    }
    for (int i = 1; i <= num_rows; i++) {
        rowmax = rowmax >= rowptr[i] ? rowmax : rowptr[i];
        rowptr[i] += rowptr[i-1];
    }
    *rowsize = rowmax;
    *ellsize = num_rows * (*rowsize);
    *diagsize = num_rows < num_columns ? num_rows : num_columns;
    return 0;
}

static int ell_from_coo(
    int num_rows,
    int num_columns,
    int64_t num_nonzeros,
    const int * rowidx,
    const int * colidx,
    const double * a,
    int64_t * rowptr,
    int64_t ellsize,
    int rowsize,
    int * ellcolidx,
    double * ella,
    double * ellad)
{
    for (int i = 0; i <= num_rows; i++) rowptr[i] = 0;
    for (int64_t k = 0; k < num_nonzeros; k++) {
        if (rowidx[k] == colidx[k]) {
            ellad[rowidx[k]-1] += a[k];
        } else {
            int i = rowidx[k]-1;
            ellcolidx[i*rowsize+rowptr[i]] = colidx[k]-1;
            ella[i*rowsize+rowptr[i]] = a[k];
            rowptr[i]++;
        }
    }
#ifdef _OPENMP
    #pragma omp for
#endif
    for (int i = 0; i < num_rows; i++) {
        int j =  i < num_columns ? i : num_columns-1;
        for (int64_t l = rowptr[i]; l < rowsize; l++) {
            ellcolidx[i*rowsize+l] = j;
            ella[i*rowsize+l] = 0.0;
        }
    }
    return 0;
}

static int ellgemv(
    int num_rows,
    double * __restrict y,
    int num_columns,
    const double * __restrict x,
    int64_t ellsize,
    int rowsize,
    const int * __restrict colidx,
    const double * __restrict a,
    const double * __restrict ad)
{
#ifdef A64FXCPU
    #pragma procedure scache_isolate_way L2=L2WAYS
    #pragma procedure scache_isolate_assign a, colidx
#endif /* A64FXCPU */

#ifdef _OPENMP
    #pragma omp for simd
#endif
    for (int i = 0; i < num_rows; i++) {
        double yi = 0;
        for (int l = 0; l < rowsize; l++)
            yi += a[i*rowsize+l] * x[colidx[i*rowsize+l]];
        y[i] += ad[i]*x[i] + yi;
    }
    return 0;
}

static int ellgemv16(
    int num_rows,
    double * __restrict y,
    int num_columns,
    const double * __restrict x,
    int64_t ellsize,
    int rowsize,
    const int * __restrict colidx,
    const double * __restrict a,
    const double * __restrict ad)
{
#ifdef A64FXCPU
    #pragma procedure scache_isolate_way L2=L2WAYS
    #pragma procedure scache_isolate_assign a, colidx
#endif /* A64FXCPU */

    if (rowsize != 16) return EINVAL;
#ifdef _OPENMP
    #pragma omp for simd
#endif
    for (int i = 0; i < num_rows; i++) {
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

    /* 2. Read the matrix from a Matrix Market file. */
    if (args.verbose > 0) {
        fprintf(stdout, "mtxfile_read: ");
        fflush(stdout);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    FILE * f;
    if ((f = fopen(args.Apath, "r")) == NULL) {
        fprintf(stderr, "%s: %s: %s\n",
                program_invocation_short_name, args.Apath, strerror(errno));
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    PAPI_UTIL_SETUP(&args.papi_opt);

    int num_rows;
    int num_columns;
    int64_t num_nonzeros;
    int64_t lines_read = 0;
    int64_t bytes_read = 0;
    err = mtxfile_fread_header(
        &num_rows, &num_columns, &num_nonzeros, f, &lines_read, &bytes_read);
    if (err) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s:%"PRId64": %s\n",
                program_invocation_short_name,
                args.Apath, lines_read+1, strerror(err));
        fclose(f);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    int * rowidx = malloc(num_nonzeros * sizeof(int));
    if (!rowidx) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        fclose(f);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    int * colidx = malloc(num_nonzeros * sizeof(int));
    if (!colidx) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(rowidx);
        fclose(f);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    double * a = malloc(num_nonzeros * sizeof(double));
    if (!a) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(colidx); free(rowidx);
        fclose(f);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    err = mtxfile_fread_data(
        num_rows, num_columns, num_nonzeros, rowidx, colidx, a, f, &lines_read, &bytes_read);
    if (err) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s:%"PRId64": %s\n",
                program_invocation_short_name,
                args.Apath, lines_read+1, strerror(err));
        free(a); free(colidx); free(rowidx);
        fclose(f);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    if (args.verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(stderr, "%'.6f seconds (%'.1f MB/s)\n",
                timespec_duration(t0, t1),
                1.0e-6 * bytes_read / timespec_duration(t0, t1));
    }
    fclose(f);

    /* 3. Convert to ELLPACK format. */
    if (args.verbose > 0) {
        fprintf(stdout, "ell_from_coo: ");
        fflush(stdout);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    int64_t * rowptr = malloc((num_rows+1) * sizeof(int64_t));
    if (!rowptr) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(a); free(colidx); free(rowidx);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    int64_t ellsize;
    int rowsize;
    int diagsize;
    err = ell_from_coo_size(
        num_rows, num_columns, num_nonzeros, rowidx, colidx, a,
        rowptr, &ellsize, &rowsize, &diagsize);
    if (err) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(err));
        free(rowptr); free(a); free(colidx); free(rowidx);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    int * ellcolidx = malloc(ellsize * sizeof(int));
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
    for (int i = 0; i < num_rows; i++) {
        for (int l = 0; l < rowsize; l++)
            ellcolidx[i*rowsize+l] = 0;
    }
    double * ella = malloc(ellsize * sizeof(double));
    if (!ella) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(ellcolidx);
        free(rowptr); free(a); free(colidx); free(rowidx);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    double * ellad = malloc(diagsize * sizeof(double));
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
    for (int i = 0; i < num_rows; i++) {
        ellad[i] = 0;
        for (int l = 0; l < rowsize; l++)
            ella[i*rowsize+l] = 0;
    }
    err = ell_from_coo(
        num_rows, num_columns, num_nonzeros, rowidx, colidx, a,
        rowptr, ellsize, rowsize, ellcolidx, ella, ellad);
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
        fprintf(stderr, "%'.6f seconds, %'d rows, %'ld nonzeros, %'d nonzeros per row\n",
                timespec_duration(t0, t1), num_rows, ellsize + num_rows, rowsize);
    }

    /* 4. allocate vectors */
    double * x = malloc(num_columns * sizeof(double));
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
    for (int j = 0; j < num_columns; j++) x[j] = 1.0;

    double * y = malloc(num_rows * sizeof(double));
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
    for (int i = 0; i < num_rows; i++) y[i] = 0.0;

    PAPI_UTIL_START("ellspmv");

    /* 5. compute the matrix-vector multiplication. */
#ifdef _OPENMP
    #pragma omp parallel
#endif
    for (int repeat = 0; repeat < args.repeat; repeat++) {
        #pragma omp master
        if (args.verbose > 0) {
            fprintf(stdout, rowsize == 16 ? "gemv16: " : "gemv: ");
            fflush(stdout);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        if (rowsize == 16) {
            err = ellgemv16(
                num_rows, y, num_columns, x, ellsize, rowsize, ellcolidx, ella, ellad);
            if (err)
                break;
        } else {
            err = ellgemv(
                num_rows, y, num_columns, x, ellsize, rowsize, ellcolidx, ella, ellad);
            if (err)
                break;
        }

        int64_t num_flops = 2*(ellsize+diagsize);
        int64_t num_bytes = num_rows*sizeof(*y) + num_columns*sizeof(*x)
            + ellsize*sizeof(*ellcolidx) + ellsize*sizeof(*ella) + diagsize*sizeof(*ellad);

        #pragma omp master
        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(stdout, "%'.6f seconds (%'.1f Gnz/s, %'.1f Gflop/s, %'.1f GB/s)\n",
                    timespec_duration(t0, t1),
                    (double) num_nonzeros * 1e-9 / (double) timespec_duration(t0, t1),
                    (double) num_flops * 1e-9 / (double) timespec_duration(t0, t1),
                    (double) num_bytes * 1e-9 / (double) timespec_duration(t0, t1));
            fflush(stdout);
        }
    }
    // stop counters
    PAPI_UTIL_FINISH();

    PAPI_UTIL_FINALIZE();

    if (err) {
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(err));
        free(y); free(x);
        free(ellad); free(ella); free(ellcolidx);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    free(x); free(ellad); free(ella); free(ellcolidx);

    /* 6. write the result vector to a file */
    if (args.ypath && !args.quiet) {
        if (args.verbose > 0) {
            fprintf(stdout, "mtxfile_write: ");
            fflush(stdout);
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        FILE * f = fopen(args.ypath, "w");
        if (!f) {
            fprintf(stderr, "%s: %s: %s\n",
                    program_invocation_short_name, strerror(errno), args.ypath);
            free(y);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        fprintf(f, "%%%%MatrixMarket vector array real general\n");
        fprintf(f, "%d\n", num_rows);
        for (int i = 0; i < num_rows; i++) fprintf(f, "%.*g\n", DBL_DIG, y[i]);
        fclose(f);
        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(stderr, "%'.6f seconds\n", timespec_duration(t0, t1));
        }
    }

    free(y);
    program_options_free(&args);
    return EXIT_SUCCESS;
}

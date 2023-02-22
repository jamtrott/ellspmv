/*
 * Benchmark program for CSR SpMV
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
 * with matrices in CSR format.
 *
 * Authors:
 *  James D. Trotter <james@simula.no>
 *  Sergej Breiter <breiter@nm.ifi.lmu.de>
 *
 *
 * History:
 *
 *  1.0 — 2023-02-22:
 *
 *   - initial version based on ellspmv.
 */

#include <errno.h>

#ifdef WITH_OPENMP
#include <omp.h>
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

const char * program_name = "csrspmv";
const char * program_version = "1.0";
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
    bool separate_diagonal;
    int repeat;
    int verbose;
    int quiet;
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
    args->separate_diagonal = false;
    args->repeat = 1;
    args->quiet = 0;
    args->verbose = 0;
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
    fprintf(f, "  A\tpath to Matrix Market file for the matrix A\n");
    fprintf(f, "  x\toptional path to Matrix Market file for the vector x\n");
    fprintf(f, "  y\toptional path for to Matrix Market file for the vector y\n");
    fprintf(f, "\n");
    fprintf(f, " Other options are:\n");
    fprintf(f, "  --separate-diagonal\t\tstore diagonal nonzeros separately\n");
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
#if defined(__FCC_version__) && defined(USE_A64FX_SECTOR_CACHE)
    fprintf(f, "Fujitsu A64FX sector cache support enabled (L2 ways: %d)\n", L2WAYS);
#endif
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
        if (strcmp(argv[0], "--separate-diagonal") == 0) {
            args->separate_diagonal = true;
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

        if (strcmp(argv[0], "-q") == 0 || strcmp(argv[0], "--quiet") == 0) {
            args->quiet = 1;
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

enum mtxsymmetry
{
    mtxgeneral,
    mtxsymmetric,
};

static int mtxfile_fread_header(
    enum mtxobject * object,
    enum mtxformat * format,
    enum mtxsymmetry * symmetry,
    int64_t * num_rows,
    int64_t * num_columns,
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
        err = freadline(linebuf, line_max, f);
        if (err) { free(linebuf); return err; }
        s = t = linebuf;
    } while (linebuf[0] == '%');

    /* parse size line */
    if (*object == mtxmatrix && *format == mtxcoordinate) {
        err = parse_int64_t(num_rows, s, &t, bytes_read);
        if (err) { free(linebuf); return err; }
        if (s == t || *t != ' ') { free(linebuf); return EINVAL; }
        if (bytes_read) (*bytes_read)++;
        s = t+1;
        err = parse_int64_t(num_columns, s, &t, bytes_read);
        if (err) { free(linebuf); return err; }
        if (s == t || *t != ' ') { free(linebuf); return EINVAL; }
        if (bytes_read) (*bytes_read)++;
        s = t+1;
        err = parse_int64_t(num_nonzeros, s, &t, bytes_read);
        if (err) { free(linebuf); return err; }
        if (s == t) { free(linebuf); return EINVAL; }
    } else if (*object == mtxvector && *format == mtxarray) {
        err = parse_int64_t(num_rows, s, &t, bytes_read);
        if (err) { free(linebuf); return err; }
        if (s == t) { free(linebuf); return EINVAL; }
    } else { free(linebuf); return EINVAL; }
    free(linebuf);
    return 0;
}

static int mtxfile_fread_matrix_coordinate_real(
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    int64_t * rowidx,
    int64_t * colidx,
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
        err = parse_int64_t(&rowidx[i], s, &t, bytes_read);
        if (err) { free(linebuf); return err; }
        if (s == t || *t != ' ') { free(linebuf); return EINVAL; }
        if (bytes_read) (*bytes_read)++;
        s = t+1;
        err = parse_int64_t(&colidx[i], s, &t, bytes_read);
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

static int mtxfile_fread_vector_array_real(
    int64_t num_rows,
    double * x,
    FILE * f,
    int64_t * lines_read,
    int64_t * bytes_read)
{
    int line_max = sysconf(_SC_LINE_MAX);
    char * linebuf = malloc(line_max+1);
    if (!linebuf) return errno;
    for (int64_t i = 0; i < num_rows; i++) {
        int err = freadline(linebuf, line_max, f);
        if (err) { free(linebuf); return err; }
        char * s = linebuf;
        char * t = s;
        err = parse_double(&x[i], s, &t, bytes_read);
        if (err) { free(linebuf); return err; }
        if (s == t) { free(linebuf); return EINVAL; }
    }
    free(linebuf);
    return 0;
}

static int csr_from_coo_size(
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    const int64_t * rowidx,
    const int64_t * colidx,
    const double * a,
    int64_t * rowptr,
    int64_t * csrsize,
    int64_t * rowsizemin,
    int64_t * rowsizemax,
    int64_t * diagsize,
    bool separate_diagonal)
{
#ifdef WITH_OPENMP
    #pragma omp for
#endif
    for (int64_t i = 0; i < num_rows; i++) rowptr[i] = 0;
    rowptr[num_rows] = 0;
    if (num_rows == num_columns && separate_diagonal) {
        for (int64_t k = 0; k < num_nonzeros; k++) {
            if (rowidx[k] != colidx[k]) rowptr[rowidx[k]]++;
        }
    } else {
        for (int64_t k = 0; k < num_nonzeros; k++) rowptr[rowidx[k]]++;
    }
    int64_t rowmin = num_rows > 0 ? rowptr[1] : 0;
    int64_t rowmax = 0;
    for (int64_t i = 1; i <= num_rows; i++) {
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

static int csr_from_coo(
    int64_t num_rows,
    int64_t num_columns,
    int64_t num_nonzeros,
    const int64_t * rowidx,
    const int64_t * colidx,
    const double * a,
    int64_t * rowptr,
    int64_t csrsize,
    int64_t rowsizemin,
    int64_t rowsizemax,
    int64_t * csrcolidx,
    double * csra,
    double * csrad,
    bool separate_diagonal)
{
    if (num_rows == num_columns && separate_diagonal) {
        for (int64_t k = 0; k < num_nonzeros; k++) {
            if (rowidx[k] == colidx[k]) {
                csrad[rowidx[k]-1] += a[k];
            } else {
                int64_t i = rowidx[k]-1;
                csrcolidx[rowptr[i]] = colidx[k]-1;
                csra[rowptr[i]] = a[k];
                rowptr[i]++;
            }
        }
    } else {
        for (int64_t k = 0; k < num_nonzeros; k++) {
            int64_t i = rowidx[k]-1;
            csrcolidx[rowptr[i]] = colidx[k]-1;
            csra[rowptr[i]] = a[k];
            rowptr[i]++;
        }
    }
    for (int64_t i = num_rows; i > 0; i--) rowptr[i] = rowptr[i-1];
    rowptr[0] = 0;
    return 0;
}

static int csrgemv(
    int64_t num_rows,
    double * __restrict y,
    int64_t num_columns,
    const double * __restrict x,
    int64_t csrsize,
    int64_t rowsizemin,
    int64_t rowsizemax,
    const int64_t * __restrict rowptr,
    const int64_t * __restrict colidx,
    const double * __restrict a)
{
#ifdef WITH_OPENMP
    #pragma omp for simd
#endif
    for (int64_t i = 0; i < num_rows; i++) {
        double yi = 0;
        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
            yi += a[k] * x[colidx[k]];
        y[i] += yi;
    }
    return 0;
}

static int csrgemvsd(
    int64_t num_rows,
    double * __restrict y,
    int64_t num_columns,
    const double * __restrict x,
    int64_t csrsize,
    int64_t rowsizemin,
    int64_t rowsizemax,
    const int64_t * __restrict rowptr,
    const int64_t * __restrict colidx,
    const double * __restrict a,
    const double * __restrict ad)
{
#if defined(__FCC_version__) && defined(USE_A64FX_SECTOR_CACHE)
    #pragma procedure scache_isolate_way L2=L2WAYS
    #pragma procedure scache_isolate_assign a, colidx
#endif

#ifdef WITH_OPENMP
    #pragma omp for simd
#endif
    for (int64_t i = 0; i < num_rows; i++) {
        double yi = 0;
        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
            yi += a[k] * x[colidx[k]];
        y[i] += ad[i]*x[i] + yi;
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
        fprintf(stderr, "mtxfile_read: ");
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    FILE * f;
    if ((f = fopen(args.Apath, "r")) == NULL) {
        fprintf(stderr, "%s: %s: %s\n",
                program_invocation_short_name, args.Apath, strerror(errno));
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    enum mtxobject object;
    enum mtxformat format;
    enum mtxsymmetry symmetry;
    int64_t num_rows;
    int64_t num_columns;
    int64_t num_nonzeros;
    int64_t lines_read = 0;
    int64_t bytes_read = 0;
    err = mtxfile_fread_header(
        &object, &format, &symmetry,
        &num_rows, &num_columns, &num_nonzeros,
        f, &lines_read, &bytes_read);
    if (err) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s:%"PRId64": %s\n",
                program_invocation_short_name,
                args.Apath, lines_read+1, strerror(err));
        fclose(f);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    int64_t * rowidx = malloc(num_nonzeros * sizeof(int64_t));
    if (!rowidx) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        fclose(f);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    int64_t * colidx = malloc(num_nonzeros * sizeof(int64_t));
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
    err = mtxfile_fread_matrix_coordinate_real(
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

    /* 3. Convert to CSR format. */
    if (args.verbose > 0) {
        fprintf(stderr, "csr_from_coo: ");
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    int64_t * csrrowptr = malloc((num_rows+1) * sizeof(int64_t));
    if (!csrrowptr) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(a); free(colidx); free(rowidx);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    int64_t csrsize;
    int64_t rowsizemin, rowsizemax;
    int64_t diagsize;
    err = csr_from_coo_size(
        num_rows, num_columns, num_nonzeros, rowidx, colidx, a,
        csrrowptr, &csrsize, &rowsizemin, &rowsizemax, &diagsize,
        args.separate_diagonal);
    if (err) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(err));
        free(csrrowptr); free(a); free(colidx); free(rowidx);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    int64_t * csrcolidx = malloc(csrsize * sizeof(int64_t));
    if (!csrcolidx) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(csrrowptr); free(a); free(colidx); free(rowidx);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
#ifdef WITH_OPENMP
    #pragma omp parallel for
#endif
    for (int64_t i = 0; i < num_rows; i++) {
        for (int64_t k = csrrowptr[i]; k < csrrowptr[i+1]; k++)
            csrcolidx[k] = 0;
    }
    double * csra = malloc(csrsize * sizeof(double));
    if (!csra) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(csrcolidx);
        free(csrrowptr); free(a); free(colidx); free(rowidx);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    double * csrad = malloc(diagsize * sizeof(double));
    if (!csrad) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(csra); free(csrcolidx);
        free(csrrowptr); free(a); free(colidx); free(rowidx);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    if (diagsize > 0) {
#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
        for (int64_t i = 0; i < num_rows; i++) csrad[i] = 0;
    }
#ifdef WITH_OPENMP
    #pragma omp parallel for
#endif
    for (int64_t i = 0; i < num_rows; i++) {
        for (int64_t k = csrrowptr[i]; k < csrrowptr[i+1]; k++)
            csra[k] = 0;
    }
    err = csr_from_coo(
        num_rows, num_columns, num_nonzeros, rowidx, colidx, a,
        csrrowptr, csrsize, rowsizemin, rowsizemax, csrcolidx, csra, csrad,
        args.separate_diagonal);
    if (err) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(err));
        free(csrad); free(csra); free(csrcolidx);
        free(csrrowptr); free(a); free(colidx); free(rowidx);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    free(a); free(colidx); free(rowidx);

    if (args.verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(stderr, "%'.6f seconds, %'"PRId64" rows, %'"PRId64" columns, %'"PRId64" nonzeros"
                ", %'"PRId64" to %'"PRId64" nonzeros per row",
                timespec_duration(t0, t1), num_rows, num_columns, csrsize+diagsize, rowsizemin, rowsizemax);
#ifdef WITH_OPENMP
        int nthreads;
        int64_t num_rows_per_thread;
        int64_t min_nonzeros_per_thread = INT64_MAX;
        int64_t max_nonzeros_per_thread = 0;
#pragma omp parallel reduction(min:min_nonzeros_per_thread) reduction(max:max_nonzeros_per_thread)
        {
            nthreads = omp_get_num_threads();
            num_rows_per_thread = (num_rows+nthreads-1)/nthreads;
            int64_t num_nonzeros = 0;
#pragma omp for
            for (int i = 0; i < num_rows; i++)
                num_nonzeros += csrrowptr[i+1]-csrrowptr[i] + (diagsize > 0 ? 1 : 0);           
            min_nonzeros_per_thread = num_nonzeros;
            max_nonzeros_per_thread = num_nonzeros;
        }
        fprintf(stderr, ", %'d threads, %'"PRId64" rows per thread, %'"PRId64" to %'"PRId64" nonzeros per thread",
                nthreads, num_rows_per_thread, min_nonzeros_per_thread, max_nonzeros_per_thread);
#endif
        fputc('\n', stderr);
    }

    /* 4. allocate vectors */
    double * x = malloc(num_columns * sizeof(double));
    if (!x) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(csrad); free(csra); free(csrcolidx); free(csrrowptr);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
    for (int64_t j = 0; j < num_columns; j++) x[j] = 1.0;

    /* read x vector from a Matrix Market file */
    if (args.xpath) {
        if (args.verbose > 0) {
            fprintf(stderr, "mtxfile_read: ");
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        FILE * f;
        if ((f = fopen(args.xpath, "r")) == NULL) {
            fprintf(stderr, "%s: %s: %s\n",
                    program_invocation_short_name, args.xpath, strerror(errno));
            free(x); free(csrad); free(csra); free(csrcolidx); free(csrrowptr);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        enum mtxobject object;
        enum mtxformat format;
        enum mtxsymmetry symmetry;
        int64_t xnum_rows;
        int64_t xnum_columns;
        int64_t xnum_nonzeros;
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        err = mtxfile_fread_header(
            &object, &format, &symmetry,
            &xnum_rows, &xnum_columns, &xnum_nonzeros,
            f, &lines_read, &bytes_read);
        if (err) {
            if (args.verbose > 0) fprintf(stderr, "\n");
            fprintf(stderr, "%s: %s:%"PRId64": %s\n",
                    program_invocation_short_name,
                    args.xpath, lines_read+1, strerror(err));
            fclose(f);
            free(x); free(csrad); free(csra); free(csrcolidx); free(csrrowptr);
            program_options_free(&args);
            return EXIT_FAILURE;
        } else if (object != mtxvector || format != mtxarray || xnum_rows != num_columns) {
            if (args.verbose > 0) fprintf(stderr, "\n");
            fprintf(stderr, "%s: %s:%"PRId64": "
                    "expected vector in array format of size %"PRId64"\n",
                    program_invocation_short_name,
                    args.xpath, lines_read+1, num_columns);
            fclose(f);
            free(x); free(csrad); free(csra); free(csrcolidx); free(csrrowptr);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        err = mtxfile_fread_vector_array_real(
            num_rows, x, f, &lines_read, &bytes_read);
        if (err) {
            if (args.verbose > 0) fprintf(stderr, "\n");
            fprintf(stderr, "%s: %s:%"PRId64": %s\n",
                    program_invocation_short_name,
                    args.xpath, lines_read+1, strerror(err));
            free(x); free(csrad); free(csra); free(csrcolidx); free(csrrowptr);
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
    }

    double * y = malloc(num_rows * sizeof(double));
    if (!y) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(x);
        free(csrad); free(csra); free(csrcolidx); free(csrrowptr);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
#ifdef WITH_OPENMP
    #pragma omp parallel for
#endif
    for (int64_t i = 0; i < num_rows; i++) y[i] = 0.0;

    /* read y vector from a Matrix Market file */
    if (args.ypath) {
        if (args.verbose > 0) {
            fprintf(stderr, "mtxfile_read: ");
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        FILE * f;
        if ((f = fopen(args.ypath, "r")) == NULL) {
            fprintf(stderr, "%s: %s: %s\n",
                    program_invocation_short_name, args.ypath, strerror(errno));
            free(y); free(x); free(csrad); free(csra); free(csrcolidx); free(csrrowptr);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        enum mtxobject object;
        enum mtxformat format;
        enum mtxsymmetry symmetry;
        int64_t ynum_rows;
        int64_t ynum_columns;
        int64_t ynum_nonzeros;
        int64_t lines_read = 0;
        int64_t bytes_read = 0;
        err = mtxfile_fread_header(
            &object, &format, &symmetry,
            &ynum_rows, &ynum_columns, &ynum_nonzeros,
            f, &lines_read, &bytes_read);
        if (err) {
            if (args.verbose > 0) fprintf(stderr, "\n");
            fprintf(stderr, "%s: %s:%"PRId64": %s\n",
                    program_invocation_short_name,
                    args.ypath, lines_read+1, strerror(err));
            fclose(f);
            free(y); free(x); free(csrad); free(csra); free(csrcolidx); free(csrrowptr);
            program_options_free(&args);
            return EXIT_FAILURE;
        } else if (object != mtxvector || format != mtxarray || ynum_rows != num_rows) {
            if (args.verbose > 0) fprintf(stderr, "\n");
            fprintf(stderr, "%s: %s:%"PRId64": "
                    "expected vector in array format of size %'"PRId64"\n",
                    program_invocation_short_name,
                    args.ypath, lines_read+1, num_rows);
            fclose(f);
            free(y); free(x); free(csrad); free(csra); free(csrcolidx); free(csrrowptr);
            program_options_free(&args);
            return EXIT_FAILURE;
        }

        err = mtxfile_fread_vector_array_real(
            num_rows, y, f, &lines_read, &bytes_read);
        if (err) {
            if (args.verbose > 0) fprintf(stderr, "\n");
            fprintf(stderr, "%s: %s:%"PRId64": %s\n",
                    program_invocation_short_name,
                    args.ypath, lines_read+1, strerror(err));
            free(y); free(x); free(csrad); free(csra); free(csrcolidx); free(csrrowptr);
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
    }

    /* 5. compute the matrix-vector multiplication. */
#ifdef WITH_OPENMP
    #pragma omp parallel
#endif
    for (int repeat = 0; repeat < args.repeat; repeat++) {
        #pragma omp master
        if (args.verbose > 0) {
            if (args.separate_diagonal) fprintf(stderr, "gemvsd: ");
            else fprintf(stderr, "gemv: ");
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        if (args.separate_diagonal) {
            err = csrgemvsd(
                num_rows, y, num_columns, x, csrsize, rowsizemin, rowsizemax, csrrowptr, csrcolidx, csra, csrad);
            if (err)
                break;
        } else {
            err = csrgemv(
                num_rows, y, num_columns, x, csrsize, rowsizemin, rowsizemax, csrrowptr, csrcolidx, csra);
            if (err)
                break;
        }

        int64_t num_flops = 2*(csrsize+diagsize);
        int64_t min_bytes = num_rows*sizeof(*y) + num_columns*sizeof(*x)
            + (num_rows+1)*sizeof(*csrrowptr) + csrsize*sizeof(*csrcolidx) + csrsize*sizeof(*csra) + diagsize*sizeof(*csrad);
        int64_t max_bytes = num_rows*sizeof(*y) + csrsize*sizeof(*x)
            + num_rows*sizeof(*csrrowptr) + csrsize*sizeof(*csrcolidx) + csrsize*sizeof(*csra)
            + diagsize*sizeof(*csrad) + diagsize*sizeof(*x);

        #pragma omp master
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
    if (err) {
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(err));
        free(y); free(x);
        free(csrad); free(csra); free(csrcolidx); free(csrrowptr);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    free(x);
    free(csrad); free(csra); free(csrcolidx); free(csrrowptr);

    /* 6. write the result vector to a file */
    if (!args.quiet) {
        if (args.verbose > 0) {
            fprintf(stderr, "mtxfile_write: ");
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        fprintf(stdout, "%%%%MatrixMarket vector array real general\n");
        fprintf(stdout, "%"PRId64"\n", num_rows);
        for (int64_t i = 0; i < num_rows; i++) fprintf(stdout, "%.*g\n", DBL_DIG, y[i]);
        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(stderr, "%'.6f seconds\n", timespec_duration(t0, t1));
        }
    }

    free(y);
    program_options_free(&args);
    return EXIT_SUCCESS;
}
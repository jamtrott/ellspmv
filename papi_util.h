/**
 * @file papi_util.h
 * @author Sergej Breiter (sergej.breiter@gmx.de)
 * @brief
 * @version 0.1
 * @date 2022-11-05
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef PAPI_UTIL_H
#define PAPI_UTIL_H

#include <stdio.h> // FILE *

enum papi_util_err_t {
   PAPI_UTIL_OK = 0,
   PAPI_UTIL_ERRNO,
   PAPI_UTIL_PARSE_ERROR,
   PAPI_UTIL_PAPI_NOT_SUPPORTED,
   PAPI_UTIL_PAPI_VERSION_MISMATCH,
   PAPI_UTIL_PAPI_ERROR };

struct papi_util_opt {
    const char *event_file;
    int print_csv;
    int print_threads;
    int print_summary;
    int print_region;
    int component;
    int multiplex;
    FILE *output;
};

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

const char * PAPI_UTIL_strerror(int err, int papierr);
int PAPI_UTIL_setup(const struct papi_util_opt *opt, int *papierr);
int PAPI_UTIL_start(const char *region_name, int *papierr);
void PAPI_UTIL_finish(void);
void PAPI_UTIL_finalize(void);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif

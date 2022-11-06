/**
 * @file papi_util.c
 * @author Sergej Breiter (sergej.breiter@gmx.de)
 * @brief
 * @version 0.1
 * @date 2022-11-05
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif /* _GNU_SOURCE */

#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <omp.h>
#include <papi.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "papi_util.h"

/* TODO: cleanup */
/* TODO: csv-output */

/* object-like defines */

#define COMMENT '#'

#define MAX_THREADS 500
#define MAX_EVENTS 50
#define MAX_FORMULAS 20

/* print defines */

#define KNRM "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[38m"
#define BOLD "\033[1m"
#define THIN "\033[0m"
#define STATSSEP \
    "====================================================================\n"
#define INFO_ERR_FMT ": ‘" KGRN BOLD "%s" THIN KNRM "‘"
#define PAPI_UTIL_ERROR_PREFIX "PAPI UTIL: "

/* function-like defines */

#define eprintf(format, ...) fprintf(stderr, PAPI_UTIL_ERROR_PREFIX format, ##__VA_ARGS__)
#define pprintf(format, ...) fprintf(_opt.output, format, ##__VA_ARGS__)

#define CHECK_PAPI_ERROR_VAARG(fn, format, ...)                                      \
    do {                                                                             \
        int ret;                                                                     \
        if ((ret = (fn)) != PAPI_OK) {                                               \
            eprintf(KGRN BOLD                                                        \
                    "%s:line %d:" THIN KNRM " In function ‘" KGRN BOLD #fn THIN KNRM \
                    "‘:\n\t" BOLD KRED "PAPI error: " KNRM THIN "%s " format " \n",  \
                    __FILE__, __LINE__, PAPI_strerror(ret), ##__VA_ARGS__);          \
            exit(EXIT_FAILURE);                                                      \
        }                                                                            \
    } while (0)

// llvm compiler does not support __VA__OPT( ) so we have to use ##__VA_ARGS__
// and do this workaround
#define CHECK_PAPI_ERROR(fn) CHECK_PAPI_ERROR_VAARG(fn, "")
#define CHECK_PAPI_ERROR_MSG(fn, msg) \
    CHECK_PAPI_ERROR_VAARG(fn, INFO_ERR_FMT, msg)

#define EXIT_PERROR(f)          \
    if (!(f))                   \
        do {                    \
            perror(#f);         \
            exit(EXIT_FAILURE); \
    } while (0)

enum papi_util_err_t { PAPI_UTIL_OK = 0,
                       PAPI_UTIL_PARSE_ERROR,
                       PAPI_UTIL_INIT_ERROR };

/***************************
 *
 * Expression Tree Code
 *
 ***************************/

typedef double (*binop_type)(double, double);

struct exptree_node {
    double value;
    char *event;
    struct exptree_node *left;
    struct exptree_node *right;
    struct exptree_node *parent;
    binop_type fn;
};

struct measurement {
    char **event_names;
    long long *values;
    double time;
};

struct user_formula {
    char *metric;
    char *unit;
    struct exptree_node *root;
};

static struct exptree_node *_make_exptree(struct exptree_node *self, char *form);
static void _destroy_exptree(struct exptree_node *self);

static int get_formula(struct user_formula *f, const char *string)
{
    char *formula;
    // parse formula string
    if (sscanf(string, " %255m[^[] [%255m[^]] %*[^=]= %255m[^\n]", &(f->metric),
               &(f->unit), &formula) != 3) {
        return PAPI_UTIL_PARSE_ERROR;
    }
    // we have a new formula - build a binary expression tree
    EXIT_PERROR(f->root = calloc(1, sizeof(struct exptree_node)));
    struct exptree_node *result = _make_exptree(f->root, formula);
    free(formula);

    // sanity check
    if (f->root != result) {
        _destroy_exptree(f->root);
        return PAPI_UTIL_PARSE_ERROR;
    }
    return PAPI_UTIL_OK;
}

static void destroy_formula(struct user_formula *f)
{
    free(f->metric);
    free(f->unit);
    _destroy_exptree(f->root);
}

/**
 * assumes null-terminated list
 */
static double _get_value(struct exptree_node *node, struct measurement *meas)
{
    if (node->event == NULL) {
        return node->value;
    }
    char **event_names = meas->event_names;

    int i = 0;
    while (*event_names) {
        if (!strcasecmp("time", node->event)) {
            return meas->time;
        } else if (!strcmp(*event_names, node->event)) {
            return (double)meas->values[i];
        }
        event_names++;
        i++;
    }

    eprintf("event not found: %s\n", node->event);
    return -1.0; // TODO
}

/**
 * @brief Evaluates binary expression tree.
 * @param node the root node
 * @param meas the measurement data
 * @return the tree evaluation
 */
static double evaluate_exptree(struct exptree_node *node, struct measurement *meas)
{
    // binary node
    if (node->left && node->right) {
        assert(node->fn != NULL);
        return node->fn(evaluate_exptree(node->left, meas),
                        evaluate_exptree(node->right, meas));
    }
    // unary node
    if (node->left) {
        return evaluate_exptree(node->left, meas);
    }
    // leaf
    return _get_value(node, meas);
}

static inline int _isoperator(char c)
{
    return c == '+' || c == '-' || c == '*' || c == '/';
}

static inline int _isfloat(char c)
{
    return isdigit(c) || c == 'e' || c == 'E' || c == '.' || c == '+' || c == '-';
}

static inline double _add(double a, double b) { return a + b; }
static inline double _sub(double a, double b) { return a - b; }
static inline double _mul(double a, double b) { return a * b; }
static inline double _div(double a, double b) { return a / b; }

static binop_type _getfunc(char c)
{
    switch (c) {
    case '+':
        return _add;
    case '-':
        return _sub;
    case '*':
        return _mul;
    case '/':
        return _div;
    default:
        return NULL; // TODO: (should never be the case)
    }
}

/**
 * @brief Allocates new node and sets parent node.
 * @param parent the parent node
 * @return new node
 */
static struct exptree_node *_make_node(struct exptree_node *parent)
{
    struct exptree_node *self;
    EXIT_PERROR(self = calloc(1, sizeof(struct exptree_node)));
    self->parent = parent;
    if (parent->left == NULL) {
        parent->left = self;
    } else if (parent->right == NULL) {
        parent->right = self;
    } else {
        // TODO: check return value in _make_exptree
        free(self);
        return NULL;
    }
    return self;
}

static char *_evname; // helper ptr for sscanf in _make_exptree
/**
 * @brief Builds calculator expression tree from formula string.
 * @param self the calling node. Set to allocated root node on first invocation
 * @param returns root node if successfull otherwise NULL
 *
 * @note it is the users responsiblity to free the tree's memory with a call to
 * @link destroy_tree
 */
static struct exptree_node *_make_exptree(struct exptree_node *self, char *string)
{
    // NUL: recursion end
    if (string[0] == 0x0) {
        return self;
    } else if (isspace(string[0])) {
        return _make_exptree(self, string + 1);
    }
    // open braces: new node and step to child
    else if (string[0] == '(') {
        struct exptree_node *new = _make_node(self);
        return _make_exptree(new, string + 1);
    }
    // float: new leaf node
    else if (isdigit(string[0])) {
        struct exptree_node *new = _make_node(self);
        if (sscanf(string, " %lf %*s", &(new->value)) != 1) {
            return NULL;
        }
        // we have successfully parsed the value of the leaf
        else {
            // now get the number of float chars
            int fltchars = 0;
            while (_isfloat(string[fltchars])) {
                fltchars++;
            }
            return _make_exptree(self, string + fltchars);
        }
    } else if (_isoperator(string[0])) {
        self->fn = _getfunc(string[0]);
        return _make_exptree(self, string + 1);
    }
    // close braces: finished current node and step back to parent
    else if (string[0] == ')') {
        return _make_exptree(self->parent, string + 1);
    }
    // event name: new leaf node
    else if (sscanf(string, " %m[^()+-*/ \n] %*s", &_evname) == 1) {
        struct exptree_node *new = _make_node(self);
        new->event = _evname;
        string += strlen(_evname);
        return _make_exptree(self, string);
    }
    return NULL;
}

static void _destroy_exptree(struct exptree_node *self)
{
    if (self->left)
        _destroy_exptree(self->left);
    if (self->right)
        _destroy_exptree(self->right);
    free(self->event);
    free(self);
}

/** other code */

/* file scope globals */

static long long _thread_values[MAX_THREADS][MAX_EVENTS];
static long long _region_values[MAX_EVENTS];
static long long _total_values[MAX_EVENTS];
static int _event_sets[MAX_THREADS];
static char *_event_names[MAX_EVENTS]; // list of event names
static struct user_formula _formulas[MAX_FORMULAS];

static double _time_start;
static _Thread_local int _thread_counter_started = 0;

static int _num_events = 0;
static int _num_threads = 0;
static int _num_formulas = 0;
static double _time_measured = 0.0;
static int _initialized = 0;

// values that can be set by user
static const char *_region_name;
static struct papi_util_opt _opt = {.event_file = NULL,
                                    .print_csv = 0,
                                    .print_threads = 0,
                                    .print_summary = 1,
                                    .print_region = 1,
                                    .multiplex = 0,
                                    .output = NULL};

/**
 * Initializes the event set for a given thread
 */
static void event_init(int *event_set, int thread_num)
{
    event_set[thread_num] = PAPI_NULL;
    CHECK_PAPI_ERROR(PAPI_create_eventset(&event_set[thread_num]));

    // assign eventset to default component (0 is CPU component)
    CHECK_PAPI_ERROR(
        PAPI_assign_eventset_component(event_set[thread_num], _opt.component));

    // set multiplex event to increase virtual counters (may degrade accuracy)
    if (_opt.multiplex)
        CHECK_PAPI_ERROR(PAPI_set_multiplex(event_set[thread_num]));

    // add named events to event set
    int event_code;
    for (int i = 0; _event_names[i] != NULL; i++) {
        CHECK_PAPI_ERROR_MSG(PAPI_event_name_to_code(_event_names[i], &event_code),
                             _event_names[i]);

        CHECK_PAPI_ERROR_MSG(PAPI_add_event(event_set[thread_num], event_code),
                             _event_names[i]);
    }
    // first thread sets number of events
    if (thread_num == 0)
        _num_events = PAPI_num_events(event_set[thread_num]);
}

/**
 *
 */
static void print_values(double time, long long *values)
{
    // RAW EVENTS
    for (int i = 0; i < _num_events; i++) {
        pprintf("%45s : %15lld\n", _event_names[i], values[i]);
    }
    pprintf("\n");

    // DERIVED EVENTS
    for (int i = 0; i < _num_formulas; i++) {
        double value = evaluate_exptree(
            _formulas[i].root,
            &(struct measurement){
                .event_names = _event_names, .values = values, .time = time});

        // TODO: stupid workaround to print avg. frequency per thread
        if ((values == _region_values || values == _total_values) && !strncasecmp(_formulas[i].metric, "frequency", strlen("frequency"))) {
            value /= omp_get_max_threads();
        }

        pprintf("%45s : %15.4lf [%s]\n", _formulas[i].metric, value,
                _formulas[i].unit);
    }
    pprintf("\n");

    if (values == _region_values || values == _total_values) {
        pprintf("%45s : %15.4lf [%s]\n", "Time", time, "s");
    }
}

static void print_values_csv(double time, long long *values)
{
    // RAW EVENTS
    for (int i = 0; i < _num_events; i++) {
        pprintf(",%lld", values[i]);
    }

    // DERIVED EVENTS
    for (int i = 0; i < _num_formulas; i++) {
        double value = evaluate_exptree(
            _formulas[i].root,
            &(struct measurement){
                .event_names = _event_names, .values = values, .time = time});

        // TODO: stupid workaround to print avg. frequency per thread
        if ((values == _region_values || values == _total_values) && !strncasecmp(_formulas[i].metric, "frequency", strlen("frequency"))) {
            value /= omp_get_max_threads();
        }

        pprintf(",%lf", value);
    }
    pprintf(",%lf\n", time);
}

static void print_header_csv()
{
    pprintf("region,threads,");
    // RAW EVENTS
    for (int i = 0; i < _num_events; i++) {
        pprintf("%s,", _event_names[i]);
    }

    // DERIVED EVENTS
    for (int i = 0; i < _num_formulas; i++) {
        pprintf("%s,", _formulas[i].metric);
    }
    pprintf("time\n");
}

/**
 * Gets PAPI event names from eventfile.
 * Returns PAPI events as list.
 * Returns NULL if event file could not be opened.
 * Sets _num_events to the number of events
 */
static char **read_eventfile(const char *event_file)
{
    // char **events = NULL;
    int num_events = 0;
    int read_formulas = 0;
    FILE *file;
    if (!(file = fopen(event_file, "r"))) {
        eprintf("fopen: %s %s\n", strerror(errno), event_file);
        return NULL;
    }

    // get events from file
    size_t n = 0;
    char *lineptr = NULL;
    while ((getline(&lineptr, &n, file)) != -1) {
        // skip empty line or comment
        if (lineptr[0] == COMMENT || strlen(lineptr) <= 1) {
            // do nothing but free lineptr in the end
        }
        // from now on parse formulas
        else if (!strncasecmp(lineptr, "formulas", strlen("formulas"))) {
            read_formulas = 1;
        }
        // get event name from line
        else if (!read_formulas) {
            // events = realloc(events, (_num_events + 1) * sizeof(char*));

            // remove newline (PAPI does not like it)
            lineptr[strlen(lineptr) - 1] = 0;
            // printf("%s\n", lineptr);
            assert(num_events < MAX_EVENTS);
            _event_names[num_events] = lineptr;
            lineptr = NULL; // cannot free this because event name still needed
            num_events++;
        }
        // get derived formula from line
        else {
            if (get_formula(&_formulas[_num_formulas], lineptr) == 0) {
                _num_formulas++;
            } else {
                eprintf("could not parse formula: %s\n", lineptr);
            }
        }
    }
    free(lineptr);
    fclose(file);

    // add NULL to terminate list
    _event_names[num_events] = NULL;
    // EXIT_PERROR(events = realloc(events, (_num_events + 1) * sizeof(char *)));
    _num_events = num_events;
    return _event_names;
}

void PAPI_UTIL_start(const char *region_name)
{
    assert(_initialized);

    _region_name = region_name;

    // erase thread and region values
    memset(_region_values, 0x0, _num_events * sizeof(long long));
    memset(_thread_values, 0x0, _num_threads * _num_events * sizeof(long long));

// everything is setup so we just have to start the counters for each thread
#pragma omp parallel
    if (PAPI_num_events(_event_sets[omp_get_thread_num()]) > 0) {
        assert(!_thread_counter_started);
        CHECK_PAPI_ERROR(PAPI_start(_event_sets[omp_get_thread_num()]));
        _thread_counter_started = 1;
    }

    _time_start = omp_get_wtime();
}

#if START_IN_PARALLEL_REGION_IMPLEMENTED
void PAPI_UTIL_thread_start(char *region_name)
{
    assert(_initialized);

#pragma omp single
    {
        _region_name = region_name;

        // erase thread and region values
        memset(_region_values, 0x0, _num_events * sizeof(long long));
        memset(_thread_values, 0x0, _num_threads * _num_events * sizeof(long long));
    }

#pragma omp barrier

    // everything is setup so we just have to start the counters for each thread
    if (PAPI_num_events(_event_sets[omp_get_thread_num()]) > 0) {
        assert(!_thread_counter_started);
        CHECK_PAPI_ERROR(PAPI_start(_event_sets[omp_get_thread_num()]));
        _thread_counter_started = 1;
    }
}
#endif

/**
 * Initializes PAPI library and events for all threads.
 */
void PAPI_UTIL_setup(const struct papi_util_opt *opt)
{
    if (_initialized) {
        eprintf("error: already initialized\n");
        return;
    }

    if (opt != NULL)
        _opt = *opt;
    if (!_opt.output)
        _opt.output = stdout;

    read_eventfile(_opt.event_file);

    // printf("eventfile: %s\n", event_file);
    if (_opt.print_csv) {
        print_header_csv();
    }

    int retval;
    retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT && retval > 0) {
        eprintf("PAPI library version mismatch!\n");
        exit(EXIT_FAILURE);
    }
    if (retval < 0) {
        eprintf("PAPI library not initialized!\n");
        exit(EXIT_FAILURE);
    }
    retval = PAPI_is_initialized();
    if (retval != PAPI_LOW_LEVEL_INITED) {
        eprintf("PAPI low level not initialized!\n");
        exit(EXIT_FAILURE);
    }

    if (_opt.multiplex)
        CHECK_PAPI_ERROR(PAPI_multiplex_init());

    CHECK_PAPI_ERROR(PAPI_thread_init(pthread_self));

#pragma omp parallel
    {
        assert(omp_get_num_threads() <= MAX_THREADS);

#pragma omp single
        _num_threads = omp_get_num_threads();

        CHECK_PAPI_ERROR(PAPI_register_thread());
    }

    // inizialize events for each thread
#pragma omp parallel
    event_init(_event_sets, omp_get_thread_num());

    _initialized = 1;
}

/**
 *
 */
void PAPI_UTIL_finish(void)
{
    double time = omp_get_wtime() - _time_start;

    if (!_initialized) {
        eprintf("error: not initialized\n");
        return;
    }
    _time_measured += time;
#pragma omp parallel
    {
        if (PAPI_num_events(_event_sets[omp_get_thread_num()]) > 0)
            CHECK_PAPI_ERROR(PAPI_stop(_event_sets[omp_get_thread_num()],
                                       _thread_values[omp_get_thread_num()]));

#pragma omp barrier

        if (_opt.print_threads) {
#pragma omp for ordered
            for (int i = 0; i < _num_threads; i++)
#pragma omp ordered
            {
                if (_opt.print_csv) {
                    pprintf("%s,%d", _region_name, omp_get_thread_num());
                    print_values_csv(time, _thread_values[omp_get_thread_num()]);
                } else {
                    pprintf(STATSSEP "   Thread %d Counters:\n" STATSSEP,
                            omp_get_thread_num());
                    print_values(time, _thread_values[omp_get_thread_num()]);
                }
            }
        }
    } // end parallel region

    // accumulate values from all threads and regions
    for (int e = 0; e < _num_events; e++) {
        // #pragma omp for reduction(+ : _region_values[e])
        for (int i = 0; i < _num_threads; i++) {
            _region_values[e] += _thread_values[i][e];
        }
        // accumulate for total summary
        _total_values[e] += _region_values[e];
    }

    if (_opt.print_region) {
        if (_opt.print_csv) {
            pprintf("%s,%d", _region_name, -1);
            print_values_csv(time, _region_values);
        } else {
            pprintf(STATSSEP "   Region %s Summary (%d Threads):\n" STATSSEP,
                    _region_name, omp_get_max_threads());
            print_values(time, _region_values);
        }
    }
}

void PAPI_UTIL_finalize(void)
{
    // assert(_initialized);
    if (!_initialized) {
        eprintf("error: not initialized\n");
        return;
    }

    if (_opt.print_summary) {
        if (_opt.print_csv) {
            pprintf("%s,%d", "total", -1);
            print_values_csv(_time_measured, _total_values);
        } else {
            pprintf(STATSSEP "   Total Summary (%d Threads):\n" STATSSEP,
                    omp_get_max_threads());
            print_values(_time_measured, _total_values);
        }
    }

    // cleanup events and buffers for thread counters
#pragma omp parallel
    {
        CHECK_PAPI_ERROR(PAPI_cleanup_eventset(_event_sets[omp_get_thread_num()]));
        CHECK_PAPI_ERROR(PAPI_destroy_eventset(&_event_sets[omp_get_thread_num()]));
    } // end parallel region

    // free event name strings and event name list
    for (char **p = _event_names; *p; p++) {
        free(*p);
        *p = NULL;
    }
    // free binary expression trees
    for (int i = 0; i < _num_formulas; i++)
        destroy_formula(&_formulas[i]);

    // free(_event_name);
    _initialized = 0;
}

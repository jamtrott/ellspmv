# Benchmark programs for sparse matrix-vector multiply
#
# Copyright (C) 2023 James D. Trotter
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see
# <https://www.gnu.org/licenses/>.
#
# Authors: James D. Trotter <james@simula.no>
#
# Last modified: 2023-03-08
#
# Benchmarking program for sparse matrix-vector multiplication (SpMV)
# with matrices in ELLPACK and CSR format.

csrspmv = csrspmv
ellspmv = ellspmv

all: $(csrspmv) $(ellspmv)
clean:
	rm -f $(csrspmv_c_objects) $(csrspmv)
	rm -f $(ellspmv_c_objects) $(ellspmv)
.PHONY: all clean

CFLAGS ?= -O2 -g -Wall
LDFLAGS ?= -lm

papi_util.o: papi_util.c papi_util.h
	$(CC) -c $(CFLAGS) $< -o $@

csrspmv_c_sources = csrspmv.c
csrspmv_c_headers = papi_util.h
csrspmv_c_objects := $(foreach x,$(csrspmv_c_sources),$(x:.c=.o))
$(csrspmv_c_objects): %.o: %.c $(csrspmv_c_headers)
	$(CC) -c $(CFLAGS) $< -o $@
$(csrspmv): $(csrspmv_c_objects) papi_util.o
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

ellspmv_c_sources = ellspmv.c
ellspmv_c_headers = papi_util.h
ellspmv_c_objects := $(foreach x,$(ellspmv_c_sources),$(x:.c=.o))
$(ellspmv_c_objects): %.o: %.c $(ellspmv_c_headers)
	$(CC) -c $(CFLAGS) $< -o $@
$(ellspmv): $(ellspmv_c_objects) papi_util.o
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

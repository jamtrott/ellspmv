# Benchmark program for sparse matrix-vector multiply (ELLPACK format)
# Copyright (C) 2022 James D. Trotter
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
# Last modified: 2022-05-13
#
# Benchmarking program for sparse matrix-vector multiplication (SpMV)
# with matrices in ELLPACK format.

ellspmv = ellspmv

all: $(ellspmv)
clean:
	rm -f $(ellspmv_c_objects) $(ellspmv)
.PHONY: all clean

CFLAGS += -g -Wall

ifndef NO_OPENMP
CFLAGS += -fopenmp -DWITH_OPENMP
endif

ellspmv_c_sources = ellspmv.c
ellspmv_c_headers =
ellspmv_c_objects := $(foreach x,$(ellspmv_c_sources),$(x:.c=.o))
$(ellspmv_c_objects): %.o: %.c $(ellspmv_c_headers)
	$(CC) -c $(CFLAGS) $< -o $@
$(ellspmv): $(ellspmv_c_objects)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

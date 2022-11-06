#!/bin/bash

#################################################
#
# build for A64FX using FCC and Sector Cache
#
#################################################

make USEPAPI=1 NO_OPENMP=1 \
    CC=fcc \
    CFLAGS="-Kfast -Kocl -Kopenmp -Koptmsg=2 -DWITH_OPENMP -DA64FXCPU -DL2WAYS=4"

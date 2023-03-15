#!/bin/bash

#################################################
#
# build for A64FX using FCC and Sector Cache
#
#################################################

CC=fcc CFLAGS="-Kfast -Kocl -Kopenmp -Koptmsg=2 -DA64FXCPU -DL2WAYS=4" USEPAPI=1 make

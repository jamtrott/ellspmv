#!/bin/bash

#################################################
#
# build for A64FX using FCC and Sector Cache
#
#################################################

CC=fcc CFLAGS="-Kfast -Kocl -Kopenmp -Koptmsg=2 -DUSE_A64FX_SECTOR_CACHE -DA64FX_SECTOR_CACHE_L2_WAYS=4" make

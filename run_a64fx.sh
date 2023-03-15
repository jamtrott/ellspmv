#!/bin/bash

#################################################
#
# run on A64FX
#
#################################################

export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Enable Fujitsu HPC extensions
export FLIB_HPCFUNC=TRUE

# Use Sector Cache
export FLIB_SCCR_CNTL=TRUE

# Use L1 Sector Cache if L2 Sector Cache unavailable
export FLIB_L1_SCCR_CNTL=FALSE

# enable huge page library
export XOS_MMM_L_HPAGE_TYPE=hugetlbfs

MATRIX=Lynx68_reordered/Lynx68_reordered.mtx
OUTVECTOR=out.mtx

# using example event file (measures memory bandwidth and dp flops)
export PAPI_UTIL_EVENTFILE=papi_util/perfgroup/a64fx/memdp.txt
./ellspmv -v ${MATRIX} ${OUTVECTOR}

#!/bin/bash
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4 
export MKL_NUM_THREADS=6
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=6
task(){
   echo $1;
   python CreateAverageXpart.py $1;
}

N=50
(
for k in {0..9999}; do 
   ((i=i%N)); ((i++==0)) && wait
   task "$k" & 
done
)

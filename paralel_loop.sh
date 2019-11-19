#!/bin/bash
task(){
   python Create3Data.py "$1";
   echo "$1"
}

N=100
(
for k in {0..9999}; do 
   ((i=i%N)); ((i++==0)) && wait
   task "$k" & 
done
)
N=4
(
for thing in a b c d e f g; do 
   ((i=i%N)); ((i++==0)) && wait
   task "$thing" & 
done
)
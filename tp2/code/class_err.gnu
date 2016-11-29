set terminal pdf
set output 'class_error.pdf'
set title "TP2"
set xlabel "Epoch"
set ylabel "Classification error"

plot "nn.dat" using 1:4 with lines title "training", "nn.dat" using 1:6 with lines title "validation"
pause 5
reread

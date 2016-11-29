set terminal pdf
set output 'nll.pdf'
set title "TP2"
set xlabel "Epoch"
set ylabel "Negative log-likelihood"

plot "nn.dat" using 1:3 with lines title "training", "nn.dat" using 1:5 with lines title "validation"
pause 5
reread

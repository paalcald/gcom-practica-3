outfile = 'apartado1.svg'
set terminal svg
set output outfile
set xrange [-1:1]
set yrange [-1:1]
plot 'vector_field.dat' using 1:2:3:4 with vectors, \
     for [i=1:10] 'orbits.dat' using (column(2*i)):(column(2*i+1)) with lines

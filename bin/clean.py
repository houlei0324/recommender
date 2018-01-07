infile = open('../results/result_movie', 'r')
outfile = open('../results/result_movie_clean', 'w')
items = {}
for line in infile:
    tmp = line.split()
    if tmp[1] not in items:
        items[tmp[1]] = 0
        outfile.write(line)
infile.close()
outfile.close()


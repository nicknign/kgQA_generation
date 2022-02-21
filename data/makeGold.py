

import sys


f = open(sys.argv[1])
lines = f.readlines()
f.close()

outlines = []
for line in lines:
    example = eval(line)
    for split_index in range(1, len(example)):
        tgt_list = example[split_index]['raw_sentence']
        outlines.append(' '.join(tgt_list)+'\n')

f = open('test-tgt','w')
f.writelines(outlines)
f.close()


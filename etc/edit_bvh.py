
f = \
    open("../Resource/interaction_motion/SMPLx/HuggingNormal_JIS.bvh", 'r') 
f_write = \
    open("../Resource/interaction_motion/SMPLx/HuggingNormal_JIS_edited.bvh", 'w')
for i, line in enumerate(f):
    if i >= 134:
        idx = line.find('.')
        line = line.replace(line[0:idx], str(int(line[0:idx]) - 10), 1)
    f_write.write(line)
        
f.close()
f_write.close()

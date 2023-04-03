# DESCRIPTION:
#
# A simple script to convert a matlab output (saved to a .mat file) into a python array & web friendly format which can
# be used as input to the simulator J and h at https://www.purdue.edu/p-bit/simulation-tool.html.

# Author: Anirudh Ghantasala



import scipy.io

# these values taken from matlab file python_helper_isingshor.m which uses shuvro's IsingGraphShor2.m
#mat = scipy.io.loadmat('JH_MATLAB.mat')
#J, h, Jr, hr, es, Prefac = mat['J'], mat['h'], mat['Jr'], mat['hr'], mat['es'], mat['Prefac']

mat = scipy.io.loadmat('jh_shor.mat')
Jr, hr = mat['realj'], mat['realh']

print('Converting to web friendly format, copy paste the output in web_friendly_jh.txt into their respective boxes on '
      'the simulator Website...')

# write J to file in web friendly format (https://www.purdue.edu/p-bit/simulation-tool.html) [[,.,.,],[,.,.,]]
f = open("web_friendly_jh.txt", 'w')
f.write('J\n')
f.write('[')
for i in range(Jr.shape[0]):
    f.write('[')
    for j in range(Jr.shape[1]):
        if j < Jr.shape[1]-1:
            f.write(repr(round(Jr[i,j], 4)) + ',')
        else:
            f.write(repr(round(Jr[i,j], 4)))
    if i < Jr.shape[0]-1:
        f.write('],')
    else:
        f.write(']')
f.write(']')


# write h to file
f.write('\nh\n')
f.write('[')

for i in range(hr.shape[0]):
    if i < hr.shape[0]-1:
        f.write(repr(hr[i, 0]) + ',')
    else:
        f.write(repr(hr[i, 0]))

f.write(']')
f.close()
print('Complete...')




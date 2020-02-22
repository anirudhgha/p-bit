import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
import pbit


fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(np.zeros((32,32)))
plt.show(block=False)

def heatmap_cur_state(state, timestep, im_height, im_width):
    #if timestep == 0:
        # create the figure
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # im = ax.imshow(state.reshape((32, 32)))
        # plt.show(block=False)
    # if timestep > 0:
        im.set_array(state.reshape((im_width, im_height)))
        fig.canvas.draw()


orig = Image.open('32x32.png')
baw = orig.convert('1')

width, height = baw.size
J_temp = np.ones((height * width, height*width))

a = baw.getpixel((0, 31))
# each pixel is either 0 or 255

# opposite pixels get weight of 1
for row in range(height):
    for col in range(width):
        cur = row*width + col
        if col+1 < width-1:
            right = row*width+(col + 1)
        else:
            right = 0
        if col - 1 > 0:
            left = row * width + (col - 1)
        else:
            left = width-1
        if row - 1 > 0:
            up = (row - 1) * width + col
        else:
            up = height-1
        if row + 1 < height-1:
            down = (row + 1) * width + col
        else:
            down = 0
        if col+1 < width and baw.getpixel((col, row)) != baw.getpixel((col + 1, row)):
            J_temp[cur, right] = -1
        if row+1 < height and baw.getpixel((col, row)) != baw.getpixel((col, row + 1)):
            J_temp[cur, down] = -1
        if col - 1 > 0 and baw.getpixel((col, row)) != baw.getpixel((col - 1, row)):
            J_temp[cur, left] = -1
        if row - 1 > 0 and baw.getpixel((col, row)) != baw.getpixel((col, row - 1)):
            J_temp[cur, up] = -1
h = np.zeros(1024)

Nt = 1000
myp = pbit.pcircuit(J_temp, h)
m_all = (myp.generate_samples(Nt, gpu=True))
end_state = m_all[Nt-1]

spacing = int(Nt/20)
m_live = np.zeros((20, width*height))
for i in range(20):
    m_live[i,:] = m_all[i*spacing,:]



#live histogram heatmap
for i in range(20):
    heatmap_cur_state(m_live[i], i, 32, 32)




final = np.array(end_state).reshape((32,32))
plt.imshow(final, cmap="gray")
plt.show()

plt.imshow(baw, cmap="gray")
plt.show()

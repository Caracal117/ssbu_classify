import matplotlib.pyplot as plt
import imageio,os
images = []
filenames=sorted((fn for fn in os.listdir('.') if fn.endswith('.jpg')))
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('gif.gif', images,duration=0.1)
print("Done")
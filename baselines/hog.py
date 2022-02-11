
# @Author: Enea Duka
# @Date: 4/28/21
import matplotlib.pyplot as plt
import torch
from skimage.feature import hog
from skimage import data, exposure

if __name__ == '__main__':
    image = data.astronaut()
    print(image.shape)
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, multichannel=True, feature_vector=True)
    fd_torch = torch.tensor(fd).squeeze()
    print(fd_torch.shape)
    print(fd_torch.max())
    print(fd_torch.min())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()
    hog_torch = torch.tensor(hog_image)
    print(hog_torch.shape)
import numpy as np

def get_gradient(img):
    """
    Calculate the 1.order gradient of an Image

    Params:
    -------
    img: Numpy array of input Image

    Returns:
    ------
    gradientx: Numpy array of horizontal gradient
    gradienty: Numpy array of vertical gradient
    """

    (n, m) = np.shape(img)
    gradientx = np.zeros((n, m))
    gradienty = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            if (i == 0 and j == 0) or (i == (n-1) and j == (m-1)) or (i == 0 and j == (m-1)) or (i == (n-1) and j == 0):
                gradientx[i, j] = img[i, j]
                gradienty[i, j] = img[i, j]
            elif i == 0 or i == (n-1):
                gradientx[i, j] = abs(img[i, j+1]-img[i, j-1])
                gradienty[i, j] = img[i, j]
            elif j == 0 or j == (m-1):
                gradientx[i, j] = img[i, j]
                gradienty[i, j] = abs(img[i+1, j]-img[i-1, j])
            else:
                gradientx[i, j] = abs(img[i, j+1]-img[i, j-1])
                gradienty[i, j] = abs(img[i+1, j]-img[i-1, j])
    return (gradientx, gradienty)

def exstend_img(img):
    """
    Exstends the boundaries of the Image

    Params:
    -------
    img: Numpy array of the Image

    Returns:
    ------
    ext_img: The exstended image

    """
    (n, m) = np.shape(img)
    ext_img  = np.zeros((n+2,m+2))

    ext_img[0,1:-1] = img[0,:]    #top
    ext_img[-1,1:-1] = img[-1,:]  #bottom
    ext_img[1:-1,0] = img[:,0]    #left
    ext_img[1:-1,-1] = img[:,-1]  #right
    ext_img[1:-1,1:-1] = img[:,:] #center
    ext_img[0,0] = img[0,0]       #UL corner
    ext_img[0,-1] = img[0,-1]     #UR corner
    ext_img[-1,0] = img[-1,0]     #LL corner
    ext_img[-1,-1] = img[-1,-1]   #LR corner

    return ext_img

def conv(kernel, img):

    """
    Calculate the convolution of an image, given a kernel

    Params:
    -------
    kernel: Numpy array of the kernel
    img: Numpy array of the Image. Has to be the same size as the kernel

    returns:
    ------
    output: int The value of the convolution
    """

    (n, m) = np.shape(kernel)
    output = 0
    for i in range(n):
        for j in range(m):
            output += kernel[i, j]*img[i, j]
    return output


def sobel_filter(img):
    """
    Applies the sobel filter to an Image

    Params:
    -------
    img: numpy array of Image

    return:
    ------
    gradientx : Numpy array of horizontal gradient
    gradienty : Numpy array of  vertifal gradient
    """

    kernelx = np.matrix([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernely = np.matrix([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    ext_img = exstend_img(img)
    (n, m) = np.shape(img)
    gradientx = np.zeros((n, m))
    gradienty = np.zeros((n, m))

    for i in range(1, n):
        for j in range(1, m):
            gradientx[i, j] = conv(kernelx, ext_img[i-1:i+2, j-1:j+2])
            gradienty[i, j] = conv(kernely, ext_img[i-1:i+2, j-1:j+2])

    return (gradientx, gradienty)


def get_directions(gradientx, gradienty):
    """
    Calculates the angle of gradientx

    params:
    --------
    gradientx: Numpy array of horizontal gradients
    gradienty: Numpy array of vertival gradients

    Returns:
    --------
    directions: Numpy array of angles
    """


    (n, m) = np.shape(gradientx)
    directions = np.zeros((n, m))
    weights = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            if gradientx[i, j] == 0:
                weights[i, j] = gradienty[i, j]
                directions[i, j] = np.pi/2
            else:
                weights[i, j] = np.sqrt(gradientx[i, j]**2+gradienty[i, j]**2)
                directions[i, j] = np.arctan(gradienty[i, j]/gradientx[i, j])
    return (directions, weights)

def hog(img, cell_size, num_bins):
    """
    Histogram of Oriented Gradients

    Params:
    -------
    img: Image to calculate HOG on
    cell_size: Size of each HOG cell
    num_bins: Number of histogram bins

    Returns:
    ------
    hog: (n,m,num_bins) array. Histogram of Oriented Gradients
    """
    (gradientx, gradienty) = sobel_filter(img)
    directions, weights = get_directions(gradientx, gradienty)
    (n, m) = np.shape(directions)
    hist = np.zeros((n/cell_size, m/cell_size, num_bins))

    for i in range(n/cell_size):
        for j in range(m/cell_size):
            img_cell = np.ndarray.flatten(directions[i*cell_size:i*cell_size+cell_size, j*cell_size:j*cell_size+cell_size])
            weights_cell = np.ndarray.flatten(weights[i*cell_size:i*cell_size+cell_size, j*cell_size:j*cell_size+cell_size])
            if max(weights_cell) == 0:
                weights_cell = weights_cell + 1
            hist[i, j] = np.histogram(img_cell,  weights=weights_cell, bins=np.linspace(-np.pi/2, np.pi/2,num=num_bins+1), density=True)[0]

    return hist

#img = np.asarray(Image.open("images/Train_face/caltech_web_crop_00003.jpg"), dtype="int")
#(gradientx, gradienty) = sobel_filter(img)
#gradients = np.gradient(img)
#(gradientx, gradienty) = get_gradient(img)
#directions, weights = get_directions(gradientx, gradienty)
#hog=hog(img, 4, 4)
#plt.figure(1)
#plt.imshow(img, cmap="gray")

# plt.figure(1)
# plt.imshow(gradientx, cmap="gray")
# plt.colorbar()

#plt.figure(3)
#plt.imshow(sobx,cmap="gray")
#plt.colorbar()
#plt.figure(4)
#plt.imshow(gradients[0],cmap="gray")

#plt.figure(5)
#plt.imshow(gradients[1],cmap="gray")
#plt.colorbar()

#plt.show()
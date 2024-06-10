import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg as la
import tensorflow as tf
import tensorflow_probability as tfp
from skimage import transform
from scipy.interpolate import interp1d

def build_covariance_matrix(variance, function, width=3):
    n = variance.shape[0]
    var = np.diag(variance.flatten())
    
    distance_map = [
        np.eye(n*n)
    ]
    
    if width > 1:
        # connect each row
        tpRow = np.zeros((n,1), dtype=np.float32)
        tpRow[1] = 1
        offdi = la.toeplitz(tpRow)
        # connect each column
        tpEdge = np.zeros((n,1), dtype=np.float32)
        tpEdge[0] = 1
        offedge = la.toeplitz(tpEdge)
        #connect diagonals
        tpDiag = np.zeros((n,1), dtype=np.float32)
        tpDiag[1] = 1
        offdiag = la.toeplitz(tpDiag)

        I = np.eye(n, dtype=np.float32)
        Ileft = np.roll(I, 1, axis=0) + np.roll(I, -1, axis=0)
        Ileft[0,n-1] = 0
        Ileft[n-1,0] = 0

        A = np.kron(I, offdi) + np.kron(Ileft, offedge)  + np.kron(Ileft, offdiag)
        A *= function(1, 1/np.log(width))
        
        distance_map.append(A)
        
    for weight in range(2, width):
        A_depth = distance_map[-1] @ distance_map[1]
        A_depth[ A_depth > 0 ] = 1.0
        for A_prev in distance_map:
            A_depth[ A_prev > 0 ] = 0.0
        
        A_depth *= function(weight, 1/np.log(width))
            
        distance_map.append(A_depth)

        
    # enforce positive semi-definite
    R = np.sum(distance_map, axis=0)
    #R = R @ R.T
    #R /= R.max()
    covariance = var @ R @ var
    
    return covariance

def _get_fourier_filter(size, filter_name):
    n = np.concatenate((np.arange(1, size / 2 + 1, 2, dtype=int),
                       np.arange(size / 2 - 1, 0, -2, dtype=int)))
    f = np.zeros(size)
    # ramp filter
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2
    
    fourier_filter = 2 * np.real(np.fft.fft(f))
    if filter_name == 'ramp':
        pass
    elif filter_name == 'shepp-logan':
        # start from first element to avoid divide by zero
        omega = np.pi * np.fft.fftfreq(size)[1:]
        fourier_filter[1:] *= np.sin(omega) / omega
    elif filter_name == 'cosine':
        freq = np.linspace(0, np.pi, size, endpoint=False)
        cosine_filter = np.fft.fftshift(np.sin(freq))
        fourier_filter *= cosine_filter
    elif filter_name == 'hamming':
        fourier_filter *= np.fft.fftshift(np.hamming(size))
    elif filter_name == 'hann':
        fourier_filter *= np.fft.fftshift(np.hanning(size))
    elif filter_name is None:
        fourier_filter[:] = 1
        
    return fourier_filter[:, np.newaxis]

def compute_gradient(sinogram, theta, reconstruction_shape, filter_name, progressbar = None):
    radon_image = tf.Variable(sinogram, dtype=tf.complex64)
    angles_count = len(theta)
    img_shape = radon_image.shape[0]
    output_size = img_shape

    fourier_filter = _get_fourier_filter(img_shape, filter_name)
    if progressbar != None:
        progressbar.value += 1
    with tf.GradientTape() as tape:
        projection = tf.transpose(tf.signal.fft(tf.transpose(radon_image))) * fourier_filter
        radon_image_filtered = tf.math.real(tf.transpose(tf.signal.ifft(tf.transpose(projection)))[:img_shape, :])

        if progressbar != None:
            progressbar.value += 1
        
        reconstructed = tf.zeros(reconstruction_shape)
        radius = output_size // 2
        xpr, ypr = np.mgrid[:output_size, :output_size] - radius
        x = np.arange(img_shape, dtype=np.float32) - img_shape // 2

        if progressbar != None:
            progressbar.value += 1
        for col, angle in zip(tf.transpose(radon_image_filtered), np.deg2rad(theta)):
            t = np.asarray(ypr * tf.cos(angle) - xpr * tf.sin(angle), dtype=np.float32)
            interpolant = tfp.math.interp_regular_1d_grid(t, x[0], x[-1], col, fill_value=0)
            reconstructed += interpolant
            if progressbar != None:
                progressbar.value += 1

        kidney_reconstructed = reconstructed * np.pi / (2 * angles_count)

    jacobian = tape.jacobian(kidney_reconstructed, radon_image)
        
    return jacobian, kidney_reconstructed

def get_neigbours_y(p, n, height, width):
    '''
    Get the neighbouring indices of the pixel p along the y-axis within 
    distance n.

    Parameter
    ---------
    p: [int]
        Index of the target pixel.
    n: int
        Distance at whitch the neighbours are considered.
    height: int
        Height of the grid.
    width: int
        Width of the grid.

    Return
    ------
    [int]
        A list of neighbouring indices in ascending order.
    '''

    assert p >= 0, 'The index of the pixel should be poitive or zero' 
    assert (height > 0 or width > 0), 'The height and width should be positive numbers'

    if n < 1:
        return []

    neighbours = []
    # move upwards
    for i in range(1, n+1):
        if p - i * width < 0:
            break
        neighbours.append(p - i * width)

    # ensure ascending order
    neighbours.reverse()

    # move downwards
    for i in range(1, n+1):
        if p + i * width >= width*height:
            break
        neighbours.append(p + i * width)

    return neighbours

def get_neigbours_x(p, n, height, width):
    '''
    Get the neighbouring indices of the pixel p along the x-axis within 
    distance n.

    Parameter
    ---------
    p: [int]
        Index of the target pixel.
    n: int
        Distance at whitch the neighbours are considered.
    height: int
        Height of the grid.
    width: int
        Width of the grid.

    Return
    ------
    [int]
        A list of neighbouring indices in ascending order.
    '''

    assert p >= 0, 'The index of the pixel should be poitive or zero' 
    assert (height > 0 or width > 0), 'The height and width should be positive numbers'

    if n < 1:
        return []

    neighbours = []
    # move left
    for i in range(1, n+1):
        if p - i < 0:
            break
        neighbours.append(p - i)

    # ensure ascending order
    neighbours.reverse()

    # move right
    for i in range(1, n+1):
        if p + i >= width:
            break
        neighbours.append(p + i)

    return neighbours

def build_covariance_y(variance, function, width=3):
    '''
    Generate a covariance matrix, which covariance is along the 
    y-dimension of a grid.

    Parameter
    ---------
    variance: array_like
        A 2D array of variances for each pixel.
    '''
    
    im_height, im_width = variance.shape
    n = im_height * im_width
    var = np.diagflat(variance.flatten())
    
    for p in range(n):
        idy = get_neigbours_y(p, width, im_height, im_width)
        
        omega = np.arange(0,width+1)
        rho = function(omega, 1/np.log(width+1))
        
        top = 1
        bottom = min(width, p//im_width)
        for idx in idy:
            if p < idx:
                # bottom
                #var[p,idx] = np.sqrt(var[p,p]) * np.sqrt(var[idx,idx]) * rho[top]
                var[idx,p] = np.sqrt(var[p,p]) * np.sqrt(var[idx,idx]) * rho[top]
                top += 1
            if p > idx:
            #    # top
                var[idx,p] = np.sqrt(var[p,p]) * np.sqrt(var[idx,idx]) * rho[bottom]
            #    var[p,idx] = np.sqrt(var[p,p]) * np.sqrt(var[idx,idx]) * bottom
                bottom -= 1
            
    return var
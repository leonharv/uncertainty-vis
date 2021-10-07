import numpy as np
from scipy import linalg as la

def exponential(value, gamma):
    return np.exp( - gamma * value)

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

def bhattacharyya(mu1, var1, mu2, var2):
    return 0.25 * np.log(
        0.25 * (
            var1/var2 + var2/var1 + 2
        )
    ) + 0.25 * (
        (mu1 - mu2)**2 / ( var1 + var2 )
    )

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

def get_neighbour_indices(p, n, width, height):
    '''
    Get all neighbour indices of p within a distance of n.
    
    example array: * * * * *
                   * + + + *
                   * + p + *
                   * + + + *
                   * * * * *
                   
    >> get_neighbour_indices(12, 1, 5, 5)
    [[ 6, 7, 8],    # + + +
     [ 11, 12, 13], # + p +
     [ 16, 17, 18]] # + + +
    
    Parameter
    ---------
    p: int
        Index of the target point.
    n: int
        Range of points.
    width: int
        Width of the underlying image.
    height: int
        Height of the underlying image.
        
    Return
    ------
    array_like
        List of the neighbour indices.
    '''
    
    assert p >= 0 and p < width* height, 'Index p must be positive and within the image range'
    assert width > 0 and height > 0, 'Image dimensions have to be positive'
    
    if n <= 0 or (width == 1 and height == 1):
        return np.array([p])
    
    current_row = p // width
    current_column = p % width
    top_row = max(current_row - n, 0)
    bottom_row = min(current_row + n, height - 1)
    left_column = max(current_column - n, 0)
    right_column = min(current_column + n, width - 1)
    
    ret = np.zeros((bottom_row - top_row + 1, right_column - left_column + 1 ))
    
    # top block
    for idx,i in enumerate(range(top_row, bottom_row + 1)):
        for jdx,j in enumerate(range(left_column, right_column + 1)):
            ret[idx,jdx] = i * width + j
            
    return ret
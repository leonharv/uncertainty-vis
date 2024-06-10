''' 
This file summarizes the following scripts:
* System Generation.ipynb
* WEPL generation.ipynb
* Error Propagation Radon transform Resolution.ipynb
'''

import numpy as np
from scipy import sparse
from scipy.interpolate import CubicSpline
import concurrent.futures
from skimage import transform
import tensorflow as tf
from error_propagation_radon_transform import utils

import sys
import logging
logger = logging.getLogger(__name__)


def loadRSP(scale=1):
    '''
    Load the default Head phantom, add padding and scale it.
    '''
    rsp = np.load('../../Data/simple_pCT/Phantoms/Head/RSP.npy')

    # padding is necessary
    phantom = np.pad(rsp[:,:,rsp.shape[2]//2], ((30*scale,30*scale), (100*scale, 100*scale)))

    x = phantom.flatten()
    RSP_shape = phantom.shape[:2]
    return x, RSP_shape

def get_mlp(spline, phi, spotx=0, shape=(130, 1026), chord_length=True):
    '''
    Generate a MLP.
    '''
    MLP_prototype = sparse.lil_matrix(shape)
    
    rotation = np.array([
        [np.cos(-phi), -np.sin(-phi)],
        [np.sin(-phi), np.cos(-phi)]
    ]).T
      
    z_start = -81
    dz = ( 2 * 81 )/shape[1]/2
    box_edge = 96 # x direction
    
    # draw the spline
    i = np.arange(-600,shape[1]+1600)
    z_mlp = i * dz + z_start
    x_mlp = spline(z_mlp)
    # rotate
    point_mlp = (rotation @ np.array([z_mlp, x_mlp])).T
    
    i_rotated = np.int64((point_mlp[:,0] - z_start) / dz/2)
    j_rotated = np.int64((point_mlp[:,1] + box_edge) * shape[0] / (2 * box_edge) )
    
    valid_i = (i_rotated >= 0) & (i_rotated < shape[1])
    valid_j = (j_rotated >= 0) & (j_rotated < shape[0])
    
    valid = valid_i & valid_j
    
    MLP_prototype[ j_rotated[valid], i_rotated[valid] ] = 1
            
    if chord_length:
        points = np.array([z_mlp[valid], x_mlp[valid]]).transpose()
        dx_dz = spline(points[:,0], 1)
        
        length = np.sqrt( dz**2 + dx_dz**2).sum()
                
        return MLP_prototype / length
    else:
        return MLP_prototype
    
def worker_spotx_future(idx, cs, phi, spotx, target_shape, chord_length):
    '''
    Worker function for concurrent executions
    '''
    return get_mlp(cs, phi, spotx, shape=target_shape, chord_length=chord_length).reshape((1,-1))

def bulkMLP_concurrent(num_angle, num_offset, num_spotx, chord_length, target_shape, max_workers=32):
    '''
    Generate many MLP based on the defined parameters.

    Parameters
    ----------
    num_angle: int
        How many angles should be generated in the range of [0,180).
    num_offset: int
        How many offsets (bending of the MLP) should be generated. If 1, no curved MLP will be generated.
    num_spotx: int
        How many parrel beams should be generated.
    chord_length: bool
        If True, the chord length will be used as values for the MLP, otherwise only 1 is set.
    target_shape: array_like
        The shape of the MLP.
    max_workers: int
        How many workers should be generated in parallel.

    Returns
    -------
    sparse.lil_matrix
        A matrix containing all MLP.
    '''
    angles = -np.linspace(0,180,num_angle) * np.pi / 180
    if num_offset == 1:
        offsets = [0]
    else:
        offsets = np.linspace(-15, 15, num_offset)
    spotxs = np.linspace(-95, 95, num_spotx)

    logger.debug('Allocating MLP matrix...')
    MLP_angles_offsets_spotx = sparse.lil_matrix((len(angles) * len(offsets) * len(spotxs), np.prod(target_shape)))
    logger.debug('Finished allocating MLP matrix.')

    future_to_idx = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers = max_workers) as executor:
        for odx,spotx in enumerate(spotxs):
            for xdx, offset in enumerate(offsets):
            
                points = np.array([
                    [-500,spotx], # beam source
                    [-400,spotx],
                    [-300,spotx],
                    [-200,spotx],
                    [-100,spotx],
                    [-81,spotx], # enter phantom

                    [81,offset + spotx] # leave phantom
                ])

                cs = CubicSpline(points[:,0], points[:,1])

                for adx,phi in enumerate(angles):
                    idx = odx * len(angles)*len(offsets) + xdx * len(angles) + adx

                    future = executor.submit(worker_spotx_future, idx, cs, phi, spotx, target_shape, chord_length)
                    future_to_idx[future] = idx
                
    for future in concurrent.futures.as_completed(future_to_idx):
        idx = future_to_idx[future]
        try:
            MLP_angles_offsets_spotx[idx,:] = future.result()
        except Exception as exc:
                print('%r generated an exception: %s' % (idx, exc))

    return MLP_angles_offsets_spotx


def calcWEPL(RSP, MLP):
    wepl = MLP @ RSP.flatten()
    return wepl


def reconstruct(wepl, num_angle, filter_name='ramp', circle=False):
    theta = np.linspace(0., 180., num_angle, endpoint=False)

    reconstructed = transform.iradon(wepl, theta=theta, filter_name=filter_name, circle=circle)
    return reconstructed

def derivate(wepl, num_angle, filter_name='ramp'):
    theta = np.linspace(0., 180., num_angle, endpoint=False)

    reconstruction_shape = (wepl.shape[0], wepl.shape[0])
    jacobian, reconstructed = utils.compute_gradient(wepl, theta, reconstruction_shape, filter_name)


def exponential(value, gamma):
    return np.exp( - gamma * value)

def createInputVariance(wepl, width=10):
    upper_edge = (wepl > 1e-2).argmax(axis=0)
    lower_edge = wepl.shape[0] - np.flip(wepl > 1e-2, 0).argmax(axis=0)

    var = np.zeros_like(wepl)
    for i in range(var.shape[1]):
        x_observed = [upper_edge[i] - wepl.shape[0]//2, upper_edge[i] + 2 - wepl.shape[0]//2, 0, lower_edge[i] - 2 - wepl.shape[0]//2, lower_edge[i] - wepl.shape[0]//2 ]

        y_observed = np.array([ 5, 3.8, 2.05, 3.8, 5 ])
        coeff = np.polyfit(x_observed, y_observed, 6)

        x = np.linspace(-wepl.shape[0]//2, wepl.shape[0]//2, wepl.shape[0])
        polynom = np.poly1d(coeff)

        y = polynom(x)

        y[:upper_edge[i]] = 0
        y[lower_edge[i]:] = 0

        var[:,i] = y

    Sigma_in = utils.build_covariance_y(var**2, function=exponential, width=width)
    return Sigma_in


def error_propataion(jacobian, Sigma_in):
    l,m,n,o = jacobian.shape
    jacobian_reshaped = tf.reshape(jacobian, (l*m, n*o))

    Sigma = jacobian_reshaped @ Sigma_in @ tf.transpose(jacobian_reshaped)

    return Sigma


def main(num_angle=179, num_offset=1, num_spotx=190, chord_length=True, filter_name='ramp'):
    x, RSP_shape = loadRSP(scale=1)

    # settings
    num_angle = 179
    num_offset = 1
    num_spotx = 180 # 180 -> Out of memory
    chord_length = True
    filter_name='ramp'
    width = 10

    max_workers = 90

    chords = 'exact' if chord_length else 'map'

    logger.info('''Parameters are:
                num_angle = {:d}
                num_offset = {:d}
                num_spotx = {:d}
                chord_length = {:s}
                filter_name = {:s}'''.format(
                    num_angle,
                    num_offset,
                    num_spotx,
                    chords,
                    filter_name
                ))

    logger.info('Start generating MLPs...')
    MLP_angles_offsets_spotx = bulkMLP_concurrent(num_angle, num_offset, num_spotx, chord_length, RSP_shape, max_workers)
    logger.info('Finished generating MLPs.')

    logger.info('Saving MLPs.')
    if chord_length:
        sparse.save_npz('../../Data/simple_pCT/MLP/MLP_angles{:d}_offset{:d}_spotx{:d}_exact_{:d}_{:d}.npz'.format(num_angle, num_offset, num_spotx, RSP_shape[0], RSP_shape[1]), MLP_angles_offsets_spotx.tocsc())
    else:
        sparse.save_npz('MLP_angles{:d}_offset{:d}_spotx{:d}_map_{:d}_{:d}.npz'.format(num_angle, num_offset, num_spotx, RSP_shape[0], RSP_shape[1]), MLP_angles_offsets_spotx.tocsc())

    logger.info('Start generating WEPLs...')
    wepl = calcWEPL(x, MLP_angles_offsets_spotx)
    wepl = np.reshape(wepl, (num_spotx, num_angle))
    logger.info('Finished generating WEPLs.')

    logger.info('Saving WEPLs.')
    np.save('../../Data/simple_pCT/WEPL/WEPL_angles{:d}_offset{:d}_spotx{:d}_exact_{:d}_{:d}.npy'.format(num_angle, num_offset, num_spotx, RSP_shape[0], RSP_shape[1]), wepl)

    logger.info('Start reconstruction...')
    reconstructed = reconstruct(wepl, num_angle, filter_name)
    logger.info('Finished reconstruction.')

    logger.info('Save reconstruction.')
    np.save('../../Data/simple_pCT/Reconstruction/Head/{:d}_{:d}/RSP_angles{:d}_offset{:d}_spotx{:d}_{:s}_{:s}.npy'.format(RSP_shape[0], RSP_shape[1], num_angle, num_offset, num_spotx, chords, filter_name), reconstructed)

    logger.info('Start derivation...')
    jacobian, _ = derivate(wepl, num_angle, filter_name)
    logger.info('Finished derivation.')

    logger.info('Save jacobian.')
    np.save('../../Data/simple_pCT/Jacobian/J_angles{:d}_offset{:d}_spotx{:d}_{:s}_{:s}_{:d}_{:d}.npy'.format(num_angle, num_offset, num_spotx, chords, filter_name, RSP_shape[0], RSP_shape[1]), jacobian)

    logger.info('Start Sigma input generation...')
    Sigma_in = createInputVariance(wepl, width)
    logger.info('Finish Sigma input generation.')

    logger.info('Start Sigma output generation...')
    Sigma_out = error_propataion(jacobian, Sigma_in)
    logger.info('Finished Sigma output generation.')

    logger.info('Save Sigma_out.')
    np.save('../../Data/simple_pCT/Sigma/Sigma_raedler_angles{:d}_offset{:d}_spotx{:d}_{:s}_{:s}_{:d}_{:d}.npy'.format(num_angle, num_offset, num_spotx, chords, filter_name, RSP_shape[0], RSP_shape[1]), Sigma_out)


if __name__ == '__main__':
    logging.basicConfig(filename='one-shot-creation.log', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info('Started')
    main()
    logger.info('Finished')
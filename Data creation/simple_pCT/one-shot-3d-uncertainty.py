import numpy as np
import tensorflow as tf
from scipy import sparse
from skimage import transform
from error_propagation_radon_transform import utils

import argparse
import sys
import os
import logging
logger = logging.getLogger(__name__)

def sliceRSP(rsp, z=0, scale=1):
    phantom = np.pad(rsp[:,:,z], ((30*scale,30*scale), (100*scale, 100*scale)))
    RSP_shape = phantom.shape[:2]
    return phantom, RSP_shape

def calcWEPL(RSP, MLP):
    wepl = MLP @ RSP.flatten()
    return wepl

def createStd(wepl, num_spotx):
    upper_edge = (wepl > 1e-2).argmax(axis=0)
    lower_edge = wepl.shape[0] - np.flip(wepl > 1e-2, 0).argmax(axis=0)

    std = np.zeros_like(wepl)
    for i in range(std.shape[1]):
        x_observed = [upper_edge[i] - num_spotx//2, upper_edge[i] + 2 - num_spotx//2, 0, lower_edge[i] - 2 - num_spotx//2, lower_edge[i] - num_spotx//2 ]

        y_observed = np.array([ 5, 3.8, 2.05, 3.8, 5 ])
        coeff = np.polyfit(x_observed, y_observed, 6)

        x = np.linspace(-num_spotx//2, num_spotx//2, num_spotx)
        polynom = np.poly1d(coeff)

        y = polynom(x)

        y[:upper_edge[i]] = 0
        y[lower_edge[i]:] = 0

        std[:,i] = y

    return std

def exponential(value, gamma):
    return np.exp( - gamma * value)

def main(num_angle=179, num_offset=1, num_spotx=190, chord_length=True, filter_name='ramp'):
    rsp = np.load('../../Data/simple_pCT/Phantoms/Head/RSP.npy')
    _, RSP_shape = sliceRSP(rsp, 0)

    width = 10
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
    
    logger.info('Loading MLPs.')
    MLP_angles_offsets_spotx = sparse.load_npz('../../Data/simple_pCT/MLP/MLP_angles{:d}_offset{:d}_spotx{:d}_exact_{:d}_{:d}.npz'.format(num_angle, num_offset, num_spotx, RSP_shape[0], RSP_shape[1]))
    
    logger.info('Loading jacobian.')
    jacobian = np.load('../../Data/simple_pCT/Jacobian/J_angles{:d}_offset{:d}_spotx{:d}_{:s}_{:s}_{:d}_{:d}.npy'.format(num_angle, num_offset, num_spotx, chords, filter_name, RSP_shape[0], RSP_shape[1]))
    l,m,n,o = jacobian.shape
    jacobian_reshaped = tf.reshape(jacobian, (l*m, n*o))

    steps = 8

    variance_out = np.empty((num_spotx,num_spotx,rsp.shape[2]//steps))
    logger.info('Start error propagation...')
    for z in range(0,rsp.shape[2],steps):
        logger.info('Start iteration {:d} of {:d} ...'.format(z, rsp.shape[2]//steps))
        x, _ = sliceRSP(rsp, z)
        logger.info('Calculate WEPL...')
        wepl = calcWEPL(x, MLP_angles_offsets_spotx)
        wepl = np.reshape(wepl, (num_spotx, num_angle))

        logger.info('Calculate Std...')
        std = createStd(wepl, num_spotx)

        logger.info('Build Covariance...')
        Sigma_in = utils.build_covariance_y(std**2, function=exponential, width=width)

        logger.info('Propagate error...')
        Sigma_out = jacobian_reshaped @ Sigma_in @ np.transpose(jacobian_reshaped)
        variance_out[:,:,z] = tf.reshape(tf.abs(tf.linalg.tensor_diag_part(Sigma_out)), (num_spotx,num_spotx))

        # if not os.path.exists('../../Data/simple_pCT/Sigma/{:d}'.format(num_angle)):
        #     os.makedirs('../../Data/simple_pCT/Sigma/{:d}'.format(num_angle))

        # np.save('../../Data/simple_pCT/Sigma/{:d}/Sigma_raedler_angles{:d}_offset{:d}_spotx{:d}_{:s}_{:s}_{:d}_{:d}_z{:d}.npy'.format(num_angle, num_angle, num_offset, num_spotx, chords, filter_name, RSP_shape[0], RSP_shape[1], z), Sigma_out)
        
    logger.info('saving variance.')
    np.save('../../Data/simple_pCT/Variance/Variance_raedler_angles{:d}_offset{:d}_spotx{:d}_{:s}_{:s}_{:d}_{:d}.npy'.format(num_angle, num_offset, num_spotx, chords, filter_name, RSP_shape[0], RSP_shape[1]), variance_out)

    logger.info('Finished error propagation.')

if __name__ == '__main__':
    logging.basicConfig(filename='one-shot-3d-uncertainty.log', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    parser = argparse.ArgumentParser(description='Create MLPs, WEPLs, Jacobian, Reconstruction and Sigma in one shot.')
    parser.add_argument('num_angle', default=178, type=int, help='Number of angles')
    parser.add_argument('num_offset', default=1, type=int, help='Number of offsets')
    parser.add_argument('num_spotx', default=130, type=int, help='Number of spotx')
    parser.add_argument('filter_name', default='ramp', choices=['ramp', 'shepp-logan', 'cosine', 'hamming', 'hann'], help='Filter')

    args = parser.parse_args()

    logger.info('Started')
    main(num_angle=args.num_angle, num_offset=args.num_offset, num_spotx=args.num_spotx, filter_name=args.filter_name)
    logger.info('Finished')
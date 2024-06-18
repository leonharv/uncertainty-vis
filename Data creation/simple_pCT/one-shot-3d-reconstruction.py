import numpy as np
from scipy import sparse
from skimage import transform

import argparse
import sys
import logging
logger = logging.getLogger(__name__)

def sliceRSP(rsp, z=0, scale=1):
    phantom = np.pad(rsp[:,:,z], ((30*scale,30*scale), (100*scale, 100*scale)))
    RSP_shape = phantom.shape[:2]
    return phantom, RSP_shape


def calcWEPL(RSP, MLP):
    wepl = MLP @ RSP.flatten()
    return wepl


def main(num_angle=179, num_offset=1, num_spotx=190, chord_length=True, filter_name='ramp'):
    rsp = np.load('../../Data/simple_pCT/Phantoms/Head/RSP.npy')
    _, RSP_shape = sliceRSP(rsp, 0)

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
        
    theta = np.linspace(0., 180., num_angle, endpoint=False)
    reconstructed = np.empty((num_spotx,num_spotx,rsp.shape[2]))

    logger.info('Start reconstruction...')
    for z in range(rsp.shape[2]):
        x, _ = sliceRSP(rsp, z)
        wepl = calcWEPL(x, MLP_angles_offsets_spotx)
        wepl = np.reshape(wepl, (num_spotx, num_angle))

        reconstructed[:,:,z] = transform.iradon(wepl, theta=theta, filter_name=filter_name, circle=True)
    logger.info('Finished reconstruction.')

    logger.info('Saving reconstruction.')
    np.save('../../Data/simple_pCT/Reconstruction/Head/{:d}_{:d}/3D/RSP_angles{:d}_offset{:d}_spotx{:d}_{:s}_{:s}.npy'.format(RSP_shape[0], RSP_shape[1], num_angle, num_offset, num_spotx, chords, filter_name), reconstructed)


if __name__ == '__main__':
    logging.basicConfig(filename='one-shot-3d-reconstruction.log', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

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

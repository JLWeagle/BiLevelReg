#!/usr/bin/env python-real

import os
import sys
import SimpleITK as sitk
from mod.utils import *
from mod.model import *

class Tester(object):
    def __init__(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)
        self._build_graph()

    def _build_graph(self):
        self.images_tf = tf.placeholder(tf.float32, shape=(1, 2, 160, 192, 224, 1))
        self.model = BiLevelNet(num_levels=2)
        self.flows_final, self.y = self.model(self.images_tf[:, 0], self.images_tf[:, 1])  # image0 image1 (fixed and moving)
        self.nn_trf = nn_trf(name='nn_trf')
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, 'D:/workspace/GraphCut/test/model/Pre-trained/model_40.ckpt')


    def test(self):
        atlas = np.load('D:/workspace/GraphCut/test/data/atlas_norm.npz')
        img1 = atlas['vol']
        img1 = np.reshape(img1, img1.shape + (1,))

        img2 = load_nii_by_name(sys.argv[1])
        images = np.array([img1, img2])  # shape(2, h, w, c, 1)
        images = np.reshape(images, (1,) + images.shape)  # shape(1, 2, h, w, c, 1)

        # Save Vol
        vol = self.sess.run(self.y, feed_dict={self.images_tf: images})
        vol = nib.Nifti1Image(vol[0, :, :, :, :], np.eye(4))
        return vol
        # nib.save(vol, 'data/warped_vol.nii.gz')


def main(input, output):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tester = Tester()
    nib.save(tester.test(), output)


if __name__ == '__main__':
    if len (sys.argv) < 3:
        print("Usage: test <input> <output>")
        sys.exit (1)
    main(sys.argv[1], sys.argv[2])

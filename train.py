import argparse
import time
from model import *
from Dataset import *
from losses import *
from utils import *


class Trainer(object):
    def __init__(self, args):
        self.args = args
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self._build_graph()

    def _build_graph(self):
        shared_keys = ['dataset_dir', 'batch_size', 'num_parallel_calls']
        shared_args = {}
        for key in shared_keys:
            shared_args[key] = getattr(self.args, key)

        # Input data
        with tf.name_scope('Data'):
            tset = BaseDataset_3D(train_or_val='train', **shared_args)
            self.images = tset.get_element()
            self.image0, self.image1 = self.images[:, 0], self.images[:, 1]

            vset = BaseDataset_3D(train_or_val='val', **shared_args)
            self.images_v, self.initializer_v = vset.get_element()
            self.image0_v, self.image1_v = self.images_v[:, 0], self.images_v[:, 1]

            self.num_batches = len(tset.samples) // self.args.batch_size
            self.num_batches_v = len(vset.samples) // self.args.batch_size

        # Model inference
        model = BiLevelNet_train(num_levels=self.args.num_levels)
        self.flows_final, self.y, self.flows, self.pypa_0, self.pypa_1 = model(self.image0, self.image1)  # image0 image1 (fixed and moving)
        self.flows_final_v, self.y_v, self.flows_v, self.pypa_0_v, self.pypa_1_v \
            = model(self.image0_v, self.image1_v, reuse=True)

        # Loss calculation
        with tf.name_scope('Loss'):
            _loss = multirobust_ncc(self.image0, self.image1, self.flows, self.args.weights, self.args.num_levels)
            _loss_v = multirobust_ncc(self.image0_v, self.image1_v, self.flows_v, self.args.weights,
                                      self.args.num_levels)
            self.ncc = ncc(self.image0, self.y)
            self.ncc_v = ncc(self.image0_v, self.y_v)

            weights_l2 = Grad(self.flows_final)
            weights_l2_v = Grad(self.flows_final_v)

            kl_loss = multirobust_MAP(self.pypa_0, self.pypa_1, self.args.weights)
            kl_loss_v = multirobust_MAP(self.pypa_0_v, self.pypa_1_v, self.args.weights)

            self.loss = _loss + self.args.gamma * kl_loss + 15 * self.ncc + weights_l2 * 10
            self.loss_v = _loss_v + self.args.gamma * kl_loss_v + 10 * self.ncc_v + weights_l2_v * 12

            self.kl_loss, self.weights_l2 = kl_loss, weights_l2
            self.kl_loss_v, self.weights_l2_v = kl_loss_v, weights_l2_v

        # Gradient descent optimization
        with tf.name_scope('Optimize'):
            self.global_step = tf.train.get_or_create_global_step()
            lr = self.args.lr
            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss, var_list=model.vars)
            with tf.control_dependencies([self.optimizer]):
                self.optimizer = tf.assign_add(self.global_step, 1)

        # Load learned model
        self.saver = tf.train.Saver(model.vars, max_to_keep=20)
        self.sess.run(tf.global_variables_initializer())
        if self.args.resume is not None:
            print(f'Loading learned model from checkpoint {self.args.resume}')
            self.saver.restore(self.sess, self.args.resume)

        tf.summary.FileWriter('./logs', graph=self.sess.graph)

    def train(self):
        train_start = time.time()
        for e in range(self.args.num_epochs):
            for i in range(self.num_batches):
                time_s = time.time()
                _, loss, ncc, reg_loss, kl_loss = self.sess.run(
                    [self.optimizer, self.loss, self.ncc, self.weights_l2, self.kl_loss])

                if i % 100 == 0:
                    batch_time = time.time() - time_s
                    kwargs = {'loss': loss, 'ncc': ncc, 'reg_loss': reg_loss,
                              'kl_loss': kl_loss, 'batch time': batch_time}
                    show_progress(e + 1, i + 1, self.num_batches, **kwargs)

            loss_vals, ncc_vals, reg_vals, kl_vals = [], [], [], []
            self.sess.run([self.initializer_v])
            for i in range(self.num_batches_v):
                image0_v, image1_v, flows_val, loss_val, ncc_val, reg_val, kl_val \
                    = self.sess.run([self.image0_v, self.image1_v, self.flows_v,
                                     self.loss_v, self.ncc_v, self.weights_l2_v, self.kl_loss_v])
                loss_vals.append(loss_val)
                ncc_vals.append(ncc_val)
                reg_vals.append(reg_val)
                kl_vals.append(kl_val)

            g_step = self.sess.run(self.global_step)
            print(
                f'\r{e+1} epoch validation, loss: {np.mean(loss_vals)}, ncc: {np.mean(ncc_vals)}, reg_loss:{np.mean(reg_vals)}, '
                f'kl_loss:{np.mean(kl_vals)}' \
                + f', global step: {g_step}, elapsed time: {time.time()-train_start} sec.')

            if not os.path.exists('model/pro-model'):
                os.mkdir('model/pro-model')
            self.saver.save(self.sess, f'model/pro-model/model_{e + 1}.ckpt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dd', '--dataset_dir', type=str,
                        help='Directory containing train dataset')
    parser.add_argument('-e', '--num_epochs', type=int, default=100,
                        help='# of epochs ')
    parser.add_argument('-b', '--batch_size', type=int, default=1,
                        help='Batch size ')
    parser.add_argument('-nc', '--num_parallel_calls', type=int, default=1,
                        help='# of parallel calls for data loading ')

    parser.add_argument('--num_levels', type=int, default=2,
                        help='# of levels for feature extraction ')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate [1e-4]')
    parser.add_argument('--weights', nargs='+', type=float,
                        default=[ 0.8, 3.2],
                        help='Weights for each pyramid loss')
    parser.add_argument('--gamma', type=float, default=0.2,
                        help='Coefficient for weight decay [4e-4]')
    parser.add_argument('-r', '--resume', type=str,
                        default= None,
                        help='Learned parameter checkpoint file [None]')

    args = parser.parse_args()
    for key, item in vars(args).items():
        print(f'{key} : {item}')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    trainer = Trainer(args)
    trainer.train()

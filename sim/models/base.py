# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from .. import losses
from ..networks import base as basenet
from ..util import logging as logutil


logger = logutil.Logger(loggee="models/base")


class Model(tf.keras.Model):
    """Uses only the parent's trackability and nothing else.
    """
    def __init__(self, config, debug=False):
        super().__init__()
        self.config = config
        self.debug = debug
        if debug:
            logger.warn("Model in debug mode; behavior may be different")
        self.net = {'main': basenet.Network()} # NOTE: insert trainable networks
        # of your model into this dictionary, values of which will be registered
        # as trainable
        self.trainable_registered = False # NOTE: before training, call
        # register_trainable() to register trainable parameters (which lie in
        # self.net)
        # Initialize loss functions and parse weights
        self.wloss = self._init_loss() # NOTE: list of weight and
        # (initialized) loss function pairs

    def _init_loss(self):
        wloss = []
        loss_str = self.config.get('DEFAULT', 'loss')
        for x in loss_str.split(','):
            loss_name, weight = self._parse_loss_and_weight(x)
            if loss_name == 'lpips':
                loss = losses.LPIPS(per_ch=False)
            elif loss_name == 'elpips':
                bs = self.config.getint('DEFAULT', 'bs')
                loss = losses.ELPIPS(bs)
            elif loss_name == 'l1':
                loss = losses.L1()
            elif loss_name == 'l2':
                loss = losses.L2()
            elif loss_name == 'ssim':
                loss = losses.SSIM(1 - 0)
            else:
                raise NotImplementedError(loss_name)
            wloss.append((weight, loss))
        return wloss

    @staticmethod
    def _parse_loss_and_weight(weight_loss_str):
        """Handles strings like '1e+2lpips' or 'l1,10barron'.
        """
        # Start from the back because looking for the longest string that
        # can be converted to a float
        for i in range(len(weight_loss_str), -1, -1):
            try:
                weight = float(weight_loss_str[:i])
            except ValueError:
                continue
            loss_name = weight_loss_str[i:]
            return loss_name, weight
        # Weight not specified
        return weight_loss_str, 1.

    def register_trainable(self):
        """Trackable objects (such as Keras sequentials and layers) must be
        directly under `self` to be registered to `trainable_variables`, so
        this function simply adds aliases directly under `self` to all nets'
        trainable variables.
        """
        registered = []
        pref = 'net_'
        for net_name, net in self.net.items():
            attr_name = pref + net_name
            assert attr_name.isidentifier(), (
                "Prepending '{pref}' to your network name '{net}' doesn't "
                "make a valid identifier; change your network name").format(
                    pref=pref, net=net_name)
            for layer_i, layer in enumerate(net.layers):
                if layer.trainable:
                    attr_name_full = attr_name + '_layer%d' % layer_i
                    assert not hasattr(self, attr_name_full), (
                        "Can't register `{}` because it is already an "
                        "attribute").format(attr_name_full)
                    setattr(self, attr_name_full, layer)
                    registered.append(attr_name_full)
        logger.info("Trainable layers registered:\n\t%s", registered)
        self.trainable_registered = True

    @staticmethod
    def _validate_mode(mode):
        allowed_modes = ('train', 'vali', 'test')
        if mode not in allowed_modes:
            raise ValueError(mode)

    def call(self, batch, mode='train'):
        """
        Returns:
            tuple:
                - **pred**
                - **gt**
                - **loss_kwargs** (*dict*) -- Keyword arguments for loss
                  computation.
                - **to_vis** (*dict*) -- Tensors to visualize.
        """
        raise NotImplementedError

    def compute_loss(self, pred, gt, **kwargs):
        """
        Returns:
            tf.Tensor: Loss.
        """
        raise NotImplementedError

    def vis_batch(self, data_dict, outdir, mode='train', dump_raw_to=None):
        raise NotImplementedError

    def compile_batch_vis(self, batch_vis_dirs, outpref, mode='train'):
        """Compiles batch visualizations into a consolidated view.

        Returns:
            str: Convinient link to your consolidated view, which will be
            logged into TensorBoard. So you should add proper file extension
            (and maybe also file viewer prefix), returning something like
            ``'http://your.file.viewer/' + outpref + '.html'``.
        """
        raise NotImplementedError

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

# pylint: disable=unsupported-assignment-operation

from os.path import join
from tqdm import tqdm
from absl import app, flags

from google3.pyglib import gfile
from google3.experimental.users.xiuming.xiuminglib import xiuminglib as xm


flags.DEFINE_string('data_dir', '', "")
FLAGS = flags.FLAGS


def main(_):
    correct_names = (
        'olat-0000-0000',
        'olat-0000-0008',
        'olat-0000-0016',
        'olat-0000-0024',
        'olat-0004-0000',
        'olat-0004-0008',
        'olat-0004-0016',
        'olat-0004-0024')

    for view_dir in tqdm(xm.os.sortglob(FLAGS.data_dir, '*_???'), desc="Views"):
        for i, correct_name in enumerate(correct_names):
            src = join(view_dir, 'rgba_olat%03d.png' % i)
            dst = join(view_dir, 'rgba_%s.png' % correct_name)
            if gfile.Exists(src):
                gfile.Rename(src, dst)


if __name__ == '__main__':
    app.run(main)

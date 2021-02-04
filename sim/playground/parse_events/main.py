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

from tqdm import tqdm
from absl import app

from google3.pyglib import gfile

import tensorflow.google.compat.v2 as tf
tf.enable_v2_behavior()


from google3.third_party.tensorflow.core.util import event_pb2


def main(_):
    f = 'events.out.tfevents.1606777424.752581.992.v2'

    dataset = tf.data.TFRecordDataset(f)

    ids, imgs = {}, {}
    for i, record in enumerate(tqdm(
            iter(dataset), total=len(list(dataset)), desc="Reading records")):
        event = event_pb2.Event.FromString(record.numpy())

        if i < 21:
            continue
        tf.io.decode_raw(
            event.summary.value[0].tensor.tensor_content, tf.float32)
        from IPython import embed; embed()

        for v in event.summary.value:
            virt_id = v.tag.split('_')[-1]

            if v.tag.startswith('image_test_virt'):
                if virt_id not in imgs:
                    imgs[virt_id] = []
                img_width = int(v.tensor.string_val[0].decode())
                img_height = int(v.tensor.string_val[1].decode())
                imgs[virt_id] += [
                    tf.io.decode_image(v.tensor.string_val[i]).numpy()
                    for i in range(2, len(v.tensor.string_val))]

            elif v.tag.startswith('id_test_virt'):
                if virt_id not in ids:
                    ids[virt_id] = []
                id_arr = tf.make_ndarray(v.tensor)
                ids[virt_id] += [id_arr[x, :] for x in range(id_arr.shape[0])]

            else:
                raise ValueError(v.tag)
    from IPython import embed; embed()

    for k, v in data.items():
        for x in v:
            from IPython import embed; embed()


if __name__ == '__main__':
    app.run(main)

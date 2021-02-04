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

def config2dict(config):
    """Assumes the configuration .ini has only the default section.
    """
    config_dict = {}
    for k, v in config.items('DEFAULT'):
        assert k not in config_dict, "Duplicate flags not allowed"
        config_dict[k] = v
    return config_dict


def get_config_ini(ckpt_path):
    return '/'.join(ckpt_path.split('/')[:-2]) + '.ini'

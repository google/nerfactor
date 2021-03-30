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

from absl import logging


logging.set_verbosity(logging.INFO)


class Logger():
    """Allows the use of (colored) prints, instead of proper logging, to avoid
    `--logtostderr` (which not only floods the console but also mutes colors).
    """
    start_str = {
        'red': '\x1b[31m',
        'green': '\x1b[32m',
        'cyan': '\x1b[36m',
        'pink': '\x1b[35m'}
    end_str = '\x1b[0m'

    def __init__(
            self, prefix="", suffix="", loggee=None, debug_mode=False,
            use_absl=False):
        self.prefix = prefix
        self.suffix = suffix
        if loggee is not None:
            self.prefix += "[%s] " % loggee
        self.debug_mode = debug_mode
        self.use_absl = use_absl

    def _format_color(self, txt, color):
        txt = self.start_str[color] + txt + self.end_str
        return txt

    def _format_content(self, *args):
        txt = args[0] % tuple(args[1:])
        txt = self.prefix + txt + self.suffix
        return txt

    def warn(self, *args, color='pink'):
        formatted = self._format_content(*args)
        if self.use_absl:
            logging.warning(formatted)
        else:
            formatted = self._format_color(formatted, color)
            print(formatted)

    def warning(self, *args, **kwargs):
        """Just an alias.
        """
        self.warn(*args, **kwargs)

    def error(self, *args, color='red'):
        formatted = self._format_content(*args)
        if self.use_absl:
            logging.error(formatted)
        else:
            formatted = self._format_color(formatted, color)
            print(formatted)

    def debug(self, *args, color='green'):
        formatted = self._format_content(*args)
        if self.use_absl:
            logging.debug(formatted)
        else:
            if self.debug_mode:
                formatted = self._format_color(formatted, color)
                print(formatted)

    def info(self, *args, color='cyan'):
        formatted = self._format_content(*args)
        if self.use_absl:
            logging.info(formatted)
        else:
            formatted = self._format_color(formatted, color)
            print(formatted)

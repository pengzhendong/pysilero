# Copyright (c) 2024, Zhendong Peng (pzd17@tsinghua.org.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from pathlib import Path

import onnxruntime as ort


class PickableSession:
    """
    This is a wrapper to make the current InferenceSession class pickable.
    """

    def __init__(self, version="v5"):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3

        assert version in ["v4", "v5"]
        model_path = Path(__file__).parent / version / "silero_vad.onnx"
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found, download it with:\n"
                f"curl -L https://modelscope.cn/models/pengzhendong/silero-vad/resolve/master/{version}/silero_vad.onnx"
                f" -o {model_path}"
            )
        self.model_path = str(model_path)
        self.init_session = partial(ort.InferenceSession, sess_options=opts, providers=["CPUExecutionProvider"])
        self.sess = self.init_session(self.model_path)

    def run(self, *args):
        return self.sess.run(None, *args)

    def __getstate__(self):
        return {"model_path": self.model_path}

    def __setstate__(self, values):
        self.model_path = values["model_path"]
        self.sess = self.init_session(self.model_path)


VERSIONS = ["v4", "v5"]


class LazySessions(dict):
    """Create InferenceSessions on first access, not at import time."""

    def __missing__(self, version):
        assert version in VERSIONS
        self[version] = session = PickableSession(version)
        return session


silero_vad = LazySessions()

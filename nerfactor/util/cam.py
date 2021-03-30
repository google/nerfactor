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

def resize_cam(cam, new_h, new_w):
    """Returns a copy of the camera with resized intrinsics.
    """
    scaled_cam = cam.Copy()
    scale_h = new_h / float(cam.ImageSizeY())
    scale_w = new_w / float(cam.ImageSizeX())
    scaled_cam.SetImageSize(new_w, new_h)
    scaled_cam.SetPrincipalPoint(
        scale_w * cam.PrincipalPointX(), scale_h * cam.PrincipalPointY())
    scaled_cam.SetFocalLength(scale_w * cam.FocalLength())
    scaled_cam.SetPixelAspectRatio(scale_h / scale_w * cam.PixelAspectRatio())
    scaled_cam.SetSkew(scale_w * cam.Skew())
    return scaled_cam

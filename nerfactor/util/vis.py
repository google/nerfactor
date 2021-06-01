from os.path import exists, join
import numpy as np

from nerfactor.util import logging as logutil, io as ioutil, img as imgutil
from third_party.xiuminglib import xiuminglib as xm

logger = logutil.Logger(loggee="util/vis")


def make_frame(
        view_dir, layout, put_text=True, put_text_param=None, data_root=None,
        rgb_embed_light=None):
    if put_text_param is None:
        put_text_param = {}
    if 'text_loc_ratio' not in put_text_param:
        put_text_param['text_loc_ratio'] = 0.05
    if 'text_size_ratio' not in put_text_param:
        put_text_param['text_size_ratio'] = 0.05
    if 'font_path' not in put_text_param:
        put_text_param['font_path'] = xm.const.Path.open_sans_regular

    layout = np.array(layout)
    if layout.ndim == 1:
        layout = np.reshape(layout, (1, -1))
    elif layout.ndim == 2:
        pass
    else:
        raise ValueError(layout.ndim)
    # Guaranteed to be 2D

    frame = []
    for row_names in layout:
        frame.append([])
        for name in row_names:
            is_render = name.startswith('rgb')
            is_nn = name == 'nn'

            # Get path
            if is_nn:
                assert data_root is not None, \
                    "When including NN, you must provide `data_root`"
                path = get_nearest_input(view_dir, data_root)
            else:
                path = join(view_dir, f'pred_{name}.png')

            if not exists(path):
                logger.warn("Skipping because of missing files:\n\t%s", path)
                return None

            img = xm.io.img.load(path)
            img = img[:, :, :3] # discards alpha
            hw = img.shape[:2]

            # Optionally, embed the light used into right top corner of render
            if is_render and rgb_embed_light is not None:
                light = rgb_embed_light
                frame_width = int(max(1 / 16 * light.shape[0], 1))
                imgutil.frame_image(light, rgb=(1, 1, 1), width=frame_width)
                light_vis_h = int(32 / 256 * hw[0]) # scale light probe size
                light = xm.img.resize(light, new_h=light_vis_h)
                img[:light.shape[0], -light.shape[1]:] = light
            # NN already has embedded light

            # Put label
            if put_text:
                font_color = (1, 1, 1) if is_render or is_nn else (0, 0, 0)
                put_text_kwargs = {
                    'label_top_left_xy': (
                        int(put_text_param['text_loc_ratio'] * hw[1]),
                        int(put_text_param['text_loc_ratio'] * hw[0])),
                    'font_size': int(
                        put_text_param['text_size_ratio'] * hw[0]),
                    'font_color': font_color,
                    'font_ttf': put_text_param['font_path']}
                if is_nn:
                    label = "Nearest Input"
                elif is_render:
                    label = "Rendering"
                elif name in ('normal', 'normals'):
                    label = "Normals"
                elif name == 'lvis':
                    label = "Visibility (mean)"
                elif name.startswith('lvis_olat_'):
                    label = "Visibility"
                elif name == 'brdf':
                    label = "BRDF"
                elif name == 'albedo':
                    label = "Albedo"
                else:
                    raise NotImplementedError(name)
                img = xm.vis.text.put_text(img, label, **put_text_kwargs)

            frame[-1].append(img)

    # Make collage
    rows = []
    for row in frame:
        try:
            rows.append(imgutil.hconcat(row))
        except:
            from IPython import embed; embed()
    frame = imgutil.vconcat(rows)

    return frame


def get_nearest_input(view_dir, data_root):
    metadata_path = join(view_dir, 'metadata.json')
    metadata = ioutil.read_json(metadata_path)
    id_ = metadata['id']
    nearest_input_path = join(data_root, id_, 'nn.png')
    return nearest_input_path

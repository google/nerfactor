from os import environ
from os.path import abspath, join, dirname


class Dir:
    tmp = environ.get('TMP_DIR', '/tmp/')

    mstatus = join(
        environ.get('MSTATUS_BACKEND_DIR', '/tmp/machine-status'), 'runtime')

    data = abspath(join(dirname(__file__), '..', 'data'))


class Path:
    cpustatus = environ.get('CPU_STATUS_FILE', '/tmp/cpu/machine_status.txt')
    gpustatus = environ.get('GPU_STATUS_FILE', '/tmp/gpu/{machine_name}')

    # Textures
    checker = join(Dir.data, 'textures', 'checker.png')

    # Images
    cameraman = join(Dir.data, 'images', 'cameraman.png')
    lenna = join(Dir.data, 'images', 'lenna.png')

    # 3D Models
    armadillo = join(Dir.data, 'models', 'armadillo.ply')
    buddha_prefix = join(Dir.data, 'models', 'buddha', 'happy_vrip')
    buddha = buddha_prefix + '.ply'
    buddha_res2 = buddha_prefix + '_res2.ply'
    buddha_res3 = buddha_prefix + '_res3.ply'
    buddha_res4 = buddha_prefix + '_res4.ply'
    bunny_prefix = join(Dir.data, 'models', 'bunny', 'bun_zipper')
    bunny = bunny_prefix + '.ply'
    bunny_res2 = bunny_prefix + '_res2.ply'
    bunny_res3 = bunny_prefix + '_res3.ply'
    bunny_res4 = bunny_prefix + '_res4.ply'
    dragon_prefix = join(Dir.data, 'models', 'dragon', 'dragon_vrip')
    dragon = dragon_prefix + '.ply'
    dragon_res2 = dragon_prefix + '_res2.ply'
    dragon_res3 = dragon_prefix + '_res3.ply'
    dragon_res4 = dragon_prefix + '_res4.ply'
    teapot = join(Dir.data, 'models', 'teapot.obj')

    # LPIPS
    lpips_weights = join(Dir.data, 'lpips', 'net-lin_alex_v0.1.pb')

    # Fonts
    open_sans_regular = join(
        Dir.data, 'fonts', 'open-sans', 'OpenSans-Regular.ttf')

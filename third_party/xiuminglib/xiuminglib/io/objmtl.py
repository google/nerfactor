# pylint: disable=len-as-condition

from os.path import basename, dirname, join
from shutil import copy
import numpy as np

from .. import os as xm_os
from ..imprt import preset_import

from ..log import get_logger
logger = get_logger()


class Obj:
    """Wavefront .obj Object.

    Face, vertex, or other indices here all start from 1.

    Attributes:
        o (str)
        v (numpy.ndarray)
        f (list)
        vn (numpy.ndarray)
        fn (list)
        vt (numpy.ndarray)
        ft (list)
        s (bool)
        mtllib (str)
        usemtl (str)
        diffuse_map_path (str)
        diffuse_map_scale (float)
    """
    def __init__(
            self, o=None, v=None, f=None, vn=None, fn=None, vt=None, ft=None,
            s=False, mtllib=None, usemtl=None, diffuse_map_path=None,
            diffuse_map_scale=1):
        """
        Args:
            o (str, optional): Object name.
            v (numpy.ndarray, optional): Vertex coordinates.
            f (list, optional): Faces' vertex indices (1-indexed), e.g.,
                ``[[1, 2, 3], [4, 5, 6], [7, 8, 9, 10], ...]``.
            vn (numpy.ndarray, optional): Vertex normals of shape N-by-3,
                normalized or not.
            fn (list, optional): Faces' vertex normal indices, e.g.,
                ``[[1, 1, 1], [], [2, 2, 2, 2], ...]``. Must be of the same
                length as ``f``.
            vt (numpy.ndarray, optional): Vertex texture coordinates of shape
                N-by-2. Coordinates must be normalized to :math:`[0, 1]`.
            ft (list, optional): Faces' texture vertex indices, e.g.,
                ``[[1, 2, 3], [4, 5, 6], [], ...]``. Must be of the same length
                as ``f``.
            s (bool, optional): Group smoothing.
            mtllib (str, optional): Material file name, e.g., ``'cube.mtl'``.
            usemtl (str, optional): Material name (defined in .mtl file).
            diffuse_map_path (str, optional): Path to diffuse texture map.
            diffuse_map_scale (float, optional): Scale of diffuse texture map.
        """
        self.mtllib = mtllib
        self.o = o
        # Vertices
        if v is not None:
            assert (len(v.shape) == 2 and v.shape[1] == 3), "'v' must be *-by-3"
        if vt is not None:
            assert (len(vt.shape) == 2 and vt.shape[1] == 2), \
                "'vt' must be *-by-2"
        if vn is not None:
            assert (len(vn.shape) == 2 and vn.shape[1] == 3), \
                "'vn' must be *-by-3"
        self.v = v
        self.vt = vt
        self.vn = vn
        # Faces
        if f is not None:
            if ft is not None:
                assert (len(ft) == len(f)), \
                    "'ft' must be of the same length as 'f' (use '[]' to fill)"
            if fn is not None:
                assert (len(fn) == len(f)), \
                    "'fn' must be of the same length as 'f' (use '[]' to fill)"
        self.f = f
        self.ft = ft
        self.fn = fn
        self.usemtl = usemtl
        self.s = s
        self.diffuse_map_path = diffuse_map_path
        self.diffuse_map_scale = diffuse_map_scale

    def load_file(self, obj_file):
        """Loads a (basic) .obj file as an object.

        Populates attributes with contents read from file.

        Args:
            obj_file (str): Path to .obj file.
        """
        fid = open(obj_file, 'r')
        lines = [l.strip('\n') for l in fid.readlines()]
        lines = [l for l in lines if len(l) > 0] # remove empty lines
        # Check if there's only one object
        n_o = len([l for l in lines if l[0] == 'o'])
        if n_o > 1:
            raise ValueError((
                ".obj file containing multiple objects is not supported "
                "-- consider using ``assimp`` instead"))
        # Count for array initializations
        n_v = len([l for l in lines if l[:2] == 'v '])
        n_vt = len([l for l in lines if l[:3] == 'vt '])
        n_vn = len([l for l in lines if l[:3] == 'vn '])
        lines_f = [l for l in lines if l[:2] == 'f ']
        n_f = len(lines_f)
        # Initialize arrays
        mtllib = None
        o = None
        v = np.zeros((n_v, 3))
        vt = np.zeros((n_vt, 2))
        vn = np.zeros((n_vn, 3))
        usemtl = None
        s = False
        f = [None] * n_f
        # If there's no 'ft' or 'fn' for a 'f', a '[]' is inserted as a
        # placeholder. This guarantees 'f[i]' always corresponds to 'ft[i]'
        # and 'fn[i]'
        ft = [None] * n_f
        fn = [None] * n_f
        # Load data line by line
        n_ft, n_fn = 0, 0
        i_v, i_vt, i_vn, i_f = 0, 0, 0, 0
        for l in lines:
            if l[0] == '#': # comment
                pass
            elif l[:7] == 'mtllib ': # mtl file
                mtllib = l[7:]
            elif l[:2] == 'o ': # object name
                o = l[2:]
            elif l[:2] == 'v ': # geometric vertex
                v[i_v, :] = [float(x) for x in l[2:].split(' ')]
                i_v += 1
            elif l[:3] == 'vt ': # texture vertex
                vt[i_vt, :] = [float(x) for x in l[3:].split(' ')]
                i_vt += 1
            elif l[:3] == 'vn ': # normal vector
                vn[i_vn, :] = [float(x) for x in l[3:].split(' ')]
                i_vn += 1
            elif l[:7] == 'usemtl ': # material name
                usemtl = l[7:]
            elif l[:2] == 's ': # group smoothing
                if l[2:] == 'on':
                    s = True
            elif l[:2] == 'f ': # face
                n_slashes = l[2:].split(' ')[0].count('/')
                if n_slashes == 0: # just f (1 2 3)
                    f[i_f] = [int(x) for x in l[2:].split(' ')]
                    ft[i_f] = []
                    fn[i_f] = []
                elif n_slashes == 1: # f and ft (1/1 2/2 3/3)
                    f[i_f] = [int(x.split('/')[0]) for x in l[2:].split(' ')]
                    ft[i_f] = [int(x.split('/')[1]) for x in l[2:].split(' ')]
                    fn[i_f] = []
                    n_ft += 1
                elif n_slashes == 2:
                    if l[2:].split(' ')[0].count('//') == 1:
                        # f and fn (1//1 2//1 3//1)
                        f[i_f] = [
                            int(x.split('//')[0]) for x in l[2:].split(' ')]
                        ft[i_f] = []
                        fn[i_f] = [
                            int(x.split('//')[1]) for x in l[2:].split(' ')]
                        n_fn += 1
                    else:
                        # f, ft and fn (1/1/1 2/2/1 3/3/1)
                        f[i_f] = [
                            int(x.split('/')[0]) for x in l[2:].split(' ')]
                        ft[i_f] = [
                            int(x.split('/')[1]) for x in l[2:].split(' ')]
                        fn[i_f] = [
                            int(x.split('/')[2]) for x in l[2:].split(' ')]
                        n_ft += 1
                        n_fn += 1
                i_f += 1
            else:
                raise ValueError("Unidentified line type: %s" % l)
        # Update self
        self.mtllib = mtllib
        self.o = o
        self.v = v
        self.vt = vt if vt.shape[0] > 0 else None
        self.vn = vn if vn.shape[0] > 0 else None
        self.f = f
        self.ft = ft if any(ft) else None # any member list not empty
        self.fn = fn if any(fn) else None
        self.usemtl = usemtl
        self.s = s

    # Print model info
    def print_info(self):
        # Basic stats
        mtllib = self.mtllib
        o = self.o
        n_v = self.v.shape[0] if self.v is not None else 0
        n_vt = self.vt.shape[0] if self.vt is not None else 0
        n_vn = self.vn.shape[0] if self.vn is not None else 0
        usemtl = self.usemtl
        s = self.s
        diffuse_map_path = self.diffuse_map_path
        diffuse_map_scale = self.diffuse_map_scale
        n_f = len(self.f) if self.f is not None else 0
        if self.ft is not None:
            n_ft = sum(len(x) > 0 for x in self.ft)
        else:
            n_ft = 0
        if self.fn is not None:
            n_fn = sum(len(x) > 0 for x in self.fn)
        else:
            n_fn = 0
        logger.info("-------------------------------------------------------")
        logger.info("Object name            'o'            %s", o)
        logger.info("Material file          'mtllib'       %s", mtllib)
        logger.info("Material               'usemtl'       %s", usemtl)
        logger.info("Diffuse texture map    'map_Kd'       %s", diffuse_map_path)
        logger.info("Diffuse map scale                     %f", diffuse_map_scale)
        logger.info("Group smoothing        's'            %r", s)
        logger.info("# geometric vertices   'v'            %d", n_v)
        logger.info("# texture vertices     'vt'           %d", n_vt)
        logger.info("# normal vectors       'vn'           %d", n_vn)
        logger.info("# geometric faces      'f x/o/o'      %d", n_f)
        logger.info("# texture faces        'f o/x/o'      %d", n_ft)
        logger.info("# normal faces         'f o/o/x'      %d", n_fn)
        # How many triangles, quads, etc.
        if n_f > 0:
            logger.info("")
            logger.info("Among %d faces:", n_f)
            vert_counts = [len(x) for x in self.f]
            for c in np.unique(vert_counts):
                howmany = vert_counts.count(c)
                logger.info("  - %d are formed by %d vertices", howmany, c)
        logger.info("-------------------------------------------------------")

    # Set vn and fn according to v and f
    def set_face_normals(self):
        """Sets face normals according to geometric vertices and their orders
        in forming faces.

        Returns:
            tuple:
                - **vn** (*numpy.ndarray*) -- Normal vectors.
                - **fn** (*list*) -- Normal faces. Each member list consists of
                  the same integer, e.g., ``[[1, 1, 1], [2, 2, 2, 2], ...]``.
        """
        n_f = len(self.f)
        vn = np.zeros((n_f, 3))
        fn = [None] * n_f
        # For each face
        for i, verts_id in enumerate(self.f):
            # Vertices must be coplanar to be valid, so we can just pick the
            # first three
            ind = [x - 1 for x in verts_id[:3]] # in .obj, index starts from 1,
            # not 0
            verts = self.v[ind, :]
            p1p2 = verts[1, :] - verts[0, :]
            p1p3 = verts[2, :] - verts[0, :]
            normal = np.cross(p1p2, p1p3)
            if np.linalg.norm(normal) == 0:
                raise ValueError((
                    "Normal vector of zero length probably due to numerical "
                    "issues?"))
            vn[i, :] = normal / np.linalg.norm(normal) # normalize
            fn[i] = [i + 1] * len(verts_id)
        # Set normals and return
        self.vn = vn
        self.fn = fn
        logger.info((
            "Face normals recalculated with 'v' and 'f' -- 'vn' and 'fn' "
            "updated"))
        return vn, fn

    # Output object to file
    def write_file(self, objpath):
        """Writes the current model to a .obj file.

        Args:
            objpath (str): Path to the output .obj.

        Writes
            - Output .obj file.
        """
        mtllib = self.mtllib
        o = self.o
        v, vt, vn = self.v, self.vt, self.vn
        usemtl = self.usemtl
        s = self.s
        f, ft, fn = self.f, self.ft, self.fn
        # mkdir if necessary
        outdir = dirname(objpath)
        xm_os.makedirs(outdir)
        # Write .obj
        with open(objpath, 'w') as fid:
            # Material file
            if mtllib is not None:
                fid.write('mtllib %s\n' % mtllib)
            # Object name
            fid.write('o %s\n' % o)
            # Vertices
            for i in range(v.shape[0]):
                fid.write('v %f %f %f\n' % tuple(v[i, :]))
            if vt is not None:
                for i in range(vt.shape[0]):
                    fid.write('vt %f %f\n' % tuple(vt[i, :]))
            if vn is not None:
                for i in range(vn.shape[0]):
                    fid.write('vn %f %f %f\n' % tuple(vn[i, :]))
            # Material name
            if usemtl is not None:
                fid.write('usemtl %s\n' % usemtl)
            # Group smoothing
            if s:
                fid.write('s on\n')
            else:
                fid.write('s off\n')
            # Faces
            if ft is None and fn is None: # just f (1 2 3)
                for v_id in f:
                    fid.write(('f' + ' %d' * len(v_id) + '\n') % tuple(v_id))
            elif ft is not None and fn is None:
                # f and ft (1/1 2/2 3/3 or 1 2 3)
                for i, v_id in enumerate(f):
                    vt_id = ft[i]
                    if len(vt_id) == len(v_id):
                        fid.write((
                            'f' + ' %d/%d' * len(v_id) + '\n') % tuple(
                                [x for pair in zip(v_id, vt_id) for x in pair]))
                    elif not vt_id:
                        fid.write(
                            ('f' + ' %d' * len(v_id) + '\n') % tuple(v_id))
                    else:
                        raise ValueError((
                            "'ft[%d]', not empty, doesn't match length of "
                            "'f[%d]'") % (i, i))
            elif ft is None and fn is not None:
                # f and fn (1//1 2//1 3//1 or 1 2 3)
                for i, v_id in enumerate(f):
                    vn_id = fn[i]
                    if len(vn_id) == len(v_id):
                        fid.write((
                            'f' + ' %d//%d' * len(v_id) + '\n') % tuple(
                                [x for pair in zip(v_id, vn_id) for x in pair]))
                    elif not vn_id:
                        fid.write(
                            ('f' + ' %d' * len(v_id) + '\n') % tuple(v_id))
                    else:
                        raise ValueError((
                            "'fn[%d]', not empty, doesn't match length of "
                            "'f[%d]'") % (i, i))
            elif ft is not None and fn is not None:
                # f, ft and fn (1/1/1 2/2/1 3/3/1 or 1/1 2/2 3/3 or
                # 1//1 2//1 3//1 or 1 2 3)
                for i, v_id in enumerate(f):
                    vt_id = ft[i]
                    vn_id = fn[i]
                    if len(vt_id) == len(v_id) and len(vn_id) == len(v_id):
                        fid.write((
                            'f' + ' %d/%d/%d' * len(v_id) + '\n') % tuple(
                                [x for triple in zip(v_id, vt_id, vn_id)
                                 for x in triple]))
                    elif len(vt_id) == len(v_id) and not vn_id:
                        fid.write((
                            'f' + ' %d/%d' * len(v_id) + '\n') % tuple(
                                [x for pair in zip(v_id, vt_id) for x in pair]))
                    elif not vt_id and len(vn_id) == len(v_id):
                        fid.write((
                            'f' + ' %d//%d' * len(v_id) + '\n') % tuple(
                                [x for pair in zip(v_id, vn_id) for x in pair]))
                    elif not vt_id and not vn_id:
                        fid.write(
                            ('f' + ' %d' * len(v_id) + '\n') % tuple(v_id))
                    else:
                        raise ValueError((
                            "If not empty, 'ft[%d]' or 'fn[%d]' doesn't match "
                            "length of 'f[%d]'") % (i, i, i))
        logger.info("Done writing to %s", objpath)


class Mtl:
    r"""Wavefront .mtl object.

    Attributes:
        mtlfile (str): Material file name, set to ``obj.mtllib``.
        newmtl (str): Material name, set to ``obj.usemtl``.
        map_Kd_path (str): Path to the diffuse map, set to
            ``obj.diffuse_map_path``.
        map_Kd_scale (float): Scale of the diffuse map, set to
            ``obj.diffuse_map_scale``.
        Ns (float)
        Ka (tuple)
        Kd (tuple)
        Ks (tuple)
        Ni (float)
        d (float)
        illum (int)
    """
    def __init__(
            self, obj, Ns=96.078431, Ka=(1, 1, 1), Kd=(0.64, 0.64, 0.64),
            Ks=(0.5, 0.5, 0.5), Ni=1, d=1, illum=2):
        r"""
        Args:
            obj (Obj): ``Obj`` object for which this ``Mtl`` object is created.
            Ns (float, optional): Specular exponent, normally
                :math:`\in[0, 1000]`.
            Ka (tuple, optional): Ambient reflectivity, each float normally
                :math:`\in[0, 1]`. Values outside increase or decrease
                relectivity accordingly.
            Kd (tuple, optional): Diffuse reflectivity. Same range as ``Ka``.
            Ks (tuple, optional): Specular reflectivity. Same range as ``Ka``.
            Ni (float, optional): Optical density, a.k.a. index of refraction
                :math:`\in[0.001, 10]`. 1 means light doesn't bend as it passes
                through. Increasing it increases the amount of bending. Glass
                has an index of refraction of about 1.5. Values of less than 1.0
                produce bizarre results and are not recommended.
            d (float, optional): Amount this material dissolves into the
                background :math:`\in[0, 1]`. 1.0 is fully opaque (default),
                and 0 is fully dissolved (completely transparent). Unlike a real
                transparent material, the dissolve does not depend upon material
                thickness, nor does it have any spectral character. Dissolve
                works on all illumination models.
            illum (int, optional): Illumination model
                :math:`\in[0, 1, ..., 10]`.
        """
        self.mtlfile = obj.mtllib
        self.newmtl = obj.usemtl
        self.map_Kd_path = obj.diffuse_map_path
        self.map_Kd_scale = obj.diffuse_map_scale
        self.Ns = Ns
        self.Ka = Ka
        self.Kd = Kd
        self.Ks = Ks
        self.Ni = Ni
        self.d = d
        self.illum = illum

    def print_info(self):
        logger.info("---------------------------------------------------------")
        logger.info("Material file                          %s", self.mtlfile)
        logger.info("Material name           'newmtl'       %s", self.newmtl)
        logger.info("Diffuse texture map     'map_Kd'       %s", self.map_Kd_path)
        logger.info("Diffuse map scale                      %f", self.map_Kd_scale)
        logger.info("Specular exponent       'Ns'           %f", self.Ns)
        logger.info("Ambient reflectivity    'Ka'           %s", self.Ka)
        logger.info("Diffuse reflectivity    'Kd'           %s", self.Kd)
        logger.info("Specular reflectivity   'Ks'           %s", self.Ks)
        logger.info("Refraction index        'Ni'           %s", self.Ni)
        logger.info("Dissolve                'd'            %f", self.d)
        logger.info("Illumination model      'illum'        %d", self.illum)
        logger.info("---------------------------------------------------------")

    def write_file(self, outdir):
        """Unit tests that can also serve as example usage.

        Args:
            outdir (str): Output directory.

        Writes
            - Output .mtl file.
        """
        cv2 = preset_import('cv2', assert_success=True)
        # Validate inputs
        assert (self.mtlfile is not None and self.newmtl is not None), \
            "'mtlfile' and 'newmtl' must not be 'None'"
        # mkdir if necessary
        xm_os.makedirs(outdir)
        # Write .mtl
        mtlpath = join(outdir, self.mtlfile)
        with open(mtlpath, 'w') as fid:
            fid.write('newmtl %s\n' % self.newmtl)
            fid.write('Ns %f\n' % self.Ns)
            fid.write('Ka %f %f %f\n' % self.Ka)
            fid.write('Kd %f %f %f\n' % self.Kd)
            fid.write('Ks %f %f %f\n' % self.Ks)
            fid.write('Ni %f\n' % self.Ni)
            fid.write('d %f\n' % self.d)
            fid.write('illum %d\n' % self.illum)
            map_Kd_path = self.map_Kd_path
            map_Kd_scale = self.map_Kd_scale
            if map_Kd_path is not None:
                fid.write('map_Kd %s\n' % basename(map_Kd_path))
                if map_Kd_scale == 1:
                    copy(map_Kd_path, outdir)
                else:
                    im = cv2.imread(map_Kd_path, cv2.IMREAD_UNCHANGED) # TODO: switch to xm.io.img
                    im = cv2.resize(im, None, fx=map_Kd_scale, fy=map_Kd_scale) # TODO: switch to xm.img
                    cv2.imwrite(join(outdir, basename(map_Kd_path)), im) # TODO: switch to xm.io.img
        logger.info("Done writing to %s", mtlpath)


def main():
    """Unit tests that can also serve as example usage."""
    objf = '../../../toy-data/obj-mtl_cube/cube.obj'
    myobj = Obj()
    myobj.print_info()
    myobj.load_file(objf)
    myobj.print_info()
    objf_reproduce = objf.replace('.obj', '_reproduce.obj')
    myobj.write_file(objf_reproduce)
    myobj.set_face_normals()
    myobj.print_info()


if __name__ == '__main__':
    main()

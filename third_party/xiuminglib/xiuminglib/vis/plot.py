# pylint: disable=blacklisted-name

from os.path import join, dirname
import numpy as np

from ..log import get_logger
logger = get_logger()

from .. import const
from ..os import makedirs, open_file


class Plot:
    def __init__(
            self,
            legend_fontsize=20,
            legend_loc=0,
            figsize=(14, 14),
            figtitle=None,
            figtitle_fontsize=20,
            xlabel=None,
            xlabel_fontsize=20,
            ylabel=None,
            ylabel_fontsize=20,
            zlabel=None,
            zlabel_fontsize=20,
            xlim=None,
            ylim=None,
            zlim=None,
            xticks=None,
            xticks_fontsize=10,
            xticks_rotation=0,
            yticks=None,
            yticks_fontsize=10,
            yticks_rotation=0,
            zticks=None,
            zticks_fontsize=10,
            zticks_rotation=0,
            grid=True,
            labels=None,
            outpath=None):
        """Plotter.

        Args:
            legend_fontsize (int, optional): Legend font size.
            legend_loc (str, optional): Legend location: ``'best'``,
                ``'upper right'``, ``'lower left'``, ``'right'``,
                ``'center left'``, ``'lower center'``, ``'upper center'``,
                ``'center'``, etc. Effective only when ``labels`` is not
                ``None``.
            figsize (tuple, optional): Width and height of the figure in inches.
            figtitle (str, optional): Figure title.
            *_fontsize (int, optional): Font size.
            ?label (str, optional): Axis labels.
            ?lim (array_like, optional): Axis min. and max. ``None`` means auto.
            ?ticks (array_like, optional): Axis tick values. ``None`` means
                auto.
            ?ticks_rotation (float, optional): Tick rotation in degrees.
            grid (bool, optional): Whether to draw grid.
            labels (list, optional): Labels.
            outpath (str, optional): Path to which the plot is saved to. Should
                end with ``'.png'``, and ``None`` means to ``const.Dir.tmp``.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        #
        self.plt = plt
        self.legend_fontsize = legend_fontsize
        self.legend_loc = legend_loc
        self.figsize = figsize
        self.figtitle = figtitle
        self.figtitle_fontsize = figtitle_fontsize
        self.xlabel = xlabel
        self.xlabel_fontsize = xlabel_fontsize
        self.ylabel = ylabel
        self.ylabel_fontsize = ylabel_fontsize
        self.zlabel = zlabel
        self.zlabel_fontsize = zlabel_fontsize
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.xticks = xticks
        self.xticks_rotation = xticks_rotation
        self.xticks_fontsize = xticks_fontsize
        self.yticks = yticks
        self.yticks_rotation = yticks_rotation
        self.yticks_fontsize = yticks_fontsize
        self.zticks = zticks
        self.zticks_rotation = zticks_rotation
        self.zticks_fontsize = zticks_fontsize
        self.grid = grid
        self.labels = labels
        self.outpath = outpath

    def _savefig(self, outpath, contents_only=False, dpi=None):
        # Make directory, if necessary
        outdir = dirname(outpath)
        makedirs(outdir)
        #
        if contents_only:
            ax = self.plt.gca()
            ax.set_position([0, 0, 1, 1])
            ax.set_axis_off()
            with open_file(outpath, 'wb') as h:
                self.plt.savefig(h, dpi=dpi)
        else:
            with open_file(outpath, 'wb') as h:
                self.plt.savefig(h, bbox_inches='tight', dpi=dpi)

    def _add_legend(self, plot_objs):
        if self.labels is None:
            return
        n_plot_objs = len(plot_objs)
        assert (len(self.labels) == n_plot_objs), (
            "Number of labels must equal number of plot objects; "
            "use None for object without a label")
        for i in range(n_plot_objs):
            plot_objs[i].set_label(self.labels[i])
        self.plt.legend(fontsize=self.legend_fontsize, loc=self.legend_loc)

    def _add_axis_labels(self, ax):
        if self.xlabel is not None:
            ax.set_xlabel(self.xlabel, fontsize=self.xlabel_fontsize)
        if self.ylabel is not None:
            ax.set_ylabel(self.ylabel, fontsize=self.ylabel_fontsize)
        if self.zlabel is not None:
            ax.set_zlabel(self.zlabel, fontsize=self.zlabel_fontsize)

    def _set_axis_ticks(self, ax):
        # FIXME: if xticks is not provided, xticks_fontsize and xticks_rotation have
        # no effect, which shouldn't be the case
        if self.xticks is not None:
            ax.set_xticklabels(
                self.xticks, fontsize=self.xticks_fontsize,
                rotation=self.xticks_rotation)
        if self.yticks is not None:
            ax.set_yticklabels(
                self.yticks, fontsize=self.yticks_fontsize,
                rotation=self.yticks_rotation)
        if self.zticks is not None:
            ax.set_zticklabels(
                self.zticks, fontsize=self.zticks_fontsize,
                rotation=self.zticks_rotation)

    def _set_axis_lim(self, ax):
        if self.xlim is not None:
            ax.set_xlim(*self.xlim)
        if self.ylim is not None:
            ax.set_ylim(*self.ylim)
        if self.zlim is not None:
            ax.set_zlim(*self.zlim)

    @staticmethod
    def _set_axes_equal(ax, xyz):
        # plt.axis('equal') not working, hence the hack of creating a cubic
        # bounding box
        x_data, y_data, z_data = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        max_range = np.array([
            x_data.max() - x_data.min(),
            y_data.max() - y_data.min(),
            z_data.max() - z_data.min()]).max()
        xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() \
            + 0.5 * (x_data.max() + x_data.min())
        yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() \
            + 0.5 * (y_data.max() + y_data.min())
        zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() \
            + 0.5 * (z_data.max() + z_data.min())
        for xb_, yb_, zb_ in zip(xb, yb, zb):
            ax.plot([xb_], [yb_], [zb_], 'w')

    def _set_title(self, ax):
        if self.figtitle is not None:
            ax.set_title(self.figtitle, fontsize=self.figtitle_fontsize)

    def bar(self, y, group_width=0.8):
        """Bar plot.

        Args:
            y (array_like): N-by-M array of N groups, each with M bars,
                or N-array of N groups, each with one bar.
            group_width (float, optional): Width allocated to each group,
                shared by all bars within the group.

        Writes
            - The bar plot.
        """
        outpath = join(const.Dir.tmp, 'bar.png') if self.outpath is None \
            else self.outpath
        fig = self.plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111)
        self._set_title(ax)
        # Ensure y is 2D, with columns representing values within groups
        # and rows across groups
        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))
        n, n_grp = y.shape
        # Group width is shared by all groups
        bar_width = group_width / n_grp
        # Assume x is evenly spaced
        x = np.arange(n)
        # Plot
        plot_objs = []
        for i in range(n_grp):
            x_ = x - 0.5 * group_width + 0.5 * bar_width + i * bar_width
            plot_obj = ax.bar(x_, y[:, i], bar_width)
            plot_objs.append(plot_obj)
        #
        self._add_legend(plot_objs)
        self.plt.grid(self.grid)
        self._add_axis_labels(ax)
        self._set_axis_ticks(ax)
        self._set_axis_lim(ax)
        self._savefig(outpath)
        self.plt.close('all')
        return outpath

    def scatter3d(
            self, xyz, colors=None, size=None, equal_axes=False, views=None):
        """3D scatter plot.

        Args:
            xyz (array_like): N-by-3 array of N points.
            colors (array_like or list(str) or str, optional): If N-array, these
                values are colormapped. If N-list, its elements should be color
                strings. If a single color string, all points use that color.
            size (int, optional): Scatter size.
            equal_axes (bool, optional): Whether to have the same scale for all
                axes.
            views (list(tuple), optional): List of elevation-azimuth angle pairs
                (in degrees). A good set of views is ``[(30, 0), (30, 45),
                (30, 90), (30, 135)]``.

        Writes
            - One or multiple (if ``views`` is provided) views of the 3D plot.
        """
        from mpl_toolkits.mplot3d import Axes3D # noqa; pylint: disable=unused-import
        #
        outpath = join(const.Dir.tmp, 'scatter3d.png') if self.outpath is None \
            else self.outpath
        fig = self.plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        self._set_title(ax)
        # Prepare kwargs to scatter()
        kwargs = {}
        need_colorbar = False
        if isinstance(colors, np.ndarray):
            kwargs['c'] = colors # will be colormapped with color map
            kwargs['cmap'] = 'viridis'
            need_colorbar = True
        elif colors is not None:
            kwargs['c'] = colors
        if size is not None:
            kwargs['s'] = size
        # Plot
        plot_objs = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], **kwargs)
        #
        self._add_legend(plot_objs)
        self.plt.grid(self.grid)
        self._add_axis_labels(ax)
        self._set_axis_ticks(ax)
        self._set_axis_lim(ax)
        if equal_axes:
            self._set_axes_equal(ax, xyz)
        if need_colorbar:
            self.plt.colorbar(plot_objs)
            # TODO: this seems to mess up equal axes
        # Save plot
        outpaths = []
        if outpath.endswith('.png'):
            if views is None:
                self._savefig(outpath)
                outpaths.append(outpath)
            else:
                for elev, azim in views:
                    ax.view_init(elev, azim)
                    self.plt.draw()
                    outpath_ = outpath[:-len('.png')] + \
                        '_elev%03d_azim%03d.png' % (elev, azim)
                    self._savefig(outpath_)
                    outpaths.append(outpath_)
        else:
            raise ValueError("`outpath` must end with '.png'")
        self.plt.close('all')
        return outpaths

    def line(self, xy, width=None, marker=None, marker_size=None):
        """Line/curve plot.

        Args:
            xy (array_like): N-by-M array of N x-values (first column) and
                their corresponding y-values (the remaining M-1 columns).
            width (float, optional): Line width.
            marker (str, optional): Marker.
            marker_size (float, optional): Marker size.

        Writes
            - The line plot.
        """
        outpath = join(const.Dir.tmp, 'line.png') if self.outpath is None \
            else self.outpath
        fig = self.plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111)
        self._set_title(ax)
        # Prepare kwargs to scatter()
        kwargs_list = []
        n_lines = xy.shape[1] - 1
        for i in range(n_lines):
            kwargs = {}
            if width is not None:
                kwargs['linewidth'] = width
            if marker is not None:
                kwargs['marker'] = marker
            if marker_size is not None:
                kwargs['markersize'] = marker_size
            kwargs_list.append(kwargs)
        # Plot
        plot_objs = []
        for i in range(n_lines):
            plot_obj = self.plt.plot(xy[:, 0], xy[:, 1 + i], **kwargs_list[i])
            assert len(plot_obj) == 1
            plot_obj = plot_obj[0]
            plot_objs.append(plot_obj)
        #
        self._add_legend(plot_objs)
        self.plt.grid(self.grid)
        self._add_axis_labels(ax)
        self._set_axis_ticks(ax)
        self._set_axis_lim(ax)
        self._savefig(outpath)
        self.plt.close('all')
        return outpath

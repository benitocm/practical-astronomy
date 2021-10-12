
"""
This module contains functions related to orbit plotting
"""
# Standard library imports

# Third party imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation
from toolz import concat

# Local application imports


class OrbitsPlot:

    def __init__(self,
                 orbs_data,
                 date_refs,
                 fig,
                 axes_limits) :

        self.orbs = orbs_data
        self.date_refs = date_refs
        self.fig = fig
        self.ax = fig.add_axes([0, 0, 1, 1], projection='3d')

        n_trajectories = len(self.orbs)

        # choose a different color for each trajectory
        colors = plt.cm.jet(np.linspace(0, 1, n_trajectories))

        # lines and points initializaiton
        lines = []
        pts = []
        for i, (name, mtx) in enumerate(self.orbs.items()):
            lines.append(self.ax.plot([], [], [], '--', c=colors[i], label=name,lw=.7))
            pts.append(self.ax.plot([], [], [], 'o', c=colors[i]))
        self.lines = list(concat(lines))
        self.pts = list(concat(pts))

        # prepare the axes limits
        self.ax.set_xlim(axes_limits)
        self.ax.set_ylim(axes_limits)
        self.ax.set_zlim(axes_limits)

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        # SUN
        self.ax.scatter3D(0,0,0, color='yellow', marker='o', lw=8, label='Sun')

        #   set the legend
        self.ax.legend(loc='upper right', prop={'size': 9})
        #ax.set_title("Tim-Sitze, Orbits of the Inner Planets")
        #animation.writer = animation.writers['ffmpeg']

        axtext = self.fig.add_axes([0.0,0.95,0.1,0.05])
        # turn the axis labels/spines/ticks off
        axtext.axis("off")

        self.time_obj = axtext.text(0.5,0.5, date_refs[0], ha="left", va="top")

    # initialization function: plot the background of each frame
    def init(self):
        for line, pt in zip(self.lines, self.pts):
            line.set_data([], [])
            line.set_3d_properties([])

            pt.set_data([], [])
            pt.set_3d_properties([])
        return self.lines + self.pts

    def animate(self, i):
        for line, pt, mtx in zip(self.lines, self.pts, self.orbs.values()):
            xs = mtx[0:i,0]        
            ys = mtx[0:i,1]
            zs = mtx[0:i,2]
            line.set_data(xs, ys)
            line.set_3d_properties(zs)
            
            x = xs[-1:]
            y = ys[-1:]
            z = zs[-1:]        
            pt.set_data(x, y)
            pt.set_3d_properties(z)
            
            self.time_obj.set_text(self.date_refs[i])

        #ax.view_init(30, 0.3 * i)
        self.fig.canvas.draw()
        return self.lines + self.pts
    

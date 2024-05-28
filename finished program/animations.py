# -*- coding: utf-8 -*-

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

'''Colored Pixelgrid animation of the evolving system'''
def color(X, X2, T):
    fig2, ax = plt.subplots(1, 3, figsize=(11, 5), gridspec_kw={'width_ratios': [20, 1, 20]})
    ax1, ax2, ax3 = ax[0], ax[1], ax[2]
    
    ax1.text(0, -0.5, "$<\sigma_i>$", size=20, bbox={'facecolor': 'white', 'pad': 3})
    ax3.text(0, -0.5, "$<\sigma_i>^2$", size=20, bbox={'facecolor': 'white', 'pad': 2.5})
    im1 = ax1.imshow(X[:,:,0], cmap="RdBu",  interpolation="nearest", animated=True)
    im2 = ax3.imshow(X[:,:,0], cmap="RdBu",  interpolation="nearest", animated=True)
    
    def animate(i):
        ax1.text(80, -0.5, "T={:.2f}".format(T), size=15, bbox={'facecolor': 'white', 'pad': 3})
        im1.set_array(X[:,:,i])
        im2.set_array(X2[:,:,i])
        return im1, im2
    
    fig2.colorbar(im1, cax=ax2)
    ax2.yaxis.set_ticks_position('left')
    fig2.tight_layout(pad=1.0)
    anim = FuncAnimation(fig2, animate, frames=30, interval=250, blit=False)
    return anim

'''B-W Pixelgrid animation of the evolving system'''
def bw(X, T):
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(X[:,:,0], cmap="Greys",  interpolation="nearest", animated=True)
    def animate(i):
        ax.text(80, -0.5, "T={:.2f}".format(T[i]), size=15, bbox={'facecolor': 'white', 'pad': 3})
        im.set_array(X[:,:,i])
        return im
    fig.tight_layout(pad=1.0)
    anim = FuncAnimation(fig, animate, frames=30, interval=250, blit=False)
    return anim

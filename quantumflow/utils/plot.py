import ipywidgets as widgets
from IPython.display import HTML, display
from matplotlib import animation, rc
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

def anim_plot(array, x=None, interval=100, bar="", figsize=(15, 3), **kwargs):
    frames = len(array)
    
    if not bar == "":
        import ipywidgets as widgets
        widget = widgets.IntProgress(min=0, max=frames, description=bar, bar_style='success',
                                     layout=widgets.Layout(width='92%'))
        display(widget)

    fig, ax = plt.subplots(figsize=figsize)
    
    if x is None:
        plt_h = ax.plot(array[0], **kwargs)
    else:
        plt_h = ax.plot(x, array[0], **kwargs) 
        
    min_last = np.min(array[-1])
    max_last = np.max(array[-1])
    span_last = max_last - min_last
        
    ax.set_ylim([min_last - span_last*0.2, max_last + span_last*0.2])

    def init():
        return plt_h

    def animate(f):
        if not bar == "":
            widget.value = f

        for i, h in enumerate(plt_h):
            if x is None:
                h.set_data(np.arange(len(array[f][:, i])), array[f][:, i], **kwargs)
            else:
                h.set_data(x, array[f][:, i], **kwargs)
        return plt_h

    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=interval,
                                   blit=True, repeat=False)

    plt.close(fig)
    rc('animation', html='html5')
    display(HTML(anim.to_html5_video(embed_limit=1024)))

    if not bar == "":
        widget.close()

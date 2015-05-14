from matplotlib import animation, pyplot as plt 


def create_animation(field, mesh, dimensions, times):
    fig = plt.figure()
    # TODO catch Dimension % 2 != 0 cases
    field = [Temp.reshape((mesh.cells**int(dimensions/2), mesh.cells**int(dimensions/2))) for Temp in field]
    im = plt.imshow(field[0], interpolation='none', vmin=0, vmax=0.01, cmap='hot')#'jet')
    plt.colorbar()
    def init():
        return im,

    def animate(i):
        im.set_array(field[i])
        return im,
    
    an = animation.FuncAnimation(fig, animate, init_func=init, frames=times)
    return an

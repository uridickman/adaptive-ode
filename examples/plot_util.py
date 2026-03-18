fs = 16
ts = 14
lw = 2



def plot(t,y,h,axs,label,color,linestyle):
    
    text_args = {
        "fontsize": fs
    }

    plot_args = {
        "color": color,
        "label": label,
        "linestyle": linestyle,
        "linewidth": lw
    }

    axs[0,0].plot(t,y[:,0],**plot_args)
    axs[0,0].set_xlabel("Time",**text_args)
    axs[0,0].set_ylabel("Y1",**text_args)

    axs[0,1].plot(t,y[:,1],**plot_args)
    axs[0,1].set_xlabel("Time",**text_args)
    axs[0,1].set_ylabel("Y2",**text_args)
    axs[0,1].legend(loc="upper right",**text_args)

    axs[1,0].plot(y[:,0],y[:,1],**plot_args)
    axs[1,0].set_xlabel("Y1",**text_args)
    axs[1,0].set_ylabel("Y2",**text_args)

    axs[1,1].plot(t[1:],h,**plot_args)
    axs[1,1].set_xlabel("Time",**text_args)
    axs[1,1].set_ylabel("Step size",**text_args)
    axs[1,1].set_yscale("log")

    for ax in axs.flatten():
        ax.tick_params("both",labelsize=ts)
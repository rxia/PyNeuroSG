"""
specify GM32 array spatial layout
"""

# manually define the layout
ch = range(1,33)     # channel
r  = [5]*5+[4]*5+[3]*6+[2]*6+[1]*5+[0]*5
c  = range(4,-1,-1)*2 + range(5,-1,-1)*2 + range(4,-1,-1)*2
layout_GM32 = dict(zip(ch, zip(r,c)))


tf_plot_layout = False
if tf_plot_layout:
    from matplotlib import pyplot as plt
    for ch, (r,c) in layout_GM32.items():
        plt.plot(c,r,'k+')
        plt.text(c,r,ch)
    plt.gca().invert_yaxis()
    plt.show()
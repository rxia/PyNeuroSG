import tkinter

def get_info (filename, frame, outfile):

    script = '''
    # Get bink video frame and store in PNG file
    lappend auto_path /usr/local/lib
    set env(DLSH_LIBRARY) /usr/local/lib/dlsh
    
    package require dlsh
    package require Bink
    load_Impro
    
    set b [bink open {}]
    scan [bink info $b] "%d %d %f %d %d" w h rate nframes ms
    bink goto $b {}
    dl_local pixels [bink getframe $b]
    set img [img_imgfromlist $pixels $w $h]
    img_writePNG $pixels $w $h {}
    bink close $b
    '''.format(filename, frame, outfile)

    t = tkinter.Tcl()
    t.eval(script)


get_info("/shared/lab/stimuli/planet_earth/450_5_sec_color_bk/3.bk2", 100, "frame.png")


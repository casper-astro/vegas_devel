from vegas_utils import vegas_status, vegas_databuf
import curses, curses.wrapper
import time

def display_status(stdscr,stat,data):
    # Set non-blocking input
    stdscr.nodelay(1)
    run = 1

    # Look like gbtstatus (why not?)
    curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_WHITE, curses.COLOR_RED)
    keycol = curses.color_pair(1)
    valcol = curses.color_pair(2)
    errcol = curses.color_pair(3)

    # Loop 
    while (run):
        # Refresh status info
        stat.read()

        # Reset screen
        stdscr.erase()

        # Draw border
        stdscr.border()

        # Get dimensions
        (ymax,xmax) = stdscr.getmaxyx()

        # Display main status info
        onecol = False # Set True for one-column format
        col = 2
        curline = 0
        stdscr.addstr(curline,col,"Current VEGAS status:", keycol);
        curline += 2
        flip=0
        for k,v in stat.hdr.items():
            if (curline < ymax-3):
                stdscr.addstr(curline,col,"%8s : "%k, keycol)
                stdscr.addstr("%s" % v, valcol)
            else:
                stdscr.addstr(ymax-3,col, "-- Increase window size --", errcol);
            if (flip or onecol):
                curline += 1
                col = 2
                flip = 0
            else:
                col = 40
                flip = 1
        col = 2
        if (flip and not onecol):
            curline += 1

        # Refresh current block info
        try:
            curblock = stat["CURBLOCK"]
        except KeyError:
            curblock=-1

        # Display current packet index, etc
        if (curblock>=0 and curline < ymax-4):
            curline += 1
            stdscr.addstr(curline,col,"Current data block info:",keycol)
            curline += 1
            data.read_hdr(curblock)
            try:
                pktidx = data.hdr[curblock]["PKTIDX"]
            except KeyError:
                pktidx = "Unknown"
            stdscr.addstr(curline,col,"%8s : " % "PKTIDX", keycol)
            stdscr.addstr("%s" % pktidx, valcol)

        # Figure out if we're folding
        foldmode = False
        try:
            foldstat = stat["FOLDSTAT"]
            curfold = stat["CURFOLD"]
            if (foldstat!="exiting"):
                foldmode = True
        except KeyError:
            foldmode = False

        # Display fold info
        if (foldmode and curline < ymax-4):
            folddata = vegas_databuf(2)
            curline += 2
            stdscr.addstr(curline,col,"Current fold block info:",keycol)
            curline += 1
            folddata.read_hdr(curfold)
            try:
                npkt = folddata.hdr[curfold]["NPKT"]
                ndrop = folddata.hdr[curfold]["NDROP"]
            except KeyError:
                npkt = "Unknown"
                ndrop = "Unknown"
            stdscr.addstr(curline,col,"%8s : " % "NPKT", keycol)
            stdscr.addstr("%s" % npkt, valcol)
            curline += 1
            stdscr.addstr(curline,col,"%8s : " % "NDROP", keycol)
            stdscr.addstr("%s" % ndrop, valcol)

        # Bottom info line
        stdscr.addstr(ymax-2,col,"Last update: " + time.asctime() \
                + "  -  Press 'q' to quit")

        # Redraw screen
        stdscr.refresh()

        # Sleep a bit
        time.sleep(0.25)

        # Look for input
        c = stdscr.getch()
        while (c != curses.ERR):
            if (c==ord('q')):
                run = 0
            c = stdscr.getch()

# Connect to vegas status, data bufs
g = vegas_status()
d = vegas_databuf()

# Wrapper calls the main func
try:
    curses.wrapper(display_status,g,d)
except KeyboardInterrupt:
    print "Exiting..."



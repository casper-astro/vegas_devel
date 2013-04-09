from sys import argv
from vegas_utils import vegas_status, vegas_databuf
import curses, curses.wrapper
import time

def display_status(stdscr,stat,instance_id,data):
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

        # Get instance_id (as a string) from status buffer
        try:
            instance_str = stat['INSTANCE']
        except KeyError:
            instance_str = '?'

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
        stdscr.addstr(curline,col, \
            " Current Status: Instance %s " % instance_str, keycol);
        curline += 2
        flip=0
#        keys = stat.hdr.keys()
#        keys.sort()
#        try:
#          keys.remove('INSTANCE')
#        except:
#          pass
#
#        if len(keys) > 0:
#            prefix = keys[0][0:3]
#        else:
#            prefix = ''
#
#        for k in keys:
#            if k[0:3] != prefix:
#                prefix = k[0:3]
#                curline += flip
#                col = 2
#                flip = 0
#                #stdscr.addch(curline, 0, curses.ACS_LTEE)
#                #stdscr.hline(curline, 1, curses.ACS_HLINE, xmax-2)
#                #stdscr.addch(curline, xmax-1, curses.ACS_RTEE)
#                curline += 1
#
#            v = stat.hdr[k]
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

# Get instance_id
try:
    instance_id = int(argv[1])
except:
    instance_id = 0

# Connect to vegas status, data bufs
try:
    g = vegas_status(instance_id)
except:
    print "Error connecting to status buffer for instance %d" % instance_id
    exit(1)
try:
    d = vegas_databuf()
except:
    print "Error connecting to data buffer for instance %d" % instance_id
    exit(1)

# Wrapper calls the main func
try:
    curses.wrapper(display_status,g,instance_id,d)
except KeyboardInterrupt:
    print "Exiting..."
except:
    print "Error reading from status buffer %d" % instance_id

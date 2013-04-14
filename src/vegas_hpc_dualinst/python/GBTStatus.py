import MySQLdb, string, sys, time, os, math
from pyslalib import slalib as s

RADTODEG    = float('57.295779513082320876798154814105170332405472466564')
DEGTORAD    = float('1.7453292519943295769236907684886127134428718885417e-2')
RADTOHRS    = float('3.8197186342054880584532103209403446888270314977710')
HRSTORAD    = float('2.6179938779914943653855361527329190701643078328126e-1')

TITLE = {
         'scan_length'        : "Duration (sec):     ", 
         "remaining"          : "Remaining:          ",
         "scan_sequence"      : "Scan: ",
         "scan_number"        : "Scan: ",
         "scan_total"         : "Scan: ",
         "scan_type"          : "Scan: ",
         "start_time"         : "Scan Start Time:    ",
         "time_to_start"      : "Scan Start Time:    ",
         "focus_offset"       : "FocusOffset(XYZ):   ",
         "focus_tilt_offset"  : "FocusOffset(XYZ T): ",
         "focus_scan"         : "Focus Y Scan:       ",
         "az_commanded"       : "Az commanded (deg): ",
         "el_commanded"       : "El commanded (deg): ",
         "az_actual"          : "Az actual (deg):    ",
         "el_actual"          : "El actual (deg):    ",
         "az_error"           : "Az error (arcsec):  ",
         "el_error"           : "El error (arcsec):  ",
         "lpcs"               : "Az/El LPCs:         ",
         "ant_motion"         : "Antenna State:      ",
         "time_to_target"     : "On Source:          ",
         "major_type"         : "Major:              ",
         "minor_type"         : "Minor:              ",
         "major"              : "Major Epoch:        ",
         "minor"              : "Minor Epoch:        ",
         "epoch"              : "Coordinate Mode:    ",
         "receiver"           : "Receiver:           ",
         "rcvr_pol"           : "Polarity:           ",
         "freq"               : "Obs Freq (MHz):     ",
         "if_rest_freq"       : "Rest Freq (MHz):    ",
         "velocity_definition": "Vel  Def:           ", 
         "velocity_frame"     : "Vel Frame:          ",
         "cal_state"          : "Cal State:          ",
         "switch_period"      : "Sw Period:          ",
         "first_if_freq"      : "Center Freq (MHz):  ",
         "wind_velocity"      : "Wind Vel (m/s):     ",
         "ambient_temperature": "Temp (c):           ",
         "source"             : "Source: ",
         "source_velocity"    : "Source Vel (km/s):  ",
         "observer"           : "Observer:           ",
         "last_update"        : "Last Update:        ",
         "operator"           : "Operator:           ",
         "utc"                : "UTC Time:           ", 
         "utc_date"           : "UTC Date:           ",
         "data_dir"           : "Project ID:         ",
         "lst"                : "LST:                ",
         "status"             : "Status:             ",
         "time_to_set"        : "Time To Set:        ",
         "j2000_major"        : "RA  J2000 (deg):    ",
         "j2000_minor"        : "Dec J2000 (deg):    "
        }

PSR_fieldNames = [
       "scan_length",
       "remaining",
       "scan_sequence",
       "scan_number",
       "start_time",
       "time_to_start",
       "az_commanded",
       "el_commanded",
       "az_actual",
       "el_actual",
       "ant_motion",
       "time_to_target",
       "major_type",
       "minor_type",
       "major",
       "minor",
       "epoch",
       "receiver",
       "rcvr_pol",
       "freq",
       "if_rest_freq",
       "cal_state",
       "switch_period",
       "first_if_freq",
       "source",
       "observer",
       "last_update",
       "utc",
       "utc_date",
       "data_dir",
       "lst",
       "status",
       "time_to_set",
       "j2000_major",
       "j2000_minor"]


class GBTStatus:
    def __init__(self):
        self.info = 'Information'
        self.db = MySQLdb.connect(passwd="w3bqu3ry",db="gbt_status",host="gbtdata.gbt.nrao.edu",user="gbtstatus")
        self.cursor = self.db.cursor()
        self.noValue = "unknown"
        self.fieldNames = PSR_fieldNames
        self.prepQuery()
        
    def getValue(self, key):
        # Set the value based on the key provided.
        if key is not None:
            if self.kvPairs.has_key(key):
                # The given key maps to a value.
                value = self.kvPairs[key]
            else:
            # The given key does not map to a value.  In this case, take the key
            # to be the literal value.
                value = key
        else:
            value = "unknown"
        return value    

    def prepQuery(self):
        try:
            # Build an query from the field names.
            queryList = ""
            self.dbfields = []
            
            # get the field names of the status table
            q = "Show Columns from status"
            self.cursor.execute(q)
            cols = self.cursor.fetchall()
            for i in cols:
                self.dbfields.append(i[0])
                
            for f in self.fieldNames:
                if f not in self.dbfields:
                    pass
                else:
                    queryList = queryList + f + ","
            queryList = queryList[0:-1]
            self.query = "select %s from status" % queryList
        
        except: #Database error - use last values from db
            print "Uh-oh!  DB error!"
            pass

    def collectKVPairs(self):
        try:
            # Run the query.
            self.cursor.execute(self.query)
            fieldValues = self.cursor.fetchall()[0]
            self.kvPairs = {}
        except: #Database error - use last values from db
            print "Uh-oh!  DB error!"
            pass

        # Build a dictionary of keyword to value mappings.
        for i, d in enumerate(self.fieldNames):
            if d not in self.dbfields:
                self.kvPairs[d] = self.noValue
            else:
                self.kvPairs[self.fieldNames[i]] = fieldValues[i]
            
    def show(self, key, title = None, units = None):
        '''Print the title and value of the item to the display'''
        # Set the value based on the key provided.
        if key is not None:
            if self.kvPairs.has_key(key):
                # The given key maps to a value.
                value = self.kvPairs[key]
            else:
                # The given key does not map to a value.  In this case, take the key
                # to be the literal value.
                value = key
                
            # The value's length is to be used as an offset.
            valueLength = len(value)
        else:
            valueLength = 0
            value = ""
        
        # Determine title's length to use as an offset.
        if title is not None:
            titleLength = len(title)
        else:
            titleLength = 0
            title = ""
        
        if units is not None:
            units = ' ' + units
        else:
            units = ""        
        
        print title, value, units

    def getScanStartInfo(self, key):

        self.scan_time_display = 1
        if self.kvPairs.has_key("start_time") and self.kvPairs.has_key("time_to_start"):

            startTime = self.kvPairs["start_time"]
            timeToStart = self.kvPairs["time_to_start"]

            if timeToStart != '0':        
                m = "%s (starts in %s)" % (startTime, timeToStart)
            else:
                m = "%s" % startTime
        else:
            m = "%s : %s" % (key, self.kvPairs[key])
            
        return m

    def getTimeToSource(self):
        timeToSource = self.kvPairs["time_to_target"]
        
        if timeToSource[0:2] != "0":
            m = "In %s" % timeToSource
        else:
            m = "Yes"    
        
        return m
    
    def getMajor(self):
        major = self.kvPairs["major"]
        if major != self.noValue:
            sexi, units = major.split('(')
            if self.kvPairs["epoch"] in ['J2000', 'B1950']:
                sexi = self.degrees2hms(float(sexi))
            else:
                return self.frmtFloat(sexi)
            #sexi = self.frmtCoordUnits(sexi)
            return sexi + ' (' + units
        else:
            return major
         
    def getMinor(self):
        minor = self.kvPairs["minor"]
        if minor != self.noValue:
            sexi, units = minor.split('(')
            if self.kvPairs["epoch"] in ['J2000', 'B1950']:
                sexi = self.degrees2dms(float(sexi))
            else:
                return self.frmtFloat(sexi)
            #sexi = self.frmtCoordUnits(sexi)
            return sexi + ' (' + units
        else:
            return minor

    def frmtCoordUnits(self, major):
        "display h:m:s for ra, but fractional degrees for other modes"
        if self.kvPairs["epoch"] in ['J2000','B1950']:
            return self.degrees2hms(major)
        else:
            return self.frmtFloat(major)
            
    def frmtFloat(self,strVal):
        if strVal == self.noValue:
            return strVal
        flVal = float(strVal)
        return '%.3f' % flVal

    def degrees2hms(self, deg):
        "converts degrees to h:m:s format"
        if deg == self.noValue:
            return self.noValue
        sign, arr = s.sla_dr2tf(4, deg*DEGTORAD)
        return "%02d:%02d:%07.4f" % (arr[0], arr[1], arr[2]+0.0001*arr[3])

    def degrees2dms(self, deg):
        "converts degrees to d:m:s format"
        if deg == self.noValue:
            return self.noValue
        sign, arr = s.sla_dr2af(4, deg*DEGTORAD)
        return sign+"%02d:%02d:%02.4f" % (arr[0], arr[1], arr[2]+0.0001*arr[3])

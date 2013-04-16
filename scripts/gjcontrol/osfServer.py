import guppi_utils
import sys
import os
import numpy as np
import subprocess
import time
import signal

def info(*args,**kwargs):
    logging.info(*args,**kwargs)
    for h in logging.getLogger().handlers:
        h.flush()

class guppiServer:
    def __init__(self, gpu):
        logging.info('init')
        self.status = guppi_utils.guppi_status()
        logging.info('got status')
        self.netbuf = guppi_utils.guppi_databuf()
        logging.info('got databuf') 
        self.daqp = None
        self.subp = None
        self.gpu = gpu
        
        
    def getData(self,block=0,size=2**20):
        logging.info('block %d' % block)
        data = self.netbuf.data(block)
        data.shape = (np.prod(data.shape),)
        return data[:size]
    
    def setParams(self, **kwargs):
        self.status.read()
        for k,v in kwargs.items():
            self.status.update(k,v)
        self.status.write()
        
    def getParam(self,param=None):
        self.status.read()
        if param:
            return {param:self.status[param]}
        else:
            return dict(self.status.items())
        
    def csCommand(self,cmd):
        logging.info('guppi_daq command: '+cmd)
        fh = open('/tmp/csguppi_daq_control','w')
        fh.write(cmd)
        fh.flush()
        fh.close()
        
    def restartDaq(self):
        logging.info("restarting daq")
        if self.daqp:
            logging.info("found existing subprocess")
            if self.daqp.poll() is None:  #if it's still running
                logging.info("subprocess is still running")
                self.csCommand('STOP')
                self.csCommand('QUIT')
                time.sleep(1)
                logging.info("killing subprocess")
                self.daqp.kill()
        try:
            os.unlink('/tmp/csguppi_daq_control')
            logging.info('removed daq control fifo')
        except Exception,e:
            logging.info('could not remove daq fifo:'+str(e))
        #source /home/gpu/gjones/puppi.sh; 
        self.daqp = subprocess.Popen(('/home/gpu/gjones/guppi_daq/src/guppi_daq_server &> /home/gpu/gjones/logs/gpu%d_daq.log' % gpu),
                                     shell=True)
        
    def pollDaq(self):
        if self.daqp:
            return self.daqp.poll()
        else:
            return False
        
    def startProc(self,cmd):
        logging.info("running " + cmd)
        if self.subp:
            try:
                logging.info("killing existing subprocess")
                self.subp.send_signal(signal.SIGINT)
                time.sleep(1)
                self.subp.kill()
                killed = True
            except:
                logging.info("exception in killing subprocess")
                killed = False
        else:
            killed = False
#        self.subp = subprocess.Popen(('%s' % (cmd,)).split(' '))
        self.daqp = subprocess.Popen(('%s &> /home/gpu/gjones/logs/dspsr%d.log' % (cmd,gpu)),shell=True)
        return killed
    
    def sendSig(self,sig):
        if self.subp:
            try:
                self.subp.send_signal(sig)
                return True
            except:
                return False
        else:
            return False
    
    def endProc(self):
        logging.info('ending subprocess')
        if self.subp:
            try:
                self.subp.kill()
            except:
                pass
            self.subp = None
            killed = True
        else:
            killed = False
        return killed
    
    def pollProc(self):
        if self.subp:
            return self.subp.poll()
        else:
            return False
        
    def ping(self):
        logging.info("got pinged")
        return True
    
    def system(self,cmd):
        logging.info("issuing: " + cmd)
        os.system(cmd)
        
        
        
if __name__ == "__main__":
    import socket
    hostname = socket.gethostname().split('.')[0]
    gpu = int(hostname[-1]) #assuming gpu0x
    
    import logging
#    import cloghandler
    logging.basicConfig(filename=('/home/gpu/gjones/logs/osfServer%d.log' % gpu),level=logging.INFO,
                        format = (hostname + ':%(levelname)s - %(asctime)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s'))
#    log = logging.getLogger()
#    log.addHandler(cloghandler.ConcurrentRotatingFileHandler('/home/gpu/gjones/logs/osfServer.log'))
    import Pyro4
    Pyro4.config.SERVERTYPE = 'multiplex'
    Pyro4.config.SOCK_REUSE = True


    name = "guppiServer%d" % gpu
    host = "gpu0%d" % gpu
    info("starting "+name+" on "+host)
    sys.stdout.flush()
    Pyro4.Daemon.serveSimple({guppiServer(gpu) : name}, ns=True, verbose=True, host=host, port=57777)
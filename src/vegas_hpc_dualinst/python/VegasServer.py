import sys
import time
import Pyro.core

from valon_synth import Synthesizer, SYNTH_A, SYNTH_B

from vegas_utils import vegas_status,vegas_databuf


class VegasServer(Pyro.core.ObjBase):
    def __init__(self,port='/dev/ttyS0'):
        Pyro.core.ObjBase.__init__(self)
        self.valon = Synthesizer(port)
        # get instance_id somehow
        self.vs = vegas_status(instance_id)
        self.dbuf2 = vegas_databuf(2)
        self.dbuf3 = vegas_databuf(3)        
        self.running = True
        
    def getTime(self):
        return time.time()
        
    def getSynthRefFreq(self):
        return self.valon.get_reference()
        
    def setSynthFreq(self,freq,chan_spacing=2.0):
        resA = self.valon.set_frequency(SYNTH_A,freq,chan_spacing=chan_spacing)
        resB = self.valon.set_frequency(SYNTH_B,freq,chan_spacing=chan_spacing)        
        return resA,resB
        
    def getSynthFreq(self):
        fA = self.valon.get_frequency(SYNTH_A)
        fB = self.valon.get_frequency(SYNTH_B)
        return fA,fB
        
    def getSynthLocked(self):
        resA = self.valon.get_phase_lock(SYNTH_A)
        resB = self.valon.get_phase_lock(SYNTH_B)
        return resA,resB
        
    def initShmem(self):
        pass
        
    def setParams(self,**kwargs):
        self.vs.read()
        for k,v in kwargs.items():
            self.vs.update(k,v)
        self.vs.write()
        
    def getParam(self,param=None):
        self.vs.read()
        if param:
            return {param:self.vs[param]}
        else:
            return dict(self.vs.items())
    def updateFromGBT(self):
        self.vs.read()
        self.vs.update_with_gbtstatus()
        self.vs.write()
        
    def getData(self,block=0,buf=2):
        if buf == 2:
            return self.dbuf2.data(block)
        else:
            return self.dbuf3.data(block)
    def quit(self):
        self.running = False
        
class VegasServerProxy:
    def __init__(self,parent):
        self.parent = parent
        class _Dummy(Pyro.core.ObjBase):
            def __init__(self):
                Pyro.core.ObjBase.__init__(self)
                
        _dd = Pyro.core.Daemon(host='localhost')
        _dd.connect(_Dummy())

        attrs = set(dir(VegasServer))
        for attr in attrs:
            def fproxy(*args,**kwargs):
                try:
                    return getattr(self.parent,attr)(*args,**kwargs)
                except Exception,e:
                    raise e
            setattr(self,attr,fproxy)
            
if __name__ == '__main__':
    try:
        hpc = int(sys.argv[1])
    except:
        print "Usage: firstargument must be integer 1-8 indicating hpc number"
        sys.exit(1)
    name = 'VegasServer%d' % hpc
    import Pyro.naming
    ns = Pyro.naming.NameServerLocator().getNS()
    old = None
    try:
        old = ns.resolve(name).getProxy()
        print "found existing ",name
    except Pyro.errors.NamingError:
        pass
    if old:
        try:
            print "quitting ",name,"... ",
            old.quit()
            print "successful!"
        except:
            print "failed."
            pass
    try:
        print "unregistering ",name," from nameserver... ",
        ns.unregister(name)
        print "done"
    except:
        print "not necessary"
        pass
       
    if hpc == 9:
        host = '10.16.96.203'
    else:
        host = 'vegas-hpc%d.gb.nrao.edu' % hpc
    daemon = Pyro.core.Daemon(host = host)
    daemon.useNameServer(ns)
    if hpc == 7:
        vs = VegasServer('/dev/ttyUSB0')
    elif hpc == 8:
        vs = VegasServer('/dev/ttyUSB0')
    else:
        vs = VegasServer()
    daemon.connect(vs,name)
    print "Starting ",name
    daemon.requestLoop(condition = lambda: vs.running, timeout=1)
    print name, "finished runnning"

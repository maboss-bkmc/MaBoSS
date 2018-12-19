
#
# class Result example using maboss communication layer: to be merged with pyMaBoSS/maboss/result.py
#

import maboss.comm

class Result:

    def __init__(self, mbcli, simulation, hexfloat = False):
        client_data = maboss.comm.ClientData(simulation.getNetwork(), simulation.getConfig())

        data = maboss.comm.DataStreamer.buildStreamData(client_data, hexfloat)
        #print "sending data", data

        data = mbcli.send(data)
        #print "received data", data

        self._result_data = maboss.comm.DataStreamer.parseStreamData(data)

    def getResultData(self):
        return self._result_data


import maboss.comm

class Result:

    def __init__(self, mbcli, simulation):
        client_data = maboss.comm.ClientData(simulation.getNetwork(), simulation.getConfig())

        data = maboss.comm.DataStreamer.buildStreamData(client_data)
        #print "sending data", data

        data = mbcli.send(data)
        #print "received data", data

        self._result_data = maboss.comm.DataStreamer.parseStreamData(data)

        print "self._result_data status=", self._result_data.getStatus(), "errmsg=", self._result_data.getErrorMessage()
        print "FP", self._result_data.getFP()
        print "Runlog", self._result_data.getRunLog()


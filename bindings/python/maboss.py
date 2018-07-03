# coding: utf-8
# MaBoSS (Markov Boolean Stochastic Simulator)
# Copyright (C) 2011-2018 Institut Curie, 26 rue d'Ulm, Paris, France
   
# MaBoSS is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
   
# MaBoSS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
   
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA 

# Module: maboss.py
# Authors: Eric Viara <viara@sysra.com>
# Date: May-July 2018

import os, sys, time, signal, socket
import atexit

MABOSS_SERVER = "../../engine/src/MaBoSS-server" # for now

LAUNCH = "LAUNCH"
MABOSS = "MaBoSS-2.0"
NETWORK = "Network:"
CONFIGURATION = "Configuration:"
CONFIGURATION_EXPRESSIONS = "Configuration-Expressions:"
CONFIGURATION_VARIABLES = "Configuration-Variables:"
RETURN = "RETURN"
STATUS = "Status:"
ERROR_MESSAGE = "Error-Message:"
STATIONARY_DISTRIBUTION = "Stationary-Distribution:"
TRAJECTORY_PROBABILITY = "Trajectory-Probability:"
TRAJECTORIES = "Trajectories:"
FIXED_POINTS = "Fixed-Points:"
RUN_LOG = "Run-Log:"

class MaBoSS:

    def __init__(self, host = None, port = None):
        self._host = host
        self._port = port
        self._client = Client(host, port)

    def launch(self, simulation):
        return Result(self._client, simulation)

    def getHost(self):
        return self._host

    def getPort(self):
        return self._port

    def getClient(self):
        return self._client

    def close(self):
        self._client.close()

class Simulation:

    def __init__(self, bndfile, cfgfile = None, cfgfiles = None):
        self._network = file_get_contents(bndfile)
        if cfgfile:
            self._config = file_get_contents(cfgfile)
        elif cfgfiles:
            self._config = ""
            for cfgfile in cfgfiles:
                self._config += file_get_contents(cfgfile)
        else:
            raise Exception("Simulation: cfgfile or cfgfiles must be set")

    def getNetwork(self):
        return self._network

    def getConfig(self):
        return self._config

class Result:

    def __init__(self, mbcli, simulation):
        client_data = ClientData(simulation.getNetwork(), simulation.getConfig())
        data = DataStreamer.buildStreamData(client_data)
        #print "send data", data
        data = mbcli.send(data)
        #print "rcv data", data
        server_data = DataStreamer.parseStreamData(data)
        print "server_data", server_data.getStatus(), server_data.getErrorMessage()
        print "server_data FP", server_data.getFP()
        print "server_data Runlog", server_data.getRunLog()

# communication

class HeaderItem:

    def __init__(self, directive, _from = None, to = None, value = None):
        self._directive = directive
        self._from = _from
        self._to = to
        self._value = value

    def getDirective(self):
        return self._directive

    def getFrom(self):
        return self._from

    def getTo(self):
        return self._to

    def getValue(self):
        return self._value

class DataStreamer:

    @staticmethod
    def buildStreamData(client_data):
        data = ""
        offset = 0
        o_offset = 0

        header = LAUNCH + " " + MABOSS + "\n"

        config_data = client_data.getConfig()
        offset += len(config_data)
        data += config_data

        (header, o_offset) = DataStreamer._add_header(header, CONFIGURATION, o_offset, offset)

        network_data = client_data.getNetwork()
        offset += len(network_data)
        data += network_data

        (header, o_offset) = DataStreamer._add_header(header, NETWORK, o_offset, offset)

        return header + "\n" + data

    @staticmethod
    def parseStreamData(ret_data):
        server_data = ServerData()
        magic = RETURN + " " + MABOSS
        magic_len = len(magic)
        if ret_data[0:magic_len] != magic:
            server_data.setStatus(1)
            server_data.setErrorMessage("magic " + magic + " not found in header")
            return server_data

        offset = magic_len
        pos = ret_data.find("\n\n", magic_len)
        if pos < 0:
            server_data.setStatus(2)
            server_data.setErrorMessage("separator double nl found in header")
            return server_data

        offset += 1
        header = ret_data[offset:pos+1]
        data  = ret_data[pos+2:]
        print "HEADER", header
        #print "DATA", data[0:200]

        header_items = []
        err_data = DataStreamer._parse_header_items(header, header_items)
        if err_data:
            server_data.setStatus(3)
            server_data.setErrorMessage(err_data)
            return server_data

        for header_item in header_items:
            directive = header_item.getDirective()
            if directive == STATUS:
                server_data.setStatus(int(header_item.getValue()))
            elif directive == ERROR_MESSAGE:
                server_data.setErrorMessage(header_item.getValue())
            else:
                data_value = data[header_item.getFrom():header_item.getTo()+1]
                if directive == STATIONARY_DISTRIBUTION:
                    server_data.setStatDist(data_value)
                elif directive == TRAJECTORY_PROBABILITY:
                    server_data.setProbTraj(data_value)
                elif directive == TRAJECTORIES:
                    server_data.setTraj(data_value)
                elif directive == FIXED_POINTS:
                    server_data.setFP(data_value)
                elif directive == RUN_LOG:
                    server_data.setRunLog(data_value)
                else:
                    server_data.setErrorMessage("unknown directive " + directive)
                    server_data.setStatus(4)
                    return server_data

        return server_data

    @staticmethod
    def _parse_header_items(header, header_items):
        opos = 0
        pos = 0
        while True:
            pos = header.find(':', opos)
            if pos < 0:
                break
            directive = header[opos:pos+1]
            opos = pos+1
            pos = header.find("\n", opos)
            if pos < 0:
                return "newline not found in header after directive " + directive

            value = header[opos:pos]
            opos = pos+1
            pos2 = value.find("-")
            if directive == STATUS or directive == ERROR_MESSAGE:
                header_items.append(HeaderItem(directive = directive, value = value))
            elif pos2 >= 0:
                header_items.append(HeaderItem(directive = directive, _from = int(value[0:pos2]), to = int(value[pos2+1:])))
            else:
                return "dash - not found in value " + value + " after directive " + directive

        return ""

    @staticmethod
    def _add_header(header, directive, o_offset, offset):
        if o_offset != offset:
            header += directive + str(o_offset) + "-" + str(offset-1) + "\n"

        return (header, offset)

class ClientData:

    def __init__(self, network = None, config = None):
        self._network = network
        self._config = config

    def getNetwork(self):
        return self._network

    def getConfig(self):
        return self._config

    def setNetwork(self, network):
        self._network = network

    def setConfig(self, config):
        self._config = config

class ServerData:

    def __init__(self):
        self._status = 0
        self._errmsg = ""
        self._stat_dist = None
        self._prob_traj = None
        self._traj = None
        self._FP = None
        self._runlog = None

    def setStatus(self, status):
        self._status = status

    def getStatus(self):
        return self._status

    def setErrorMessage(self, errmsg):
        self._errmsg = errmsg

    def getErrorMessage(self):
        return self._errmsg

    def setStatDist(self, data_value):
        self._stat_dist = data_value

    def getStatDist(self):
        return self._stat_dist

    def setProbTraj(self, data_value):
        self._prob_traj = data_value

    def getProbTraj(self):
        return self._prob_traj

    def setTraj(self, data_value):
        self._traj = data_value

    def getTraj(self):
        return self._traj

    def setFP(self, data_value):
        self._FP = data_value

    def getFP(self):
        return self._FP

    def setRunLog(self, data_value):
        self._runlog = data_value

    def getRunLog(self):
        return self._runlog

class Client:
    
    SERVER_NUM = 1 # for now

    def __init__(self, host = None, port = None):
        self._host = host
        self._port = port
        self._pid = None
        self._mb = bytearray()
        self._mb.append(0)
        self._pidfile = None

        if host == None:
            if port == None:
                port = '/tmp/MaBoSS_pipe_' + str(os.getpid()) + "_" + str(Client.SERVER_NUM)

            self._pidfile = '/tmp/MaBoSS_pidfile_' + str(os.getpid()) + "_" + str(Client.SERVER_NUM)
            Client.SERVER_NUM += 1

            try:
                pid = os.fork()
            except OSError, e:
                print >> sys.stderr, "error fork:", e
                return

            if pid == 0:
                try:
                    args = [MABOSS_SERVER, "--host", "localhost", "--port", port, "--pidfile", self._pidfile]
                    os.execv(MABOSS_SERVER, args)
                except e:
                    print >> sys.stderr, "error execv:", e

            self._pid = pid
            atexit.register(self.close)
            server_started = False
            MAX_TRIES = 20
            TIME_INTERVAL = 0.1
            for try_cnt in range(MAX_TRIES):
                if os.path.exists(self._pidfile):
                    server_started = True
                    break
                time.sleep(TIME_INTERVAL)

            if not server_started:
                raise Exception("MaBoSS server on port " + port + " not started after " + str(MAX_TRIES*TIME_INTERVAL) + " seconds")

            self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._socket.connect(port)
        else:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.connect((host, port))
            
    def send(self, data):
        self._socket.send(data)
        self._term()
        SIZE = 4096
        ret_data = ""
        while True:
            databuf = self._socket.recv(SIZE)
            if not databuf or len(databuf) <= 0:
                break
            ret_data += databuf

        return ret_data

    def _term(self):
        self._socket.send(self._mb)

    def close(self):
        if self._pid != None:
            print "kill", self._pid
            os.kill(self._pid, signal.SIGTERM)
            if self._pidfile:
                os.remove(self._pidfile)
            self._pid = None

#        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#        self._socket.connect((host, port))
#        print "Connection on {}".format(port)
#        self._socket.send('coucou les lapins')

def file_get_contents(filename):
    if not os.path.isfile(filename):
        raise Exception(filename + " is not a valid file")
    fd = os.open(filename, os.O_RDONLY)
    if fd >= 0:
        stat = os.fstat(fd)
        contents = os.read(fd, stat.st_size)
        os.close(fd)
        return contents
    raise Exception(file + " is not readable")











import maboss.comm, maboss.simul, maboss.result

mbcli = maboss.comm.MaBoSSClient()
TESTLOC = "../../engine/tests/"

simulation = maboss.simul.Simulation(bndfile = TESTLOC + "/cellcycle.bnd", cfgfiles = [TESTLOC + "/cellcycle_runcfg.cfg", TESTLOC + "/cellcycle_runcfg-thread_1-simple.cfg"])

result = mbcli.launch(simulation) # call Result(mbcli, simulation)



import maboss

mb = maboss.MaBoSS()
TESTLOC = "../../engine/tests/"

simulation = maboss.Simulation(bndfile = TESTLOC + "/cellcycle.bnd", cfgfiles = [TESTLOC + "/cellcycle_runcfg.cfg", TESTLOC + "/cellcycle_runcfg-thread_1-simple.cfg"])

result = mb.launch(simulation) # call Result(mb, simulation)


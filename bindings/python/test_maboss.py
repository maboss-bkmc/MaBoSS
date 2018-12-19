
# env:
# export MABOSS_SERVER=../../engine/src/MaBoSS-server

import maboss.comm, maboss.simul, maboss.result

mbcli = maboss.comm.MaBoSSClient()

TESTLOC = "../../engine/tests/"

simulation = maboss.simul.Simulation(bndfile = TESTLOC + "/cellcycle.bnd", cfgfiles = [TESTLOC + "/cellcycle_runcfg.cfg", TESTLOC + "/cellcycle_runcfg-thread_1-simple.cfg"])

result = mbcli.run(simulation) # call Result(mbcli, simulation)

result_data = result.getResultData()

print "result_data status=", result_data.getStatus(), "errmsg=", result_data.getErrorMessage()
print "FP", result_data.getFP()
print "Runlog", result_data.getRunLog()



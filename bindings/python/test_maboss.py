
# env:
# export MABOSS_SERVER=../../engine/src/MaBoSS-server

import maboss.comm, maboss.simul, maboss.result

mbcli = maboss.comm.MaBoSSClient()

TESTLOC = "../../engine/tests/"

simulation = maboss.simul.Simulation(bndfile = TESTLOC + "/cellcycle.bnd", cfgfiles = [TESTLOC + "/cellcycle_runcfg.cfg", TESTLOC + "/cellcycle_runcfg-thread_1-simple.cfg"])

# True for hexfloat
result = mbcli.run(simulation, True) # call Result(mbcli, simulation)

result_data = result.getResultData()

print "result_data status=", result_data.getStatus(), "errmsg=", result_data.getErrorMessage()
print "FP", result_data.getFP()
print "Runlog", result_data.getRunLog()
print "ProbTraj", result_data.getProbTraj()
print "StatDist", result_data.getStatDist()



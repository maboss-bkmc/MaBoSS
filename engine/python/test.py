import maboss_module

sim = maboss_module.MaBoSSSim("metastasis.bnd", "metastasis.cfg")

print(sim)
print(type(sim))
print(dir(sim))

print(sim.run())
print(sim.get_last_probtraj())
print(sim.get_states())
print(sim.get_raw_probtrajs())

del sim

sim2 = maboss_module.MaBoSSSim("metastasis.bnd", "metastasis.cfg")
sim2.run()
print(sim2.get_last_probtraj())


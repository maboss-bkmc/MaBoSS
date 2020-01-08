import cmaboss_128n
from unittest import TestCase

class TestCMaBoSS(TestCase):

    def test_load_model(self):
        sim = cmaboss_128n.MaBoSSSim(network="../tests/ewing/ewing_full.bnd", config="../tests/ewing/ewing.cfg")
        res = sim.run()
        res.get_states()
        res.get_nodes()
        res.get_last_states_probtraj()
        res.get_raw_probtrajs()
        res.get_last_nodes_probtraj()
        res.get_raw_nodes_probtrajs()
        res.get_fp_table()

    def test_load_model_str(self):
        with open("../tests/ewing/ewing_full.bnd", "r") as bnd, open("../tests/ewing/ewing.cfg", "r") as cfg:    
            sim = cmaboss_128n.MaBoSSSim(network_str=bnd.read(),config_str=cfg.read())
            res = sim.run()
            res.get_states()
            res.get_nodes()
            res.get_last_states_probtraj()
            res.get_raw_probtrajs()
            res.get_last_nodes_probtraj()
            res.get_raw_nodes_probtrajs()
            res.get_fp_table()

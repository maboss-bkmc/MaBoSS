import cmaboss
from unittest import TestCase

class TestCMaBoSS(TestCase):

    def test_load_model(self):
        sim = cmaboss.MaBoSSSim(network="../examples/metastasis.bnd", config="../examples/metastasis.cfg")
        res = sim.run()
        res.get_states()
        res.get_nodes()
        res.get_last_states_probtraj()
        res.get_raw_probtrajs()
        res.get_last_nodes_probtraj()
        res.get_raw_nodes_probtrajs()
        res.get_fp_table()

    def test_load_model_str(self):
        with open("../examples/metastasis.bnd", "r") as bnd, open("../examples/metastasis.cfg", "r") as cfg:    
            sim = cmaboss.MaBoSSSim(network_str=bnd.read(),config_str=cfg.read())
            res = sim.run()
            res.get_states()
            res.get_nodes()
            res.get_last_states_probtraj()
            res.get_raw_probtrajs()
            res.get_last_nodes_probtraj()
            res.get_raw_nodes_probtrajs()
            res.get_fp_table()

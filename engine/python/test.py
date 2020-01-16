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

    def test_load_model_error(self):
        with self.assertRaises(cmaboss.BNException):
            cmaboss.MaBoSSSim(network="../examples/metastasis-error.bnd", config="../examples/metastasis.cfg")

    def test_last_states(self):
        expected = {
            '<nil>': 0.206, 'Apoptosis -- CellCycleArrest': 0.426, 'CellCycleArrest': 0.068, 
            'Migration -- Metastasis -- Invasion -- CellCycleArrest': 0.3
        }

        expected_nodes = {
            'Metastasis': 0.3, 'Invasion': 0.3, 'Migration': 0.3, 'Apoptosis': 0.426, 'CellCycleArrest': 0.794
        }

        sim = cmaboss.MaBoSSSim(network="../examples/metastasis.bnd", config="../examples/metastasis.cfg")
        res = sim.run(only_last_state=False)

        for key, value in res.get_last_states_probtraj().items():
            self.assertAlmostEqual(value, expected[key])
        
        for key, value in res.get_last_nodes_probtraj().items():
            self.assertAlmostEqual(value, expected_nodes[key])
        
        simfinal = cmaboss.MaBoSSSim(network="../examples/metastasis.bnd", config="../examples/metastasis.cfg")
        resfinal = simfinal.run(only_last_state=True)
    
        for key, value in resfinal.get_last_states_probtraj().items():
            self.assertAlmostEqual(value, expected[key])

        for key, value in resfinal.get_last_nodes_probtraj().items():
            self.assertAlmostEqual(value, expected_nodes[key])
           
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

    def test_load_model_str_error(self):
        with open("../examples/metastasis-error.bnd", "r") as bnd, open("../examples/metastasis.cfg", "r") as cfg:    
            with self.assertRaises(cmaboss.BNException):
                cmaboss.MaBoSSSim(network_str=bnd.read(),config_str=cfg.read())

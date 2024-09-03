import cmaboss_128n
import numpy as np
from unittest import TestCase

class TestCMaBoSS(TestCase):

    def test_load_model(self):
        sim = cmaboss_128n.MaBoSSSim(network="../tests/ewing/ewing_full.bnd", config="../tests/ewing/ewing.cfg")
        res = sim.run()
        res.get_probtraj()
        res.get_last_probtraj()
        res.get_nodes_probtraj()
        res.get_last_nodes_probtraj()
        res.get_fp_table()

    def test_load_model_error(self):
        with self.assertRaises(cmaboss_128n.BNException):
            sim = cmaboss_128n.MaBoSSSim(network="../tests/ewing/ewing_full-error.bnd", config="../tests/ewing/ewing.cfg")

    def test_last_states(self):
        expected = {
            '<nil>': 0.206, 'Apoptosis -- CellCycleArrest': 0.426, 'CellCycleArrest': 0.068, 
            'Migration -- Metastasis -- Invasion -- CellCycleArrest': 0.3
        }

        expected_nodes = {
            'Metastasis': 0.3, 'Invasion': 0.3, 'Migration': 0.3, 'Apoptosis': 0.426, 'CellCycleArrest': 0.794
        }

        sim = cmaboss_128n.MaBoSSSim(network="../tests/metastasis/metastasis.bnd", config="../tests/metastasis/metastasis.cfg")
        res = sim.run(only_last_state=False)
        
        raw_res, _, states, _ = res.get_last_probtraj()
        for i, value in enumerate(np.nditer(raw_res)):
            self.assertAlmostEqual(value, expected[states[i]])
        
        raw_nodes_res, _, nodes, _ = res.get_last_nodes_probtraj()
        for i, value in enumerate(np.nditer(raw_nodes_res)):
            self.assertAlmostEqual(value, expected_nodes[nodes[i]])
        
        simfinal = cmaboss_128n.MaBoSSSim(network="../tests/metastasis/metastasis.bnd", config="../tests/metastasis/metastasis.cfg")
        resfinal = simfinal.run(only_last_state=True)    
        
        raw_res, _, states, _ = res.get_last_probtraj()
        for i, value in enumerate(np.nditer(raw_res)):
            self.assertAlmostEqual(value, expected[states[i]])
        
        raw_nodes_res, _, nodes, _ = res.get_last_nodes_probtraj()
        for i, value in enumerate(np.nditer(raw_nodes_res)):
            self.assertAlmostEqual(value, expected_nodes[nodes[i]])
        
    def test_load_model_str(self):
        with open("../tests/ewing/ewing_full.bnd", "r") as bnd, open("../tests/ewing/ewing.cfg", "r") as cfg:    
            sim = cmaboss_128n.MaBoSSSim(network_str=bnd.read(),config_str=cfg.read())
            res = sim.run()
            res.get_probtraj()
            res.get_last_probtraj()
            res.get_nodes_probtraj()
            res.get_last_nodes_probtraj()
            res.get_fp_table()

    def test_load_model_str_error(self):
        with open("../tests/ewing/ewing_full-error.bnd", "r") as bnd, open("../tests/ewing/ewing.cfg", "r") as cfg:    
            with self.assertRaises(cmaboss_128n.BNException):
                cmaboss_128n.MaBoSSSim(network_str=bnd.read(),config_str=cfg.read())

    def test_observed_graph(self):
        sim = cmaboss_128n.MaBoSSSim("../tests/sizek/sizek.bnd", "../tests/sizek/sizek_wgraph.cfg")
        sim.update_parameters(sample_count=500)
        res = sim.run()
        values, states = res.get_observed_graph()
        
        ref_values = np.array([
            [   0.,  117., 3887.,    0.,  901.,    0.,    0.,    0.],
            [2511.,    0.,    0.,   18.,    0.,   37.,    0.,    0.],
            [ 514.,    0.,    0.,   11.,    0.,    0., 3357.,    0.],
            [   0.,   34.,   17.,    0.,    0.,    0.,    0.,    8.],
            [1716.,    0.,    0.,    0.,    0., 1779.,   71.,    0.],
            [   0., 2449.,    0.,    0.,  489.,    0.,    0.,    9.],
            [   0.,    0.,   12.,    0., 2200.,    0.,    0., 1217.],
            [   0.,    0.,    0.,   30.,    0., 1191.,    5.,    0.],
        ])
        ref_states = ['<nil>', 'CyclinB', 'CyclinE', 'CyclinB -- CyclinE', 'CyclinA', 'CyclinB -- CyclinA', 'CyclinE -- CyclinA', 'CyclinB -- CyclinE -- CyclinA']
        
        for state, ref_state in zip(states, ref_states):
            self.assertEqual(state, ref_state)
            
        self.assertTrue(np.allclose(values, ref_values))

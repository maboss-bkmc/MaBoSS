import cmaboss
import numpy as np
import pandas as pd
from unittest import TestCase

class TestCMaBoSS(TestCase):

    def test_load_model(self):
        net = cmaboss.MaBoSSNet(network="../examples/metastasis.bnd")
        cfg = cmaboss.MaBoSSCfg(net, "../examples/metastasis.cfg")
        sim = cmaboss.MaBoSSSim(net=net, cfg=cfg)
        res = sim.run()

    def test_load_model_cellcycle(self):
        net = cmaboss.MaBoSSNet(network="../tests/cellcycle/cellcycle.bnd")
        cfg = cmaboss.MaBoSSCfg(net, "../tests/cellcycle/cellcycle_runcfg.cfg", "../tests/cellcycle/cellcycle_runcfg-thread_6.cfg")
        sim = cmaboss.MaBoSSSim(net=net, cfg=cfg)
        res = sim.run()

    def test_simulate(self):
        sim = cmaboss.MaBoSSSim(network="../examples/metastasis.bnd", config="../examples/metastasis.cfg")
        res = sim.run()
        res.get_probtraj()
        res.get_last_probtraj()
        res.get_nodes_probtraj()
        res.get_last_nodes_probtraj()
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

        raw_res, states, _ = res.get_last_probtraj()
        for i, value in enumerate(np.nditer(raw_res)):
            self.assertAlmostEqual(value, expected[states[i]])
        
        raw_nodes_res, nodes, _ = res.get_last_nodes_probtraj()
        for i, value in enumerate(np.nditer(raw_nodes_res)):
            self.assertAlmostEqual(value, expected_nodes[nodes[i]])
        
        simfinal = cmaboss.MaBoSSSim(network="../examples/metastasis.bnd", config="../examples/metastasis.cfg")
        resfinal = simfinal.run(only_last_state=True)
    
        raw_res, states, _ = res.get_last_probtraj()
        for i, value in enumerate(np.nditer(raw_res)):
            self.assertAlmostEqual(value, expected[states[i]])
        
        raw_nodes_res, nodes, _ = res.get_last_nodes_probtraj()
        for i, value in enumerate(np.nditer(raw_nodes_res)):
            self.assertAlmostEqual(value, expected_nodes[nodes[i]])
           
    def test_load_model_str(self):
        with open("../examples/metastasis.bnd", "r") as bnd, open("../examples/metastasis.cfg", "r") as cfg:    
            sim = cmaboss.MaBoSSSim(network_str=bnd.read(),config_str=cfg.read())
            res = sim.run()
            res.get_probtraj()
            res.get_last_probtraj()
            res.get_nodes_probtraj()
            res.get_last_nodes_probtraj()
            res.get_fp_table()

    def test_load_model_str_error(self):
        with open("../examples/metastasis-error.bnd", "r") as bnd, open("../examples/metastasis.cfg", "r") as cfg:    
            with self.assertRaises(cmaboss.BNException):
                cmaboss.MaBoSSSim(network_str=bnd.read(),config_str=cfg.read())
                
                
    def test_popmaboss(self):
        sim = cmaboss.PopMaBoSSSim("../examples/popmaboss/Fork.bnd", "../examples/popmaboss/Fork.cfg")
        res = sim.run()

        table = pd.DataFrame(*res.get_probtraj())

        table_expected = pd.DataFrame(
            np.array(
                [[0.09550711, 0.02469615, 0.16469156, 0.15948447, 0.09704259,
                    0.15443514, 0.10494093, 0.00507406, 0.19412798],
                [0.33333333, 0.33333333, 0.        , 0.        , 0.33333333,
                    0.        , 0.        , 0.        , 0.        ],
                [0.33333333, 0.33333333, 0.        , 0.        , 0.33333333,
                    0.        , 0.        , 0.        , 0.        ],
                [0.33333333, 0.33333333, 0.        , 0.        , 0.33333333,
                    0.        , 0.        , 0.        , 0.        ]]
                , dtype=np.float64
            ),
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([
                '[{A -- C:2},{A -- B:1}]', '[{A -- C:3}]', '[{A:1},{A -- C:1},{A -- B:1}]', 
                '[{A:1},{A -- C:2}]', '[{A -- B:3}]', '[{A:1},{A -- B:2}]', '[{A:2},{A -- C:1}]', 
                '[{A:2},{A -- B:1}]', '[{A:3}]'
            ])
        )
        
        pd.testing.assert_frame_equal(table.sort_index(axis=1), table_expected.sort_index(axis=1), check_exact=False)

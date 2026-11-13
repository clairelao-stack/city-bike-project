"""Small feasibility tests; no Gurobi license is needed."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heuristic import plan_transfers, improve_warm_start
from warm_start import check_warm_start, summarize_start


class WarmStartTests(unittest.TestCase):
    def setUp(self):
        self.data = {'stations': [
            dict(id='A', I0=5, I_star=0, G=-5, C=10),
            dict(id='B', I0=0, I_star=5, G=5, C=10),
            dict(id='C', I0=3, I_star=0, G=-3, C=10),
            dict(id='D', I0=0, I_star=3, G=3, C=10)]}
        self.depots = [dict(truck_id='T1', depot_id='P1'), dict(truck_id='T2', depot_id='P2')]
        points = dict(A=0, B=1, C=2, D=3, P1=-1, P2=4)
        self.distances = {}
        for depot in self.depots:
            ids = ['A', 'B', 'C', 'D', depot['depot_id']]
            for i in ids:
                for j in ids:
                    if i != j:
                        self.distances[i,j,depot['truck_id']] = abs(points[i]-points[j])
        routes = {
            'T1': dict(path=['P1','A','B','P1'], transfers={'A':-5,'B':5},
                       loads={'A':5,'B':0}, distance_km=4),
            'T2': dict(path=['P2','C','D','P2'], transfers={'C':-3,'D':3},
                       loads={'C':3,'D':0}, distance_km=4)}
        self.start = summarize_start(self.data['stations'], routes, 5, 20, 1.5, 5)

    def validate(self, start, capacity=5, budget=20):
        return check_warm_start(start,self.data,self.depots,self.distances,capacity,budget,1.5,5)

    def test_known_feasible_plan(self):
        result = self.validate(self.start)
        self.assertEqual(result['total_mismatch'],0)
        self.assertEqual(result['objective'],12)

    def test_unmatched_pickups_stay_at_station(self):
        transfers, loads = plan_transfers(['A','B'], {'A':-10,'B':3}, 5)
        self.assertEqual(transfers,{'A':-3,'B':3})
        self.assertEqual(loads,{'A':3,'B':0})

    def test_delivery_before_pickup_is_rejected(self):
        start = copy.deepcopy(self.start)
        start['routes']['T1']['path']=['P1','B','A','P1']
        with self.assertRaises(ValueError):
            self.validate(start)

    def test_changed_capacity_and_budget_are_checked(self):
        with self.assertRaises(ValueError):
            self.validate(self.start,capacity=2)
        with self.assertRaises(ValueError):
            self.validate(self.start,budget=1)

    def test_repeated_station_is_rejected(self):
        start=copy.deepcopy(self.start)
        start['routes']['T1']['path'].insert(2,'A')
        with self.assertRaises(ValueError):
            self.validate(start)

    def test_search_keeps_feasible_non_worsening_plan(self):
        best, report = improve_warm_start(self.start,self.data,self.depots,self.distances,
                                          5,20,1.5,5,seconds=2,max_iterations=200)
        self.validate(best)
        self.assertLessEqual(best['objective'],self.start['objective'])
        self.assertLessEqual(report['iterations'],200)


if __name__ == '__main__':
    unittest.main()

import unittest
import pandas as pd
from fantasyTeam import Team
import numpy as np
import os
from waiverWireSimulator import WaiverWireSimulator

class Test(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        # Create a sample waiver wire dataframe
        self.waiver_wire_data = {
            'Name': ['Player A', 'Player B', 'Player C', 'Player D', 'Player E', 'Player F'],
            'Position': ['RB', 'WR', 'TE', 'QB', 'DST', 'RB'],
            'Team': ['Team A', 'Team B', 'Team C', 'Team D', 'Team E', 'Team F'],
            'ByeWeek': [5, 6, 7, 8, 9, 10],
            'AverageDraftPositionPPR': [20, 30, 40, 50, 60, 100],
            'PointsPerGame': [10, 16, 8, 14, 5, 11],
            'Status': ['Healthy', 'Healthy', 'Healthy', 'Out', 'Healthy', 'Healthy'],
            'ProjectedFantasyPoints': [10, 16, 8, 14, 5, 11],
            'FantasyPoints': [0, 0, 0, 0, 0, 0]
        }
        self.waiver_wire_df = pd.DataFrame(self.waiver_wire_data)
        self.waiver_wire_simulator = WaiverWireSimulator(self.waiver_wire_df)
        self.waiver_wire_simulator.waiver_wire.dropna()
        self.waiver_wire_simulator.waiver_wire = self.waiver_wire_simulator.waiver_wire.astype(
            {
                'Name': 'string',
                'Position': 'string',  
                'Team': 'string',
                'ByeWeek': 'Int64',
                'AverageDraftPositionPPR': 'float64',
                'PointsPerGame': 'float64',
                'Status': 'string',
                'ProjectedFantasyPoints': 'float64',
                'FantasyPoints': 'float64'
            }
        )

        # Create a sample team
        self.team = Team('Test Team', 1)
        self.team.addPickToRoster('RB', 'Player 1', 1, 10, 'Team X', 5, 10.0, 'Healthy')
        self.team.addPickToRoster('WR', 'Player 2', 2, 20, 'Team Y', 6, 12.0, 'Healthy')
        self.team.addPickToRoster('QB', 'Player 3', 3, 30, 'Team Z', 7, 11.0, 'Healthy')
        self.team.addPickToRoster('TE', 'Player 4', 4, 40, 'Team W', 8, 14.0, 'Healthy')
        self.team.addToBench('Player 5', 'RB', 5, 50, 'Team V', 9, 5.0, 'Healthy', 5.0, 0)

    def test_determineDrop(self):
        self.waiver_wire_simulator.week = 3
        for i in range(6):
            self.team.addToBench(f'Test Player {i}', 'WR', i, 10.5, 'Team A', 5, 15.0, 'Healthy', 15.0, 0)
        drop_player = self.waiver_wire_simulator.determineDrop(self.team)
        self.assertEqual(drop_player['Name'], 'Player 5')

    def test_determineAdd(self):
        self.waiver_wire_simulator.week = 3
        add_player = self.waiver_wire_simulator.determineAdd(self.team, 'RB')
        self.assertEqual(add_player['Name'], 'Player F')

    def test_shouldAddDrop(self):
        self.waiver_wire_simulator.week = 3
        result = self.waiver_wire_simulator.shouldAddDrop(self.team, 'RB')
        self.assertIsNotNone(result)
        drop_player, add_player = result
        self.assertIsNone(drop_player)
        self.assertEqual(add_player['Name'], 'Player F')

    def test_determineSwaps(self):
        self.waiver_wire_simulator.week = 3
        self.team.positionsInNeed = ['RB2', 'WR2']
        self.team.rosterStatus = 0
        swaps = self.waiver_wire_simulator.determineSwaps(self.team)
        self.assertEqual(len(swaps), 2) 
        # 2 was calculated from 2 pos in need and streaming K and not 
        # DST but since no K in wire, doesnt contribute
        drop_player1, add_player1 = swaps[0]
        drop_player2, add_player2 = swaps[1]
        self.assertIsNone(drop_player1)
        self.assertEqual(add_player1['Name'], 'Player F')
        self.assertIsNone(drop_player2)
        self.assertEqual(add_player2['Name'], 'Player B')

    def test_addPlayerToWaiverWire(self):
        self.waiver_wire_simulator.week = 3
        player_to_add = self.team.roster[self.team.roster['Name'] == 'Player 5'].iloc[0]
        self.waiver_wire_simulator.addPlayerToWaiverWire(player_to_add)
        self.assertIn('Player 5', self.waiver_wire_simulator.waiver_wire['Name'].values)

    def test_removePlayerFromWaiverWire(self):
        self.waiver_wire_simulator.week = 3
        self.waiver_wire_simulator.removePlayerFromWaiverWire({'Name': 'Player A'})
        self.assertNotIn('Player A', self.waiver_wire_simulator.waiver_wire['Name'].values)

    def test_addDrop(self):
        self.waiver_wire_simulator.week = 3
        for i in range(6):
            self.team.addToBench(f'Test Player {i}', 'WR', i, 10.5, 'Team A', 5, 15.0, 'Healthy')
        add_player = self.waiver_wire_simulator.determineAdd(self.team, 'WR')
        drop_player = self.waiver_wire_simulator.determineDrop(self.team)
        self.team.waiverwirestatus = 1
        self.waiver_wire_simulator.addDrop(self.team, add_player, drop_player)
        self.assertIn(add_player['Name'], self.team.roster['Name'].values)
        self.assertNotIn(drop_player['Name'], self.team.roster['Name'].values)

if __name__ == '__main__':
    unittest.main()
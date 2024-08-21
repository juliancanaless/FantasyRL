import unittest
import pandas as pd
from fantasyTeam import Team
pd.options.mode.chained_assignment = None 


class TestTeam(unittest.TestCase):

    def setUp(self):
        self.team = Team('Test Team', 1)

    def test_initialization(self):
        self.assertEqual(self.team.name, 'Test Team')
        self.assertEqual(self.team.draftPick, 1)
        self.assertEqual(self.team.roster.shape, (16, 11))
        self.assertTrue('FantasyPosition' in self.team.roster.columns)

    def test_draft_strategy(self):
        qb_strat, rb_strat, wr_strat, te_strat, k_strat, dst_strat = self.team.strategy
        self.assertIn(qb_strat, ['EarlyRoundQB', 'MidRoundQB', 'LateRoundQB'])
        self.assertIn(rb_strat, ['ZeroRB', 'HeroRB', 'None'])
        self.assertIn(wr_strat, ['ZeroWR', 'None'])
        self.assertIn(te_strat, ['EarlyRoundTE', 'MidRoundTE', 'LateRoundTE'])
        self.assertIn(k_strat, ['EarlyK', 'MidK', 'LateK'])
        self.assertIn(dst_strat, ['EarlyDST', 'MidDST', 'LateDST'])

    def test_add_pick_to_roster(self):
        self.team.addPickToRoster('RB', 'Test Player', 1, 10.5, 'Team A', 5, 15.0, 'ACT')
        self.assertEqual(self.team.roster.loc[self.team.roster['FantasyPosition'] == 'RB1', 'Name'].values[0], 'Test Player')

    def test_add_pick_to_roster_mock_draft(self):
        self.team.addPickToRoster('WR', 'C. Lamb', 9, 2, 'Cowboys', 7, 0, 'ACT')
        self.team.addPickToRoster('RB', 'J. Gibbs', 12, 13, 'Lions', 5, 0, 'ACT')
        self.team.addPickToRoster('RB', 'J. Jacobs', 29, 30, 'Packers', 10, 0, 'ACT')
        self.team.addPickToRoster('WR', 'N. Collins', 32, 23, 'Texans', 14, 0, 'ACT')
        self.team.addPickToRoster('WR', 'D. Smith', 49, 43, 'Eagles', 5, 0, 'ACT')
        self.team.addPickToRoster('QB', 'C. Stroud', 52, 44, 'Texans', 14, 0, 'ACT')
        self.team.addPickToRoster('TE', 'K. Pitts', 69, 64, 'Falcons', 12, 0, 'ACT')
        self.team.addPickToRoster('WR', 'R. Rice', 72, 78, 'Chiefs', 6, 0, 'ACT')
        self.team.addPickToRoster('WR', 'J. Reed', 89, 74, 'Packers', 10, 0, 'ACT')
        self.team.addPickToRoster('RB', 'E. Elliot', 92, 130, 'Cowboys', 7, 0, 'ACT')
        self.team.addPickToRoster('RB', 'A. Ekeler', 109, 85, 'Commanders', 14, 0, 'ACT')
        self.team.addPickToRoster('DST', '49ers D/ST', 112, 212, '49ers', 9, 0, 'ACT')
        self.team.addPickToRoster('QB', 'T. Lawrence', 129, 102, 'Jaguars', 12, 0, 'ACT')
        self.team.addPickToRoster('K', 'B. Aubrey', 132, 121, 'Cowboys', 7, 0, 'ACT')
        self.team.addPickToRoster('TE', 'C. Kmet', 149, 136, 'Bears', 7, 0, 'ACT')
        self.team.addPickToRoster('WR', 'C. Samuel', 152, 106, 'Bills', 12, 0, 'ACT')
        
        df = pd.DataFrame(
            {
                'FantasyPosition': ['QB', 'RB1', 'RB2', 'WR1', 'WR2', 'TE', 'FLEX', 'K', 'DST',
                                    'BE1', 'BE2', 'BE3', 'BE4', 'BE5', 'BE6', 'BE7'],
                'Name': ['C. Stroud', 'J. Gibbs', 'J. Jacobs', 'C. Lamb', 'N. Collins', 'K. Pitts', 'D. Smith', 'B. Aubrey', '49ers D/ST',
                         'R. Rice', 'J. Reed', 'E. Elliot', 'A. Ekeler', 'T. Lawrence', 'C. Kmet', 'C. Samuel'],
                'Position': ['QB', 'RB', 'RB', 'WR', 'WR', 'TE', 'WR', 'K', 'DST',
                             'WR', 'WR', 'RB', 'RB', 'QB', 'TE', 'WR'],
                'PickNumber': [52, 12, 29, 9, 32, 69, 49, 132, 112, 72, 89, 92, 109, 129, 149, 152],
                'AverageDraftPositionPPR': [44, 13, 30, 2, 23, 64, 43, 121, 212, 78, 74, 130, 85, 102, 136, 106],
                'Team': ['Texans', 'Lions', 'Packers', 'Cowboys', 'Texans', 'Falcons', 'Eagles', 'Cowboys', '49ers',
                         'Chiefs', 'Packers', 'Cowboys', 'Commanders', 'Jaguars', 'Bears', 'Bills'],
                'ByeWeek': [14, 5, 10, 7, 14, 12, 5, 7, 9, 6, 10, 7, 14, 12, 7, 12],
                'PointsPerGame': [0.0] * 16,
                'Status': ['ACT'] * 16,
                'ProjectedFantasyPoints': [None] * 16,
                'FantasyPoints': [None] * 16
            }
        )

        df = df.astype(
            {
                'FantasyPosition': 'string',
                'Name': 'string',
                'Position': 'string',
                'PickNumber': 'Int64',  
                'AverageDraftPositionPPR': 'float64',
                'Team': 'string',
                'ByeWeek': 'Int64',
                'PointsPerGame': 'float64',  
                'Status': 'string',
                'ProjectedFantasyPoints': 'float64',
                'FantasyPoints': 'float64'
            }
        ) 
        
        pd.testing.assert_frame_equal(self.team.roster.reset_index(drop=True), df)

    def test_add_to_bench(self):
        self.team.addToBench('Test Player', 'WR', 1, 10.5, 'Team A', 5, 15.0, 'ACT')
        self.assertEqual(self.team.roster.loc[self.team.roster['FantasyPosition'] == 'BE1', 'Name'].values[0], 'Test Player')

    def test_get_bench(self):
        self.team.addToBench('Test Player', 'WR', 1, 10.5, 'Team A', 5, 15.0, 'ACT')
        bench = self.team.getBench()
        self.assertEqual(bench.iloc[0]['Name'], 'Test Player')

    def test_get_bench_mock_draft(self):
        self.team.addPickToRoster('WR', 'C. Lamb', 9, 2, 'Cowboys', 7, 0, 'ACT')
        self.team.addPickToRoster('RB', 'J. Gibbs', 12, 13, 'Lions', 5, 0, 'ACT')
        self.team.addPickToRoster('RB', 'J. Jacobs', 29, 30, 'Packers', 10, 0, 'ACT')
        self.team.addPickToRoster('WR', 'N. Collins', 32, 23, 'Texans', 14, 0, 'ACT')
        self.team.addPickToRoster('WR', 'D. Smith', 49, 43, 'Eagles', 5, 0, 'ACT')
        self.team.addPickToRoster('QB', 'C. Stroud', 52, 44, 'Texans', 14, 0, 'ACT')
        self.team.addPickToRoster('TE', 'K. Pitts', 69, 64, 'Falcons', 12, 0, 'ACT')
        self.team.addPickToRoster('WR', 'R. Rice', 72, 78, 'Chiefs', 6, 0, 'ACT')
        self.team.addPickToRoster('WR', 'J. Reed', 89, 74, 'Packers', 10, 0, 'ACT')
        self.team.addPickToRoster('RB', 'E. Elliot', 92, 130, 'Cowboys', 7, 0, 'ACT')
        self.team.addPickToRoster('RB', 'A. Ekeler', 109, 85, 'Commanders', 14, 0, 'ACT')
        self.team.addPickToRoster('DST', '49ers D/ST', 112, 212, '49ers', 9, 0, 'ACT')
        self.team.addPickToRoster('QB', 'T. Lawrence', 129, 102, 'Jaguars', 12, 0, 'ACT')
        self.team.addPickToRoster('K', 'B. Aubrey', 132, 121, 'Cowboys', 7, 0, 'ACT')
        self.team.addPickToRoster('TE', 'C. Kmet', 149, 136, 'Bears', 7, 0, 'ACT')
        self.team.addPickToRoster('WR', 'C. Samuel', 152, 106, 'Bills', 12, 0, 'ACT')
        bench = self.team.getBench()
        expected_bench = pd.DataFrame(
            {
                'FantasyPosition': ['BE1', 'BE2', 'BE3', 'BE4', 'BE5', 'BE6', 'BE7'],
                'Name': ['R. Rice', 'J. Reed', 'E. Elliot', 'A. Ekeler', 'T. Lawrence', 'C. Kmet', 'C. Samuel'],
                'Position': ['WR', 'WR', 'RB', 'RB', 'QB', 'TE', 'WR'],
                'PickNumber': [72, 89, 92, 109, 129, 149, 152],
                'AverageDraftPositionPPR': [78, 74, 130, 85, 102, 136, 106],
                'Team': ['Chiefs', 'Packers', 'Cowboys', 'Commanders', 'Jaguars', 'Bears', 'Bills'],
                'ByeWeek': [6, 10, 7, 14, 12, 7, 12],
                'PointsPerGame': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'Status': ['ACT', 'ACT', 'ACT', 'ACT', 'ACT', 'ACT', 'ACT'],
                'ProjectedFantasyPoints': [None] * 7,
                'FantasyPoints': [None] * 7
            }
        )
        
        expected_bench = expected_bench.astype(
            {
                'FantasyPosition': 'string',
                'Name': 'string',
                'Position': 'string',
                'PickNumber': 'Int64',  
                'AverageDraftPositionPPR': 'float64',
                'Team': 'string',
                'ByeWeek': 'Int64',
                'PointsPerGame': 'float64',  
                'Status': 'string',
                'ProjectedFantasyPoints': 'float64',
                'FantasyPoints': 'float64'
            }
        )
        # Check if the bench DataFrame matches the expected DataFrame
        pd.testing.assert_frame_equal(bench.reset_index(drop=True), expected_bench)
        

    def test_drop_player(self):
        self.team.addPickToRoster('WR', 'C. Lamb', 9, 2, 'Cowboys', 7, 0, 'ACT')
        self.team.addPickToRoster('RB', 'J. Gibbs', 12, 13, 'Lions', 5, 0, 'ACT')
        self.team.addPickToRoster('RB', 'J. Jacobs', 29, 30, 'Packers', 10, 0, 'ACT')
        self.team.addPickToRoster('WR', 'N. Collins', 32, 23, 'Texans', 14, 0, 'ACT')
        self.team.addPickToRoster('WR', 'D. Smith', 49, 43, 'Eagles', 5, 0, 'ACT')
        self.team.dropPlayer('C. Lamb')
        self.team.dropPlayer('J. Jacobs')
        self.assertTrue(pd.isna(self.team.roster.loc[self.team.roster['FantasyPosition'] == 'WR1', 'Name'].values[0]))
        self.assertTrue(pd.isna(self.team.roster.loc[self.team.roster['FantasyPosition'] == 'RB2', 'Name'].values[0]))

    def test_is_bench_full(self):
        for i in range(7):
            self.team.addToBench(f'Test Player {i}', 'WR', i, 10.5, 'Team A', 5, 15.0, 'ACT')
        self.assertTrue(self.team.isBenchFull())
        self.team.dropPlayer('Test Player 1')
        self.assertFalse(self.team.isBenchFull())

    def test_injured_player(self):
        self.team.addPickToRoster('RB', 'Test Player', 1, 10.5, 'Team A', 5, 15.0, 'Out')
        self.assertTrue(self.team.injuredPlayer())

    def test_injured_active_players(self):
        self.team.addPickToRoster('RB', 'Test Player 1', 1, 10.5, 'Team A', 5, 15.0, 'Out')
        self.team.addPickToRoster('RB', 'Test Player 2', 1, 10.5, 'Team A', 5, 15.0, 'Out')
        self.team.addPickToRoster('WR', 'Test Player 3', 1, 10.5, 'Team A', 5, 15.0, 'ACT')
        injured_players = self.team.injuredActivePlayers()
        self.assertEqual(injured_players.iloc[0]['Name'], 'Test Player 1')
        self.assertEqual(injured_players.iloc[1]['Name'], 'Test Player 2')

    def test_swap_players_bench_active(self):
        '''
        Tests case 1, active same pos <-> bench same pos
        '''
        self.team.addPickToRoster('RB', 'Player 1', 1, 10.5, 'Team A', 5, 15.0, 'ACT')
        self.team.addToBench('Player 2', 'RB', 2, 12.5, 'Team B', 6, 13.0, 'ACT')
        player1_row = self.team.roster[self.team.roster['Name'] == 'Player 1'].iloc[0]
        player2_row = self.team.roster[self.team.roster['Name'] == 'Player 2'].iloc[0]
        self.team.swapPlayers(player1_row, player2_row)
        self.assertEqual(self.team.roster.loc[self.team.roster['FantasyPosition'] == 'RB1', 'Name'].values[0], 'Player 2')
        self.assertEqual(self.team.roster.loc[self.team.roster['FantasyPosition'] == 'BE1', 'Name'].values[0], 'Player 1')

    def test_swap_players2(self):
        '''
        Tests case 2 and 3
        '''
        self.team.addPickToRoster('WR', 'C. Lamb', 9, 2, 'Cowboys', 7, 0, 'ACT')
        self.team.addPickToRoster('RB', 'J. Gibbs', 12, 13, 'Lions', 5, 0, 'ACT')
        self.team.addPickToRoster('RB', 'J. Jacobs', 29, 30, 'Packers', 10, 0, 'ACT')
        self.team.addPickToRoster('WR', 'N. Collins', 32, 23, 'Texans', 14, 0, 'ACT')
        self.team.addPickToRoster('WR', 'D. Smith', 49, 43, 'Eagles', 5, 0, 'ACT')
        self.team.addPickToRoster('QB', 'C. Stroud', 52, 44, 'Texans', 14, 0, 'ACT')
        self.team.addPickToRoster('TE', 'K. Pitts', 69, 64, 'Falcons', 12, 0, 'ACT')
        self.team.addPickToRoster('WR', 'R. Rice', 72, 78, 'Chiefs', 6, 0, 'ACT')
        self.team.addPickToRoster('WR', 'J. Reed', 89, 74, 'Packers', 10, 0, 'ACT')
        self.team.addPickToRoster('RB', 'E. Elliot', 92, 130, 'Cowboys', 7, 0, 'ACT')
        self.team.addPickToRoster('RB', 'A. Ekeler', 109, 85, 'Commanders', 14, 0, 'ACT')
        self.team.addPickToRoster('DST', '49ers D/ST', 112, 212, '49ers', 9, 0, 'ACT')
        self.team.addPickToRoster('QB', 'T. Lawrence', 129, 102, 'Jaguars', 12, 0, 'ACT')
        self.team.addPickToRoster('K', 'B. Aubrey', 132, 121, 'Cowboys', 7, 0, 'ACT')
        self.team.addPickToRoster('TE', 'C. Kmet', 149, 136, 'Bears', 7, 0, 'ACT')
        self.team.addPickToRoster('WR', 'C. Samuel', 152, 106, 'Bills', 12, 0, 'ACT')
        player1_row = self.team.roster[self.team.roster['FantasyPosition'] == 'FLEX'].iloc[0]
        player2_row = self.team.roster[self.team.roster['Name'] == 'C. Kmet'].iloc[0]
        self.team.swapPlayers(player1_row, player2_row)
        self.assertEqual(self.team.roster.loc[self.team.roster['FantasyPosition'] == 'FLEX', 'Name'].values[0], 'C. Kmet')
        self.assertEqual(self.team.roster.loc[self.team.roster['FantasyPosition'] == 'BE6', 'Name'].values[0], 'D. Smith')
        self.team.swapPlayers(player1_row, player2_row)
        self.assertEqual(self.team.roster.loc[self.team.roster['FantasyPosition'] == 'FLEX', 'Name'].values[0], 'D. Smith')
        self.assertEqual(self.team.roster.loc[self.team.roster['FantasyPosition'] == 'BE6', 'Name'].values[0], 'C. Kmet')
        player1_row = self.team.roster[self.team.roster['FantasyPosition'] == 'FLEX'].iloc[0]
        player2_row = self.team.roster[self.team.roster['Name'] == 'T. Lawrence'].iloc[0]
        self.assertFalse(self.team.swapPlayers(player1_row, player2_row))
        player1_row = self.team.roster[self.team.roster['FantasyPosition'] == 'WR1'].iloc[0]
        player2_row = self.team.roster[self.team.roster['FantasyPosition'] == 'WR2'].iloc[0]
        self.team.swapPlayers(player1_row, player2_row)
        self.assertEqual(self.team.roster.loc[self.team.roster['FantasyPosition'] == 'WR1', 'Name'].values[0], 'N. Collins')
        self.assertEqual(self.team.roster.loc[self.team.roster['FantasyPosition'] == 'WR2', 'Name'].values[0], 'C. Lamb')


    def test_update_roster(self):
        self.team.addPickToRoster('WR', 'C. Lamb', 9, 2, 'Cowboys', 7, 22, 'ACT')
        self.team.addPickToRoster('RB', 'J. Gibbs', 12, 13, 'Lions', 5, 15, 'ACT')
        self.team.addPickToRoster('RB', 'J. Jacobs', 29, 30, 'Packers', 10, 18, 'ACT')
        self.team.addPickToRoster('WR', 'N. Collins', 32, 23, 'Texans', 14, 23, 'Out')
        self.team.addPickToRoster('WR', 'D. Smith', 49, 43, 'Eagles', 5, 16, 'ACT')
        self.team.addPickToRoster('QB', 'C. Stroud', 52, 44, 'Texans', 14, 22, 'Out')
        self.team.addPickToRoster('TE', 'K. Pitts', 69, 64, 'Falcons', 12, 15, 'ACT')
        self.team.addPickToRoster('WR', 'R. Rice', 72, 78, 'Chiefs', 6, 0, 'Out')
        self.team.addPickToRoster('WR', 'J. Reed', 89, 74, 'Packers', 10, 12, 'ACT')
        self.team.addPickToRoster('RB', 'E. Elliot', 92, 130, 'Cowboys', 7, 13, 'ACT')
        self.team.addPickToRoster('RB', 'A. Ekeler', 109, 85, 'Commanders', 14, 16, 'ACT')
        self.team.addPickToRoster('DST', '49ers D/ST', 112, 212, '49ers', 9, 10, 'ACT')
        self.team.addPickToRoster('QB', 'T. Lawrence', 129, 102, 'Jaguars', 12, 19, 'ACT')
        self.team.addPickToRoster('K', 'B. Aubrey', 132, 121, 'Cowboys', 7, 12, 'ACT')
        self.team.addPickToRoster('TE', 'C. Kmet', 149, 136, 'Bears', 7, 20, 'ACT')
        self.team.addPickToRoster('WR', 'C. Samuel', 152, 106, 'Bills', 12, 15, 'ACT')
        self.team.roster['ProjectedFantasyPoints'] = 0.0
        self.team.currentWeek = 4
        self.team.updateRoster()
        self.assertEqual(self.team.roster.loc[self.team.roster['FantasyPosition'] == 'RB1', 'Name'].values[0], 'J. Jacobs')
        self.assertEqual(self.team.roster.loc[self.team.roster['FantasyPosition'] == 'RB2', 'Name'].values[0], 'A. Ekeler')
        self.assertEqual(self.team.roster.loc[self.team.roster['FantasyPosition'] == 'WR1', 'Name'].values[0], 'C. Lamb')
        self.assertEqual(self.team.roster.loc[self.team.roster['FantasyPosition'] == 'WR2', 'Name'].values[0], 'D. Smith')
        self.assertEqual(self.team.roster.loc[self.team.roster['FantasyPosition'] == 'TE', 'Name'].values[0], 'C. Kmet')
        self.assertEqual(self.team.roster.loc[self.team.roster['FantasyPosition'] == 'QB', 'Name'].values[0], 'T. Lawrence')
        self.assertEqual(self.team.roster.loc[self.team.roster['FantasyPosition'] == 'FLEX', 'Name'].values[0], 'C. Samuel')
        self.assertEqual(self.team.roster.loc[self.team.roster['FantasyPosition'] == 'K', 'Name'].values[0], 'B. Aubrey')
        self.assertEqual(self.team.roster.loc[self.team.roster['FantasyPosition'] == 'DST', 'Name'].values[0], '49ers D/ST')
        self.team.roster.loc[self.team.roster['Name'] == 'J. Gibbs', 'ProjectedFantasyPoints'] = 19
        self.team.updateRoster()
        self.assertEqual(self.team.roster.loc[self.team.roster['FantasyPosition'] == 'RB1', 'Name'].values[0], 'J. Gibbs')

if __name__ == '__main__':
    unittest.main()

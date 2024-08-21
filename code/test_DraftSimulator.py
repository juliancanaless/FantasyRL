import unittest
import pandas as pd
from draftSimulator import DraftSimulator
from fantasyTeam import Team
import numpy as np
import os

class Test(unittest.TestCase):

    def setUp(self):
        np.random.seed(42) 

        script_dir = os.path.dirname(__file__)
        data_dir = os.path.join(script_dir, '..', 'data')
        data_file_path = os.path.join(data_dir, 'ppr-adp-2024.csv')
        weekly_stats_path = os.path.join(data_dir, 'weekly-stats-2023.csv')

        myTeam = Team("MahomiBear", 3)
        leagueMembers = [
            ('Gridiron Gladiators', 10),
            ('Pigskin Pirates', 2),
            ('Touchdown Titans', 4),
            ('Blitzkrieg Bandits', 1),
            ('End Zone Avengers', 7),
            ('Hail Mary Heroes', 5),
            ('Fumble Force', 9),
            ('Red Zone Renegades', 6),
            ('Sack Masters', 8)
        ]

        self.draft = DraftSimulator(path=data_file_path, myTeam=myTeam,
                                    leagueMembers=leagueMembers, leagueSize=len(leagueMembers) + 1,
                                    numRounds=16, model=None, stats=weekly_stats_path)
        
    def test_initialization(self):
        self.assertEqual(self.draft.me.name, 'MahomiBear')
        self.assertEqual(self.draft.teams[0].name, 'Blitzkrieg Bandits')
        self.assertEqual(self.draft.teams[5].name, 'Red Zone Renegades')
        self.assertEqual(self.draft.teams[9].name, 'Gridiron Gladiators')
        self.assertEqual(self.draft.draftPicksBoard.iloc[10]['team'], 'Gridiron Gladiators')
        self.assertEqual(self.draft.draftPicksBoard.iloc[13]['team'], 'End Zone Avengers')
        self.assertEqual(self.draft.draftBoard.iloc[0]['Name'], 'Christian McCaffrey')

    def test_determineRequiredPositions(self):
        self.draft.teams[0].posFreqMap = {'QB': 2, 'RB': 5, 'WR': 5, 'TE': 2, 'DST': 0, 'K': 0}
        remaining_rounds = 2
        required_positions_set = self.draft._determineRequiredPositions(self.draft.teams[0], remaining_rounds)
        expected_set = {'DST', 'K'}
        self.assertEqual(required_positions_set, expected_set)

        self.draft.teams[0].posFreqMap = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 0, 'DST': 0, 'K': 0}
        remaining_rounds = 11
        required_positions_set = self.draft._determineRequiredPositions(self.draft.teams[0], remaining_rounds)
        expected_set = set()
        self.assertEqual(required_positions_set, expected_set)

    def test_selectTopPlayerByPositionSet(self):

        position_set = {'RB'}
        player_name, player_team, player_position, bye_week, status, avgadp = self.draft._selectTopPlayerByPositionSet(position_set, self.draft.teams[0], 'early')
        
        self.assertEqual(player_name, 'Christian McCaffrey')
        self.assertEqual(player_position, 'RB')
        self.assertEqual(player_team, 'SF')
        self.assertEqual(bye_week, 9)
        self.assertEqual(status, 'ACT')
        self.assertEqual(avgadp, 1)
        self.assertFalse(self.draft.draftBoard.loc[self.draft.draftBoard['Name'] == 'Christian McCaffrey', 'Available'].values[0])

        position_set = {'WR', 'RB'}
        player_name, player_team, player_position, bye_week, status, avgadp = self.draft._selectTopPlayerByPositionSet(position_set, self.draft.teams[1], 'early')
        
        self.assertEqual(player_name, 'CeeDee Lamb')
        self.assertEqual(player_position, 'WR')
        self.assertEqual(player_team, 'DAL')
        self.assertEqual(bye_week, 7)
        self.assertEqual(status, 'ACT')
        self.assertEqual(avgadp, 2)
        self.assertFalse(self.draft.draftBoard.loc[self.draft.draftBoard['Name'] == 'Christian McCaffrey', 'Available'].values[0])
        self.assertFalse(self.draft.draftBoard.loc[self.draft.draftBoard['Name'] == 'CeeDee Lamb', 'Available'].values[0])


    def test_full_draft_process(self):
        # Run the full draft simulation
        draftTeams = self.draft.teams.copy()
        for _ in range(self.draft.numRounds):
            for team in draftTeams:
                # add conditional for 'me' when model created
                response = self.draft.otherTeamSelection(team)
                if response:
                    player_name, playerTeam, position, byeWeek, status, avgadp = response
                    team.addPickToRoster(position, player_name, self.draft.currentPick,
                                            avgadp, playerTeam, byeWeek, 0, status)
                else:
                    print('error occured in selecting draft pick')
                self.draft.currentPick += 1
            self.draft.currentRound += 1
            draftTeams.reverse()

        # Check that the draft board is updated correctly
        team_list = self.draft.draftPicksBoard.head(20)['team'].tolist()
        draftOrdering = [team.name for team in self.draft.teams]
        reverseOrder = reversed(draftOrdering)
        for teamName in reverseOrder:
            draftOrdering.append(teamName)
        self.assertEqual(draftOrdering, team_list)

        # Check that each team's roster has the correct number of players
        for team in self.draft.teams:
            rostered_players = team.roster['Name'].dropna().tolist()
            self.assertEqual(len(rostered_players), 16)

        # Check the draft picks board for correct order and picks
        self.assertEqual(self.draft.draftPicksBoard.shape[0], self.draft.numRounds * self.draft.numTeams)

        # Check that the snake pattern is followed correctly
        for round_num in range(1, self.draft.numRounds + 1):
            round_picks = self.draft.draftPicksBoard[self.draft.draftPicksBoard['draftRound'] == round_num]
            if round_num % 2 == 1:  # Odd rounds
                expected_order = [team.name for team in self.draft.teams]
            else:  # Even rounds (reversed order)
                expected_order = [team.name for team in reversed(self.draft.teams)]
            self.assertEqual(round_picks['team'].tolist(), expected_order)


if __name__ == '__main__':
    unittest.main()


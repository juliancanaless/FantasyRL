import pandas as pd
import numpy as np
from waiverWireSimulator import WaiverWireSimulator
from fantasyTeam import Team
from typing import Dict
from typing import List

PLAYOFF_START_WEEK = 14

class SeasonSimulator:

    def __init__(self, teams: list[Team], weekly_info_path: str, waiver_wire_df: pd.DataFrame):
        """
        Initialize the season simulator with teams, their drafted rosters, and points data.
        
        :param teams: List of team names
        :param draft_rosters: Dictionary where keys are team names and values are dataframes of drafted players
        :param weekly_info_df: Dataframe containing the points scored by each player each week
        """
        self.teams = teams
        self.teamNames = [team.name for team in self.teams]
        if len(self.teams) == 8:
            self.numWeeks = 16
        else:
            self.numWeeks = 17
        self.num_teams = len(teams)
        if self.num_teams == 8:
            self.playoff_style = '4Player'
            self.toilet_bowl_style = '4Player'
        elif self.num_teams == 10:
            self.playoff_style = '6Player'
            self.toilet_bowl_style = '4Player'
        elif self.num_teams == 12:
            self.playoff_style = '6Player'
            self.toilet_bowl_style = '6Player'
        else:
            raise ValueError("Only 8, 10, and 12 team leagues are supported")
        self.weekly_info_df = pd.read_csv(weekly_info_path)
        self.standings = self._create_standings_df()
        self.matchups = self._create_matchups()
        self.playoff_standings = self._create_playoff_standings_df()
        self.waiverWire = WaiverWireSimulator(waiver_wire_df)
        self.waiverWireOrdering = self.standings['Team'].to_numpy()[::-1]
        self.team_dict: Dict[str, Team] = {team.name: team for team in self.teams}

        # playoff/toilet bowl instance variables
        self.playoff_teams = []
        self.eliminated_teams = []
        self.week15_points = {}
        self.quarterfinal_teams = []
        self.semifinal_teams = []
        
        self.toilet_bowl_teams = []
        self.eliminated_toilet_bowl_teams = []
        self.week15_points_toilet_bowl = {}
        self.toilet_bowl_quarterfinal_teams = []
        self.toilet_bowl_semifinal_teams = []
        self.week16_points = {}
        self.week16_points_toilet_bowl = {}

    def _create_playoff_standings_df(self):
        '''
        Create a playoff standings that contains final winners and losers.
        '''
        standings = pd.DataFrame({
            'Team': self.teamNames,
            'Rank': np.full(shape=len(self.teamNames), fill_value=99, dtype=int)
        })
        return standings

    def _create_standings_df(self):
        """
        Create a standings dataframe to track wins, losses, and other stats.
        """
        standings = pd.DataFrame({
            'Team': self.teamNames,
            'Wins': np.zeros(len(self.teamNames), dtype=int),
            'Losses': np.zeros(len(self.teamNames), dtype=int),
            'Points For': np.zeros(len(self.teamNames), dtype=float),
            'Points Against': np.zeros(len(self.teamNames), dtype=float)
        })
        return standings
    
    def _create_matchups(self):
        """
        Create a schedule of matchups for the season.
        Each team plays a different pre-determined team each week.
        The matchups repeat until playoffs start at week 14.
        """
        num_teams = len(self.teamNames)
        if num_teams % 2 != 0:
            raise ValueError("Number of teams must be even for round-robin scheduling.")

        # Generate round-robin schedule
        rounds = []
        teams = self.teamNames[:]
        for _ in range(num_teams - 1):
            round = []
            for j in range(num_teams // 2):
                round.append((teams[j], teams[num_teams - 1 - j]))
            teams.insert(1, teams.pop())  # Rotate teams
            rounds.append(round)

        # Create the matchups dictionary
        matchups = {week: [] for week in range(1, self.numWeeks + 1)}

        # Fill in the matchups for each week, repeating rounds until playoffs start
        week = 1
        while week < PLAYOFF_START_WEEK:
            for round in rounds:
                if week >= PLAYOFF_START_WEEK:
                    break
                matchups[week] = round
                week += 1

        return matchups
    
    def _update_points_per_game(self, week: int):
        '''
        Update running totals for waiver wire and roster
        '''
        filtered_weekly_info_df = self.weekly_info_df[(self.weekly_info_df['Week'] <= week) & (self.weekly_info_df['Status'] != 'Out')]
        filtered_weekly_info_df = filtered_weekly_info_df.sort_values(by=['Name', 'Week'])

        # Calculate the rolling average points per game for each player with a window of 3
        filtered_weekly_info_df['PointsPerGame'] = filtered_weekly_info_df.groupby('Name')['FantasyPoints'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)

        # Calculate the number of games played by each player
        games_played = filtered_weekly_info_df.groupby('Name').size().reset_index(name='games_played')

        # Get the latest PointsPerGame for each player up to the given week
        latest_points_per_game = filtered_weekly_info_df.groupby('Name').tail(1)[['Name', 'PointsPerGame']]

        # Merge the games played into the latest_points_per_game DataFrame
        latest_points_per_game = latest_points_per_game.merge(games_played, on='Name', how='left')

        # Set PointsPerGame to 0 for players with fewer than 3 games
        latest_points_per_game.loc[latest_points_per_game['games_played'] < 3, 'PointsPerGame'] = 0

        # Replace NaN values with 0.0
        latest_points_per_game['PointsPerGame'] = latest_points_per_game['PointsPerGame'].fillna(0.0)

        # Merge the running averages into the waiver wire DataFrame
        updated_waiver_wire_df = self.waiverWire.waiver_wire.merge(latest_points_per_game[['Name', 'PointsPerGame']], on='Name', how='left', suffixes=('', '_new'))

        # Update the PointsPerGame column
        updated_waiver_wire_df['PointsPerGame'] = updated_waiver_wire_df['PointsPerGame_new']
        updated_waiver_wire_df.drop(columns=['PointsPerGame_new'], inplace=True)
        self.waiverWire.waiver_wire = updated_waiver_wire_df

        for team in self.teams:
            roster = team.roster
            updated_rosters = roster.merge(latest_points_per_game[['Name', 'PointsPerGame']], on='Name', how='left', suffixes=('', '_new'))
            updated_rosters['PointsPerGame'] = updated_rosters['PointsPerGame_new'].fillna(0.0)
            updated_rosters.drop(columns=['PointsPerGame_new'], inplace=True)
            team.roster = updated_rosters
        
    def update_player_status_points(self, week: int):
        '''
        Updates the statuses of all players in waiver wire and rosters for upcoming week.
        '''
        weekly_info_current_week = self.weekly_info_df[self.weekly_info_df['Week'] == week]

        # Update waiver wire
        for idx, player in self.waiverWire.waiver_wire.iterrows():
            update_info = weekly_info_current_week[weekly_info_current_week['Name'] == player['Name']]
            if player['ByeWeek'] == week:
                self.waiverWire.waiver_wire.at[idx, 'ProjectedFantasyPoints'] = 0.0
                self.waiverWire.waiver_wire.at[idx, 'Status'] = 'INA'
            elif not update_info.empty:
                self.waiverWire.waiver_wire.at[idx, 'Status'] = update_info.iloc[0]['Status']
                self.waiverWire.waiver_wire.at[idx, 'ProjectedFantasyPoints'] = update_info.iloc[0]['ProjectedFantasyPoints']
            else:
                self.waiverWire.waiver_wire.at[idx, 'ProjectedFantasyPoints'] = 0.0
                self.waiverWire.waiver_wire.at[idx, 'Status'] = 'INA'

        # Update team rosters
        for team in self.teams:
            for idx, player in team.roster.iterrows():
                update_info = weekly_info_current_week[weekly_info_current_week['Name'] == player['Name']]
                if player['ByeWeek'] == week:
                    team.roster.at[idx, 'ProjectedFantasyPoints'] = 0.0
                    team.roster.at[idx, 'Status'] = 'INA'
                elif not update_info.empty:
                    team.roster.at[idx, 'Status'] = update_info.iloc[0]['Status']
                    team.roster.at[idx, 'ProjectedFantasyPoints'] = update_info.iloc[0]['ProjectedFantasyPoints']
                else:
                    team.roster.at[idx, 'ProjectedFantasyPoints'] = 0.0
                    team.roster.at[idx, 'Status'] = 'INA'

    def simulate_week(self, week: int):
        """
        Simulate a given week of matchups.
        """
        weekly_matchups = self.matchups[week]
        week_points = self.weekly_info_df[self.weekly_info_df['Week'] == week]
        
        for team1, team2 in weekly_matchups:
            team1_points = self._calculate_team_points(team1, week_points)
            team2_points = self._calculate_team_points(team2, week_points)
            
            self._update_standings(team1, team1_points, team2, team2_points)
            
            print(f"Week {week} - {team1} vs {team2}: {team1_points} - {team2_points}")
    
    def _calculate_team_points(self, team: Team, week_points: pd.DataFrame):
        """
        Calculate the total points for a team based on their roster and the points data for the week.
        
        :param team: Team name
        :param week_points: Dataframe of points scored by players in the given week
        :return: Total points scored by the team
        """
        teamObject = self.team_dict[team]
        team_points = 0

        updated_roster_df = teamObject.roster.merge(week_points[['Name', 'FantasyPoints']], on='Name', how='left', suffixes=('', '_new'))
        updated_roster_df['FantasyPoints'] = updated_roster_df['FantasyPoints_new'].fillna(updated_roster_df['FantasyPoints'])
        updated_roster_df.drop(columns=['FantasyPoints_new'], inplace=True)
        
        teamObject.roster = updated_roster_df

        bench = teamObject.getBench()
        bench_names = bench['Name'].tolist()
        activePlayers = teamObject.roster[~teamObject.roster['Name'].isin(bench_names)]

        for player in activePlayers['Name']:
            player_points = week_points[week_points['Name'] == player]['FantasyPoints']
            if not player_points.empty:
                team_points += player_points.values[0]
        return team_points
    
    def _update_standings(self, team1: Team, team1_points: float, team2: Team, team2_points: float):
        """
        Update the standings based on the results of a matchup.
        """
        if team1_points > team2_points:
            winner, loser = team1, team2
        else:
            winner, loser = team2, team1
        
        self.standings.loc[self.standings['Team'] == winner, 'Wins'] += 1
        self.standings.loc[self.standings['Team'] == loser, 'Losses'] += 1
        self.standings.loc[self.standings['Team'] == team1, 'Points For'] += team1_points
        self.standings.loc[self.standings['Team'] == team1, 'Points Against'] += team2_points
        self.standings.loc[self.standings['Team'] == team2, 'Points For'] += team2_points
        self.standings.loc[self.standings['Team'] == team2, 'Points Against'] += team1_points
        self.standings.sort_values(by=['Wins', 'Points For'], ascending=[False, False], inplace=True)

    def simulate_playoffs(self, week: int):
        """
        Simulate playoffs based on the standings.
        """
        if self.playoff_style == '4Player':
            self.playoff_teams = self.standings.head(4)['Team'].values.tolist()
            self.toilet_bowl_teams = self.standings.tail(4)['Team'].values.tolist()
        else:
            self.playoff_teams = self.standings.head(6)['Team'].values.tolist()
            if self.toilet_bowl_style == '4Player':
                self.toilet_bowl_teams = self.standings.tail(4)['Team'].values.tolist()
            else:
                self.toilet_bowl_teams = self.standings.tail(6)['Team'].values.tolist()

        if self.playoff_style == '4Player':
            self._simulate_4player_playoffs(week)
        else:
            self._simulate_6player_playoffs(week)

        self._simulate_toilet_bowl(week)

    def _simulate_4player_playoffs(self, week: int):
        """
        Simulate 4-team playoffs.
        """
        if week == PLAYOFF_START_WEEK:
            seed1, seed2, seed3, seed4 = self.playoff_teams
            week_points = self.weekly_info_df[self.weekly_info_df['Week'] == week]

            seed1_points = self._calculate_team_points(seed1, week_points)
            seed4_points = self._calculate_team_points(seed4, week_points)

            seed2_points = self._calculate_team_points(seed2, week_points)
            seed3_points = self._calculate_team_points(seed3, week_points)

            if seed1_points > seed4_points:
                seed1_advance = seed1
                seed4_eliminated = seed4
            else:
                seed1_advance = seed4
                seed4_eliminated = seed1

            if seed2_points > seed3_points:
                seed2_advance = seed2
                seed3_eliminated = seed3
            else:
                seed2_advance = seed3
                seed3_eliminated = seed2

            self.semifinal_teams = [seed1_advance, seed2_advance]
            self.eliminated_teams = [seed3_eliminated, seed4_eliminated]
        elif week == 15:
            seed1_advance, seed2_advance = self.semifinal_teams
            week_points = self.weekly_info_df[self.weekly_info_df['Week'] == week]

            seed1_points = self._calculate_team_points(seed1_advance, week_points)
            seed2_points = self._calculate_team_points(seed2_advance, week_points)
            self.week15_points = {seed1_advance: seed1_points, seed2_advance: seed2_points}

            seed3_eliminated, seed4_eliminated = self.eliminated_teams
            seed3_points = self._calculate_team_points(seed3_eliminated, week_points)
            seed4_points = self._calculate_team_points(seed4_eliminated, week_points)

            thirdPlace = seed3_eliminated if seed3_points > seed4_points else seed4_eliminated

            self.playoff_standings.loc[self.playoff_standings['Team'] == thirdPlace, 'Rank'] = 3
            self.playoff_standings.loc[self.playoff_standings['Team'] == (seed3_eliminated if thirdPlace != seed3_eliminated else seed4_eliminated), 'Rank'] = 4

        else:
            seed1_advance, seed2_advance = self.semifinal_teams
            week_points = self.weekly_info_df[self.weekly_info_df['Week'] == week]

            seed1_points = self._calculate_team_points(seed1_advance, week_points)
            seed2_points = self._calculate_team_points(seed2_advance, week_points)

            seed1_total_points = self.week15_points[seed1_advance] + seed1_points
            seed2_total_points = self.week15_points[seed2_advance] + seed2_points

            winner = seed1_advance if seed1_total_points > seed2_total_points else seed2_advance

            self.playoff_standings.loc[self.playoff_standings['Team'] == winner, 'Rank'] = 1
            self.playoff_standings.loc[self.playoff_standings['Team'] == (seed1_advance if winner != seed1_advance else seed2_advance), 'Rank'] = 2

            print(f"4-Player Playoff Final: {seed1_advance} vs {seed2_advance} - Winner: {winner}")

    def _simulate_6player_playoffs(self, week: int):
        """
        Simulate 6-team playoffs.
        """
        if week == PLAYOFF_START_WEEK:
            seed1, seed2, seed3, seed4, seed5, seed6 = self.playoff_teams
            week_points = self.weekly_info_df[self.weekly_info_df['Week'] == week]

            # Quarterfinals: Seed 3 vs Seed 6 and Seed 4 vs Seed 5
            seed3_points = self._calculate_team_points(seed3, week_points)
            seed6_points = self._calculate_team_points(seed6, week_points)

            if seed3_points > seed6_points:
                seed3_advance = seed3
                seed6_eliminated = seed6
            else:
                seed3_advance = seed6
                seed6_eliminated = seed3

            seed4_points = self._calculate_team_points(seed4, week_points)
            seed5_points = self._calculate_team_points(seed5, week_points)

            if seed4_points > seed5_points:
                seed4_advance = seed4
                seed5_eliminated = seed5
            else:
                seed4_advance = seed5
                seed5_eliminated = seed4

            self.quarterfinal_teams = [seed1, seed2, seed3_advance, seed4_advance]
            self.eliminated_teams = [seed5_eliminated, seed6_eliminated]
        elif week == 15:
            seed1, seed2, seed3_advance, seed4_advance = self.quarterfinal_teams
            week_points = self.weekly_info_df[self.weekly_info_df['Week'] == week]

            # Semifinals: Seed 1 vs Seed 4_advance and Seed 2 vs Seed 3_advance and seed5_eliminated vs seed6_eliminated
            seed1_points = self._calculate_team_points(seed1, week_points)
            seed4_points = self._calculate_team_points(seed4_advance, week_points)

            if seed1_points > seed4_points:
                seed1_advance = seed1
                seed4_eliminated = seed4_advance
            else:
                seed1_advance = seed4_advance
                seed4_eliminated = seed1

            seed2_points = self._calculate_team_points(seed2, week_points)
            seed3_points = self._calculate_team_points(seed3_advance, week_points)

            if seed2_points > seed3_points:
                seed2_advance = seed2
                seed3_eliminated = seed3_advance
            else:
                seed2_advance = seed3_advance
                seed3_eliminated = seed2

            self.semifinal_teams = [seed1_advance, seed2_advance]

            seed5_eliminated, seed6_eliminated = self.eliminated_teams
            seed5_eliminated_points = self._calculate_team_points(seed5_eliminated, week_points)
            seed6_eliminated_points = self._calculate_team_points(seed6_eliminated, week_points)

            if seed5_eliminated_points > seed6_eliminated_points:
                fifth_place = seed5_eliminated
                sixth_place = seed6_eliminated
            else:
                fifth_place = seed6_eliminated
                sixth_place = seed5_eliminated

            self.playoff_standings.loc[self.playoff_standings['Team'] == fifth_place, 'Rank'] = 5
            self.playoff_standings.loc[self.playoff_standings['Team'] == sixth_place, 'Rank'] = 6

            self.eliminated_teams = [seed3_eliminated, seed4_eliminated]
        elif week == 16:
            seed1_advance, seed2_advance = self.semifinal_teams
            week_points = self.weekly_info_df[self.weekly_info_df['Week'] == week]

            # Final week setup: Collect points
            seed1_points = self._calculate_team_points(seed1_advance, week_points)
            seed2_points = self._calculate_team_points(seed2_advance, week_points)

            self.week16_points = {seed1_advance: seed1_points, seed2_advance: seed2_points}

            seed3_eliminated, seed4_eliminated = self.eliminated_teams
            seed3_points = self._calculate_team_points(seed3_eliminated, week_points)
            seed4_points = self._calculate_team_points(seed4_eliminated, week_points)

            thirdPlace = seed3_eliminated if seed3_points > seed4_points else seed4_eliminated

            self.playoff_standings.loc[self.playoff_standings['Team'] == thirdPlace, 'Rank'] = 3
            self.playoff_standings.loc[self.playoff_standings['Team'] == (seed3_eliminated if thirdPlace != seed3_eliminated else seed4_eliminated), 'Rank'] = 4


        else:
            seed1_advance, seed2_advance = self.semifinal_teams
            week_points = self.weekly_info_df[self.weekly_info_df['Week'] == week]

            seed1_points = self._calculate_team_points(seed1_advance, week_points)
            seed2_points = self._calculate_team_points(seed2_advance, week_points)

            seed1_total_points = self.week16_points[seed1_advance] + seed1_points
            seed2_total_points = self.week16_points[seed2_advance] + seed2_points

            winner = seed1_advance if seed1_total_points > seed2_total_points else seed2_advance

            self.playoff_standings.loc[self.playoff_standings['Team'] == winner, 'Rank'] = 1
            self.playoff_standings.loc[self.playoff_standings['Team'] == (seed1_advance if winner != seed1_advance else seed2_advance), 'Rank'] = 2

            print(f"6-Player Playoff Final: {seed1_advance} vs {seed2_advance} - Winner: {winner}")

    def _simulate_toilet_bowl(self, week: int):
        """
        Simulate the Toilet Bowl.
        """
        if self.toilet_bowl_style == '4Player':
            self._simulate_toilet_bowl_4player(week)
        else:
            self._simulate_toilet_bowl_6player(week)

    def _simulate_toilet_bowl_4player(self, week: int):
        """
        Simulate the 4-team Toilet Bowl.
        """
        if self.num_teams == 8:
            start = 5
        else:
            start = 7

        if week == PLAYOFF_START_WEEK:
            seed4, seed3, seed2, seed1 = self.toilet_bowl_teams
            week_points = self.weekly_info_df[self.weekly_info_df['Week'] == week]

            seed1_points = self._calculate_team_points(seed1, week_points)
            seed4_points = self._calculate_team_points(seed4, week_points)

            seed2_points = self._calculate_team_points(seed2, week_points)
            seed3_points = self._calculate_team_points(seed3, week_points)

            if seed1_points > seed4_points:
                seed1_advance = seed1
                seed4_eliminated = seed4
            else:
                seed1_advance = seed4
                seed4_eliminated = seed1

            if seed2_points > seed3_points:
                seed2_advance = seed2
                seed3_eliminated = seed3
            else:
                seed2_advance = seed3
                seed3_eliminated = seed2

            self.toilet_bowl_semifinal_teams = [seed1_advance, seed2_advance]
            self.eliminated_toilet_bowl_teams = [seed3_eliminated, seed4_eliminated]
        elif week == 15:
            seed1_advance, seed2_advance = self.toilet_bowl_semifinal_teams
            week_points = self.weekly_info_df[self.weekly_info_df['Week'] == week]

            seed1_points = self._calculate_team_points(seed1_advance, week_points)
            seed2_points = self._calculate_team_points(seed2_advance, week_points)
            self.week15_points_toilet_bowl = {seed1_advance: seed1_points, seed2_advance: seed2_points}
            seed3_eliminated, seed4_eliminated = self.eliminated_toilet_bowl_teams
            seed3_points = self._calculate_team_points(seed3_eliminated, week_points)
            seed4_points = self._calculate_team_points(seed4_eliminated, week_points)

            thirdPlace = seed3_eliminated if seed3_points > seed4_points else seed4_eliminated

            self.playoff_standings.loc[self.playoff_standings['Team'] == thirdPlace, 'Rank'] = start + 1
            self.playoff_standings.loc[self.playoff_standings['Team'] == (seed3_eliminated if thirdPlace != seed3_eliminated else seed4_eliminated), 'Rank'] = start
        else:
            seed1_advance, seed2_advance = self.toilet_bowl_semifinal_teams
            week_points = self.weekly_info_df[self.weekly_info_df['Week'] == week]

            seed1_points = self._calculate_team_points(seed1_advance, week_points)
            seed2_points = self._calculate_team_points(seed2_advance, week_points)

            seed1_total_points = self.week15_points_toilet_bowl[seed1_advance] + seed1_points
            seed2_total_points = self.week15_points_toilet_bowl[seed2_advance] + seed2_points

            winner = seed1_advance if seed1_total_points > seed2_total_points else seed2_advance

            self.playoff_standings.loc[self.playoff_standings['Team'] == winner, 'Rank'] = start + 3
            self.playoff_standings.loc[self.playoff_standings['Team'] == (seed1_advance if winner != seed1_advance else seed2_advance), 'Rank'] = start + 2

            print(f"4-Player Toilet Bowl Final: {seed1_advance} vs {seed2_advance} - Winner: {winner}")

    def _simulate_toilet_bowl_6player(self, week: int):
        """
        Simulate the 6-team Toilet Bowl.
        """
        if week == PLAYOFF_START_WEEK:
            seed6, seed5, seed4, seed3, seed2, seed1 = self.toilet_bowl_teams
            week_points = self.weekly_info_df[self.weekly_info_df['Week'] == week]

            seed3_points = self._calculate_team_points(seed3, week_points)
            seed6_points = self._calculate_team_points(seed6, week_points)

            if seed3_points > seed6_points:
                seed3_advance = seed3
                seed6_eliminated = seed6
            else:
                seed3_advance = seed6
                seed6_eliminated = seed3

            seed4_points = self._calculate_team_points(seed4, week_points)
            seed5_points = self._calculate_team_points(seed5, week_points)

            if seed4_points > seed5_points:
                seed4_advance = seed4
                seed5_eliminated = seed5
            else:
                seed4_advance = seed5
                seed5_eliminated = seed4

            self.toilet_bowl_semifinal_teams = [seed1, seed2, seed3_advance, seed4_advance]
            self.eliminated_toilet_bowl_teams = [seed5_eliminated, seed6_eliminated]
        elif week == 15:
            seed1, seed2, seed3_advance, seed4_advance = self.toilet_bowl_semifinal_teams
            week_points = self.weekly_info_df[self.weekly_info_df['Week'] == week]

            seed1_points = self._calculate_team_points(seed1, week_points)
            seed4_points = self._calculate_team_points(seed4_advance, week_points)

            if seed1_points > seed4_points:
                seed1_advance = seed1
                seed4_eliminated = seed4_advance
            else:
                seed1_advance = seed4_advance
                seed4_eliminated = seed1

            seed2_points = self._calculate_team_points(seed2, week_points)
            seed3_points = self._calculate_team_points(seed3_advance, week_points)

            if seed2_points > seed3_points:
                seed2_advance = seed2
                seed3_eliminated = seed3_advance
            else:
                seed2_advance = seed3_advance
                seed3_eliminated = seed2

            self.toilet_bowl_semifinal_teams = [seed1_advance, seed2_advance]

            seed5_eliminated, seed6_eliminated = self.eliminated_toilet_bowl_teams
            seed5_eliminated_points = self._calculate_team_points(seed5_eliminated, week_points)
            seed6_eliminated_points = self._calculate_team_points(seed6_eliminated, week_points)

            if seed5_eliminated_points > seed6_eliminated_points:
                fifth_place = seed5_eliminated
                sixth_place = seed6_eliminated
            else:
                fifth_place = seed6_eliminated
                sixth_place = seed5_eliminated

            self.playoff_standings.loc[self.playoff_standings['Team'] == fifth_place, 'Rank'] = 7
            self.playoff_standings.loc[self.playoff_standings['Team'] == sixth_place, 'Rank'] = 8

            self.eliminated_toilet_bowl_teams = [seed3_eliminated, seed4_eliminated]
        elif week == 16:
            seed1_advance, seed2_advance = self.toilet_bowl_semifinal_teams
            week_points = self.weekly_info_df[self.weekly_info_df['Week'] == week]

            seed1_points = self._calculate_team_points(seed1_advance, week_points)
            seed2_points = self._calculate_team_points(seed2_advance, week_points)

            self.week16_points_toilet_bowl = {seed1_advance: seed1_points, seed2_advance: seed2_points}

            seed3_eliminated, seed4_eliminated = self.eliminated_toilet_bowl_teams
            seed3_points = self._calculate_team_points(seed3_eliminated, week_points)
            seed4_points = self._calculate_team_points(seed4_eliminated, week_points)

            thirdPlace = seed3_eliminated if seed3_points > seed4_points else seed4_eliminated

            self.playoff_standings.loc[self.playoff_standings['Team'] == thirdPlace, 'Rank'] = 9
            self.playoff_standings.loc[self.playoff_standings['Team'] == (seed3_eliminated if thirdPlace != seed3_eliminated else seed4_eliminated), 'Rank'] = 10

        else:
            seed1_advance, seed2_advance = self.toilet_bowl_semifinal_teams
            week_points = self.weekly_info_df[self.weekly_info_df['Week'] == week]

            seed1_points = self._calculate_team_points(seed1_advance, week_points)
            seed2_points = self._calculate_team_points(seed2_advance, week_points)

            seed1_total_points = self.week16_points_toilet_bowl[seed1_advance] + seed1_points
            seed2_total_points = self.week16_points_toilet_bowl[seed2_advance] + seed2_points

            winner = seed1_advance if seed1_total_points > seed2_total_points else seed2_advance

            self.playoff_standings.loc[self.playoff_standings['Team'] == winner, 'Rank'] = 11
            self.playoff_standings.loc[self.playoff_standings['Team'] == (seed1_advance if winner != seed1_advance else seed2_advance), 'Rank'] = 12

            print(f"6-Player Toilet Bowl Final: {seed1_advance} vs {seed2_advance} - Winner: {winner}")


    def update_rosters(self):
        '''
        Updates all team's rosters for the upcoming week.
        '''
        waiverWireOrdering: List[Team] = []
        for name in self.waiverWireOrdering:
            waiverWireOrdering.append(self.team_dict[name])
        for team in waiverWireOrdering:
            team.determineWeekWaiverWireStatus()
            # print(team)
            # print(f'K streaming: {team.streamK}')
            # print(f'DST streaming: {team.streamDST}')
            # print(f'Waiver wire status: {team.waiverwirestatus}')
            updated = False
            updatedK = False
            updatedDST = False
            while not team.updateRoster() or team.waiverwirestatus or team.streamK or team.streamDST:
                # needs waiver wire
                # 1. failed to update because roster is missing something and must use waiver wire
                # 2. trying to improve roster and has positive waiver wire status
                # print(f'roster status: {team.rosterStatus}')
                # if not team.rosterStatus:
                    # print(team.positionsInNeed)
                pairs = self.waiverWire.determineSwaps(team)
                # print(pairs)

                for pair in pairs:
                    self.waiverWire.addDrop(team, pair[1], pair[0])
                if not updated:
                    team.waiverwirestatus = 0
                    if team.streamK:
                        team.streamK = not team.streamK
                        updatedK = True
                    if team.streamDST:
                        team.streamDST = not team.streamDST
                        updatedDST = True
                    updated = True
            if updatedK:
                team.streamK = not team.streamK
            if updatedDST:
                team.streamDST = not team.streamDST
                

    def simulate_season(self):
        """
        Simulate the entire season week by week.
        """
        playoff_start = PLAYOFF_START_WEEK
        for week in self.matchups.keys():
            for team in self.teams:
                team.currentWeek = week
                team.positionsInNeed = []
                team.goingToDrop = []
                team.goingToAdd = []
            self.waiverWire.week = week
            if week == playoff_start:
                break
            self.update_player_status_points(week)
            self.update_rosters()
            self.simulate_week(week)
            self._update_points_per_game(week)
            self.waiverWireOrdering = self.standings['Team'].to_numpy()[::-1]
        for playoff_week in range(playoff_start, self.numWeeks + 1):
            self.update_player_status_points(playoff_week)
            self.update_rosters()
            self.simulate_playoffs(playoff_week)
            self._update_points_per_game(playoff_week)
        print("\nFinal Season Standings:")
        print(self.standings)
        print("\nFinal Playoff Standings:")
        self.playoff_standings.sort_values(by='Rank', ascending=True, inplace=True)
        print(self.playoff_standings)
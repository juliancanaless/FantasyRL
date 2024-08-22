from draftSimulator import DraftSimulator
from fantasyTeam import Team
from seasonSimulator import SeasonSimulator
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import os
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from draftSimulator import max_positions
pd.options.mode.chained_assignment = None 

pos_to_fantpos_mapping = {
    'QB': ['QB'],
    'RB': ['RB1', 'RB2', 'FLEX'],
    'WR': ['WR1', 'WR2', 'FLEX'],
    'TE': ['TE', 'FLEX'],
    'K': ['K'],
    'DST': ['DST']
}

positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DST']
fant_positions = ['QB', 'RB1', 'RB2', 'WR1', 'WR2', 'TE', 'FLEX', 'K', 'DST', 'BE1', 'BE2', 'BE3', 'BE4', 'BE5', 'BE6', 'BE7']

class FantasyFootballEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, board_path, weekly_stats_path, weekly_info_path, team_name=None, team_pick=None, leagueMembers=None):
        '''
        There are two modes that the FantasyFootballEnv can be utilized. 

        1) Input all team_name, team_pick, and leagueMembers parameters to utilize the environment
        to simulate a specific league.

        2) Do not input a team_name, team_pick, or leagueMembers parameters and utilize the environment
        as a random generator of leagues. 
        '''
        super(FantasyFootballEnv, self).__init__()

        self.np_random = np.random.RandomState(42)

        self.board_path = board_path
        self.weekly_stats_path = weekly_stats_path
        self.current_dir = os.getcwd()
        self.data_dir = os.path.join(self.current_dir, '..', 'data')
        self.weekly_info_path = weekly_info_path

        if team_name is not None and team_pick is not None and leagueMembers is not None:
            self.team: Team = Team(team_name, team_pick)
            self.draft: DraftSimulator = DraftSimulator(path=board_path, myTeam=self.team,
                                    leagueMembers=leagueMembers, leagueSize=len(leagueMembers) + 1,
                                    numRounds=16, stats=weekly_stats_path)
            self.draftTeams: list[Team] = self.draft.teams.copy()
            self.snake = self.draftTeams.copy()
        else:
            newNumTeams = np.random.choice(a=[8,10,12], size=None, replace=True, p=[.25, .35, .4])
            low = 1
            high = newNumTeams
            randomized_numbers = np.random.permutation(np.arange(low, high+1))
            myTeam = Team('Team1', randomized_numbers[0])
            leagueMembers = []
            for i in range(low+1, high+1):
                team_name = f'Team{i}'
                team_picknum = randomized_numbers[i-1]
                team = (team_name, team_picknum)
                leagueMembers.append(team)
            self.draft: DraftSimulator = DraftSimulator(path=self.board_path, myTeam=myTeam,
                                        leagueMembers=leagueMembers, leagueSize=len(leagueMembers) + 1,
                                        numRounds=16, stats=self.weekly_stats_path)
            self.team: Team = myTeam
            self.draftTeams: list[Team] = self.draft.teams.copy()
            self.snake = self.draftTeams.copy()

        # define action and obs space

        stats_low = np.ones(((len(positions)-2)*10*21, 48)) * -100
        stats_high = np.ones(((len(positions)-2)*10*21, 48)) * 1000

        draftboard_low = np.zeros((len(positions)*10, 9))
        draftboard_high = np.ones((len(positions)*10, 9)) * 2500

        roster_low = np.ones((len(self.team.roster), len(self.team.roster.columns))) * -10
        roster_high = np.ones((len(self.team.roster), len(self.team.roster.columns))) * 2500


        self.observation_space = spaces.Dict({
            'stats': spaces.Box(low=stats_low, high=stats_high, dtype=np.float32),
            'draftboard': spaces.Box(low=draftboard_low, high=draftboard_high, dtype=np.float32),
            'roster': spaces.Box(low=roster_low, high=roster_high, dtype=np.float32)
        })

        self.shared_label_encoders = {
            'Name': LabelEncoder(),
            'Position': LabelEncoder(),
            'Team': LabelEncoder(),
            'Status': LabelEncoder(),
            'FantasyPosition': LabelEncoder(),
            'player_display_name': LabelEncoder(),
            'recent_team': LabelEncoder(),
            'season_type': LabelEncoder(),
            'opponent_team': LabelEncoder()
        }

        # fit on fant positions for later usage
        self.shared_label_encoders['FantasyPosition'].fit(fant_positions)

        # creates instance variables stats_ and draftBoard_ that are ready to use for torch
        # update_draftBoard will need to be called right before an action is needed to be taken
        self._process_data()

        self.action_space = spaces.Discrete(len(self.draftBoard_))

        self.current_team_idx = 0
        self.current_step = 0
        self.state = self._get_state()

    def update_action_space(self, obs: pd.DataFrame):
        '''
        Updates the action space - removes drafted players from the agent's action space 
        '''
        self.action_space = spaces.Discrete(obs.shape[0])

    def _encode_categorical_data_with_shared_encoders(self, df: pd.DataFrame, columns):
        """
        Encodes the categorical columns using shared label encoders across all dataframes.
        """
        res = df.copy(deep=True)
        for col in columns:
            le = self.shared_label_encoders[col]
            # Fit encoder if not already fitted
            if not hasattr(le, 'classes_') or le.classes_.size == 0:
                res[col] = le.fit_transform(df[col].astype(str))
            else:
                res[col] = le.transform(df[col].astype(str))
        return res
    
    def _process_data(self):
        '''
        Preprocesses data for usage in agent's DeepQNetwork
        '''
        # encode draftboard
        self._process_draftboard()

        categorical_columns_stats_mapping = {
            'player_display_name': 'Name',
            'position': 'Position',
            'recent_team': 'Team',
            'season_type': 'season_type',
            'opponent_team': 'opponent_team'
        }

        # encode stats
        self.stats_ = self.draft.stats.copy(deep=True)

        for original_col, encoder_key in categorical_columns_stats_mapping.items():
            le = self.shared_label_encoders[encoder_key]
            if not hasattr(le, 'classes_') or le.classes_.size == 0:
                self.stats_[original_col] = le.fit_transform(self.stats_[original_col].astype(str))
            else:
                self.stats_[original_col] = le.transform(self.stats_[original_col].astype(str))

    def _process_draftboard(self):
        '''
        Pre-processes draftboard data for agent's DeepQNetwork
        '''
        # column to encode
        categorical_columns_draftboard = ['Name', 'Team', 'Position', 'Status']

        # Encode draftboard dataframe using shared encoders
        self.draftBoard_ = self._encode_categorical_data_with_shared_encoders(
            self.draft.draftBoard, categorical_columns_draftboard
        )
        # convert position rank to float type
        self._keep_only_numeric_and_convert_to_float(self.draftBoard_, 'PositionRank')
        self.draftBoard_['Available'] = self.draftBoard_['Available'].astype(int)

    def _keep_only_numeric_and_convert_to_float(self, df: pd.DataFrame, col_name):
        '''
        niche method for PositionRank col in draftboard
        '''
        if col_name in df.columns:
            df[col_name] = df[col_name].astype(str)
            # Remove all non-numeric characters
            df[col_name] = df[col_name].str.replace(r'\D', '', regex=True)
            # Convert the column to float
            df[col_name] = df[col_name].astype(float)
    
    def _update_roster(self):
        categorical_columns_roster = ['FantasyPosition', 'Name', 'Position', 'Team', 'Status']
        roster = self.team.roster.dropna()
        self.roster_ = self._encode_categorical_data_with_shared_encoders(
            roster, categorical_columns_roster
        )

        fixed_roster_size = len(self.team.roster)  # Example fixed size
        current_size = self.roster_.shape[0]

        if current_size < fixed_roster_size:
            # Create a DataFrame with -1 values for missing rows
            num_missing_rows = fixed_roster_size - current_size
            columns = self.roster_.columns
            
            # Create a DataFrame of -1s for the missing rows
            missing_rows = pd.DataFrame(-1, index=range(num_missing_rows), columns=columns)
            
            # Concatenate the missing rows to the existing roster_
            self.roster_ = pd.concat([self.roster_, missing_rows], ignore_index=True)
        
    
    def _get_state(self):
        self._update_roster()
        return {
            'stats': self.stats_,
            'draftboard': self.draftBoard_,
            'roster': self.roster_
        }
    
    def _pad_dataframe(self, df, target_shape):
        """
        Pads the DataFrame `df` with -1s to match the target_shape while keeping it as a DataFrame.
        """
        current_shape = df.shape
        if current_shape[0] < target_shape[0]:
            # Create padding DataFrame with -1s
            padding = pd.DataFrame(
                -1, 
                index=range(target_shape[0] - current_shape[0]), 
                columns=df.columns
            )
            # Concatenate the original DataFrame with the padding
            padded_df = pd.concat([df, padding], ignore_index=True)
        else:
            padded_df = df

        return padded_df
    
    def get_observation(self):
        draftboard_df = self.state['draftboard']
        
        filter_draftboard_df = draftboard_df[draftboard_df['Available'] == 1]
        top_players = []

        # Iterate over each position and get the top 10 players for each
        for pos in positions:
            pos_encoding = self.shared_label_encoders['Position'].transform([pos])[0]
            top_players_pos = filter_draftboard_df[filter_draftboard_df['Position'] == pos_encoding].head(10)
            top_players.append(top_players_pos)

        # Combine all top players into a single DataFrame
        draftboard_obs = pd.concat(top_players)

        self.update_action_space(draftboard_obs)

        roster_obs = self.state['roster']
        draftboard_names = draftboard_obs['Name']
        # fix stats obs --> wrong bc draft and stats have different labels
        stats_obs = self.stats_[self.stats_['player_display_name'].isin(draftboard_names)]

        stats_obs = self._pad_dataframe(stats_obs, self.observation_space['stats'].shape)
        draftboard_obs = self._pad_dataframe(draftboard_obs, self.observation_space['draftboard'].shape)
        roster_obs = self._pad_dataframe(roster_obs, self.observation_space['roster'].shape)

        return {
            'stats': stats_obs,
            'draftboard': draftboard_obs,
            'roster': roster_obs
        }
    
    def _run_draft(self):
        agent_team_name = self.team.name 

        for _ in range(self.draft.currentRound, self.draft.numRounds+1):
            for idx in range(self.current_team_idx, len(self.snake)):
                team = self.snake[idx]
                if team.name == agent_team_name:
                    self.state = self._get_state()
                    self.current_team_idx = (idx + 1) % len(self.snake)
                    if self.current_team_idx == 0:
                        self.draft.currentRound += 1
                        self.snake.reverse()
                    return  # Halt the draft process to allow the agent to make a pick
                else:
                    response = self.draft.otherTeamSelection(team)
                    if response:
                        player_name, playerTeam, position, byeWeek, status, avgadp = response
                        team.addPickToRoster(position, player_name, self.draft.currentPick,
                                            avgadp, playerTeam, byeWeek, 0, status)
                        print(f"{team.name} selection at pick {self.draft.currentPick}, round {self.draft.currentRound}: {player_name}, {position}")
                        player_name_encoding = self.shared_label_encoders['Name'].transform([player_name])[0]
                        self.draftBoard_.loc[self.draftBoard_['Name'] == player_name_encoding, 'Available'] = 0
                    else:
                        print('error occurred in selecting draft pick')
                    self.draft.currentPick += 1
            self.draft.currentRound += 1
            self.snake.reverse()
            self.current_team_idx = 0

    
    def step(self, action):
        done = False
        reward = 0
        observation = self.get_observation()

        if self.draft.currentRound <= self.draft.numRounds:
            # enough draft capital
            player_name = self.shared_label_encoders['Name'].inverse_transform([action])[0]
            player_position = self.draft.draftBoard.loc[self.draft.draftBoard['Name'] == player_name, 'Position'].values[0]
            if self.team.posFreqMap[player_position] >= max_positions[player_position]:
                # Invalid action because it violates the max_positions constraint
                reward -= 10  # Penalize for invalid action
                self.current_step += 1
                return observation, reward, done
            
            # Check required positions for the remaining rounds
            remaining_rounds = self.draft.numRounds - self.draft.currentRound + 1
            required_positions_set = self.draft._determineRequiredPositions(self.team, remaining_rounds)

            if len(required_positions_set) > 0 and player_position not in required_positions_set:
                print(f'model choice invalid action - {required_positions_set}')
                # Invalid action because it violates the required_positions constraint
                reward -= 10  # Penalize for invalid action
                self.current_step += 1
                return observation, reward, done
            
            # add case where bench is full and player chosen does not add to active roster
            fant_positions = pos_to_fantpos_mapping[player_position]
            available_in_active = False
            for fant_pos in fant_positions:
                if pd.isna(self.team.roster.loc[self.team.roster['FantasyPosition'] == fant_pos, 'Name'].values[0]):
                    available_in_active = True
                    break
            if self.team.isBenchFull() and not available_in_active:
                # invalid action because there is no space
                print(f'model choice invalid action - not enough room in roster for pick')
                reward -= 10
                self.current_step += 1
                return observation, reward, done
            
            # can draft player
            self.draft.mySelection(player_name)

            self.draftBoard_.loc[self.draftBoard_['Name'] == action, 'Available'] = 0

            # Continue the draft after the agent's pick
            self._run_draft()
            self.current_step += 1
            observation = self.get_observation()

        else:
            # draft done
            waiver_wire_df = self.draft.constructWaiverWire()
            waiver_wire_df.drop(columns=['Rank'], inplace=True)

            self.season: SeasonSimulator = SeasonSimulator(teams=self.draft.teams, weekly_info_path=self.weekly_info_path, waiver_wire_df=waiver_wire_df)
            self.season.simulate_season()
            placement = self.season.playoff_standings.loc[self.season.playoff_standings['Team'] == self.draft.me.name, 'Rank'].values[0]
            reward = self._calculate_reward(placement)
            observation = self.get_observation()
            done = True

        return observation, reward, done
    
    def reset(self):
        # reset environment
        newNumTeams = np.random.choice(a=[8,10,12], size=None, replace=True, p=[.25, .35, .4])
        low = 1
        high = newNumTeams
        randomized_numbers = np.random.permutation(np.arange(low, high+1))
        myTeam = Team('Team1', randomized_numbers[0])
        leagueMembers = []
        for i in range(low+1, high+1):
            team_name = f'Team{i}'
            team_picknum = randomized_numbers[i-1]
            team = (team_name, team_picknum)
            leagueMembers.append(team)
        self.draft: DraftSimulator = DraftSimulator(path=self.board_path, myTeam=myTeam,
                                    leagueMembers=leagueMembers, leagueSize=len(leagueMembers) + 1,
                                    numRounds=16, stats=self.weekly_stats_path)
        self.team: Team = myTeam
        self.draftTeams: list[Team] = self.draft.teams.copy()
        self.snake = self.draftTeams.copy()
        self.shared_label_encoders = {
            'Name': LabelEncoder(),
            'Position': LabelEncoder(),
            'Team': LabelEncoder(),
            'Status': LabelEncoder(),
            'FantasyPosition': LabelEncoder(),
            'player_display_name': LabelEncoder(),
            'recent_team': LabelEncoder(),
            'season_type': LabelEncoder(),
            'opponent_team': LabelEncoder()
        }
        self.shared_label_encoders['FantasyPosition'].fit(fant_positions)
        self._process_data()
        self.current_team_idx = 0
        self.current_step = 0
        self.state = self._get_state()
        return self.state
    
    # def render(self, mode='human', close=False):
    #     self.draft.render()  # Implement render in DraftSimulator
    #     for team in self.draft.teams:
    #         team.render()  # Implement render in Team
    #     self.season.render()  # Implement render in SeasonSimulator
    #     self.waiverWire.render()  # Implement render in WaiverWireSimulator

    def _calculate_reward(self, placement: int):
        reward = (((self.draft.numTeams-placement+1)**2) / (self.draft.numTeams**2)) * 10
        P = -10
        if placement == self.draft.numTeams:
            reward = P
        return reward
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.preprocessing import LabelEncoder

stratsByPos = {'QB': ['EarlyRoundQB', 'MidRoundQB', 'LateRoundQB'], 
               'TE': ['EarlyRoundTE', 'MidRoundTE', 'LateRoundTE'], 
               'RB': ['ZeroRB', 'HeroRB', 'None'], 
               'WR': ['ZeroWR', 'None'],
               'K': ['EarlyK', 'MidK', 'LateK'],
               'DST': ['EarlyDST', 'MidDST', 'LateDST']
               }


# strats by stage
stratsByStage = {'early': ['HeroRB', 'EarlyRoundQB', 'EarlyRoundTE'],
                 'middle': ['MidRoundQB', 'MidRoundTE'],
                 'earlyLate': ['LateRoundQB', 'LateRoundTE', 'EarlyK', 'EarlyDST'],
                 'midLate': ['MidK', 'MidDST'],
                 'lateLate': ['LateK', 'LateDST']
                }

position_mapping = {
    'QB': ['QB'],
    'RB1': ['RB'],
    'RB2': ['RB'],
    'WR1': ['WR'],
    'WR2': ['WR'],
    'TE': ['TE'],
    'FLEX': ['WR', 'RB', 'TE'],
    'K': ['K'],
    'DST': ['DST']
}


class Team:
    def __init__(self, name: str, draftPick: int) -> None:
        self.name = name
        self.draftPick = draftPick
        self.roster = self._createRosterDF()
        self.strategy = self._draftStrategy()
        self.strategiesLeft = self._determineStratsLeft()
        self.posFreqMap = {'QB': 0, 'WR': 0, 'RB': 0, 'TE': 0, 'K': 0, 'DST': 0}
        self.picksNeeded = self._calcResPicksByRound()
        self.waiverWireActivity = np.random.beta(2, 6)
        self.currentWeek = 1
        self.streamK = self._streamK()
        self.streamDST = self._streamDST()
        self.waiverwirestatus = 0
        self.positionsInNeed = []
        self.goingToDrop = []
        self.goingToAdd = []
        self.rosterStatus = 1

    def _createRosterDF(self):
        '''
        Creates a roster df that will serve as an important instance variable.
        '''
        roster = pd.DataFrame(
            {
                'FantasyPosition': ['QB', 'RB1', 'RB2', 'WR1', 'WR2', 'TE', 'FLEX', 'K', 'DST',
                             'BE1', 'BE2', 'BE3', 'BE4', 'BE5', 'BE6', 'BE7'],
                'Name': [None] * 16,
                'Position': [None] * 16,
                'PickNumber': [None] * 16,
                'AverageDraftPositionPPR': [None] * 16,
                'Team': [None] * 16,
                'ByeWeek': [None] * 16,
                'PointsPerGame': [None] * 16,
                'Status': [None] * 16,
                'ProjectedFantasyPoints': [0] * 16,
                'FantasyPoints': [0] * 16
            }
        )

        roster = roster.astype(
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
        return roster
    
    def _draftStrategy(self):
        '''
        Determines the team's draft strategy. The strategy is selected randomly on a discrete 
        distribution of what general players behave.
        '''
        qb_prob = [0.3, 0.6, 0.1]
        te_prob = [0.2, 0.75, 0.05]
        rb_prob = [0.25, 0.25, 0.5]
        wr_prob = [0.05, 0.95]
        k_prob = [0.1, 0.5, 0.4]
        dst_prob = [0.2, 0.4, 0.4]
        qb_strat = np.random.choice(stratsByPos['QB'], p=qb_prob)
        te_strat = np.random.choice(stratsByPos['TE'], p=te_prob)
        rb_strat = np.random.choice(stratsByPos['RB'], p=rb_prob)
        wr_strat = np.random.choice(stratsByPos['WR'], p=wr_prob)
        k_strat = np.random.choice(stratsByPos['K'], p=k_prob)
        dst_strat = np.random.choice(stratsByPos['DST'], p=dst_prob)
        return (qb_strat, rb_strat, wr_strat, te_strat, k_strat, dst_strat)
    
    def _streamK(self):
        '''
        Sets an instance variable that states whether a team 
        will stream kickers weekly or not.
        '''
        _, _, _, _, k_strat, _ = self.strategy
        if k_strat == 'EarlyK':
            return False
        else:
            return True
        
    def _streamDST(self):
        '''
        Sets an instance variable that states whether a team 
        will stream defense/special teams weekly or not.
        '''
        _, _, _, _, _, dst_strat = self.strategy
        if dst_strat == 'EarlyDST':
            return False
        else:
            return True

    def _determineStratsLeft(self):
        '''
        Returns a mapping of how many strategies they have left within each stage of the draft.
        This is meant to help a team determine whether they must act immiediately or not in order
        to fulfill their draft strategy.
        '''
        qb_strat, rb_strat, wr_strat, te_strat, k_strat, dst_strat = self.strategy
        strats = [qb_strat, rb_strat, wr_strat, te_strat, k_strat, dst_strat]
        early = [strat in strats for strat in stratsByStage['early']]
        mid = [strat in strats for strat in stratsByStage['middle']]
        earlylate = [strat in strats for strat in stratsByStage['earlyLate']]
        midlate = [strat in strats for strat in stratsByStage['midLate']]
        latelate = [strat in strats for strat in stratsByStage['lateLate']]
        mapping = {'early': early, 'middle': mid, 'earlyLate': earlylate,
                   'midLate': midlate, 'lateLate': latelate
                   }
        return mapping

    
    def _calcResPicksByRound(self):
        '''
        Determines the number of picks needed within each stage to fulfill strategy requirements.
        '''
        qb_strat, rb_strat, wr_strat, te_strat, k_strat, dstk_strat = self.strategy
        strats = [qb_strat, rb_strat, wr_strat, te_strat, k_strat, dstk_strat]
        numEarlyRoundStrats = sum(strat in strats for strat in stratsByStage['early'])
        numMiddleRoundStrats = sum(strat in strats for strat in stratsByStage['middle'])
        numearlylate = sum(strat in strats for strat in stratsByStage['earlyLate'])
        nummidlate = sum(strat in strats for strat in stratsByStage['midLate'])
        numlatelate = sum(strat in strats for strat in stratsByStage['lateLate'])
        mapping = {'early': numEarlyRoundStrats, 'middle': numMiddleRoundStrats, 'earlyLate': numearlylate,
                   'midLate': nummidlate, 'lateLate': numlatelate
                   }
        return mapping

    def addPickToRoster(self, pos: str, name: str, pick: int, avgadp: float, team: str, bye: int, ppg: float, status: str):
        '''
        Adds a pick to the roster.
        '''
        if pos in ['RB', 'WR', 'TE']:
            if pos == 'RB':
                if pd.isna(self.roster.loc[self.roster['FantasyPosition'] == 'RB1', 'Name']).values[0]:
                    self.posFreqMap[pos] += 1
                    self.roster.loc[self.roster['FantasyPosition'] == 'RB1', ['Name', 'Position', 'PickNumber', 'AverageDraftPositionPPR', 'Team', 'ByeWeek', 'PointsPerGame', 'Status']] = [name, pos, pick, avgadp, team, bye, ppg, status]
                elif pd.isna(self.roster.loc[self.roster['FantasyPosition'] == 'RB2', 'Name']).values[0]:
                    self.posFreqMap[pos] += 1
                    self.roster.loc[self.roster['FantasyPosition'] == 'RB2', ['Name', 'Position', 'PickNumber', 'AverageDraftPositionPPR', 'Team', 'ByeWeek', 'PointsPerGame', 'Status']] = [name, pos, pick, avgadp, team, bye, ppg, status]
                elif pd.isna(self.roster.loc[self.roster['FantasyPosition'] == 'FLEX', 'Name']).values[0]:
                    self.posFreqMap[pos] += 1
                    self.roster.loc[self.roster['FantasyPosition'] == 'FLEX', ['Name', 'Position', 'PickNumber', 'AverageDraftPositionPPR', 'Team', 'ByeWeek', 'PointsPerGame', 'Status']] = [name, pos, pick, avgadp, team, bye, ppg, status]
                else:
                    self.addToBench(name, pos, pick, avgadp, team, bye, ppg, status)
            elif pos == 'WR':
                if pd.isna(self.roster.loc[self.roster['FantasyPosition'] == 'WR1', 'Name']).values[0]:
                    self.posFreqMap[pos] += 1
                    self.roster.loc[self.roster['FantasyPosition'] == 'WR1', ['Name', 'Position', 'PickNumber', 'AverageDraftPositionPPR', 'Team', 'ByeWeek', 'PointsPerGame', 'Status']] = [name, pos, pick, avgadp, team, bye, ppg, status]
                elif pd.isna(self.roster.loc[self.roster['FantasyPosition'] == 'WR2', 'Name']).values[0]:
                    self.posFreqMap[pos] += 1
                    self.roster.loc[self.roster['FantasyPosition'] == 'WR2', ['Name', 'Position', 'PickNumber', 'AverageDraftPositionPPR', 'Team', 'ByeWeek', 'PointsPerGame', 'Status']] = [name, pos, pick, avgadp, team, bye, ppg, status]
                elif pd.isna(self.roster.loc[self.roster['FantasyPosition'] == 'FLEX', 'Name']).values[0]:
                    self.posFreqMap[pos] += 1
                    self.roster.loc[self.roster['FantasyPosition'] == 'FLEX', ['Name', 'Position', 'PickNumber', 'AverageDraftPositionPPR', 'Team', 'ByeWeek', 'PointsPerGame', 'Status']] = [name, pos, pick, avgadp, team, bye, ppg, status]
                else:
                    self.addToBench(name, pos, pick, avgadp, team, bye, ppg, status)
            elif pos == 'TE':
                if pd.isna(self.roster.loc[self.roster['FantasyPosition'] == 'TE', 'Name']).values[0]:
                    self.posFreqMap[pos] += 1
                    self.roster.loc[self.roster['FantasyPosition'] == 'TE', ['Name', 'Position', 'PickNumber', 'AverageDraftPositionPPR', 'Team', 'ByeWeek', 'PointsPerGame', 'Status']] = [name, pos, pick, avgadp, team, bye, ppg, status]
                elif pd.isna(self.roster.loc[self.roster['FantasyPosition'] == 'FLEX', 'Name']).values[0]:
                    self.posFreqMap[pos] += 1
                    self.roster.loc[self.roster['FantasyPosition'] == 'FLEX', ['Name', 'Position', 'PickNumber', 'AverageDraftPositionPPR', 'Team', 'ByeWeek', 'PointsPerGame', 'Status']] = [name, pos, pick, avgadp, team, bye, ppg, status]
                else:
                    self.addToBench(name, pos, pick, avgadp, team, bye, ppg, status)
        else:
            if pd.isna(self.roster.loc[self.roster['FantasyPosition'] == pos, 'Name']).values[0]:
                self.posFreqMap[pos] += 1
                self.roster.loc[self.roster['FantasyPosition'] == pos, ['Name', 'Position', 'PickNumber', 'AverageDraftPositionPPR', 'Team', 'ByeWeek', 'PointsPerGame', 'Status']] = [name, pos, pick, avgadp, team, bye, ppg, status]
            else:
                self.addToBench(name, pos, pick, avgadp, team, bye, ppg, status)

    def addToBench(self, name: str, pos: str, pick: int, avgadp: float, team: str, bye: int, ppg: float, status: str, proj: Optional[float]=None, pts: Optional[float]=None):
        '''
        Adds a pick to the bench.
        '''
        bench = self.getBench()
        
        # Check for null values only in the 'Name' column
        null_data = bench[bench['Name'].isnull()]
        
        if self.isBenchFull():
            print("No more bench spots available")
            return False
        else:
            if null_data.empty:
                print("No empty bench spots available")
                return False
            
            benchSpot = null_data.head(1)['FantasyPosition'].values[0]
            
            if proj is None and pts is None:
                self.roster.loc[self.roster['FantasyPosition'] == benchSpot, ['Name', 'Position', 'PickNumber', 'AverageDraftPositionPPR', 'Team', 'ByeWeek', 'PointsPerGame', 'Status']] = [name, pos, pick, avgadp, team, bye, ppg, status]
            else:
                self.roster.loc[self.roster['FantasyPosition'] == benchSpot, ['Name', 'Position', 'PickNumber', 'AverageDraftPositionPPR', 'Team', 'ByeWeek', 'PointsPerGame', 'Status', 'ProjectedFantasyPoints', 'FantasyPoints']] = [name, pos, pick, avgadp, team, bye, ppg, status, proj, pts]
            
            self.posFreqMap[pos] += 1  
            return True

    
    def getBench(self):
        '''
        Returns a df of the bench players
        '''
        bench_spots = ['BE1', 'BE2', 'BE3', 'BE4', 'BE5', 'BE6', 'BE7']
        bench = self.roster[self.roster['FantasyPosition'].isin(bench_spots)]
        return bench


    def dropPlayer(self, playerName: Optional[str]):
        '''
        Drop the player from roster.
        Sets row to None
        '''
        if playerName is None:
            return
        player_row = self.roster[self.roster['Name'] == playerName]
        player_pos = player_row['Position'].values[0]
        if player_pos:
            self.roster.loc[self.roster['Name'] == playerName, 
                            ['Name', 'Position', 'PickNumber', 'AverageDraftPositionPPR', 
                            'Team', 'ByeWeek', 'PointsPerGame', 'Status', 
                            'ProjectedFantasyPoints', 'FantasyPoints']] = None
            self.posFreqMap[player_pos] -= 1
        else:
            print('Player position is empty')

    def isBenchFull(self):
        '''
        Determines if the bench is full
        '''
        bench = self.getBench()

        if bench['Name'].isnull().sum() > 0:
            return False
        return True
    
    def injuredPlayer(self):
        '''
        Determine if there are players in active roster that are designated as Out.
        '''
        if self.roster[self.roster['Status'] == 'Out'].empty:
            return False
        return True
    
    def injuredActivePlayers(self):
        '''
        Return a DF of the active players with an Out designation
        '''
        bench = self.getBench()
        bench_names = bench['Name'].tolist()
        activePlayers = self.roster[~self.roster['Name'].isin(bench_names)]
        return activePlayers[activePlayers['Status'] == 'Out']
    
    def swapPlayers(self, player1_row: Optional[pd.Series], player2_row: Optional[pd.Series]):
        '''
        Swaps one player with another within a roster
        '''
        player1_name = str(player1_row['Name'])
        player1_pos = str(player1_row['Position'])
        player1_fant_pos = str(player1_row['FantasyPosition'])
        player2_name = str(player2_row['Name'])
        player2_pos = str(player2_row['Position'])
        player2_fant_pos = str(player2_row['FantasyPosition'])

        bench = self.getBench()
        bench_names = bench['Name'].dropna().tolist()
        FLEX_pos = ['WR', 'RB', 'TE']

        if pd.isna(player1_name) or pd.isna(player2_name):
            return False

        # CASES WHERE SWAPPING IS VALID

        # case 1: active same pos <-> bench same pos
        player1_in_bench = player1_name in bench_names
        player2_in_bench = player2_name in bench_names
        one_in_bench = player1_in_bench ^ player2_in_bench
        one_in_bench_compatible = player1_pos == player2_pos

        # case 2: flex <--> {wr, rb, te} either in bench or active
        player1_is_FLEX = player1_fant_pos == 'FLEX'
        player2_is_FLEX = player2_fant_pos == 'FLEX'
        one_is_FLEX = player1_is_FLEX ^ player2_is_FLEX
        one_is_FLEX_pos_valid = (player1_is_FLEX and player2_pos in FLEX_pos) or (player2_is_FLEX and player1_pos in FLEX_pos)

        # Additional check for FLEX compatibility
        if one_is_FLEX and ((player1_is_FLEX and player2_pos not in FLEX_pos) or (player2_is_FLEX and player1_pos not in FLEX_pos)):
            return False

        # case 3: WR1 <-> WR2 or RB1 <-> RB2
        players_wr = 'WR' in player1_fant_pos and 'WR' in player2_fant_pos
        players_rb = 'RB' in player1_fant_pos and 'RB' in player2_fant_pos

        if (one_in_bench and one_in_bench_compatible) or (one_is_FLEX and one_is_FLEX_pos_valid) or (players_wr) or (players_rb):
            picknum_1 = player1_row['PickNumber']
            avgadp_1 = player1_row['AverageDraftPositionPPR']
            team_1 = player1_row['Team']
            bye_1 = player1_row['ByeWeek']
            ppg_1 = player1_row['PointsPerGame']
            status_1 = player1_row['Status']
            projected_pts_1 = player1_row['ProjectedFantasyPoints']
            pts_1 = player1_row['FantasyPoints']

            picknum_2 = player2_row['PickNumber']
            avgadp_2 = player2_row['AverageDraftPositionPPR']
            team_2 = player2_row['Team']
            bye_2 = player2_row['ByeWeek']
            ppg_2 = player2_row['PointsPerGame']
            status_2 = player2_row['Status']
            projected_pts_2 = player2_row['ProjectedFantasyPoints']
            pts_2 = player2_row['FantasyPoints']

            # Find the indices of the players to be swapped
            idx1 = self.roster[self.roster['Name'] == player1_name].index[0]
            idx2 = self.roster[self.roster['Name'] == player2_name].index[0]

            # Store the new values for the rows
            new_values_1 = [player2_name, player2_pos, picknum_2, avgadp_2, team_2, bye_2, ppg_2, status_2, projected_pts_2, pts_2]
            new_values_2 = [player1_name, player1_pos, picknum_1, avgadp_1, team_1, bye_1, ppg_1, status_1, projected_pts_1, pts_1]

            # Update the DataFrame rows
            self.roster.loc[idx1, ['Name', 'Position', 'PickNumber', 'AverageDraftPositionPPR', 'Team', 'ByeWeek', 'PointsPerGame', 'Status', 'ProjectedFantasyPoints', 'FantasyPoints']] = new_values_1
            self.roster.loc[idx2, ['Name', 'Position', 'PickNumber', 'AverageDraftPositionPPR', 'Team', 'ByeWeek', 'PointsPerGame', 'Status', 'ProjectedFantasyPoints', 'FantasyPoints']] = new_values_2

            return True
        else:
            return False
        
    def updateRoster(self):
        '''
        Updates roster for the upcoming week. 
        If players on active roster are injured, swap them with healthy players.
        If there are not enough bench players to fill spot, updates roster status to 0. 
        This will indicate to waiver wire that this team will use it to 
        get players and update afterwards. Will also update positions in need if applicable.
        '''
        # for each fantasy position, compare the player in the role to the top ranked player in roster
        # if top player is in fantasy position and healthy, keep same and go to next
        # if not, swap current with best. 
        # if there are not enough players, update pos in need, return False and external main will go waiver wire for roster updates
        # 

        ### Update what happens when player swaps in here cuz roster needs to be updated

        # Get bench players and active players
        bench = self.getBench()
        bench_names = bench['Name'].dropna().tolist()
        activePlayers = self.roster[~self.roster['Name'].isin(bench_names)]
        # print(self.roster)

        def custom_sort_key(row):
            # Before week 3, only use ProjectedFantasyPoints and AverageDraftPositionPPR
            if self.currentWeek < 4:
                return (row['ProjectedFantasyPoints'])
            # After week 3, use PointsPerGame only if ProjectedFantasyPoints is 0.0
            else:
                projected_fantasy_points = row['ProjectedFantasyPoints']
                if projected_fantasy_points == 0.0:
                    return (row['PointsPerGame'])
                return (row['ProjectedFantasyPoints'])

        # Get and sort healthy players by position
        def get_sorted_roster(position, minimum_length=0):
            roster = self.roster[(self.roster['Position'] == position) & (self.roster['Status'] == 'ACT')]
            roster['sort_key'] = roster.apply(custom_sort_key, axis=1)
            roster.sort_values(by='sort_key', ascending=False, inplace=True, ignore_index=True)
            roster.drop(columns='sort_key', inplace=True)
            return roster, (roster.iloc[minimum_length:] if len(roster) > minimum_length else pd.DataFrame())

        qb_roster, _ = get_sorted_roster('QB')
        wr_roster, rest_wrs = get_sorted_roster('WR', 2)
        rb_roster, rest_rbs = get_sorted_roster('RB', 2)
        te_roster, rest_tes = get_sorted_roster('TE', 1)
        k_roster, _ = get_sorted_roster('K')
        dst_roster, _ = get_sorted_roster('DST')

        # Create FLEX roster from remaining WR, RB, TE
        FLEX_roster = pd.concat([rest_wrs, rest_rbs, rest_tes], axis=0, ignore_index=True)
        if not FLEX_roster.empty:
            FLEX_roster['sort_key'] = FLEX_roster.apply(custom_sort_key, axis=1)
            FLEX_roster.sort_values(by='sort_key', ascending=False, inplace=True, ignore_index=True)
            FLEX_roster.drop(columns='sort_key', inplace=True)

        # Map positions to top player
        player_map = {
            'QB': qb_roster.iloc[0] if len(qb_roster) > 0 else None,
            'RB1': rb_roster.iloc[0] if len(rb_roster) > 0 else None,
            'RB2': rb_roster.iloc[1] if len(rb_roster) > 1 else None,
            'WR1': wr_roster.iloc[0] if len(wr_roster) > 0 else None,
            'WR2': wr_roster.iloc[1] if len(wr_roster) > 1 else None,
            'TE': te_roster.iloc[0] if len(te_roster) > 0 else None,
            'FLEX': FLEX_roster.iloc[0] if len(FLEX_roster) > 0 else None,
            'K': k_roster.iloc[0] if len(k_roster) > 0 else None,
            'DST': dst_roster.iloc[0] if len(dst_roster) > 0 else None
        }

        # Swap players if needed and check for injuries
        for fant_pos, top_player in player_map.items():
            if len(activePlayers[activePlayers['FantasyPosition'] == fant_pos]) > 0:
                active_player = activePlayers[activePlayers['FantasyPosition'] == fant_pos].iloc[0]
            else:
                active_player = None

            if active_player is not None:
                active_player_name = str(active_player['Name'])
            else:
                active_player_name = None

            if top_player is not None:
                top_player_name = str(top_player['Name'])
            else:
                top_player_name = None

            if active_player_name != top_player_name:
                if top_player_name and active_player_name:
                    self.swapPlayers(active_player, top_player)
                    # Update activePlayers after swap
                    bench = self.getBench()
                    bench_names = bench['Name'].tolist()
                    activePlayers = self.roster[~self.roster['Name'].isin(bench_names)]
                else:
                    if not (fant_pos == 'K' and self.streamK) and not (fant_pos == 'DST' and self.streamDST):
                        # print(fant_pos)
                        # print(f'top player: {top_player_name}')
                        # print(f'active player: {active_player_name}')
                        # if fant_pos == 'QB':
                        #     print(qb_roster)
                        # if fant_pos == 'RB2':
                        #     print(rb_roster)
                        # print(player_map)
                        self.positionsInNeed.append(fant_pos)

        if len(self.positionsInNeed) > 0:
            self.rosterStatus = 0
            return False
        self.rosterStatus = 1
        return True
        

    def determineWeekWaiverWireStatus(self):
        '''
        Sets the waiver wire status for the team.
        '''
        proba = np.random.normal(0.26, 0.18)
        if proba <= self.waiverWireActivity:
            self.waiverwirestatus = 1
        else:
            self.waiverwirestatus = 0

    def __repr__(self):
        return f"Team({self.name})"
    
    def get_state(self):
        roster_state = self.roster.to_numpy(dtype=np.float32)
        return roster_state
    
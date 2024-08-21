import pandas as pd
from fantasyTeam import Team
from typing import Optional

positions = ['QB', 'RB1', 'RB2', 'WR1', 'WR2', 'TE', 'FLEX', 'K', 'DST']
offense_positions = ['QB', 'RB1', 'WR1', 'TE']
position_mapping = {
    'QB': 'QB',
    'RB1': 'RB',
    'RB2': 'RB',
    'WR1': 'WR',
    'WR2': 'WR',
    'TE': 'TE',
    'FLEX': 'FLEX',
    'K': 'K',
    'DST': 'DST'
}

class WaiverWireSimulator():

    def __init__(self, waiverWire: pd.DataFrame):
        self.waiver_wire = waiverWire
        self.waiver_wire = self.waiver_wire.astype(
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
        self.waiver_wire.dropna()
        self.week = 1
        self._sortWaiverWire()

    def _sortWaiverWire(self):
        '''
        Sorts the waiver wire attribute.
        '''
        def custom_sort_key(row):
            # Before week 4, only use ProjectedFantasyPoints and AverageDraftPositionPPR
            if self.week < 4:
                return (row['ProjectedFantasyPoints'])
            # After week 4, use PointsPerGame only if ProjectedFantasyPoints is 0.0
            else:
                projected_fantasy_points = row['ProjectedFantasyPoints']
                if projected_fantasy_points == 0.0:
                    return (row['PointsPerGame'])
                return (row['ProjectedFantasyPoints'])
        

        # Apply the custom sort key
        self.waiver_wire['sort_key'] = self.waiver_wire.apply(custom_sort_key, axis=1)
        self.waiver_wire.sort_values(by='sort_key', ascending=False, inplace=True, ignore_index=True)
        self.waiver_wire.drop(columns='sort_key', inplace=True)

    def determineDrop(self, team: Team, pos: Optional[str] = None):
        '''
        Determines which player in roster to drop using efficiency sort ket.
        If there is space on the bench, returns None.
        '''
        def custom_sort_key(row):
            if self.week < 4:
                return (row['AverageDraftPositionPPR'])
            else:
                points_per_game = row['PointsPerGame']
                projected_fantasy_points = row['ProjectedFantasyPoints']
                if projected_fantasy_points == 0.0:
                    return (points_per_game)
                
                combined_score = (0.7 * points_per_game) + (0.3 * projected_fantasy_points)
                return (combined_score)
        
        if not team.isBenchFull():
            return None
        bench = team.getBench()
        bench = bench[~bench['Name'].isin(team.goingToDrop)]
        bench_1 = bench[bench['PickNumber'] == 0]
        bench_2 = bench[bench['PickNumber'] > 45]
        bench = pd.concat(objs=[bench_1, bench_2], ignore_index=True)
        if pos is not None:
            # also means that waiver wire status is on
            bench_players_same_position = bench[bench['Position'] == pos]
            if bench_players_same_position.empty:
                # If no backups are available, find the position with the highest frequency in the bench
                pos_counts = bench['Position'].value_counts()
                if not pos_counts.empty:
                    # Determine the position to drop based on the highest frequency
                    max_count = pos_counts.max()
                    most_frequent_positions = pos_counts[pos_counts == max_count].index.tolist()
                    # Use priority order if there are ties
                    priority_order = ['DST', 'K', 'TE', 'WR', 'RB', 'QB']
                    for priority_pos in priority_order:
                        if priority_pos in most_frequent_positions:
                            pos_to_drop = priority_pos
                            break
                    bench_players_same_position = bench[bench['Position'] == pos_to_drop]
                else:
                    bench_players_same_position = bench
            bench_players_same_position['sort_key'] = bench_players_same_position.apply(custom_sort_key, axis=1)
            if self.week < 4:
                bench_players_same_position.sort_values(by='sort_key', ascending=False, inplace=True, ignore_index=True)
            else:
                bench_players_same_position.sort_values(by='sort_key', inplace=True, ignore_index=True)
            bench_players_same_position.drop(columns='sort_key', inplace=True)
            bottom_bench_player = bench_players_same_position.iloc[0] if not bench_players_same_position.empty else None
            return bottom_bench_player
        
        bench['sort_key'] = bench.apply(custom_sort_key, axis=1)
        if self.week < 4:
            bench.sort_values(by='sort_key', ascending=False, inplace=True, ignore_index=True)
        else:
            bench.sort_values(by='sort_key', inplace=True, ignore_index=True)
        bench.drop(columns='sort_key', inplace=True)
        if not bench.empty:
            return bench.iloc[0]
        else:
            return None
        
    def determineAdd(self, team: Team, pos: str):
        '''
        Determines which player to potentially add from waiver wire filtered by position.
        '''
        def custom_sort_key(row):
            # Before week 3, only use ProjectedFantasyPoints and AverageDraftPositionPPR
            if self.week < 4:
                return (row['ProjectedFantasyPoints'])
            # After week 3, use PointsPerGame only if ProjectedFantasyPoints is 0.0
            else:
                projected_fantasy_points = row['ProjectedFantasyPoints']
                if projected_fantasy_points == 0:
                    return (row['PointsPerGame'])
                return (row['ProjectedFantasyPoints'])
            
        filtered_waiverwire = self.waiver_wire[self.waiver_wire['Position'] == pos]
        filtered_waiverwire = filtered_waiverwire[~filtered_waiverwire['Name'].isin(team.goingToAdd)]
        positionsNeeded = []
        for fant_pos in team.positionsInNeed:
            positionsNeeded.append(position_mapping[fant_pos])
        if pos in ['K', 'DST'] or pos in positionsNeeded:
            filtered_waiverwire = filtered_waiverwire[filtered_waiverwire['Status'] == 'ACT']
        filtered_waiverwire['sort_key'] = filtered_waiverwire.apply(custom_sort_key, axis=1)
        filtered_waiverwire.sort_values(by='sort_key', ascending=False, inplace=True, ignore_index=True)
        filtered_waiverwire.drop(columns='sort_key', inplace=True)
        if not filtered_waiverwire.empty:
            return filtered_waiverwire.iloc[0]
        else:
            return None
        
        
    def shouldAddDrop(self, team: Team, pos: str):
        '''
        Determines if a team is better off dropping someone on the roster for someone on 
        the waiver wire based on position. If there hasnt been 3 games played at least, returns
        None in order to minimize extreme variation as long as the team is not requesting a 
        streaming position or is in need of a position.
        '''
        def custom_sort_key(row):
            # Before week 3, only use ProjectedFantasyPoints and AverageDraftPositionPPR
            if self.week < 4:
                return (row['ProjectedFantasyPoints'])
            # After week 3, use PointsPerGame only if ProjectedFantasyPoints is 0.0
            else:
                projected_fantasy_points = row['ProjectedFantasyPoints']
                if projected_fantasy_points == 0.0:
                    return (row['PointsPerGame'])
                return (row['ProjectedFantasyPoints'])
        # special cases for streaming kickers/DST
        if (self.week < 3 and pos not in ['DST', 'K']) and (team.rosterStatus):
            return None
        if pos == 'FLEX':
            top_waiver_player_rb = self.determineAdd(team, 'RB')
            if top_waiver_player_rb is not None and not isinstance(top_waiver_player_rb, pd.DataFrame):
                top_waiver_player_rb = pd.DataFrame([top_waiver_player_rb])

            top_waiver_player_wr = self.determineAdd(team, 'WR')
            if top_waiver_player_wr is not None and not isinstance(top_waiver_player_wr, pd.DataFrame):
                top_waiver_player_wr = pd.DataFrame([top_waiver_player_wr])

            top_waiver_player_te = self.determineAdd(team, 'TE')
            if top_waiver_player_te is not None and not isinstance(top_waiver_player_te, pd.DataFrame):
                top_waiver_player_te = pd.DataFrame([top_waiver_player_te])
            
            bottom_bench_player = self.determineDrop(team)
            # dont need to consider waiver wire status as that function will never request FLEX 
            # these only come from positions in need
            if bottom_bench_player is not None and not isinstance(bottom_bench_player, pd.DataFrame):
                bottom_bench_player = pd.DataFrame([bottom_bench_player])

            positions = []
            for fant_pos in team.positionsInNeed:
                position = position_mapping[fant_pos]
                positions.append(position)
            
            if ((bottom_bench_player is not None) and ((top_waiver_player_rb is not None) or (top_waiver_player_wr is not None) or (top_waiver_player_te is not None))):
                merged_players = pd.concat(
                    objs=[top_waiver_player_rb[['Name', 'Position', 'AverageDraftPositionPPR', 'Status', 'PointsPerGame', 'ProjectedFantasyPoints', 'FantasyPoints']],
                        top_waiver_player_wr[['Name', 'Position', 'AverageDraftPositionPPR', 'Status', 'PointsPerGame', 'ProjectedFantasyPoints', 'FantasyPoints']],
                        top_waiver_player_te[['Name', 'Position', 'AverageDraftPositionPPR', 'Status', 'PointsPerGame', 'ProjectedFantasyPoints', 'FantasyPoints']],
                        bottom_bench_player[['Name', 'Position', 'AverageDraftPositionPPR', 'Status', 'PointsPerGame', 'ProjectedFantasyPoints', 'FantasyPoints']]], 
                        ignore_index=True)
                merged_players['sort_key'] = merged_players.apply(custom_sort_key, axis=1)
                merged_players.sort_values(by='sort_key', ascending=False, inplace=True, ignore_index=True)
                merged_players.drop(columns='sort_key', inplace=True)

                if str(merged_players.iloc[0]['Position']) == 'WR':
                        top_waiver_player_flex = top_waiver_player_wr
                elif str(merged_players.iloc[0]['Position']) == 'RB':
                    top_waiver_player_flex = top_waiver_player_rb
                else:
                    top_waiver_player_flex = top_waiver_player_te
                
                if str(merged_players.iloc[0]['Name']) != str(bottom_bench_player.iloc[0]['Name']):
                    return (bottom_bench_player, top_waiver_player_flex)
                if (not team.rosterStatus and pos in positions):
                    if merged_players.iloc[0] is not None and not isinstance(merged_players.iloc[0], pd.DataFrame):
                        top_waiver_player_flex = pd.DataFrame([merged_players.iloc[0]])
                    return (bottom_bench_player, top_waiver_player_flex)
            return None
        else:
            top_waiver_player = self.determineAdd(team, pos)
            if top_waiver_player is not None and not isinstance(top_waiver_player, pd.DataFrame):
                top_waiver_player = pd.DataFrame([top_waiver_player])
            if pos == 'K' and team.streamK:
                return (None, top_waiver_player)
            if pos == 'DST' and team.streamDST:
                return (None, top_waiver_player)
            if team.waiverwirestatus:
                bottom_bench_player = self.determineDrop(team, pos)
            else:
                bottom_bench_player = self.determineDrop(team)
            
            if bottom_bench_player is not None and not isinstance(bottom_bench_player, pd.DataFrame):
                bottom_bench_player = pd.DataFrame([bottom_bench_player])
        
            positions = []
            for fant_pos in team.positionsInNeed:
                position = position_mapping[fant_pos]
                positions.append(position)
            if bottom_bench_player is None and top_waiver_player is not None:
                return (bottom_bench_player, top_waiver_player)
            if ((bottom_bench_player is not None) and (top_waiver_player is not None)):
                merged_players = pd.concat(
                    objs=[top_waiver_player[['Name', 'Position', 'AverageDraftPositionPPR', 'Status', 'PointsPerGame', 'ProjectedFantasyPoints', 'FantasyPoints']],
                        bottom_bench_player[['Name', 'Position', 'AverageDraftPositionPPR', 'Status', 'PointsPerGame', 'ProjectedFantasyPoints', 'FantasyPoints']]], 
                        ignore_index=True)
                merged_players['sort_key'] = merged_players.apply(custom_sort_key, axis=1)
                merged_players.sort_values(by='sort_key', ascending=False, inplace=True, ignore_index=True)
                merged_players.drop(columns='sort_key', inplace=True)
                
                if str(merged_players.iloc[0]['Name']) != str(bottom_bench_player.iloc[0]['Name']):
                    return (bottom_bench_player, top_waiver_player)
                if (not team.rosterStatus and pos in positions):
                    return (bottom_bench_player, top_waiver_player)
            return None
    
    def determineSwaps(self, team: Team):
        '''
        Determines which player from the waiver wire to add based on needs.
        Returns a list of tuples. 
        The tuples has the roster player listed first and the waiver wire player listed second.
        '''
        self.waiver_wire['PointsPerGame'] = self.waiver_wire['PointsPerGame'].fillna(0.0)
        self.waiver_wire['ProjectedFantasyPoints'] = self.waiver_wire['ProjectedFantasyPoints'].fillna(0.0)
        pairs = []
        if not team.rosterStatus:
            # came to waiver wire because needs it to fix roster status
            positions_in_need_copy = team.positionsInNeed.copy() 
            for fant_pos in positions_in_need_copy:
                pos = position_mapping[fant_pos]
                pair = self.shouldAddDrop(team, pos)
                if pair is not None:
                    pairs.append(pair)
                    if pair[1] is not None:
                        team.goingToAdd.append(str(pair[1]['Name'].iloc[0]))
                    if pair[0] is not None:
                        team.goingToDrop.append(str(pair[0]['Name'].iloc[0]))
                    # print(f'positions in need before: {team.positionsInNeed}')
                    team.positionsInNeed.remove(fant_pos)
                    # print(f'after: {team.positionsInNeed}')
        if team.waiverwirestatus:
            # came to waiver wire because wants to get better backups
            for fant_pos in offense_positions:
                pos = position_mapping[fant_pos]
                pair = self.shouldAddDrop(team, pos)
                if pair is not None:
                    pairs.append(pair)
                    if pair[1] is not None:
                        team.goingToAdd.append(str(pair[1]['Name'].iloc[0]))
                    if pair[0] is not None:
                        team.goingToDrop.append(str(pair[0]['Name'].iloc[0]))
        if team.streamK:
            # stream kicker
            pair = self.shouldAddDrop(team, 'K')
            if pair:
                pairs.append(pair)
                team.goingToAdd.append(str(pair[1]['Name'].iloc[0]))
                if pair[0] is not None:
                    team.goingToDrop.append(str(pair[0]['Name'].iloc[0]))
        if team.streamDST:
            # stream dst
            pair = self.shouldAddDrop(team, 'DST')
            if pair is not None:
                pairs.append(pair)
                if pair[1] is not None:
                    team.goingToAdd.append(str(pair[1]['Name'].iloc[0]))
                if pair[0] is not None:
                    team.goingToDrop.append(str(pair[0]['Name'].iloc[0]))
        return pairs

    def addPlayerToWaiverWire(self, player: pd.DataFrame):
        if not isinstance(player, pd.DataFrame) or player.shape[0] != 1:
            raise ValueError("The player parameter should be a pandas DataFrame with a single row")

        required_columns = ['Name', 'Team', 'ByeWeek', 'Position', 'AverageDraftPositionPPR', 'Status', 
                            'PointsPerGame', 'ProjectedFantasyPoints', 'FantasyPoints']

        for col in required_columns:
            if col not in player.columns:
                raise ValueError(f"The player DataFrame is missing the required column: {col}")

        # Extract the values from the single row DataFrame
        player_row = player.iloc[0]
        playerName = player_row['Name']
        team = player_row['Team']
        byeWeek = player_row['ByeWeek']
        position = player_row['Position']
        avgadp = player_row['AverageDraftPositionPPR']
        status = player_row['Status']
        ppg = player_row['PointsPerGame']
        proj = player_row['ProjectedFantasyPoints']
        pts = player_row['FantasyPoints']

        new_row = pd.DataFrame([{
            'Name': playerName,
            'Team': team,
            'ByeWeek': byeWeek,
            'Position': position,
            'AverageDraftPositionPPR': avgadp,
            'Status': status,
            'PointsPerGame': ppg,
            'ProjectedFantasyPoints': proj,
            'FantasyPoints': pts
        }])

        # Append the new row to the waiver_wire DataFrame using pd.concat
        self.waiver_wire = pd.concat([self.waiver_wire, new_row], ignore_index=True)



    def removePlayerFromWaiverWire(self, player: pd.DataFrame):
        player_name = player['Name'].values[0]
        self.waiver_wire = self.waiver_wire[~self.waiver_wire['Name'].isin([player_name])]

    def addDrop(self, team: Team, addPlayer: pd.DataFrame, dropPlayer: pd.DataFrame):
        '''
        Given a team, a requested player in the waiver wire, and a player in the team's roster,
        exchanges the two players. If dropPlayer is None, adds to bench. This was determined earlier
        and should not cause errors.

        If the player being added is streaming a kicker or DST, checks if the proposed player from waiver 
        wire has more projected points than the rostered player and only exchanges if true.
        '''
        if addPlayer is None:
            return
        playerName = addPlayer['Name'].values[0]
        position = addPlayer['Position'].values[0]
        playerTeam = addPlayer['Team'].values[0]
        byeWeek = addPlayer['ByeWeek'].values[0]
        ppg = addPlayer['PointsPerGame'].values[0]
        status = addPlayer['Status'].values[0]
        avgadp = addPlayer['AverageDraftPositionPPR'].values[0]
        proj = addPlayer['ProjectedFantasyPoints'].values[0]
        pts = addPlayer['FantasyPoints'].values[0]
        # case where DST and Kickers are being streamed
        if position == 'K':
            if team.streamK:
                rostered = team.roster[team.roster['FantasyPosition'] == 'K']
                if not rostered.empty:
                    rostered_proj = rostered['ProjectedFantasyPoints'].values[0]
                else:
                    rostered_proj = -1
                if proj > rostered_proj:
                    # swap players
                    self.addPlayerToWaiverWire(rostered)
                    idx = team.roster[team.roster['FantasyPosition'] == 'K'].index[0]
                    new_values = [playerName, position, 0, avgadp, playerTeam, byeWeek, ppg, status, proj, pts]
                    team.roster.loc[idx, ['Name', 'Position', 'PickNumber', 'AverageDraftPositionPPR', 'Team', 'ByeWeek', 'PointsPerGame', 'Status', 'ProjectedFantasyPoints', 'FantasyPoints']] = new_values
                    self.removePlayerFromWaiverWire(addPlayer)
            else:
                rostered = team.roster[team.roster['FantasyPosition'] == 'K']
                if not rostered.empty:
                    rostered_proj = rostered['ProjectedFantasyPoints'].values[0]
                else:
                    rostered_proj = -1
                if proj > rostered_proj:
                    # print(f'executing waiver wire trade')
                    # print(f'{team.name}, week {self.week}')
                    # drop proposed player
                    if dropPlayer is not None and not dropPlayer.empty:
                        # print(f'roster before player drop\n {team.roster}')
                        dropPlayerName = dropPlayer['Name'].values[0]
                        team.dropPlayer(dropPlayerName)
                        self.addPlayerToWaiverWire(dropPlayer)
                        # print(f'roster after player drop\n {team.roster}')
                    # player has more proj but not enough space in bench
                    if not team.isBenchFull():
                        # print(f'adding player {playerName} to team roster')
                        team.addToBench(playerName, position, 0, avgadp, playerTeam, byeWeek, ppg, status, proj, pts)
                        self.removePlayerFromWaiverWire(addPlayer)

        elif position == 'DST':
            if team.streamDST:
                rostered = team.roster[team.roster['FantasyPosition'] == 'DST']
                if not rostered.empty:
                    rostered_proj = rostered['ProjectedFantasyPoints'].values[0]
                else:
                    rostered_proj = -1
                if proj > rostered_proj:
                    # swap players
                    self.addPlayerToWaiverWire(rostered)
                    idx = team.roster[team.roster['FantasyPosition'] == 'DST'].index[0]
                    new_values = [playerName, position, 0, avgadp, playerTeam, byeWeek, ppg, status, proj, pts]
                    team.roster.loc[idx, ['Name', 'Position', 'PickNumber', 'AverageDraftPositionPPR', 'Team', 'ByeWeek', 'PointsPerGame', 'Status', 'ProjectedFantasyPoints', 'FantasyPoints']] = new_values
                    self.removePlayerFromWaiverWire(addPlayer)
            else:
                rostered = team.roster[team.roster['FantasyPosition'] == 'DST']
                if not rostered.empty:
                    rostered_proj = rostered['ProjectedFantasyPoints'].values[0]
                else:
                    rostered_proj = -1
                if proj > rostered_proj:
                    # print(f'executing waiver wire trade')
                    # print(f'{team.name}, week {self.week}')
                    if dropPlayer is not None and not dropPlayer.empty:
                        # print(f'roster before player drop\n {team.roster}')
                        dropPlayerName = dropPlayer['Name'].values[0]
                        team.dropPlayer(dropPlayerName)
                        self.addPlayerToWaiverWire(dropPlayer)
                        # print(f'roster after player drop\n {team.roster}')
                    if not team.isBenchFull():
                        team.addToBench(playerName, position, 0, avgadp, playerTeam, byeWeek, ppg, status, proj, pts)
                        self.removePlayerFromWaiverWire(addPlayer)
        else:
            # add drop based on pair
            # print(f'executing waiver wire trade')
            # print(f'{team.name}, week {self.week}')
            if dropPlayer is not None and not dropPlayer.empty:
                # print(f'roster before player drop\n {team.roster}')
                dropPlayerName = dropPlayer['Name'].values[0]
                team.dropPlayer(dropPlayerName)
                self.addPlayerToWaiverWire(dropPlayer)
            if not team.isBenchFull():
                team.addToBench(playerName, position, 0, avgadp, playerTeam, byeWeek, ppg, status, proj, pts)
                self.removePlayerFromWaiverWire(addPlayer)
        self._sortWaiverWire
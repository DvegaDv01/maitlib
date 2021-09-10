import os
import sys
from fastai.tabular.all import *
from pandas.tseries.offsets import DateOffset

def gamma_theta(data):
    """Set up and calculate gamma theta for all data"""
    df = deepcopy(data)
    S_2 = np.square(df.loc[:,'active_underlying_price'])
    df.loc[:,'cash_gamma'] = df.loc[:,'gamma'] * S_2 / 100
    million = 1000000
    df.loc[:,'gamma_theta'] = (df.loc[:,'theta'] / df.loc[:,'cash_gamma']) * million
    
    return df



def relative_strike(data, column_name='active_underlying_price'):
    X = []
    for df in data:
        df.loc[:,'relative_strike'] = df.loc[:,'strike'] / df.loc[:,column_name]
        X.append(df)
    return X



def strike_filter(data):
    X = []
    for df in data:
        x = df[(df.loc[:,'relative_strike'] > 0.84) & (df.loc[:,'relative_strike'] < 1.115)].copy(deep=True)
        x.sort_values(by=['quote_datetime', 'strike'], inplace=True)
        x.reset_index(drop=True, inplace=True)
        X.append(x)
    return X


def time_as_pct_of_year(dataframe, column_name='quote_datetime', maturity=None):
    """pass in a dataframe."""
    timedata = pd.DataFrame()
    df = dataframe.copy(deep=True)
    
    t = pd.to_datetime(df.loc[:,column_name])
        
    # Finding the amount of time left until the next standard SPX expiration    
    standard_ex = pd.to_datetime(df.loc[:,column_name]).dt.strftime('%Y-%m').map(lambda x: np.busday_offset(x, 2,roll='forward',weekmask='Fri')).map(lambda x: pd.Timestamp(x).replace(hour=9, minute=30))
    following_mnth = pd.to_datetime(df.loc[:,column_name]).dt.strftime('%Y-%m').apply(lambda x: pd.to_datetime(x)  + DateOffset(months=1))
    next_standard_ex = pd.to_datetime(following_mnth).dt.strftime('%Y-%m').map(lambda x: np.busday_offset(x, 2,roll='forward',weekmask='Fri')).map(lambda x: pd.Timestamp(x).replace(hour=9, minute=30))
    
    days_until_standard_ex = (standard_ex - t).dt.days + ((((standard_ex - t).dt.seconds / 60) / 60) / 24)
    days_until_next_standard_ex = (next_standard_ex - t).dt.days + ((((next_standard_ex - t).dt.seconds / 60) / 60) / 24)
    
    timedata.loc[:, 'SPX_standard_expiration_dte'] = pd.Series([x if x > 0 else y for x, y in zip(days_until_standard_ex, days_until_next_standard_ex)])            
    
    # Finding the amount of time left in the year
    yr_end = pd.to_datetime(df.loc[:,column_name]).dt.strftime('%Y-%m').map(lambda x: pd.Timestamp(x).replace(month=12, day=31, hour=23, minute=59, second=59))
    remaining_days_in_yr = (yr_end - t).dt.days + ((((yr_end - t).dt.seconds / 60) / 60) / 24)
    timedata.loc[:, 'quotetime_as_pct_of_yr'] = 1 - (remaining_days_in_yr / 365)

    if maturity is not None:
        # including the nearest expiration date in final dataset to account for time series maturity relationships   
        expiry = pd.DataFrame(np.full_like(t.values, maturity))
        expiry.columns = ['expiry']
        # Finding the amount of time left until the next expiration
        expiration = pd.to_datetime(expiry.loc[:,'expiry']).dt.strftime('%Y-%m-%d').map(lambda x: pd.Timestamp(x).replace(hour=16, minute=14, second=59))    
        timedata.loc[:, 'days_until_nearest_expiration'] = (expiration - t).dt.days + ((((expiration - t).dt.seconds / 60) / 60) / 24)
        
    
    return timedata


def ratio_vertical_spread_pnl(NTMdf, ATMdf):
    
    d10_vega = NTMdf.loc[:, 'vega']
    atmvega = ATMdf.loc[:, 'vega']
    ratio = (atmvega/d10_vega).round()
    tot_longvega = ratio * d10_vega
    tot_longprem = ratio * NTMdf.loc[:, 'ask']
    shortPrem = ATMdf.loc[:, 'bid']
    initialPnl = (shortPrem - tot_longprem)[0]
    PnL = ((shortPrem - tot_longprem) - initialPnl)
    
    return PnL



def delta_near_the_money(data:pd.DataFrame, optiondelta: float):

    # pass in a dataframe; concatenate all data into one df first if applicable
    # the At-The_money data will be return
    dataframes = [dataframe.reset_index(drop=True) for index, dataframe in data.groupby(['quote_datetime','expiration'])]
    
    catm = [calls[(calls.loc[:, 'option_type']=='C') | (calls.loc[:, 'option_type']=='c')].reset_index(drop=True) for calls in dataframes]
    patm = [puts[(puts.loc[:, 'option_type']=='P') | (puts.loc[:, 'option_type']=='p')].reset_index(drop=True) for puts in dataframes]

    delta_50 = 0.50
    ntm_calls, ntm_puts = [], []
    for call, put in zip(catm, patm):
        ntmc = call.iloc[[call.loc[:, 'delta'].abs().sub(optiondelta).abs().idxmin()], :].copy(deep=True)
        ntmp = put.iloc[[put.loc[:, 'delta'].abs().sub(optiondelta).abs().idxmin()], :].copy(deep=True)
        
        d50 = put.iloc[[put.loc[:, 'delta'].abs().sub(delta_50).abs().idxmin()], :].copy(deep=True)
        skew_nD = ((ntmp.loc[:, 'implied_volatility'].values  - ntmc.loc[:, 'implied_volatility'].values) / d50.loc[:, 'implied_volatility'].values).round(2)
        
        ntmc.loc[:, 'ndelta_skew'] = skew_nD
        ntmp.loc[:, 'ndelta_skew'] = skew_nD
        ntm_calls.append(ntmc), ntm_puts.append(ntmp)
    
    # returns a concatenated dataframe of all at the money strikes at each quote time for each maturity
    # use pandas to seperate df into preferred parts
    calls = pd.concat(ntm_calls)
    puts = pd.concat(ntm_puts)
   
    NTM_C = [calls[(pd.to_datetime(calls.loc[:,'quote_datetime']).dt.strftime('%Y-%m-%d')==day)].reset_index(drop=True) for day in pd.to_datetime(calls.loc[:,'quote_datetime']).dt.strftime('%Y-%m-%d').unique()]
    NTM_P = [puts[(pd.to_datetime(puts.loc[:,'quote_datetime']).dt.strftime('%Y-%m-%d')==day)].reset_index(drop=True) for day in pd.to_datetime(puts.loc[:,'quote_datetime']).dt.strftime('%Y-%m-%d').unique()]
    
    return NTM_C, NTM_P     




def fixed_strike(data:pd.DataFrame, useDeltas=False, fDelta=0.5, market_time='09:45:00'):
    # does the same job as ATM function but instead for the starting fixed strike
    
    # pass in a dataframe; concatenate all data into one df first if applicable
    # the At-The_money data will be return
    if useDeltas==False:
        K, S = 'strike', 'active_underlying_price'        
        Lam_func = lambda x: x.iloc[[abs(x.loc[:,K].sub(x.loc[:,S])).idxmin()],:].copy(deep=True)
    else:
        K, S = 'delta', fDelta
        Lam_func = lambda x: x.iloc[[x.loc[:, K].abs().sub(fDelta).abs().idxmin()], :].copy(deep=True)
    
    dataframes = [dataframe.reset_index(drop=True) for index, dataframe in data.groupby('expiration')]
    
    catm = [calls[(calls.loc[:, 'option_type']=='C') | (calls.loc[:, 'option_type']=='c')].reset_index(drop=True) for calls in dataframes]
    patm = [puts[(puts.loc[:, 'option_type']=='P') | (puts.loc[:, 'option_type']=='p')].reset_index(drop=True) for puts in dataframes]
    delta_25 = 0.25
    delta_50 = 0.50
    
    fixed_calls, fixed_puts = [], []
    for call, put in zip(catm, patm):
        market_open = f"{pd.Timestamp(call.loc[:,'quote_datetime'].unique()[0]).strftime('%Y-%m-%d')} {market_time}"
        fixed_call = call[(call.loc[:, 'quote_datetime']==market_open)].copy(deep=True).reset_index(drop=True)
        fixed_put = put[(put.loc[:, 'quote_datetime']==market_open)].copy(deep=True).reset_index(drop=True)
        
        fixed_call_strike = Lam_func(fixed_call)
        fixed_put_strike = Lam_func(fixed_put)
        x = fixed_call_strike.strike.values
        x2 = fixed_put_strike.strike.values
        # finds index of strike closest to spot
        fc = call.loc[(call.loc[:, 'strike'] == float(x))].copy(deep=True).reset_index(drop=True)
        fp = put.loc[(put.loc[:, 'strike'] == float(x2))].copy(deep=True).reset_index(drop=True)
        
        fixed_calls.append(fc), fixed_puts.append(fp)
    
    # returns a concatenated dataframe of all at the money strikes at each quote time for each maturity
    # use pandas to seperate df into preferred parts
    calls = pd.concat(fixed_calls)
    puts = pd.concat(fixed_puts)
    
    
   
    F_C = [calls[(pd.to_datetime(calls.loc[:,'quote_datetime']).dt.strftime('%Y-%m-%d')==day)].reset_index(drop=True) for day in pd.to_datetime(calls.loc[:,'quote_datetime']).dt.strftime('%Y-%m-%d').unique()]
    F_P = [puts[(pd.to_datetime(puts.loc[:,'quote_datetime']).dt.strftime('%Y-%m-%d')==day)].reset_index(drop=True) for day in pd.to_datetime(puts.loc[:,'quote_datetime']).dt.strftime('%Y-%m-%d').unique()]
    
    return F_C, F_P



def fixed_maturity_selection(FIXED_data, maturity:int):
    
    fixed_maturity = [fixed_strike[(fixed_strike.loc[:,'expiration']==fixed_strike.loc[:,'expiration'].unique()[maturity])].reset_index(drop=True) for fixed_strike in FIXED_data]
    
    return pd.concat(fixed_maturity).reset_index(drop=True)




def pct_near_the_money(data:pd.DataFrame, pct: float, otm=True):
    
    if otm==True:
        call_pct = pct + 1
        put_pct = 1 - pct
    else:
        call_pct = 1 - pct
        put_pct = pct + 1
    
    # pass in a dataframe; concatenate all data into one df first if applicable
    # the At-The_money data will be return
    dataframes = [dataframe.reset_index(drop=True) for index, dataframe in data.groupby(['quote_datetime','expiration'])]
    
    catm = [calls[(calls.loc[:, 'option_type']=='C') | (calls.loc[:, 'option_type']=='c')].reset_index(drop=True) for calls in dataframes]
    patm = [puts[(puts.loc[:, 'option_type']=='P') | (puts.loc[:, 'option_type']=='p')].reset_index(drop=True) for puts in dataframes]
    delta_25 = 0.25
    delta_50 = 0.50
    
    ntm_calls, ntm_puts = [], []
    for call, put in zip(catm, patm):
        ntmc = call.iloc[[abs(call.loc[:,'relative_strike'].sub(call_pct)).idxmin()],:].copy(deep=True)
        ntmp = put.iloc[[abs(put.loc[:,'relative_strike'].sub(put_pct)).idxmin()],:].copy(deep=True)
        
        c25 = call.iloc[[call.loc[:, 'delta'].abs().sub(delta_25).abs().idxmin()], :].copy(deep=True)
        p25 = put.iloc[[put.loc[:, 'delta'].abs().sub(delta_25).abs().idxmin()], :].copy(deep=True)
        d50 = put.iloc[[put.loc[:, 'delta'].abs().sub(delta_50).abs().idxmin()], :].copy(deep=True)
        skew = ((p25.loc[:, 'implied_volatility'].values  - c25.loc[:, 'implied_volatility'].values) / d50.loc[:, 'implied_volatility'].values).round(2)
        
#        ntmc.loc[:, 'delta_skew'] = skew
#        ntmp.loc[:, 'delta_skew'] = skew
        ntm_calls.append(ntmc), ntm_puts.append(ntmp)
    
    # returns a concatenated dataframe of all at the money strikes at each quote time for each maturity
    # use pandas to seperate df into preferred parts
    calls = pd.concat(ntm_calls)
    puts = pd.concat(ntm_puts)
    
    
    
   
    NTM_C = [calls[(pd.to_datetime(calls.loc[:,'quote_datetime']).dt.strftime('%Y-%m-%d')==day)].reset_index(drop=True) for day in pd.to_datetime(calls.loc[:,'quote_datetime']).dt.strftime('%Y-%m-%d').unique()]
    NTM_P = [puts[(pd.to_datetime(puts.loc[:,'quote_datetime']).dt.strftime('%Y-%m-%d')==day)].reset_index(drop=True) for day in pd.to_datetime(puts.loc[:,'quote_datetime']).dt.strftime('%Y-%m-%d').unique()]
    
    return NTM_C, NTM_P



def quick_at_the_money_old(data:pd.DataFrame):
    # pass in a dataframe; concatenate all data into one df first if applicable
    # the At-The_money data will be return
    dataframes = [dataframe.reset_index(drop=True) for index, dataframe in data.groupby(['quote_datetime','expiration'])]
    
    catm = [calls[(calls.loc[:, 'option_type']=='C') | (calls.loc[:, 'option_type']=='c')].reset_index(drop=True) for calls in dataframes]
    patm = [puts[(puts.loc[:, 'option_type']=='P') | (puts.loc[:, 'option_type']=='p')].reset_index(drop=True) for puts in dataframes]
    
    atm_calls = [df.iloc[[abs(df.loc[:,'strike'].sub(df.loc[:,'active_underlying_price'])).idxmin()],:] for df in catm]
    atm_puts = [df.iloc[[abs(df.loc[:,'strike'].sub(df.loc[:,'active_underlying_price'])).idxmin()],:] for df in patm]
    
    # returns a concatenated dataframe of all at the money strikes at each quote time for each maturity
    # use pandas to seperate df into preferred parts
    calls = pd.concat(atm_calls)
    puts = pd.concat(atm_puts)
    
    
    #df = pd.concat(atm)
    #ATM = [df[(pd.to_datetime(df.loc[:,'quote_datetime']).dt.strftime('%Y-%m-%d')==day)].reset_index(drop=True) for day in pd.to_datetime(df.loc[:,'quote_datetime']).dt.strftime('%Y-%m-%d').unique()]
    
    ATM_C = [calls[(pd.to_datetime(calls.loc[:,'quote_datetime']).dt.strftime('%Y-%m-%d')==day)].reset_index(drop=True) for day in pd.to_datetime(calls.loc[:,'quote_datetime']).dt.strftime('%Y-%m-%d').unique()]
    ATM_P = [puts[(pd.to_datetime(puts.loc[:,'quote_datetime']).dt.strftime('%Y-%m-%d')==day)].reset_index(drop=True) for day in pd.to_datetime(puts.loc[:,'quote_datetime']).dt.strftime('%Y-%m-%d').unique()]
    
    return ATM_C,ATM_P


def quick_at_the_money(data:pd.DataFrame):
    # pass in a dataframe; concatenate all data into one df first if applicable
    # the At-The_money data will be return
    dataframes = [dataframe.reset_index(drop=True) for index, dataframe in data.groupby(['quote_datetime','expiration'])]
    
    catm = [calls[(calls.loc[:, 'option_type']=='C') | (calls.loc[:, 'option_type']=='c')].reset_index(drop=True) for calls in dataframes]
    patm = [puts[(puts.loc[:, 'option_type']=='P') | (puts.loc[:, 'option_type']=='p')].reset_index(drop=True) for puts in dataframes]
    delta_25 = 0.25
    delta_50 = 0.50
    
    atm_calls, atm_puts = [], []
    for call, put in zip(catm, patm):
        # finds index of strike closest to spot
        atmc = call.iloc[[abs(call.loc[:,'strike'].sub(call.loc[:,'active_underlying_price'])).idxmin()],:].copy(deep=True)
        atmp = put.iloc[[abs(put.loc[:,'strike'].sub(put.loc[:,'active_underlying_price'])).idxmin()],:].copy(deep=True)
        
        c25 = call.iloc[[call.loc[:, 'delta'].abs().sub(delta_25).abs().idxmin()], :].copy(deep=True)
        p25 = put.iloc[[put.loc[:, 'delta'].abs().sub(delta_25).abs().idxmin()], :].copy(deep=True)
        d50 = put.iloc[[put.loc[:, 'delta'].abs().sub(delta_50).abs().idxmin()], :].copy(deep=True)
        skew = ((p25.loc[:, 'implied_volatility'].values  - c25.loc[:, 'implied_volatility'].values) / d50.loc[:, 'implied_volatility'].values).round(2)
        
        atmc.loc[:, 'delta_skew'] = skew
        atmp.loc[:, 'delta_skew'] = skew
        atm_calls.append(atmc), atm_puts.append(atmp)
    
    # returns a concatenated dataframe of all at the money strikes at each quote time for each maturity
    # use pandas to seperate df into preferred parts
    calls = pd.concat(atm_calls)
    puts = pd.concat(atm_puts)
    
    
   
    ATM_C = [calls[(pd.to_datetime(calls.loc[:,'quote_datetime']).dt.strftime('%Y-%m-%d')==day)].reset_index(drop=True) for day in pd.to_datetime(calls.loc[:,'quote_datetime']).dt.strftime('%Y-%m-%d').unique()]
    ATM_P = [puts[(pd.to_datetime(puts.loc[:,'quote_datetime']).dt.strftime('%Y-%m-%d')==day)].reset_index(drop=True) for day in pd.to_datetime(puts.loc[:,'quote_datetime']).dt.strftime('%Y-%m-%d').unique()]
    
    return ATM_C, ATM_P



def ntm_maturity_selection(NTM_data, maturity:int):
    
    ntm_maturity = [ntm[(ntm.loc[:,'expiration']==ntm.loc[:,'expiration'].unique()[maturity])].reset_index(drop=True) for ntm in NTM_data]
    
    return pd.concat(ntm_maturity).reset_index(drop=True)



def atm_maturity_selection(ATM_data, maturity:int):
    
    atm_maturity = [atm[(atm.loc[:,'expiration']==atm.loc[:,'expiration'].sort_values().unique()[maturity])].reset_index(drop=True) for atm in ATM_data]
    
    return pd.concat(atm_maturity).reset_index(drop=True)




def smile_old(data):
    # for single data frames
    if isinstance(data, pd.DataFrame):
        # cvix and pvix are each a list of dataframes that has been decomposed into 
        # mulitple dataframes; where each dataframe now represents data for a specific maturity every 15mins.
        cvix = [(e, data[(data.expiration==e) & (data.option_type=='C') | (data.option_type=='c')]) for e in data.expiration.sort_values().unique()]
        pvix = [(e, data[(data.expiration==e) & (data.option_type=='P') | (data.option_type=='p')]) for e in data.expiration.sort_values().unique()]

        bench=[0.85, 0.90, 0.93, 0.95, 0.97, 0.99, 1.03, 1.05, 1.10]
        b_columns = ['85%', '90%','93%','95%','97%','99%','103%','105%','110%']
        call_list, put_list = [],[]
        for X,Y in zip(cvix,pvix):
            call_df,put_df = X[1].copy(deep=True),Y[1].copy(deep=True)
            
            call_df.loc[:,'relative_strike'] = call_df.loc[:, 'strike'].copy(deep=True)/call_df.loc[:, 'active_underlying_price'].copy(deep=True)
            put_df.loc[:,'relative_strike'] = put_df.loc[:, 'strike'].copy(deep=True)/put_df.loc[:, 'active_underlying_price'].copy(deep=True)            
            call_df.reset_index(drop=True, inplace=True)
            put_df.reset_index(drop=True, inplace=True)
            
            call_bench_strike = [np.ones_like(call_df.loc[:,'relative_strike'])*n for n in bench]
            put_bench_strike = [np.ones_like(put_df.loc[:,'relative_strike'])*n for n in bench]

            c_bench_df = pd.DataFrame(call_bench_strike).T
            c_bench_df.columns = ['85%', '90%','93%','95%','97%','99%','103%','105%','110%']
            c_bench_df.index = call_df.loc[:,'relative_strike'].index

            p_bench_df = pd.DataFrame(put_bench_strike).T
            p_bench_df.columns = ['85%', '90%','93%','95%','97%','99%','103%','105%','110%']
            p_bench_df.index = put_df.loc[:,'relative_strike'].index

            call_df2 = pd.DataFrame.join(call_df,c_bench_df).reset_index(drop=True)
            put_df2 = pd.DataFrame.join(put_df,p_bench_df).reset_index(drop=True)

            call_groupby = [j.reset_index(drop=True) for i,j in call_df2.groupby('quote_datetime')]
            put_groupby = [j.reset_index(drop=True) for i,j in put_df2.groupby('quote_datetime')]

            # This function may cause an issue with columns names becoming different from b_columns variable above
            
            call_rstrike = [df.iloc[[abs(df.loc[:,'relative_strike'].sub(df.loc[:,c])).idxmin(axis=1)],:].copy(deep=True) for df in call_groupby for c in b_columns]
            put_rstrike = [df.iloc[[abs(df.loc[:,'relative_strike'].sub(df.loc[:,c])).idxmin(axis=1)],:].copy(deep=True) for df in put_groupby for c in b_columns]            
                    
                    
            c_rstrike = pd.concat(call_rstrike)
            p_rstrike = pd.concat(put_rstrike)
            
            # Data may have NaN values after concatenation.
            # keeping columns with no NaN values [i.e ~df.isnull().all()] could be a solution
            # or try seperating concatenation to occur after each complete pass through each item in b_columns

            c_rstrike.loc[:, 'relative_strike'] = c_rstrike.loc[:, 'relative_strike'].round(2)
            p_rstrike.loc[:, 'relative_strike'] = p_rstrike.loc[:, 'relative_strike'].round(2)

            call_smile_dataframe = c_rstrike #.pivot_table(values='implied_volatility',index='quote_datetime', columns='relative_strike')
            put_smile_dataframe = p_rstrike #.pivot_table(values='implied_volatility',index='quote_datetime', columns='relative_strike')


            
#             call_smile_dataframe.columns = b_columns   keep
#             put_smile_dataframe.columns = b_columns   keep
            call_list.append(call_smile_dataframe)
            put_list.append(put_smile_dataframe)


        
        
        cmaturity, pmaturity = [e for e,_ in cvix],[e for e,_ in pvix]
        c_ticker_smile = call_list
        p_ticker_smile = put_list
        
#         c_ticker_smile = [s.pivot_table(values='implied_volatility',index='quote_datetime', columns='strike') for s in call_list]
#         p_ticker_smile = [s.pivot_table(values='implied_volatility',index='quote_datetime', columns='strike') for s in put_list]
        
        
    else:
        cmaturity, c_ticker_smile, pmaturity, p_ticker_smile = None,None,None,None
    
    return cmaturity, c_ticker_smile, pmaturity, p_ticker_smile


def dte_helper(data):

    data.loc[:,'expiration'] = pd.to_datetime(data.loc[:,'expiration']).dt.strftime('%Y-%m-%d').map(lambda x: pd.Timestamp(x).replace(hour=16, minute=15, second=0))
    data.loc[:,'dte'] = pd.to_datetime(data.loc[:,'expiration']) - pd.to_datetime(data.loc[:,'quote_datetime'])
    data.loc[:, 'full_dte'] = (pd.to_datetime(data.loc[:,'expiration']) - pd.to_datetime(data.loc[:,'quote_datetime'])).dt.days
    
    return data

def expiry_hlpr(data):

    expiry = data.iloc[[data.loc[:, 'full_dte'].abs().sub(data.SPX_standard_expiration_dte.round()[0]).abs().idxmin()], :].copy(deep=True).expiration
    expiry_list = data.expiration.sort_values().unique()
    term_idx = np.where(expiry_list==expiry.item())[0].item()

    return [term_idx]


def maturity_hlpr(maturity_idx):

    for x in maturity_idx:
        if x==1:
            mats = [x,2,3,12,19,20]           
        elif x==2:
            mats = [1,x,3,12,19,20]

        elif x==3:
            mats = [1,2,x,12,19,20]
            
        elif x==12:
            mats = [1,2,3,x,19,20]
            
        elif x==19:
            mats = [1,2,3,12,x,20]
            
        elif x==20:
            mats = [1,2,3,12,19,x]
        else:
            mats = [1,2,3,12,19,20]
            mats.sort()
    return mats



def smile(data):
    # for single data frames
    if isinstance(data, pd.DataFrame):
        
        new = data.copy(deep=True)
        new.loc[:,'relative_strike'] = new.loc[:, 'strike'].copy(deep=True)/new.loc[:, 'active_underlying_price'].copy(deep=True)

        bench=[0.85, 0.90, 0.93, 0.95, 0.97, 0.99, 1.03, 1.05, 1.10]
        b_columns = ['85%', '90%','93%','95%','97%','99%','103%','105%','110%']
        option_bench_strike = [np.ones_like(new.loc[:,'relative_strike'])*n for n in bench]
        option_bench_df = pd.DataFrame(option_bench_strike).T
        option_bench_df.columns = ['85%', '90%','93%','95%','97%','99%','103%','105%','110%']
        option_bench_df.index = new.loc[:,'relative_strike'].index
        
        option_df = pd.DataFrame.join(new,option_bench_df).reset_index(drop=True)

        
        c1, p1 = option_df[(option_df.option_type=='C') | (option_df.option_type=='c')], option_df[(option_df.option_type=='P') | (option_df.option_type=='p')]
        

        c2 = [c1[(c1.expiration==e)] for e in c1.expiration.sort_values().unique()]
        p2 = [p1[(p1.expiration==e)] for e in p1.expiration.sort_values().unique()]

        call_list, put_list = [],[]
        for x, y in zip(c2,p2):
            call_df,put_df = x.copy(deep=True),y.copy(deep=True)
            
            call_df.reset_index(drop=True, inplace=True)
            put_df.reset_index(drop=True, inplace=True)
            
            call_groupby = [quotetime.reset_index(drop=True) for i,quotetime in call_df.groupby('quote_datetime')]
            put_groupby = [quotetime.reset_index(drop=True) for i,quotetime in put_df.groupby('quote_datetime')]
        
            
            call_rstrike = [frame.iloc[[abs(frame.loc[:,'relative_strike'].sub(frame.loc[:,c])).idxmin(axis=1)],:].copy(deep=True) for frame in call_groupby for c in b_columns]
            put_rstrike = [frame.iloc[[abs(frame.loc[:,'relative_strike'].sub(frame.loc[:,c])).idxmin(axis=1)],:].copy(deep=True) for frame in put_groupby for c in b_columns]            
               
                    
            c_rstrike = pd.concat(call_rstrike)
            p_rstrike = pd.concat(put_rstrike)


            c_rstrike.loc[:, 'relative_strike'] = c_rstrike.loc[:, 'relative_strike'].round(2)
            p_rstrike.loc[:, 'relative_strike'] = p_rstrike.loc[:, 'relative_strike'].round(2)

            call_smile_dataframe = c_rstrike
            put_smile_dataframe = p_rstrike

            call_list.append(call_smile_dataframe)
            put_list.append(put_smile_dataframe)


        c_ticker_smile = call_list
        p_ticker_smile = put_list       
        
    else:
        c_ticker_smile, p_ticker_smile = None, None
    
    return c_ticker_smile, p_ticker_smile



    
    
    
    
def rearrange_concat(dataset:list):
    # Concatenate reorganized data
    X = []
    for i in range(0,7,1):
        x = []
        for day in dataset:
            x.append(day[i])
        X.append(pd.concat(x))

    data = [df.reset_index(drop=True) for df in X]
    return data



def skew_theta_func(pre_smile, atm_data):
    # pre_smile are a list of data frames with the same nth tenor across all available quote_datetimes - tenors specifically selected
    # by the "mats" variable
    df = pre_smile
    atm = atm_data
    skew_theta = []
    for atm_df, nth_tenor in zip(atm, df):
        X = []
        for t in atm_df.quote_datetime.values:
            x_maturity = nth_tenor[nth_tenor.loc[:,'quote_datetime'].values == t].copy(deep=True)
            x_maturity.sort_values(by='strike',inplace=True)
            x_maturity.loc[:,'skew_theta'] = abs(atm_df[atm_df.loc[:,'quote_datetime'].values == t].loc[:,'gamma_theta'].values) - x_maturity.loc[:,'gamma_theta']
            X.append(x_maturity)
        data = pd.concat(X)
        data.reset_index(drop=True, inplace=True)
        skew_theta.append(data)
    return skew_theta




def variable_surface(calls_smile_data, puts_smile_data, variable='implied_volatility'):
    """use variable_surface func to extract gamma_theta and cash_gamma columns along with any other desired 'skew,' ie. IV. """
    calls = [(j.reset_index(drop=True).loc[0,'quote_datetime'],j.reset_index(drop=True).loc[0,'expiration'],*j.reset_index(drop=True).loc[:,'skew_theta'],*j.reset_index(drop=True).loc[:,variable]) for df in calls_smile_data for i,j in df.groupby('quote_datetime')]
    puts = [(j.reset_index(drop=True).loc[0,'quote_datetime'],j.reset_index(drop=True).loc[0,'expiration'],*j.reset_index(drop=True).loc[:,'skew_theta'],*j.reset_index(drop=True).loc[:,variable]) for df in puts_smile_data for i,j in df.groupby('quote_datetime')]
     
    data_columns = ['quote_datetime', 'maturity','85_skew_theta', '90_skew_theta','93_skew_theta','95_skew_theta','97_skew_theta','99_skew_theta','103_skew_theta','105_skew_theta','110_skew_theta', '85%', '90%','93%','95%','97%','99%','103%','105%','110%']
    call_smile_dataframes,put_smile_dataframes = [],[]
    
    for c,p in zip(calls,puts):
        cl = pd.DataFrame(c).T
        pt = pd.DataFrame(p).T
        cl.columns = data_columns
        pt.columns = data_columns
        
        # the line of code below appends all data, which will have the "same" data 7 times (7 different maturities)
        # maybe include dte in calls and puts to make it more flexible to plot data of the course of multiple days
        call_smile_dataframes.append(cl)
        put_smile_dataframes.append(pt)
    call_smile_dataframes = pd.concat(call_smile_dataframes)
    put_smile_dataframes = pd.concat(put_smile_dataframes)
    
    return call_smile_dataframes.reset_index(drop=True), put_smile_dataframes.reset_index(drop=True)



def quick_bookdepth_extraction(df, level_cn='level2', combine=False):

    # For reference the order of raw string data: ['exchange_id', 'bid_size', 'bid', 'ask_size', 'ask']
    
    level2 = df.loc[:, level_cn].copy(deep=True).apply(lambda x: np.char.split(x, sep=";"))
    level2_sep = level2.apply(lambda x: np.char.split(x, sep=","))

    bid_liquidity = level2_sep.apply(lambda x: pd.Series(x).apply(lambda y: int(y[1]) * (float(y[2]) * 100)))
    ask_liquidity = level2_sep.apply(lambda x: pd.Series(x).apply(lambda y: int(y[3]) * (float(y[4]) * 100)))    
    
    bid_liquidity.loc[:, 'total_bid_liq'] = bid_liquidity.sum(axis=1)
    ask_liquidity.loc[:, 'total_ask_liq'] = ask_liquidity.sum(axis=1)
    
    bid_data = bid_liquidity.loc[:, 'total_bid_liq'].copy(deep=True)
    ask_data = ask_liquidity.loc[:, 'total_ask_liq'].copy(deep=True)
    data = bid_data.to_frame().merge(ask_data, left_index=True, right_index=True)
    data.loc[:, 'best_offer_liq'] = (df.loc[:, 'ask'] * 100) * df.loc[:, 'ask_size']
    data.loc[:, 'best_bid_liq'] = (df.loc[:, 'bid'] * 100) * df.loc[:, 'bid_size']
    
    # ask (what buyers pay) - bid (what sellers receive): How much is being bought minus how much is being sold
    # So when net liq is neg: more being sold, When positive: more being bought
    data.loc[:, 'net_liquidity'] = (data.loc[:, 'total_ask_liq'] - data.loc[:, 'total_bid_liq'])
    
    if combine==True:
        data = data.merge(df, left_index=True, right_index=True)
    else:
        pass
    
    return data


def fixed_surface(Data, variable_nm='implied_volatility', deltas = [0.05, 0.10, 0.15, 0.25, 0.30, 0.40, 0.50, 0.65, 0.70, 0.80, 0.9], terms=[1,2,3]):
    c,p = [],[]
    atmc, atmp = quick_at_the_money(Data)
    atm_list = len(Data.loc[:, 'expiration'].sort_values().unique())
    ATMC = [atm_maturity_selection(atmc, i) for i in terms]
    ATMP = [atm_maturity_selection(atmp, i) for i in terms]
    for dlt in deltas:
        c_options,p_options = [],[] 
        xc, xp = fixed_strike(Data, useDeltas=True, fDelta=dlt)

        l1 = len(Data.loc[:, 'expiration'].sort_values().unique())
        XC = [atm_maturity_selection(xc, i) for i in range(0,l1)]
        XP = [atm_maturity_selection(xp, i) for i in range(0,l1)]
        TT = [i for i in range(0,len(terms))]
        for T,C,P,ATMC_,ATMP_ in zip(TT,XC,XP,ATMC,ATMP):
            colnames = [variable_nm,'local_vol_skew','black_scholes_vol','black_scholes_vol_skew','local_vol_chg','realised_skew', 'skew_profit', 'vanna', 'vommaP', 'delta','elasticity', 'mid_price','delta_skew','charm']
            cname = [f"T_{T}_c{int(dlt*100)}d_fixed_strike_{col}" for col in colnames]
            pname = [f"T_{T}_p{int(dlt*100)}d_fixed_strike_{col}" for col in colnames]
            
            temp_names = ['quote_datetime',variable_nm,'mid_price','vanna', 'vommaP', 'delta','Elasticity','charm']
            # defined df frame
            cimplieds = C.loc[:, temp_names].copy(deep=True).reset_index(drop=True)
            pimplieds = P.loc[:, temp_names].copy(deep=True).reset_index(drop=True)

            
            # Local vol
            cimplieds.loc[:, {cname[0]}] = cimplieds.loc[:, 'implied_volatility'].round(5)
            pimplieds.loc[:, {pname[0]}] = pimplieds.loc[:, 'implied_volatility'].round(5)

            # black_scholes vol
            cimplieds.loc[:, {cname[2]}] = (cimplieds.loc[:, 'implied_volatility'] + ATMC_.loc[:, 'implied_volatility'])/2
            pimplieds.loc[:, {pname[2]}] = (pimplieds.loc[:, 'implied_volatility'] + ATMP_.loc[:, 'implied_volatility'])/2

            # Local vol skew
            cimplieds.loc[:, {cname[1]}] = cimplieds.loc[:, 'implied_volatility'] - ATMC_.loc[:, 'implied_volatility'].round(5)
            pimplieds.loc[:, {pname[1]}] = pimplieds.loc[:, 'implied_volatility'] - ATMP_.loc[:, 'implied_volatility'].round(5)

            # black_scholes skew
            cimplieds.loc[:, {cname[3]}] = cimplieds.loc[:, cname[2]] - ATMC_.loc[:, 'implied_volatility'].round(5)
            pimplieds.loc[:, {pname[3]}] = pimplieds.loc[:, pname[2]] - ATMP_.loc[:, 'implied_volatility'].round(5)
            
            # Local vol chg
            cimplieds.loc[:, {cname[4]}] = cimplieds.loc[:, cname[0]].diff().round(4)
            pimplieds.loc[:, {pname[4]}] = pimplieds.loc[:, pname[0]].diff().round(4)

            # Realised Skew: black_scholes vol (t1 - t0) / Local vol (t1 -t0)
            cimplieds.loc[:, {cname[5]}] = cimplieds.loc[:, cname[2]].diff() / cimplieds.loc[:, cname[0]].diff().round(5)
            pimplieds.loc[:, {pname[5]}] = pimplieds.loc[:, pname[2]].diff() / pimplieds.loc[:, pname[0]].diff().round(5)
                
            # Skew Profit: Realised Skew - 1
            cimplieds.loc[:, {cname[6]}] = (cimplieds.loc[:, cname[5]] - 1).round(4)
            pimplieds.loc[:, {pname[6]}] = (pimplieds.loc[:, pname[5]] - 1).round(4)
                
            # delta
            cimplieds.loc[:, {cname[9]}] = cimplieds.loc[:, 'delta'].round(3)
            pimplieds.loc[:, {pname[9]}] = pimplieds.loc[:, 'delta'].round(3)
            
            # volga
            cimplieds.loc[:, {cname[8]}] = cimplieds.loc[:, 'vommaP'].round(6)
            pimplieds.loc[:, {pname[8]}] = pimplieds.loc[:, 'vommaP'].round(6)
            
            # Vanna
            cimplieds.loc[:, {cname[7]}] = cimplieds.loc[:, 'vanna'].round(6)
            pimplieds.loc[:, {pname[7]}] = pimplieds.loc[:, 'vanna'].round(6)

            # Charm
            cimplieds.loc[:, {cname[13]}] = cimplieds.loc[:, 'charm'].round(6)
            pimplieds.loc[:, {pname[13]}] = pimplieds.loc[:, 'charm'].round(6)            

            # elasticity
            cimplieds.loc[:, {cname[10]}] = cimplieds.loc[:, 'Elasticity'].round(6)
            pimplieds.loc[:, {pname[10]}] = pimplieds.loc[:, 'Elasticity'].round(6)
            
            # Mid price
            cimplieds.loc[:, {cname[11]}] = cimplieds.loc[:, 'mid_price'].round(3)
            pimplieds.loc[:, {pname[11]}] = pimplieds.loc[:, 'mid_price'].round(3)

            
            if dlt==0.50:
                
                # Delta skew
                cimplieds.loc[:, {cname[12]}] = ATMC_.loc[:, 'delta_skew'].round(3)
                pimplieds.loc[:, {pname[12]}] = ATMP_.loc[:, 'delta_skew'].round(3)
            else:
                pass
            
            
                
            cimplieds.drop(columns=temp_names[1:],inplace=True)
            pimplieds.drop(columns=temp_names[1:],inplace=True)
            # set time to index for faster merging when using pandas
            cimplieds.set_index('quote_datetime', inplace=True)
            pimplieds.set_index('quote_datetime', inplace=True)
            c_options.append(cimplieds)
            p_options.append(pimplieds)
        o1 = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True),c_options)
        o2 = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True),p_options)        
        c.append(o1)
        p.append(o2)
    o1 = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True),c)
    o2 = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True),p)
    
    o1.loc[:,'active_underlying_price'] = ATMC[0].loc[:,'active_underlying_price'].values
    o2.loc[:,'active_underlying_price'] = ATMP[0].loc[:,'active_underlying_price'].values

    # spot chg
    o1.loc[:,'spot_chg'] = o1.loc[:,'active_underlying_price'] - o1.reset_index(drop=True).loc[0,'active_underlying_price']
    o2.loc[:,'spot_chg'] = o2.loc[:,'active_underlying_price'] - o2.reset_index(drop=True).loc[0,'active_underlying_price']
    

    return o1,o2
    





def bookdepth_extraction(df, level_cn='level2', adj_col=None, min_c=0):
    # Wrangle Level 2 data to calculate total dollar amount for each bid and ask place across all exchange id's
    # adj_col controls how far the bid/ask depth extends
    level2 = df.loc[:, level_cn].copy(deep=True).apply(lambda x: np.char.split(x, sep=";"))
    level2_sep = level2.apply(lambda x: np.char.split(x, sep=","))
    datasets = []
    for i, n in enumerate(level2_sep):
        
        # since level2 represents an entry for each row of the inputted data, 'i' is the matching index value
        # and is used in the batch names to represent which row the level2 data is extracted from
        
        depth = pd.Series(n).apply(lambda x: pd.Series(x))
        depth.columns = ['exchange_id', 'bid_size', 'bid', 'ask_size', 'ask']

        depth.loc[:, 'bid_size'] = depth.loc[:, 'bid_size'].astype('int')
        depth.loc[:, 'bid'] = depth.loc[:, 'bid'].astype('float')
        depth.loc[:, 'ask_size'] = depth.loc[:, 'ask_size'].astype('int')
        depth.loc[:, 'ask'] = depth.loc[:, 'ask'].astype('float')

        depth.loc[:, 'bid_liquidity'] = depth.loc[:, 'bid_size'] * (depth.loc[:, 'bid'] * 100)
        depth.loc[:, 'ask_liquidity'] = depth.loc[:, 'ask_size'] * (depth.loc[:, 'ask'] * 100)
    

        fields = ['bid', 'ask']
        
        for c in fields:
            batch, prices = [],[]
            
            for num, price in enumerate(depth.loc[:,c].sort_values().unique()):
                
                df_batch = depth[(depth.loc[:,c]==price)].copy(deep=True)
                x = df_batch.loc[:, f'{c}_liquidity'].sum()
                data = pd.DataFrame(pd.Series(x))
                data.columns = [f'{c}_{num}_total_liquidity']
                batch.append(data)
                prices.append(price)
                
            batch_name = f'{c}_batch_{i}'
            datasets.append({batch_name:[batch,prices]})

    dim = 0
    depth_bid = [reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True), data[key][dim]) for data in datasets for key in data.keys() if key.startswith('bid')]
    depth_ask = [reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True), data[key][dim]) for data in datasets for key in data.keys() if key.startswith('ask')]
    bid_book, ask_book = pd.concat(depth_bid), pd.concat(depth_ask)
    bid_book.reset_index(drop=True, inplace=True)
    ask_book.reset_index(drop=True, inplace=True)
    bid_book, ask_book = bid_book.iloc[:, :adj_col], ask_book.iloc[:, :adj_col]
    
    book = bid_book.merge(ask_book, left_index=True, right_index=True)
    
    price_dim = 1
    bid_price = [pd.DataFrame(data[key][price_dim]).T for data in datasets for key in data.keys() if key.startswith('bid')]
    ask_price = [pd.DataFrame(data[key][price_dim]).T for data in datasets for key in data.keys() if key.startswith('ask')]
    bid_price_book, ask_price_book = pd.concat(bid_price), pd.concat(ask_price)
    bid_price_book.reset_index(drop=True, inplace=True)
    ask_price_book.reset_index(drop=True, inplace=True)
    bid_price_book.columns, ask_price_book.columns = [f'bid_price_{n}' for n in bid_price_book.columns],[f'ask_price_{n}' for n in ask_price_book.columns]
    bid_price_book, ask_price_book = bid_price_book.iloc[:, :adj_col], ask_price_book.iloc[:, :adj_col]
    
    
    price_book = bid_price_book.merge(ask_price_book, left_index=True, right_index=True)
    
    level2_book = book.merge(price_book, left_index=True, right_index=True)
    bid_num, ask_num = int(len(bid_price_book.columns)), int(len(ask_price_book.columns))
    

    # ask (what buyers pay) - bid (what sellers receive): How much is being bought minus how much is being sold
    # So when net liq is neg: more being sold, When positive: more being bought
    if adj_col==None:
        # adj_col = min(bid_num, ask_num) - 1
        #level2_book.loc[:, 'net_liquidity'] = level2_book.loc[:, 'ask_0_total_liquidity':f'ask_{min_c}_total_liquidity'].sum(axis=1) - level2_book.loc[:, 'bid_0_total_liquidity':f'bid_{min_c}_total_liquidity'].sum(axis=1)
        # test
        level2_book.loc[:, 'net_liquidity'] = pd.DataFrame([level2_book.loc[:, c] for c in level2_book.columns if c.startswith('ask') and c.endswith('_total_liquidity')]).T.sum(axis=1) - pd.DataFrame([level2_book.loc[:, c] for c in level2_book.columns if c.startswith('bid') and c.endswith('_total_liquidity')]).T.sum(axis=1)
        
        
    else:
        adj_col = adj_col - 1
        level2_book.loc[:, 'net_liquidity'] = level2_book.loc[:, 'ask_0_total_liquidity':f'ask_{adj_col}_total_liquidity'].sum(axis=1) - level2_book.loc[:, 'bid_0_total_liquidity':f'bid_{adj_col}_total_liquidity'].sum(axis=1)

    return level2_book


def main(path: Union[str, Path], filenames=None, start_slice=None,end_slice=None, mats = [1,2,12,13,18,19,20]):
    # Grab 7 key Maturities; VIX requires a special maturity list
    # spy: ticker
    #SPY:TIKCER
    dataset_day = []

    for fn in filenames[start_slice:end_slice]:
        dataframe = pd.read_csv(path/fn)
        dataframe_ = [gamma_theta(df) for df in [dataframe]]
        ticker = relative_strike(dataframe_)
        TIKCER = strike_filter(ticker)
        # Next line create lists of lists, where the first index bracket represents a different day and the 2nd bracket represents a different maturity
        ticker_c_ticker_smile, ticker_p_ticker_smile = list(zip(*[smile(df) for df in TIKCER]))

        # Get ATM data to use for Skew Theta calculation;
        TIKCER_preATM = pd.concat(TIKCER)
        TIKCER_preATM.reset_index(drop=True, inplace=True)

        TIKCER_ATM_C, TIKCER_ATM_P = quick_at_the_money(TIKCER_preATM)
        TIKCER_FIXED_C, TIKCER_FIXED_P = fixed_strike(TIKCER_preATM)
        TIKCER_NTM_C, TIKCER_NTM_P = pct_near_the_money(TIKCER_preATM, pct=0.03)

        # Next line create lists of At-The-Money data for 7 different maturities (mats)
        # And Extract Level 2 column
        TIKCER_calls_atm_maturities, TIKCER_puts_atm_maturities = [atm_maturity_selection(TIKCER_ATM_C, i) for i in mats], [atm_maturity_selection(TIKCER_ATM_P, i) for i in mats]
        TIKCER_calls_fixed_maturities, TIKCER_puts_fixed_maturities = [atm_maturity_selection(TIKCER_FIXED_C, i) for i in mats], [atm_maturity_selection(TIKCER_FIXED_P, i) for i in mats]
        TIKCER_calls_ntm_maturities, TIKCER_puts_ntm_maturities = [atm_maturity_selection(TIKCER_NTM_C, i) for i in mats], [atm_maturity_selection(TIKCER_NTM_P, i) for i in mats]

        depth_call_datasets = [bookdepth_extraction(data) for data in TIKCER_calls_atm_maturities]
        depth_put_datasets = [bookdepth_extraction(data) for data in TIKCER_puts_atm_maturities]
        depth_call_fixed_datasets = [bookdepth_extraction(data) for data in TIKCER_calls_fixed_maturities]
        depth_put_fixed_datasets = [bookdepth_extraction(data) for data in TIKCER_puts_fixed_maturities]
        depth_call_ntm_datasets = [bookdepth_extraction(data) for data in TIKCER_calls_ntm_maturities]
        depth_put_ntm_datasets = [bookdepth_extraction(data) for data in TIKCER_puts_ntm_maturities]

        # Need to merge all (ATM, Fixed and NTM) to combine both the depth book and fields such as IV and delta; for each dataset
        call_datasets = [atm.iloc[:,:].merge(book, left_index=True, right_index=True) for atm,book in zip(TIKCER_calls_atm_maturities, depth_call_datasets)]
        put_datasets = [atm.iloc[:,:].merge(book, left_index=True, right_index=True) for atm,book in zip(TIKCER_puts_atm_maturities, depth_put_datasets)]
        fixedcall_datasets = [fixed.iloc[:,:].merge(book, left_index=True, right_index=True) for fixed,book in zip(TIKCER_calls_fixed_maturities, depth_call_fixed_datasets)]
        fixedput_datasets = [fixed.iloc[:,:].merge(book, left_index=True, right_index=True) for fixed,book in zip(TIKCER_puts_fixed_maturities, depth_put_fixed_datasets)]
        ntmcall_datasets = [ntm.iloc[:,:].merge(book, left_index=True, right_index=True) for ntm,book in zip(TIKCER_calls_ntm_maturities, depth_call_ntm_datasets)]
        ntmput_datasets = [ntm.iloc[:,:].merge(book, left_index=True, right_index=True) for ntm,book in zip(TIKCER_puts_ntm_maturities, depth_put_ntm_datasets)]

        # concat
        call_datasets, put_datasets = pd.concat(call_datasets), pd.concat(put_datasets)
        call_fixed_datasets, put_fixed_datasets = pd.concat(fixedcall_datasets), pd.concat(fixedput_datasets)
        call_ntm_datasets, put_ntm_datasets = pd.concat(ntmcall_datasets), pd.concat(ntmput_datasets)

        call_datasets.reset_index(drop=True, inplace=True)
        put_datasets.reset_index(drop=True, inplace=True)
        call_fixed_datasets.reset_index(drop=True, inplace=True), put_fixed_datasets.reset_index(drop=True, inplace=True)
        call_ntm_datasets.reset_index(drop=True, inplace=True), put_ntm_datasets.reset_index(drop=True, inplace=True)
       
        # drop
        colns = ['quote_datetime', 'underlying_symbol', 'root', 'option_type', 'expiration', 'underlying_bid', 'underlying_ask', 'implied_underlying_price', 'level2', 'relative_strike']
        call_datasets.drop(columns=colns, inplace=True)
        put_datasets.drop(columns=colns, inplace=True)

        delcolumns = ['underlying_symbol', 'quote_datetime', 'root', 'expiration', 'option_type', 'open', 'high', 'low', 'close','bid_size','bid', 'ask_size', 'ask', 'gamma', 'theta', 'rho', 'underlying_bid', 'underlying_ask', 'active_underlying_price', 'implied_underlying_price', 'level2', 'relative_strike']
        call_fixed_datasets.drop(columns=delcolumns, inplace=True), put_fixed_datasets.drop(columns=delcolumns, inplace=True)
        call_ntm_datasets.drop(columns=delcolumns, inplace=True), put_ntm_datasets.drop(columns=delcolumns, inplace=True)



        # Extract only the defined maturities above (mats), from each available day of data in the ticker_smile lists
        # Not neccessary to do this for fixed and ntm, only atm
        csmile_gammatheta = [[day[i] for i in mats] for day in ticker_c_ticker_smile]
        psmile_gammatheta = [[day[i] for i in mats] for day in ticker_p_ticker_smile]
        callsmile_gammatheta_con = rearrange_concat(csmile_gammatheta)
        putsmile_gammatheta_con = rearrange_concat(psmile_gammatheta)

        # List length should be equal as theyre should pull data from the same exact maturirites; each list item shall represent a unique maturity
        len(TIKCER_calls_atm_maturities)==len(callsmile_gammatheta_con)

        c_skew_theta, p_skew_theta = skew_theta_func(callsmile_gammatheta_con,TIKCER_calls_atm_maturities),skew_theta_func(putsmile_gammatheta_con,TIKCER_puts_atm_maturities)
        iv_call_smile, iv_put_smile = variable_surface(c_skew_theta, p_skew_theta)
        rho_call_smile, rho_put_smile = variable_surface(c_skew_theta, p_skew_theta, variable='rho')
        rho_call_smile, rho_put_smile = rho_call_smile.loc[:, '85%':'110%'].add_suffix('_rho'), rho_put_smile.loc[:, '85%':'110%'].add_suffix('_rho')

        vega_call_smile, vega_put_smile = variable_surface(c_skew_theta, p_skew_theta, variable='vega')
        vega_call_smile, vega_put_smile = vega_call_smile.loc[:, '85%':'110%'].add_suffix('_vega'), vega_put_smile.loc[:, '85%':'110%'].add_suffix('_vega')
        
        
        # Merge; include fixed and ntm here
        call_ = iv_call_smile.merge(vega_call_smile, left_index=True, right_index=True)
        put_ = iv_put_smile.merge(vega_put_smile, left_index=True, right_index=True)
        
        call = call_.merge(rho_call_smile, left_index=True, right_index=True)
        put = put_.merge(rho_put_smile, left_index=True, right_index=True)
        # merge fixed and ntm
        c_fixed_ntm_combo = call_fixed_datasets.merge(call_ntm_datasets, left_index=True, right_index=True, suffixes=('_fixed', '_ntm'))
        p_fixed_ntm_combo = put_fixed_datasets.merge(put_ntm_datasets, left_index=True, right_index=True, suffixes=('_fixed', '_ntm'))

        # merge atm with depth book
        calls = call.merge(call_datasets, left_index=True, right_index=True)
        puts = put.merge(put_datasets, left_index=True, right_index=True)

        # merge all
        calls.merge(c_fixed_ntm_combo, left_index=True, right_index=True)
        puts.merge(p_fixed_ntm_combo, left_index=True, right_index=True)
        
        
#        l = [TIKCER_calls_atm_maturities, TIKCER_puts_atm_maturities,c_skew_theta, p_skew_theta]
        l = [calls, puts]
        dataset_day.append(l)
    c,p = 0,1
    calls, puts = [day[c] for day in dataset_day], [day[p] for day in dataset_day]

    # Set quote_datetime as index to isolate from suffixes, add term suffixes before merging; changing data from long to wide. 
    C = [[df.set_index('quote_datetime').iloc[:, :][(df.set_index('quote_datetime').loc[:, 'maturity']==expiry)].iloc[:, 1:].add_suffix(f'_term_{term}')  for term, expiry in enumerate(df.loc[:, 'maturity'].sort_values().unique())] for df in calls]
    P = [[df.set_index('quote_datetime').iloc[:, :][(df.set_index('quote_datetime').loc[:, 'maturity']==expiry)].iloc[:, 1:].add_suffix(f'_term_{term}')  for term, expiry in enumerate(df.loc[:, 'maturity'].sort_values().unique())] for df in puts]
    
    CALLS = [reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True), data) for data in C]
    PUTS = [reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True), data) for data in P]
    
    spotdrop_c = [[c for c in df.columns if c.startswith('active_underlying_price')] for df in CALLS]
    spotdrop_p = [[c for c in df.columns if c.startswith('active_underlying_price')] for df in PUTS]
    
    CALLS = [df.drop(columns=spotdrop_c[i][1:]) for i, df in enumerate(CALLS)]
    PUTS = [df.drop(columns=spotdrop_p[i][1:]) for i, df in enumerate(PUTS)]

    
    return CALLS, PUTS

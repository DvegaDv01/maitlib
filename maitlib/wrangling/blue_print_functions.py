#from fastai.tabular.all import *
#import plotly.graph_objects as go
import pandas as pd
import sys
from pandas.tseries.holiday import *
from pandas.tseries.offsets import *


def add_remove_features_old(df):
    #any new columns added to this function should potentially also be added to 
    xf = pd.DataFrame()
    data = df.copy()
    data.drop(columns=['underlying_symbol','implied_underlying_price',
                       'underlying_ask', 'underlying_bid', 'root', 'level2'], inplace=True)
    data['start_yr'] = [pd.Timestamp(i).replace(month=1, day=1, hour=0, minute=0, second=0) 
                      for i in data.loc[:,'quote_datetime']]
    data['end_yr'] = [i.replace(month=12, day=31,year= i.year, hour=23, minute=59) 
                      for i in data.loc[:,'start_yr']]    
    data['expiration'] = [pd.Timestamp(i).replace(hour=16, minute=0) 
                      for i in data.loc[:,'expiration']]
    data['dte'] = data.loc[:,'expiration'].subtract(pd.to_datetime(data.loc[:,'quote_datetime']))
    data['t'] = data.loc[:,'dte'].dt.days
    data['mid_price'] = (data['bid'] + data['ask'])/2
    data['mid_size'] = (data['bid_size'] + data['ask_size'])/2
    data['implied_volatility'] = data.implied_volatility * 100
    return data



def concat_trading_dte(df, num_of_trade_dte):
    # used below in add_remove_function

    data = df.copy(deep=True)
    data = {e:data[(data.loc[:,'expiration']==e)] for e in num_of_trade_dte.keys()}
    l = []
    for e in num_of_trade_dte.keys():
        data[e]['trading_DTE'] = num_of_trade_dte[e]
        l.append(pd.DataFrame(data[e]))
    return pd.concat(l)



class TradingCalendar(AbstractHolidayCalendar):
    # Month and Day parameters must be set otherwise Nonetype will return for holiday.dates
    rules = [
     Holiday('NewYears', month=1, day=1, observance=sunday_to_monday),
     Holiday('MartinLutherKingJr', month=1, day=6, offset=WeekOfMonth(week=2, weekday= 0)),
     Holiday('Presidents Day', month=2, day=1, offset=WeekOfMonth(week=2, weekday= 0)),
     Holiday('Good Friday', month=1, day=1, offset=[Easter(), Day(-2)]),
     Holiday('Memorial Day', month=5, day=1, offset=LastWeekOfMonth(weekday=0)),
     Holiday('Independence Day', month=7, day=4, observance=nearest_workday),
     Holiday('Labor Day', month=9, day=1, offset=WeekOfMonth(week=0, weekday= 0)),
     Holiday('Thanksgiving', month=11, day=1, offset=WeekOfMonth(week=3, weekday= 3)),
     Holiday('Christmas', month=12, day=25, observance=nearest_workday)]





def add_remove_features(df):    
    
    traders_calendar = TradingCalendar()

    #any new columns added to this function should potentially also be added to 
    trading_calendar = CustomBusinessDay(calendar=traders_calendar)
    # Tried to fix the assignment issue
    xf = df.copy(deep=True)
    data = xf.copy(deep=True)
    dr_columns = ['underlying_symbol','underlying_ask', 'underlying_bid', 'root', 'level2']
    for i in dr_columns: 
        if i in data.columns:
            data.drop(columns=i, inplace=True)
    else:
        pass
    data['start_yr'] = [pd.Timestamp(i).replace(month=1, day=1, hour=0, minute=0, second=0) 
                      for i in data.loc[:,'quote_datetime']]
    data['end_yr'] = [i.replace(month=12, day=31,year= i.year, hour=23, minute=59) 
                      for i in data.loc[:,'start_yr']]
    
    start_yr = pd.Timestamp(data.loc[:,'quote_datetime'].unique()[0]).replace(month=1, day=1, hour=0, minute=0, second=0)
    yr_of_last_contract = pd.Timestamp(data.loc[:,'expiration'].unique()[-1]).replace(month=12, day=31, hour=23, minute=59)
    trading_days = pd.date_range(start=start_yr, end=yr_of_last_contract, freq=trading_calendar).to_frame(index=False)
    expiry = pd.to_datetime(data.loc[:,'expiration'].sort_values().unique()).to_series()
    current_time = pd.to_datetime(df.loc[:,'quote_datetime'].unique())
   

    # expiry_trading_day returns a tuple of which day of the Trading calendar year, that the expiry falls on with expiration
    # Example '2019-11-29 00:00:00' is the 230th day on the Trading Calendar
    unverified_expiry_trading_day = [(e,trading_days[trading_days.iloc[:,0] == e.strftime('%Y-%m-%d')].index) for e in expiry]
    expiry_trading_day= [(i,e[0]) for i,e in unverified_expiry_trading_day if e.size != 0]
    current_time_trading_day = [trading_days[trading_days.iloc[:,0] == t.strftime('%Y-%m-%d')].index[0] for t in current_time]
    # trading_DTE returns the number of trading days until expiration
    trading_DTE = {t.strftime('%Y-%m-%d'):(d - current_time_trading_day[0]) for t,d in expiry_trading_day}
    standard_trading_hours = [abs(current_time[-2] - t) for t in current_time][:-1]
    
    
    data['mid_price'] = (data['bid'] + data['ask'])/2
    data['mid_size'] = (data['bid_size'] + data['ask_size'])/2
    data['implied_volatility'] = data.implied_volatility
    
    result = concat_trading_dte(data, trading_DTE)
    result['expiration'] = [pd.Timestamp(i).replace(hour=16, minute=0) 
                      for i in result.loc[:,'expiration']]
    
    # To run data quality on correct expirations provided by Cboe, return 'unverified_expiry_trading_day' variable
    # and make sure it matches 'expiry_trading_day' if it does not that means Cboe provided data that is not consistent
    # with the trading calendar
    return (result, expiry_trading_day, unverified_expiry_trading_day)
    



# Seperate data into two sets: calls and puts and returns a DATAFRAME (NOT list of dfs) for each
# Multilevel indexing might be better for this function
def sep_cp(data):
    df, _, _ = add_remove_features(data)
    x = df.groupby(['option_type','quote_datetime'])
    calls,puts = [],[]
    for i,j in x:
        if i[0]=='C':
            calls.append(j)
        elif i[0]=='P':
            puts.append(j)
#   This should be a class or something, dont repeat code
    c_shape =np.array(calls).shape
    c_indx = c_shape[0] * c_shape[1]
    c_coln = c_shape[2]
    p_shape =np.array(puts).shape
    p_indx = p_shape[0] * c_shape[1]
    p_coln = p_shape[2]
    calls_df = pd.DataFrame(np.array(calls).reshape(c_indx,c_coln))
    puts_df = pd.DataFrame(np.array(puts).reshape(p_indx,p_coln))
    calls_df.columns=df.columns
    puts_df=df.columns
    return calls_df,puts_df




# Current working function
def bounded_pct(df,p):
    # df should be a dataframe, not list
    x = df.sort_values(by=['strike'])
    pct = (int(p)/100) * x.active_underlying_price
    hstrike = x.active_underlying_price + pct
    lstrike = x.active_underlying_price - pct
    x = x[(x.strike>=lstrike) & (x.strike<=hstrike)]
    
    return x.sort_values(by=['quote_datetime','expiration','strike'])



#def adjust_expiration(data):
#    X = []
#    for df in data:
#        adjusted_dates = [(e, pd.offsets.Week(weekday=4).rollback(e).strftime('%Y-%m-%d')) for e in df.expiration.unique() if pd.to_datetime(e).dayofweek==5]
#        df.replace(adjusted_dates[0], adjusted_dates[1])
#        X.append(df)
#    return X




# Testing function to get rid of weird Nans; so far works perfect
# after removing column renaming for t column
# Multilevel indexing might be better for this function
def vol_surface_setup_old(data,p):
    ''' This function keeps all data into seperate  df instead on one combined'''
    surface_df = pd.DataFrame()
    for i,df in enumerate(data):
        df = bounded_pct(df,p)
        vol_surface_data = df.loc[:,['quote_datetime','option_type','expiration',
                                     'strike','implied_volatility','t','active_underlying_price']].reset_index(drop=True)
        vol_surface_data.rename(columns = {'t': f'dte'}, inplace=True)
        surface_df = pd.concat([surface_df,vol_surface_data])
        # df_list returns a list of dfs for each expiration full of all strikes,
        # looping through each timestamp
    df_list = [surface_df[(surface_df.quote_datetime==i) & (surface_df.expiration==j)] 
               for i in surface_df.quote_datetime.unique() for j in surface_df.expiration.unique()]
       
    return surface_df, df_list # Use df_list to create futures contract for each month



# consider renaming to data_setup; although returns vol surf data, other data incuded. Or seperate df_list instructions into a seperate function
def vol_surface_setup_old2(data,p):
# Multilevel indexing might be better for this function
    df = bounded_pct(data,p)
    vol_surface_data = df.loc[:,['quote_datetime','option_type','expiration','strike','mid_price',
                                 'implied_volatility','t','active_underlying_price']].reset_index(drop=True)
    vol_surface_data = vol_surface_data.sort_values(by=['quote_datetime','expiration','strike'])
    vol_surface_data.rename(columns = {'t': f'dte'}, inplace=True)
    df_list = [vol_surface_data[(vol_surface_data.quote_datetime==i) & (vol_surface_data.expiration==j)] 
               for i in vol_surface_data.quote_datetime.unique() for j in vol_surface_data.expiration.unique()]
       
    return vol_surface_data, df_list # Use df_list to create futures contract for each month



def vol_surface_setup(data,p):
    '''Use data_setup function instead and pass True to vol surf arguement'''
# Multilevel indexing might be better for this function
    df = add_remove_features(bounded_pct(data.copy(),p))
#   May want to rid of this column filtering line 
    vol_surface_data = df.loc[:,['quote_datetime','option_type','expiration','strike','mid_price',
                                 'implied_volatility','t','active_underlying_price']].reset_index(drop=True)
    vol_surface_data.rename(columns = {'t': f'dte'}, inplace=True)
    df_list = [[(vol_surface_data[(vol_surface_data.quote_datetime==i) & (vol_surface_data.expiration==j)]) 
                for j in vol_surface_data.expiration.unique()] for i in vol_surface_data.quote_datetime.unique()]
    df_tup = [(str(j[i].quote_datetime.unique()).strip("[]''"),pd.concat(df_list[i])) 
              for i,j in enumerate(df_list)]
    df_dict = {i:j for i,j in df_tup}

       
    return vol_surface_data, df_dict


def data_setup(data, p, vol_surface_data=False):
    '''returns tuple if vol_surface_data=True
        otherwise a dict'''
# Multilevel indexing might be better for this function
    data_source = data.copy(deep=True)
    df, expiry_trading_day, unverified_expiry_trading_day = add_remove_features(bounded_pct(data_source,p))
    expiry_trading_day = [pd.Timestamp(e).replace(hour=16, minute=0) 
                         for e, _ in expiry_trading_day]
    
    #df.rename(columns = {'t': f'time_to_expiry'}, inplace=True)
    df_list = [[(df[(df.quote_datetime==i) & (df.expiration==j)]) 
               for j in expiry_trading_day] for i in df.quote_datetime.unique()]
    
    result = [pd.concat(i)
              for i in df_list]
    result = {str(i.quote_datetime.unique()).strip("[]''"):i for i in result}
##    result = {i:j for i,j in df_tup}

    if vol_surface_data==True:
        v = df.loc[:,['quote_datetime','option_type','expiration','strike','mid_price',
                                         'implied_volatility','time_to_expiry','active_underlying_price']].reset_index(drop=True)
        result = (result,v) 
    else:
        pass
    return (result, expiry_trading_day)



def term_structure_data(df):
# Multilevel indexing might be better for this function    
    x = [pd.DataFrame(df.groupby('strike').get_group(i)) for i in df.strike.unique()]


    calls = [[p.groupby(['option_type','quote_datetime']).get_group(('C',t)) 
              for t in p.quote_datetime.unique()] for o,p in enumerate(x)]
    puts = [[p.groupby(['option_type','quote_datetime']).get_group(('P',t)) 
             for t in p.quote_datetime.unique()] for o,p in enumerate(x)]
    
    return calls, puts # a list of dataframes ready to plot term structures




# Currently only saves one plot but does work.

def vol_surface_plot(surface_df):
    # surface_df should be a list of dataframes. ie df_list from volsurfdata
    for d,g in enumerate(x[:4]):
        df = g[d].loc[:,['t','strike','implied_volatility']].reset_index(drop=True)
        fig = go.Figure(data=[go.Mesh3d(x=df.loc[:,'t'],y=df.loc[:,'strike'],
                                    z=df.loc[:,'implied_volatility'], color='cyan', opacity=0.50)])
        fig.write_html('/mnt/c/Users/danfo/ben/plots/test_plots')
    
# Do the same when creating function for plotting the term structure
# plot should be 2-D tho, simliar to code below; Not 3-D
'''xxx.plot(x='expiration', y='implied_volatility',
                 figsize=(10,6), legend=False)'''





# Multilevel indexing might be better for this function
# this function needs to be verified before implimenting
# it seems off. Make this a callback function using fastai
# def contracts_quantile(df):
#     """this function calculates the total num of option contracts
#     across all expirations then indivdually calculate the quantile
#     of each expiry accross total"""
#     x = df.groupby('expiration')
#     ql = np.arange(0,1,.05).round(2)  # range to return ventile (quantile)
#     contr_tot = [(i,j) for i in df.expiration.unique() # return tuple of each exiration and the 
#                  for j in [len(x.get_group(i))]]       # total number of option contracts for
#     contr_tot = pd.DataFrame(contr_tot)                # the particular expiration
#     qvals = pd.qcut(contr_tot[1], q=20, labels=ql)
    
#     return f"quantile, num of contracts, expiration: ", list(zip(qvals, contr_tot[1], contr_tot[0]))



def option_chain(df, values, index, columns):
    '''Index should be Date, exipry/dte and strike
    col should be option_type
    values should atleast be mid_price. Can inclue Implied Vol & others
    '''
    return df.pivot_table(values=values, index=index, columns=columns)



def all_option_chains_old(data_list,indx,col,values):
    full_chain = [option_chain(i,indx,col,values) for i in data_list]
    return full_chain


def all_option_chains(data_list, values, index, columns):
    if type(data_list) == dict:
        full_chain = {dt:option_chain(data_list[dt], values=values, index=index, columns=columns) for dt in data_list.keys()}
        
    elif type(data_list) == list:
        full_chain = [option_chain(i, values=values, index=index, columns=columns) for i in data_list]
    return full_chain



def min_difference_old(df,indx, col,values):
    if type(df)== list:
        data = []
        for d in df:
            sliced_data = d.loc[:,'mid_price']
            d['call_put_price_diff'] = (sliced_data['C'] - 
                                           sliced_data['P']).abs()
            # Remember all expirations may have different strike prices thus changing the mid price.
            # Expect different ATM strike across different expirations even at the same quote_datetime

            data.append(d)
    else:
        # takes data add columns then create pivot table in form of option chain
        data = add_remove_features(df.copy())
        data = option_chain(data,indx, col, values)
        # Use multilevel indexing to slice through data
        sliced_data = data.loc[(slice(None),slice(None)),'mid_price']
        CPdiff = np.array((sliced_data['C'] - sliced_data['P']).abs())
        data['call_put_price_diff'] = CPdiff
    
    return data



def min_difference(df,indx, col,values):
    if type(df)== dict:
        data = {}
        for d in df.keys():
            data_dict = df.copy()
            sliced_data = data_dict[d].loc[:,'mid_price'].copy()
            data_dict[d]['call_put_price_diff'] = (sliced_data['C'] - 
                                           sliced_data['P']).abs()
            # Remember all expirations may have different strike prices thus changing the mid price.
            # Expect different ATM strike across different expirations even at the same quote_datetime

            data.update({d:data_dict[d]})
        return data
    else:
        # takes data add columns then create pivot table in form of option chain
        # Use multilevel indexing to slice through data
        data = df.copy()
        sliced_data = data.loc[(slice(None),slice(None)),'mid_price']
        CPdiff = np.array((sliced_data['C'] - sliced_data['P']).abs())
        data['call_put_price_diff'] = CPdiff
    
    return data
    




def forward_price(df):
    K = df.loc[df.loc[:,'call_put_price_diff']==df.loc[:,'call_put_price_diff'].min()].index[0][2]
    r = 0.08
    t = None
    delta = df.loc[:,'call_put_price_diff'].min()
    return K + np.exp(r*t) * delta




def at_the_money(data_list):
    ''' Remember all expirations may have different strike prices thus changing with strike is ATM.
    Expect different ATM strike across different expirations even at the same quote_datetime'''
    if isinstance(data_list, list):
        atm = []
        for df in data_list:
            near_the_money = df[df.loc[:,'call_put_price_diff'].eq(df.loc[:,'call_put_price_diff'].min())].index[0][2]
            atm.append(near_the_money)     
        
    elif isinstance(data_list, dict):
        data = [[data_list[i].loc[(i, j)].loc[:,'call_put_price_diff'].idxmin(), i, j] for i in data_list.keys() 
                for j in data_list[i].index.get_level_values('expiration').unique()]
        df = pd.DataFrame(data)
        group = df.groupby([2])
        # A dictionary of ATM strikes at each expiry for each timestamp
        atm = {str(i):j for i,j in group}        

    return atm


def atm_dictionary(data_dict, ATM):
    '''data_dict should be the result of the data_setup func
       & ATM should be the result of the at_the_money function
       which uses the result of the min_diff func
       which uses the result of all_option_chains'''
    d_list = [data_dict[i][(data_dict[i].loc[:,'expiration'].isin([str(j.iloc[2])])) &
                           (data_dict[i].loc[:,'quote_datetime'].isin([j.iloc[1]])) & (data_dict[i].loc[:,'strike'].isin([j.iloc[0]]))] 
              for i in data_dict.keys() for ix in ATM for _,j in ATM[ix].iterrows()
              if data_dict[i][(data_dict[i].loc[:,'expiration'].isin([str(j.iloc[2])])) &
                              (data_dict[i].loc[:,'quote_datetime'].isin([j.iloc[1]])) &
                              (data_dict[i].loc[:,'strike'].isin([j.iloc[0]]))].empty ==False]
    atm_dict = {(i.loc[:,'quote_datetime'].values[0], str(pd.Period(i.loc[:,'expiration'].values[0], freq='s'))):i
                for i in d_list}
    
    return atm_dict



def atm_series(data:dict, quote_time, expiry):
    
    time_structure_list = [[data[i, j] for j in expiry] for i in quote_time]
    time_structure = [pd.concat(i) for i in time_structure_list]
    
    atm_vol_list = [[data[i, j] for i in quote_time] for j in expiry]
    atm_vol = [pd.concat(i) for i in atm_vol_list]
    
    return time_structure, atm_vol



def fix_datetime(x, hour=None, minute=None, second=None):
    # Quick utility func to adjust date time
    return pd.Timestamp(x).replace(hour=hour, minute=minute, second=second)



def yr_unit_time(data_list, set_index=True):
    # FOR TERM STRUCTURE; set proper time to be used as index and x axis on plots
    l = []
    secs_in_1day = (60 * 60 * 24)
    mins_in_1day = 1440
    for i in data_list:
        # 1 equaling to 1 year, 0.5 equaling to 6 months
        decaying_time = pd.to_datetime(i.loc[:, 'expiration']) - pd.to_datetime(i.loc[:, 'quote_datetime'])
        i.loc[:, 'trading_DTE'] = i.loc[:, 'trading_DTE'] + ((decaying_time.dt.seconds/60)/mins_in_1day)
        i.loc[:, 'expiry_as_pct_of_yr'] = (i.loc[:, 'trading_DTE'].astype(float) * secs_in_1day)/ (secs_in_1day * 252)
        i.loc[:, 'quotetime_as_pct_of_yr'] = quotetime_as_pct_of_year(i)
        l.append(i)
    return l


def filterSPY7(spydf:list):
    
    return [df[df.loc[:,'root']!='SPY7'].sort_values(by='expiration').reset_index(drop=True) for df in spydf]



def quotetime_as_pct_of_year(df, column_name='quote_datetime'):
    """pass in a dataframe.
    For column name you may need to pass (i.e 'quote_datetime_atm_s1c') for some
    datasets"""     
    tn = pd.to_datetime(df.loc[:,column_name].copy(deep=True))
    yr_end = [pd.Timestamp(time).replace(month=12, day=31, hour=23, minute=59, second=59) 
              for time in df.loc[:,column_name].copy(deep=True)]
    time_df = pd.DataFrame([pd.Series(tn),pd.Series(yr_end)]).T
    time_df.columns= ['quote_datetime','yr_end']
    time_df.loc[:,'time_diff'] = time_df.loc[:, 'yr_end'] - time_df.loc[:, 'quote_datetime']
    time_df.loc[:,'remaining_days_in_yr'] = time_df.loc[:, 'time_diff'].dt.days + ((time_df.loc[:, 'time_diff'].dt.seconds/60)/60)/24
    time_df.loc[:, 'quotetime_as_pct_of_yr'] = time_df.loc[:,'remaining_days_in_yr']/365
    
    return 1 - time_df.loc[:, 'quotetime_as_pct_of_yr']



def clock_time(data_list, set_index=True):
    # FOR ATM VOL; set proper time to be used as index and x axis on plots
    l = []
    for i in data_list:
        i.loc[:, 'clock_time'] = [fix_datetime(row).strftime('%H:%M:%S') for row in i.loc[:, 'quote_datetime']]
        if set_index==True:
            i.set_index('clock_time', inplace=True)
        else:
            pass
        l.append(i)
    return l




def is_third_friday(dt_object):
    """This function main use is for SPX options
    NOT SPY but add as feature to both"""
    
    # dt_object should be string
    dt_object = pd.to_datetime(dt_object)
    t_weekday = dt_object.weekday()
    t_day = dt_object.day
    return t_weekday == 4 & 15 <= t_day <=21



def near_next_after_term(data_list):
    # list should be a list of option chain dataframes
    # other terms print provides rest of terms after near & next
    near_term = data_list[0].index[0][1]
    next_term = data_list[1].index[0][1]
    other_terms = []
    for i in data_list[2:]:
        other_terms.append(i.index[0][1])
    return near_term,next_term,other_terms


class Auto_cond:
    def __init__(self, col, val, c_op):
        self.opp = ' & '
        self.ss = ''
        self.col = col
        self.val = val
        self.c_op = c_op
        self.cl = len(self.col)


    def auto_cond(self):
        '''Takes columns or col:{List} comparing operator or c_op:{List} and values or val:{List} and
        automate the creation of conditionals required to a filter df.
        Returns a conditional statement
        Example: x = Auto_cond(col, val, c_op).auto_cond()'''
        
        for i,j in enumerate(zip(self.col,self.val,self.c_op)):

            if ((i!=self.cl-1) and (i!=0)):

                if isinstance(self.val[i], str):
                    self.s_str = f"({self.col[i]}{self.c_op[i]}'{self.val[i]}')"
                    self.ss += self.opp + self.s_str

                elif isinstance(self.val[i], (int, float)):
                    self.s_int = f"({self.col[i]}{self.c_op[i]}{self.val[i]})"
                    self.ss += self.opp + self.s_int

            elif (i==0):

                if isinstance(self.val[i], str):
                    self.s_str = f"({self.col[i]}{self.c_op[i]}'{self.val[i]}')"
                    self.ss += self.s_str

                elif isinstance(self.val[i], (int, float)):
                    self.s_int = f"({self.col[i]}{self.c_op[i]}{self.val[i]})"
                    self.ss += self.s_int

            elif (i==self.cl-1):

                if isinstance(self.val[i], str):
                    self.s_str = f"({self.col[i]}{self.c_op[i]}'{self.val[i]}')"
                    self.ss += self.opp + self.s_str

                elif isinstance(self.val[i], (int, float)):
                    self.s_int = f"({self.col[i]}{self.c_op[i]}{self.val[i]})"
                    self.ss += self.opp + self.s_int


        return self.ss
    
    def conditional_concat(self, data, **kwargs):
        '''Takes param data:{dict} and iterates each key, applying
        the conditionals defined by auto_cond function, to each
        value:{DataFrame} in dictionary keys then concatenates
        into one df.
        Make sure col & val is defined prior to calling this func.
        Example: x = Auto_cond(col,val,c_op)
                 x.conditional_concat(data) '''
    
        cc = pd.concat([data[j][data[j].eval(Auto_cond(self.col, self.val, self.c_op).auto_cond())]
                        for i,j in enumerate(data.keys())])
        return cc





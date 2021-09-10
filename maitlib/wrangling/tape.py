import os
import sys
from fastai.tabular.all import *
sys.path.append('/home/ben/MaitlandOak-linux/maitai')
from maitai.skewtheta import *
from maitai.greeks import *
from pandas.tseries.offsets import DateOffset
import awswrangler as wr
import boto3
import boto3.session



def load_frame(filepath,fnames,use_cols=False,cols=None ):
    data = []
    for fn in fnames:
        print(f'Reading: {fn}')

        if use_cols==True:
            df = pd.read_csv(filepath/fn, index_col=False, usecols=cols)
            data.append(df)
            
        else:
            df = pd.read_csv(filepath/fn)
            data.append(df)
    return data



def load_tape(X, filenames=None, rate_path=None, inx=None):
    cols = ['underlying_symbol', 'quote_datetime', 'sequence_number', 'root', 'expiration', 'strike', 'option_type', 'exchange_id', 'trade_size', 'trade_price', 'trade_condition_id', 'canceled_trade_condition_id', 'best_bid', 'best_ask', 'trade_iv', 'trade_delta', 'underlying_bid', 'underlying_ask', 'number_of_exchanges']
    
    tape = []
    for _,row in X.reset_index(drop=True).loc[inx:inx,:].iterrows():
        
        tape_slice1 = row.tape[0]
        tape_slice2 = row.tape[1]
        path1 = row.tape_path

       
        tape_df = load_frame(path1,os.listdir(path1)[tape_slice1:tape_slice2], use_cols=True, cols=cols)
        tape_df = pd.concat(tape_df).reset_index(drop=True)

        
        tape_data = tape_greeks(tape_df, rate_path)
                     
        tape.append(tape_data)
        
    return tape


def load_vix(X, filenames=None, inx=None,spot_only=True):
    
    vix = []
    for _,row in X.reset_index(drop=True).loc[inx:inx,:].iterrows():
        
        vix_slice1 = row.vix[0]
        vix_slice2 = row.vix[1]
        path1 = row.vix_path

        vix_df = load_frame(path1,os.listdir(path1)[vix_slice1:vix_slice2])
        vix_df = pd.concat(vix_df).reset_index(drop=True)
        vix_df.loc[:, 'quote_datetime'] = pd.to_datetime(vix_df.loc[:, 'quote_datetime'])

        if spot_only==True:
            vix_df = vix_df.loc[:,['quote_datetime','active_underlying_price']]
            
        vix.append(vix_df)
        
    return vix


def load_calcs(X, filenames=None, ratepath=None, run_greeks=False, inx=None, root='SPY'):
    
    calcs = []
    for _,row in X.reset_index(drop=True).loc[inx:inx,:].iterrows():
        
        calcs_slice1 = row.option_calcs[0]
        calcs_slice2 = row.option_calcs[1]
        path1 = row.calcs_path
        
        
        calcs_df = load_frame(path1,os.listdir(path1)[calcs_slice1:calcs_slice2])
        calcs_df = pd.concat(calcs_df).reset_index(drop=True)

        calcs_df = calcs_df[(calcs_df.root==root)].reset_index(drop=True)
        calcs_df.loc[:, 'mid_price'] = (calcs_df.loc[:, 'bid'] + calcs_df.loc[:, 'ask'])/2
        

        if run_greeks==True:
            
            rates = pd.read_csv(ratepath/os.listdir(ratepath)[1])
            rates.loc[:, 'Date'] = pd.to_datetime(rates.loc[:, 'Date'])            
            rdate = pd.to_datetime(calcs_df.loc[0, 'quote_datetime']).strftime('%Y-%m-%d')
            rates_date = rates[(rates.loc[:, 'Date']==rdate)]
            RatesDate = pd.DataFrame(np.repeat(rates_date.values, len(calcs_df.index), axis=0), columns=rates_date.columns)
            calcs_df = calcs_df.merge(RatesDate.iloc[:, 1:], left_index=True, right_index=True)
            calcs_df = dte_helper(calcs_df)

            r = RatesDate.loc[0, '10 yr']
            C_ = calcs_df[(calcs_df.loc[:, 'option_type']=='C') | (calcs_df.loc[:, 'option_type']=='c')].reset_index(drop=True)
            P_ = calcs_df[(calcs_df.loc[:, 'option_type']=='P') | (calcs_df.loc[:, 'option_type']=='p')].reset_index(drop=True)
            calls,puts = OptionGreeks(C_, r=r).run(), OptionGreeks(P_, r=r).run()
            Calls, Puts = C_.merge(calls, left_index=True, right_index=True), P_.merge(puts, left_index=True, right_index=True)
            calcs_df = pd.concat([Calls,Puts]).reset_index(drop=True)
        
        calcs.append(calcs_df)
        
    return calcs


def tape_data_prep(option_tape):
    tape_dataframe = add_trade_columns(option_tape)
    tape_dataframe_ = net_exposures(tape_dataframe)
    timesets = create_timesets()
    calls,puts = between_time_setup(tape_dataframe_, timesets=timesets)
    c,p = [dataframe.reset_index(drop=True) for index, dataframe in calls.groupby(['quote_datetime','expiration', 'strike'])],[dataframe.reset_index(drop=True) for index, dataframe in puts.groupby(['quote_datetime','expiration', 'strike'])]        

    wrangle_calls, wrangle_puts = consolidate_trades(c),consolidate_trades(p)
    wrangle_calls.reset_index(drop=True, inplace=True), wrangle_puts.reset_index(drop=True, inplace=True);
    c1 = delta_buckets(wrangle_calls)
    p1 = delta_buckets(wrangle_puts)

    option_data = c1.set_index('quote_datetime').merge(p1.set_index('quote_datetime'), left_index=True, right_index=True, suffixes=('_calls','_puts')).reset_index()
    o_df = pd.concat([wrangle_calls, wrangle_puts]).sort_values(by=['quote_datetime', 'expiration','strike']).reset_index(drop=True)
    exposures = higher_order_exposure(o_df)

    D1 = option_data.set_index('quote_datetime').merge(exposures.set_index('quote_datetime'), left_index=True,right_index=True).reset_index()
    return D1


def option_calc_data_prep(calc_dataframe, vix_dataframe):
    
    vix_df = vix_spot_match(calc_dataframe,vix_dataframe)
    spot_vix = [(spot,vix_df[(vix_df.quote_datetime==spot)].vix_spot.unique().item()) for spot in vix_df.quote_datetime.unique()]
    spot_vix = pd.DataFrame(spot_vix, columns=['quote_datetime','vix_spot'])
    
    calcs_5min = dte_helper(calc_dataframe)
    df_ = [gamma_theta(gdataframe_) for gdataframe_ in [calcs_5min]]
    data = pd.concat(df_).reset_index(drop=True)
    standex = time_as_pct_of_year(data)
    stan_dataframe = standex.merge(data, left_index=True, right_index=True)
    maturity_idx = expiry_hlpr(stan_dataframe)
    maturities = maturity_hlpr(maturity_idx)
    # Fixed Strike, BS IV & SKEW
    call_fixed_surface,put_fixed_surface = fixed_surface(stan_dataframe, terms=maturities)
    
    o = call_fixed_surface.merge(put_fixed_surface.drop(columns=['active_underlying_price', 'spot_chg']), left_index=True, right_index=True, suffixes=(None, None)).reset_index()
    o.loc[:, 'quote_datetime'] = pd.to_datetime(o.loc[:, 'quote_datetime'])
    spot_vix.loc[:, 'quote_datetime'] = pd.to_datetime(spot_vix.loc[:, 'quote_datetime'])
    o = o.merge(spot_vix, left_on='quote_datetime', right_on='quote_datetime')
#    o.loc[:,'vix_spot'] = spot_vix.loc[:, 'vix_spot']

    return o
    



def vix_spot_match(df, vix):
    v = []
    for t in df.loc[:, 'quote_datetime'].sort_values().unique():

        like_x = df[(df.loc[:,'quote_datetime']==t)]
        vix_ = vix[(vix.loc[:,'quote_datetime']==t)].reset_index(drop=True)
        vx = pd.DataFrame(np.full_like(like_x.iloc[:,0].values, vix_.loc[0, 'active_underlying_price'], dtype=float))
        vt = pd.DataFrame(np.full_like(like_x.iloc[:,0].values, vix_.loc[0, 'quote_datetime']))        
        vx_ = vt.merge(vx, left_index=True, right_index=True)
        v.append(vx_)
    v_ = pd.concat(v).reset_index(drop=True)
    v_.columns = ['quote_datetime','vix_spot']
    v_.loc[:,'quote_datetime'] = pd.to_datetime(v_.loc[:,'quote_datetime'])
    return v_

def net_exposures(df):

    data = df.copy(deep=True)
    
    l = []
    s = []
    lg,sg,lc,sc,lv,sv,lvn,svn,lvt,svt,lsp, ssp, lzm, szm, ldvdv, sdvdv, lvega, svega, lely, sely = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
    
    for option_type, onBidAsk, tradeDelta, tradeGamma, tradeCharm, tradeVolga, tradeVanna, tradeVeta, tradeSpeed, tradeZomma, tradeDvannaDvol, tradeVega, tradeElasticity in zip(data.loc[:, 'option_type'], data.loc[:, 'onBidAsk'].values, data.loc[:, 'total_trade_delta'].values, data.loc[:, 'total_trade_Gamma'].values, data.loc[:, 'total_trade_charm'].values, data.loc[:, 'total_trade_volga'].values, data.loc[:, 'total_trade_vanna'].values, data.loc[:, 'total_trade_veta'].values,data.loc[:, 'total_trade_speed'].values, data.loc[:, 'total_trade_zomma'].values, data.loc[:, 'total_trade_DvannaDvol'].values, data.loc[:, 'total_trade_vega'].values, data.loc[:, 'total_trade_elasticity'].values
):
        
        # Assumption: Long Call
        if (option_type == 'C') and (onBidAsk == 'A'):
            long_delta = tradeDelta
            long_gamma = tradeGamma
            long_charm = tradeCharm
            long_volga = tradeVolga
            long_vanna = tradeVanna
            long_veta = tradeVeta
            long_speed = tradeSpeed
            long_zomma = tradeZomma
            long_DvannaDvol = tradeDvannaDvol
            long_vega = tradeVega
            long_elasticity = tradeElasticity 

            short_delta = 0
            short_gamma = 0
            short_charm = 0             
            short_volga = 0
            short_vanna = 0
            short_veta = 0
            short_speed = 0
            short_zomma = 0
            short_DvannaDvol = 0
            short_vega = 0
            short_elasticity = 0
            

            
            
        # Assumption: Short Call
        elif (option_type == 'C') and (onBidAsk == 'B'):
            long_delta = 0
            long_gamma = 0
            long_charm = 0
            long_volga = 0
            long_vanna = 0
            long_veta = 0
            long_speed = 0
            long_zomma = 0
            long_DvannaDvol = 0 
            long_vega = 0
            long_elasticity = 0
            
            short_delta = tradeDelta
            short_gamma = tradeGamma
            short_charm = tradeCharm
            short_volga = tradeVolga
            short_vanna = tradeVanna
            short_veta = tradeVeta
            short_speed = tradeSpeed
            short_zomma = tradeZomma
            short_DvannaDvol= tradeDvannaDvol
            short_vega = tradeVega
            short_elasticity = tradeElasticity
            
            
        # Assumption: Long Put            
        elif (option_type == 'P') and (onBidAsk == 'B'):
            long_delta = tradeDelta
            long_gamma = tradeGamma
            long_charm = tradeCharm
            long_volga = tradeVolga
            long_vanna = tradeVanna
            long_veta = tradeVeta
            long_speed = tradeSpeed
            long_zomma = tradeZomma
            long_DvannaDvol = tradeDvannaDvol
            long_vega = tradeVega
            long_elasticity = tradeElasticity

            short_delta = 0
            short_gamma = 0
            short_charm = 0
            short_volga = 0
            short_vanna = 0
            short_veta = 0
            short_speed = 0
            short_zomma = 0
            short_DvannaDvol = 0
            short_vega = 0 
            short_elasticity = 0

        # Assumption: Short Put            
        elif (option_type == 'P') and (onBidAsk == 'A'):
            long_delta = 0
            long_gamma = 0
            long_charm = 0
            long_volga = 0
            long_vanna = 0
            long_veta = 0
            long_speed = 0
            long_zomma = 0
            long_DvannaDvol = 0
            long_vega = 0
            long_elasticity = 0
            

            short_delta = tradeDelta
            short_gamma = tradeGamma
            short_charm = tradeCharm
            short_volga = tradeVolga
            short_vanna = tradeVanna
            short_veta = tradeVeta
            short_speed = tradeSpeed
            short_zomma = tradeZomma
            short_DvannaDvol= tradeDvannaDvol
            short_vega = tradeVega
            short_elasticity = tradeElasticity
            
        elif (onBidAsk != 'B') and (onBidAsk != 'A') or (onBidAsk == 'B') and (onBidAsk == 'A'):
            long_delta = 0
            long_gamma = 0
            long_charm = 0
            long_volga = 0
            long_vanna = 0
            long_veta = 0
            long_speed = 0
            long_zomma = 0
            long_DvannaDvol = 0
            long_vega = 0
            long_elasticity = 0
            

            short_delta = 0
            short_gamma = 0
            short_charm = 0
            short_volga = 0
            short_vanna = 0
            short_veta = 0
            short_speed = 0
            short_zomma = 0
            short_DvannaDvol = 0
            short_vega = 0
            short_elasticity = 0
            
            
        l.append(long_delta)
        s.append(short_delta)
        lg.append(long_gamma)
        sg.append(short_gamma)
        lc.append(long_charm)
        sc.append(short_charm)
        lv.append(long_volga)
        sv.append(short_volga)
        lvn.append(long_vanna)
        svn.append(short_vanna)
        lvt.append(long_veta)
        svt.append(short_veta)
        lsp.append(long_speed)
        ssp.append(short_speed)
        lzm.append(long_zomma)
        szm.append(short_zomma)
        ldvdv.append(long_DvannaDvol)
        sdvdv.append(short_DvannaDvol)
        lvega.append(long_vega)
        svega.append(short_vega)
        lely.append(long_elasticity)
        sely.append(short_elasticity)
        
        
    data.loc[:, 'long_delta'] = l
    data.loc[:, 'short_delta'] = s
    data.loc[:, 'long_gamma'] = lg
    data.loc[:, 'short_gamma'] = sg
    data.loc[:, 'long_charm'] = lc
    data.loc[:, 'short_charm'] = sc
    data.loc[:, 'long_volga'] = lv
    data.loc[:, 'short_volga'] = sv    
    data.loc[:, 'long_vanna'] = lvn
    data.loc[:, 'short_vanna'] = svn
    data.loc[:, 'long_veta'] = lvt
    data.loc[:, 'short_veta'] = svt
    data.loc[:, 'long_speed'] = lsp
    data.loc[:, 'short_speed'] = ssp
    data.loc[:, 'long_zomma'] = lzm
    data.loc[:, 'short_zomma'] = szm
    data.loc[:, 'long_dvannadvol'] = ldvdv
    data.loc[:, 'short_dvannadvol'] = sdvdv
    data.loc[:, 'long_vega'] = lvega
    data.loc[:, 'short_vega'] = svega
    data.loc[:, 'long_elasticity'] = lely
    data.loc[:, 'short_elasticity'] = sely
    
    
    
    
    data.loc[:, 'long_delta'] = data.loc[:, 'long_delta'].abs()
    data.loc[:, 'short_delta'] = data.loc[:, 'short_delta'].abs()
    
    data.loc[:, 'long_gamma'] = data.loc[:, 'long_gamma'].abs()
    data.loc[:, 'short_gamma'] = data.loc[:, 'short_gamma'].abs()
    data.loc[:, 'long_charm'] = data.loc[:, 'long_charm'].abs()
    data.loc[:, 'short_charm'] = data.loc[:, 'short_charm'].abs()
    data.loc[:, 'long_volga'] = data.loc[:, 'long_volga'].abs()
    data.loc[:, 'short_volga'] = data.loc[:, 'short_volga'].abs()
    data.loc[:, 'long_vanna'] = data.loc[:, 'long_vanna'].abs()
    data.loc[:, 'short_vanna'] = data.loc[:, 'short_vanna'].abs()
    data.loc[:, 'long_veta'] = data.loc[:, 'long_veta'].abs()
    data.loc[:, 'short_veta'] = data.loc[:, 'short_veta'].abs()
    data.loc[:, 'long_speed'] = data.loc[:, 'long_speed'].abs()
    data.loc[:, 'short_speed'] = data.loc[:, 'short_speed'].abs()
    data.loc[:, 'long_zomma'] = data.loc[:, 'long_zomma'].abs()
    data.loc[:, 'short_zomma'] = data.loc[:, 'short_zomma'].abs()
    data.loc[:, 'long_dvannadvol'] = data.loc[:, 'long_dvannadvol'].abs()
    data.loc[:, 'short_dvannadvol'] = data.loc[:, 'short_dvannadvol'].abs()
    data.loc[:, 'long_vega'] = data.loc[:, 'long_vega'].abs()
    data.loc[:, 'short_vega'] = data.loc[:, 'short_vega'].abs()
    data.loc[:, 'long_elasticity'] = data.loc[:, 'long_elasticity'].abs()
    data.loc[:, 'short_elasticity'] = data.loc[:, 'short_elasticity'].abs()
    
    data.sort_values(by='quote_datetime', inplace=True)
    
    return data.reset_index(drop=True)



def fn_date(s,k=None,k_sub=None):
    # kth occurrence
    k = k - k_sub
    l = []
    for i in s:

        seperated_string = i.split('_')
        unused_str = seperated_string[:k]
        date = seperated_string[k]
        if date.endswith('.zip'):
            date = date.split('.')[0]
        else:
            pass
        l.append(date)
    return np.array(l)


def index_selector(alldates_path, alldates,uniquedates):
    """all dates should include files uploaded in 5 min increments but may be full day and the same as 'uniquedates'
        uniquedates should only include each unique date"""
    selected_indices, included, missing =[], [], []
    
    for i in uniquedates:
        ind = np.where(alldates == i)[0]
        if ind.size != 0:
            strt = ind[0]
            fin = ind[-1] + 1

            selected_indices = [f'{i}', (strt,fin), alldates_path]
            included.append(selected_indices)
        else:
            missing.append(f'{i} not in list')
        
        
    return included, missing


def create_timesets():
    hr, minute_interval = list(range(9,17,1)), list(range(0,60,5))
    minute_interval[0]='00'
    minute_interval[1]='05'
    minute_interval.append(59)
    mn = []
    for hour in hr:
        mm = []
        for minute in minute_interval:
            if minute==59:
                hour += 1
                minute = '00'
                mm.append(f"{hour}:{minute}") 
            else:
                mm.append(f"{hour}:{minute}")
        mn.append(mm)
    time_intervals = mn
    time_intervals[0] = time_intervals[0][6:]
    time_intervals[-1] = time_intervals[-1][:4]
    five_min_interval = [t for l in time_intervals for t in l]
    intervals = [(t,t+1) for t in range(0,len(five_min_interval),1)]
    intervals = intervals[:-1]
    timesets = [(five_min_interval[t1],five_min_interval[t2]) for t1, t2 in intervals]
    [timesets.remove((ti,tj)) for ti,tj in timesets if ti==tj]
    return timesets


def between_time_setup(df, timesets):
    df.loc[:,'quote_datetime'] = pd.to_datetime(df.loc[:,'quote_datetime'])
    calls = df[df.loc[:, 'option_type']=='C'].copy(deep=True)
    puts = df[df.loc[:, 'option_type']=='P'].copy(deep=True)
    c,p = [],[]
    for t1, t2 in timesets:
        Calls = calls.set_index('quote_datetime').between_time(t1, t2).reset_index()
        Puts = puts.set_index('quote_datetime').between_time(t1, t2).reset_index()
        
        Calls.loc[:,'datetime'] = pd.to_datetime(Calls.loc[:,'quote_datetime'].dt.strftime('%Y-%m-%d').map(lambda x: f"{x} {t2}"))
        Puts.loc[:,'datetime'] = pd.to_datetime(Puts.loc[:,'quote_datetime'].dt.strftime('%Y-%m-%d').map(lambda x: f"{x} {t2}"))

        c.append(Calls)
        p.append(Puts)
    C,P = pd.concat(c),pd.concat(p)
    C.loc[:, 'quote_datetime_tick'] = C.loc[:, 'quote_datetime']
    C.loc[:, 'quote_datetime'] = pd.to_datetime(C.loc[:, 'datetime']) 
    C.drop(columns=['datetime'], inplace=True)

    P.loc[:, 'quote_datetime_tick'] = P.loc[:, 'quote_datetime']
    P.loc[:, 'quote_datetime'] = pd.to_datetime(P.loc[:, 'datetime'])
    P.drop(columns=['datetime'], inplace=True)

    return C.reset_index(drop=True),P.reset_index(drop=True)



def consolidate_trades(data):

    dataframe_X = []
    
    for dataframe in data:
        first_quote_datetime = dataframe.loc[0, 'quote_datetime_tick']
        last_quote_datetime = dataframe.loc[:,'quote_datetime_tick'][len(dataframe.quote_datetime_tick) - 1]
        quote_datetime = dataframe.loc[0, 'quote_datetime']
        expiration = dataframe.loc[0, 'expiration']
        strike = dataframe.loc[0, 'strike']
        option_type = dataframe.loc[0, 'option_type']
        contracts_traded = dataframe.loc[:, 'trade_size'].sum()
        number_of_trades = len(dataframe)
        avg_num_of_contracts_per_trade = dataframe.loc[:, 'trade_size'].mean()
        avg_trade_price = dataframe.loc[:, 'trade_price'].mean()
        avg_trade_delta = dataframe.loc[:, 'trade_delta'].mean()
        avg_trade_iv = dataframe.loc[:, 'trade_iv'].mean()
        num_onbid = len(['T'  for o in dataframe.loc[:, 'onBidAsk'] if o =='B'])
        num_onask = len(['T'  for o in dataframe.loc[:, 'onBidAsk'] if o =='A'])
        num_intermarket = len(['T'  for o in dataframe.loc[:, 'onBidAsk'] if o =='I'])
        num_gt_ask = len(['T'  for o in dataframe.loc[:, 'onBidAsk'] if o =='G'])
        num_lt_bid = len(['T'  for o in dataframe.loc[:, 'onBidAsk'] if o =='L'])
        num_other = len(['T'  for o in dataframe.loc[:, 'onBidAsk'] if o =='O'])
        amt_of_deltas = dataframe.loc[:, 'total_trade_delta'].sum()
        notional_amount = dataframe.loc[:, 'total_trade_notional'].sum()
        long_delta = dataframe.loc[:, 'long_delta'].sum()
        short_delta = dataframe.loc[:, 'short_delta'].sum()
        
        long_gamma = dataframe.loc[:, 'long_gamma'].sum()
        short_gamma = dataframe.loc[:, 'short_gamma'].sum()
              
        long_charm = dataframe.loc[:, 'long_charm'].sum()
        short_charm = dataframe.loc[:, 'short_charm'].sum()

        long_volga = dataframe.loc[:, 'long_volga'].sum()
        short_volga = dataframe.loc[:, 'short_volga'].sum()

        long_vanna = dataframe.loc[:, 'long_vanna'].sum()
        short_vanna = dataframe.loc[:, 'short_vanna'].sum()

        long_veta = dataframe.loc[:, 'long_veta'].sum()
        short_veta = dataframe.loc[:, 'short_veta'].sum()
        
        long_speed = dataframe.loc[:, 'long_speed'].sum()
        short_speed = dataframe.loc[:, 'short_speed'].sum()

        long_zomma = dataframe.loc[:, 'long_zomma'].sum()
        short_zomma = dataframe.loc[:, 'short_zomma'].sum()

        long_dvannadvol = dataframe.loc[:, 'long_dvannadvol'].sum()
        short_dvannadvol = dataframe.loc[:, 'short_dvannadvol'].sum()

        long_vega = dataframe.loc[:, 'long_vega'].sum()
        short_vega = dataframe.loc[:, 'short_vega'].sum()

        long_elasticity = dataframe.loc[:, 'long_elasticity'].sum()
        short_elasticity = dataframe.loc[:, 'short_elasticity'].sum()

        
        df = pd.DataFrame([first_quote_datetime,last_quote_datetime,quote_datetime,expiration,strike,option_type,contracts_traded,number_of_trades,avg_num_of_contracts_per_trade,avg_trade_price,avg_trade_delta,avg_trade_iv,num_onbid,num_onask,num_intermarket,num_gt_ask,num_lt_bid,num_other,amt_of_deltas,notional_amount,long_delta,short_delta,long_gamma,short_gamma,long_charm,short_charm,long_volga,short_volga,long_vanna,short_vanna,long_veta,short_veta,long_speed,short_speed,long_zomma,short_zomma,long_dvannadvol,short_dvannadvol,long_vega,short_vega,long_elasticity,short_elasticity]).T
        
        df.columns = ['first_quote_datetime','last_quote_datetime','quote_datetime','expiration','strike','option_type','contracts_traded','number_of_trades','avg_num_of_contracts_per_trade','avg_trade_price','avg_trade_delta','avg_trade_iv','num_onbid','num_onask','num_intermarket','num_gt_ask','num_lt_bid','num_other','amt_of_deltas','notional_amount','long_delta','short_delta','long_gamma','short_gamma','long_charm','short_charm','long_volga','short_volga','long_vanna','short_vanna','long_veta','short_veta','long_speed','short_speed','long_zomma','short_zomma','long_dvannadvol','short_dvannadvol','long_vega','short_vega','long_elasticity','short_elasticity']
        
        
        dataframe_X.append(df)

    Data = pd.concat(dataframe_X)
    return Data
                                





def add_trade_columns(data):
    
    df = data.copy(deep=True)
    l = []
    for price,bid,ask in zip(df.loc[:, 'trade_price'].values,df.loc[:, 'best_bid'].values,df.loc[:, 'best_ask'].values):
        if price == bid and price!= ask:
            onbidask = 'B'
        elif price == ask and price!= bid:
            onbidask = 'A'
        elif bid < price < ask:
            onbidask = 'I'
        elif bid > price:
            onbidask = 'L'
        elif ask < price:
            onbidask = 'G'
        else:
            onbidask = 'O'
        l.append(onbidask)
    
    df.loc[:, 'onBidAsk'] = l    
    
    df.loc[:, 'total_trade_delta'] = df.loc[:, 'trade_delta'] * df.loc[:, 'trade_size']
    df.loc[:, 'total_trade_Gamma'] = df.loc[:, 'gammaP'] * df.loc[:, 'trade_size']
    df.loc[:, 'total_trade_charm'] = df.loc[:, 'charm'] * df.loc[:, 'trade_size']    
    df.loc[:, 'total_trade_speed'] = df.loc[:, 'speedP'] * df.loc[:, 'trade_size']
    df.loc[:, 'total_trade_elasticity'] = df.loc[:, 'Elasticity'] * df.loc[:, 'trade_size']
    df.loc[:, 'total_trade_vega'] = df.loc[:, 'vegaP'] * df.loc[:, 'trade_size']
    df.loc[:, 'total_trade_volga'] = df.loc[:, 'vommaP'] * df.loc[:, 'trade_size']    
    df.loc[:, 'total_trade_vanna'] = df.loc[:, 'vanna'] * df.loc[:, 'trade_size']    
    df.loc[:, 'total_trade_DvannaDvol'] = df.loc[:, 'DvannaDvol'] * df.loc[:, 'trade_size']    
    df.loc[:, 'total_trade_veta'] = df.loc[:, 'vega_bleed'] * df.loc[:, 'trade_size']
    df.loc[:, 'total_trade_zomma'] = df.loc[:, 'zommaP'] * df.loc[:, 'trade_size']

    df.loc[:, 'total_trade_notional'] = (df.loc[:, 'trade_price'] * 100) * df.loc[:, 'trade_size']
    df.loc[:, 'quote_datetime'] = pd.to_datetime(df.loc[:, 'quote_datetime'])
    
    
    return df

def higher_order_exposure(option_data):
    
    dummydata = [option_data[(option_data.loc[:,'quote_datetime']==i)] for i in option_data.loc[:,'quote_datetime'].unique()]
    quote_datetime = [pd.to_datetime(i.loc[:, 'quote_datetime'].unique()) for i in dummydata]
    XX =[]
    for df in dummydata:
        delta_exposure = df.loc[:, 'long_delta'].sum() - df.loc[:, 'short_delta'].sum()

        gamma_exposure = df.loc[:, 'long_gamma'].sum() - df.loc[:, 'short_gamma'].sum()

        charm_exposure = df.loc[:, 'long_charm'].sum() - df.loc[:, 'short_charm'].sum()

        volga_exposure = df.loc[:, 'long_volga'].sum() - df.loc[:, 'short_volga'].sum()

        vanna_exposure = df.loc[:, 'long_vanna'].sum() - df.loc[:, 'short_vanna'].sum()

        veta_exposure = df.loc[:, 'long_veta'].sum() - df.loc[:, 'short_veta'].sum()

        speed_exposure = df.loc[:, 'long_speed'].sum() - df.loc[:, 'short_speed'].sum()

        zomma_exposure = df.loc[:, 'long_zomma'].sum() - df.loc[:, 'short_zomma'].sum()

        dvannadvol_exposure = df.loc[:, 'long_dvannadvol'].sum() - df.loc[:, 'short_dvannadvol'].sum()

        vega_exposure = df.loc[:, 'long_vega'].sum() - df.loc[:, 'short_vega'].sum()

        elasticity_exposure = df.loc[:, 'long_elasticity'].sum() - df.loc[:, 'short_elasticity'].sum()
        
        xx = pd.DataFrame([delta_exposure,gamma_exposure,charm_exposure,speed_exposure,vega_exposure,volga_exposure,vanna_exposure,veta_exposure,zomma_exposure,dvannadvol_exposure,elasticity_exposure]).T
        xx.columns=['delta_exposure','gamma_exposure','charm_exposure','speed_exposure','vega_exposure','volga_exposure','vanna_exposure','veta_exposure','zomma_exposure','dvannadvol_exposure','elasticity_exposure']
        XX.append(xx)
    data_ = pd.concat(XX).reset_index(drop=True).cumsum()
    data_.loc[:,'quote_datetime'] = quote_datetime
    return data_

def tape_greeks(tape_df, rate_path=None):
    tape_df.loc[:, 'active_underlying_price'] = (tape_df.loc[:, 'underlying_ask'] + tape_df.loc[:,'underlying_bid'])/2
    tape_df.loc[:, 'implied_volatility'] = tape_df.loc[:, 'trade_iv']
    tape_df.loc[:, 'ask'] = tape_df.loc[:, 'best_ask']
    tape_df.loc[:, 'bid'] = tape_df.loc[:, 'best_bid']
#    tape_df.loc[:, 'delta'] = tape_df.loc[:, 'trade_delta']


    rates = pd.read_csv(rate_path/os.listdir(rate_path)[1])
    rates.loc[:, 'Date'] = pd.to_datetime(rates.loc[:, 'Date'])
    rdate = pd.to_datetime(tape_df.loc[0, 'quote_datetime']).strftime('%Y-%m-%d')
    rates_date = rates[(rates.loc[:, 'Date']==rdate)]
    RatesDate = pd.DataFrame(np.repeat(rates_date.values, len(tape_df.index), axis=0), columns=rates_date.columns)
    tape_df = tape_df.merge(RatesDate.iloc[:, 1:], left_index=True, right_index=True)
    tape_df = dte_helper(tape_df)

    r = RatesDate.reset_index(drop=True).loc[0, '10 yr']

    C_ = tape_df[(tape_df.loc[:, 'option_type']=='C') | (tape_df.loc[:, 'option_type']=='c')].reset_index(drop=True)
    P_ = tape_df[(tape_df.loc[:, 'option_type']=='P') | (tape_df.loc[:, 'option_type']=='p')].reset_index(drop=True)
    calls,puts = OptionGreeks(C_, r=r).run(), OptionGreeks(P_, r=r).run()
    Calls, Puts = C_.merge(calls, left_index=True, right_index=True), P_.merge(puts, left_index=True, right_index=True)
    tape_df = pd.concat([Calls,Puts]).reset_index(drop=True)
    return tape_df


def delta_buckets(data_set):

    deltalist = [n/10 for n in range(1,10)]
    ds = []
    
    for t in data_set.loc[:, 'quote_datetime'].unique():
        
        storedlist = []
        for i in deltalist:
            ii = round(i - .1, 2)
            data = data_set[(data_set.loc[:, 'quote_datetime']==t) & (data_set.loc[:, 'avg_trade_delta'].abs()>=ii) & (data_set.loc[:, 'avg_trade_delta'].abs()<=i)].copy(deep=True)
            num_of_unique_expiries = len(data.loc[:,'expiration'].unique())
            traded_liquidity = ((data.loc[:,'avg_trade_price'] * 100) * data.loc[:,'contracts_traded']).sum()
            amt_of_traded_deltas = ((data.loc[:,'avg_trade_delta'] * 100) * data.loc[:,'contracts_traded']).sum()
            avg_trade_delta = data.loc[:,'avg_trade_delta'].mean()
            avg_trade_iv = data.loc[:,'avg_trade_iv'].mean()
            contracts_traded = data.loc[:,'contracts_traded'].sum()
            num_of_trades = data.loc[:,'number_of_trades'].sum()
            avg_num_of_contracts_per_trade = data.loc[:,'avg_num_of_contracts_per_trade'].mean()
            num_onbid = data.loc[:,'num_onbid'].sum()
            num_onask = data.loc[:,'num_onask'].sum()
            num_intermarket = data.loc[:,'num_intermarket'].sum()
            num_gt_ask = data.loc[:,'num_gt_ask'].sum()
            num_lt_bid = data.loc[:,'num_lt_bid'].sum()
            num_other = data.loc[:,'num_other'].sum()
            T = t
#            testdf.loc[:, ['num_of_unique_expiries','traded_liquidity','amt_of_traded_deltas','avg_trade_iv','contracts_traded','num_of_trades','avg_num_of_contracts_per_trade','num_onbid','num_onask','num_intermarket','num_gt_ask','num_lt_bid','num_other']]
            
            colnames = f'_{int(ii*100)}-{int(i*100)}delta_bucket'
            aa = pd.DataFrame([num_of_unique_expiries,traded_liquidity,amt_of_traded_deltas,avg_trade_delta,avg_trade_iv,contracts_traded,num_of_trades,avg_num_of_contracts_per_trade,num_onbid,num_onask,num_intermarket,num_gt_ask,num_lt_bid,num_other]).T
            aa.columns = ['num_of_unique_expiries','traded_liquidity','amt_of_traded_deltas','avg_trade_delta','avg_trade_iv','contracts_traded','num_of_trades','avg_num_of_contracts_per_trade','num_onbid','num_onask','num_intermarket','num_gt_ask','num_lt_bid','num_other']
            aa = aa.add_suffix(f'_{int(ii*100)}-{int(i*100)}d_bucket')
            aa.loc[:,'quote_datetime'] = T
            storedlist.append(aa)
#        zzz = pd.DataFrame(storedlist, columns=['num_of_unique_expiries','traded_liquidity','amt_of_traded_deltas','avg_trade_delta','avg_trade_iv','contracts_traded','num_of_trades','avg_num_of_contracts_per_trade','num_onbid','num_onask','num_intermarket','num_gt_ask','num_lt_bid','num_other','colnames'])
        ds.append(storedlist)
    dd = pd.concat([reduce(lambda  left,right: pd.merge(left,right,left_index=True, how='outer', right_index=True), xx) for xx in ds]).reset_index(drop=True)
    return dd
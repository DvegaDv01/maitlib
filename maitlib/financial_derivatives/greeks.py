import os
import sys
from fastai.tabular.all import *
from pandas.tseries.offsets import DateOffset
from scipy.stats import norm


# calculate the greeks
# ['delta', 'gamma', 'vega', 'theta', 'rho', 'lambdaa','epsilon','vanna','charm','volga','DvegaDtime','veta','color','ultima','speed','zomma']

class OptionGreeks:

    def __init__(self, df, r=0.0058, q=0.0):
        # black-scholes inputs
        # b = r - q; but some equations ask for r - b
        # which is the same as r - (r - q)
        self.rate = torch.tensor(r, requires_grad=True)
        self.dividend = torch.tensor(q, requires_grad=True)
        self.b = self.rate - self.dividend
        
        self.df = df
        self.df.loc[:,'expiration'] = pd.to_datetime(df.loc[:,'expiration']).dt.strftime('%Y-%m-%d').map(lambda x: pd.Timestamp(x).replace(hour=16, minute=15, second=0))                
        self.df.loc[:,'dte'] = self.df.loc[:,'expiration'] - pd.to_datetime(self.df.loc[:,'quote_datetime'])
        self.maturity = self.df.expiration
        self.quote_datetime = self.df.quote_datetime
        self.spot = torch.tensor(self.df.active_underlying_price.values, requires_grad=True)
        self.strike = torch.tensor(self.df.strike.values, requires_grad=True)
        self.vol = torch.tensor(self.df.implied_volatility.values, requires_grad=True)
        self.time = torch.tensor((self.df.dte.dt.days/365).values, requires_grad=True)
        
        self.price = torch.tensor(((self.df.bid + self.df.ask)/2).values, requires_grad=True)
        
        #self.cboe_gamma = torch.tensor(self.df.gamma.values, requires_grad=True)
        #self.cboe_vega = torch.tensor(self.df.vega.values, requires_grad=True)

        # utility functions
        self.cdf = torch.distributions.Normal(0,1).cdf
        self.pdf = lambda x: torch.distributions.Normal(0,1).log_prob(x).exp()
    
            # black-scholes formula
        self.d1 = (torch.log(self.spot/self.strike) + ((self.rate - self.dividend) + self.vol*self.vol/2)*self.time) / (self.vol*torch.sqrt(self.time))
        self.d2 = (torch.log(self.spot/self.strike) + ((self.rate - self.dividend) - self.vol*self.vol/2)*self.time) / (self.vol*torch.sqrt(self.time))

    '''vanna: the change in delta for a 1 pt change in volatility
       gammaP: the amount delta will chg for a 1% move in spot(instead of the traditional 1pt move)
       zommaP: the amount gamma chg for a 1% move in volatility (offers a good indication of which strikes to use in higher/lower iv option valuation)
       speedP: the amount gamma chg for a 1% move in spot (a high-speed value indicates that gamma is very sensitive to changes in the underlying asset)
       colorP(Gamma pct bleed or GammaTheta): the change in gamma with repspect to small changes in time to maturity
       volga(vega convexity or vomma): the sensitivity of vega, to changes in implied volatility. Divided by 10,000 to rep a one pct pt chg in vol
       ultima: volga[vomma]'s sensitivity to a change in volatility. Divide by 1m to get metric of 1 vol pt move
       '''
        
    def run(self):
        greek_columns = ['Strike', 'Delta','vanna', 'charm', 'DvannaDvol', 'Elasticity', 'Gamma', 'gammaP', 'zommaP', 'speedP', 'colorP', 'Vega', 
                'vegaP', 'volga', 'vommaP', 'ultima', 'vega_bleed', 'DdeltaDvar', 'Theta', 'Bleed_Offset_Volatility']

        if self.df.option_type.all()=='C':
            o = OptionGreeks(self.df).call()
            g = pd.DataFrame([i.detach().numpy() for i in o], index=greek_columns).T
            
        elif self.df.option_type.all()=='P':
            o = OptionGreeks(self.df).put()
            g = pd.DataFrame([i.detach().numpy() for i in o], index=greek_columns).T
        
        
        return g

    def call(self):

        ov = self.spot * torch.exp(-self.dividend*self.time) * self.cdf(self.d1) - self.strike*torch.exp(-self.rate*self.time) * self.cdf(self.d2) 
        
        # run backpropagation to calculate option greeks
#        ov.backward()
        
        # Delta related Greeks
        delta = (torch.exp((self.b - self.rate)*self.time)) * self.cdf(self.d1)
        vanna = (((-torch.exp((self.b-self.rate)*self.time) * self.d2) / self.vol) * self.pdf(self.d1))
        charm = -torch.exp((self.b-self.rate)*self.time) *  (self.pdf(self.d1) * ((self.b/(self.vol*torch.sqrt(self.time))) - (self.d2/(2*self.time)))  + (self.b-self.rate)*self.cdf(self.d1))
        DvannaDvol = (vanna * (1/self.vol) * ((self.d1*self.d2) - (self.d1/self.d2) -1)) / 10000
        Elasticity = torch.exp((self.b-self.rate)*self.time) * self.cdf(self.d1) * (self.spot/self.price)
        
        # Gamma related Greeks
        gamma = self.pdf(self.d1) * torch.exp((self.b - self.rate)*self.time) / (self.spot * self.vol * torch.sqrt(self.time))
        gammaP = self.pdf(self.d1) * torch.exp((self.b - self.rate)*self.time) / (100 * self.vol * torch.sqrt(self.time))
        zommaP = gammaP * ((self.d1 * self.d2 - 1)/self.vol)
        speedP = -(gammaP * (1 + ( self.d1/(self.vol * torch.sqrt(self.time) )))) / self.spot
        colorP = gammaP * (self.rate - self.b + ((self.b * self.d1)/(self.vol * torch.sqrt(self.time))) + ((1 - self.d1 * self.d2)/(2 * self.time)))

        # Vega related Greeks
        vega = self.spot * torch.exp((self.b - self.rate)* self.time) * self.pdf(self.d1) * torch.sqrt(self.time)
        vegaP = (self.vol/10) * vega
        volga = (vega * ( (self.d1 * self.d2) / self.vol))
        vommaP = vegaP * ( (self.d1 * self.d2) / self.vol)
        ultima = ((vega * ( (self.d1 * self.d2) / self.vol)) * (1/self.vol) * (self.d1*self.d2 - (self.d1/self.d2) - (self.d2/self.d1) - 1))
        vega_bleed = (vega * ((self.rate - self.b) + ((self.b*self.d1)/(self.vol*torch.sqrt(self.time))) - ((1 + self.d1*self.d2)/(2*self.time)))) / 36500
        DdeltaDvar = -self.spot * torch.exp((self.b - self.rate)*self.time) * self.pdf(self.d1) * (self.d2/(2 * (self.vol*self.vol)))

        # Call Theta (one day time decay)
        theta = (((-self.spot*torch.exp((self.b-self.rate)*self.time) * self.pdf(self.d1) * self.vol) / (2*torch.sqrt(self.time)))  -  ((self.b-self.rate) * self.spot*torch.exp((self.b-self.rate)*self.time)*self.cdf(self.d1))  -  self.rate*self.strike*torch.exp(-self.rate*self.time)*self.cdf(self.d2)) / 365       
        Bleed_Offset_Volatility = theta/vega

        
        epsilon = self.dividend.grad
        strike_greek = self.strike.grad
        
        return [self.strike, delta, vanna/100, charm, DvannaDvol, Elasticity, gamma, gammaP, zommaP, speedP, colorP, vega/100, vegaP, volga/10000, vommaP, ultima, vega_bleed, DdeltaDvar, theta, Bleed_Offset_Volatility]

    def put(self):

        ov = self.strike*torch.exp(-self.rate*self.time) * self.cdf(-self.d2) - self.spot * torch.exp(-self.dividend*self.time) * self.cdf(-self.d1)
        
        # run backpropagation to calculate option greeks
#        ov.backward()
        
        # Delta related Greeks        
        delta = (torch.exp((self.b - self.rate)*self.time)) * (self.cdf(self.d1) - 1)
        vanna = (((-torch.exp((self.b-self.rate)*self.time) * self.d2) / self.vol) * self.pdf(self.d1))
        charm =  -torch.exp((self.b-self.rate)*self.time) *  (self.pdf(self.d1) * ((self.b/(self.vol*torch.sqrt(self.time))) - (self.d2/(2*self.time)))  - (self.b-self.rate)*self.cdf(-self.d1))
        DvannaDvol = (vanna * (1/self.vol) * ((self.d1*self.d2) - (self.d1/self.d2) -1)) / 10000
        Elasticity = torch.exp((self.b-self.rate)*self.time) * (self.cdf(self.d1) - 1) * (self.spot/self.price)
        
        # Gamma related Greeks        
        gamma = self.pdf(self.d1) * torch.exp((self.b - self.rate)*self.time) / (self.spot * self.vol * torch.sqrt(self.time))
        gammaP = self.pdf(self.d1) * torch.exp((self.b - self.rate)*self.time) / (100 * self.vol * torch.sqrt(self.time))
        zommaP = gammaP * ((self.d1 * self.d2 - 1)/self.vol)
        speedP = -(gammaP * (1 + ( self.d1/(self.vol * torch.sqrt(self.time) )))) / self.spot        
        colorP = gammaP * (self.rate - self.b + ((self.b * self.d1)/(self.vol * torch.sqrt(self.time))) + ((1 - self.d1 * self.d2)/(2 * self.time)))
        
        # Vega related Greeks
        vega = self.spot * torch.exp((self.b - self.rate)* self.time) * self.pdf(self.d1) * torch.sqrt(self.time)
        vegaP = (self.vol/10) * vega
        volga = (vega * ( (self.d1 * self.d2) / self.vol))
        vommaP = vegaP * ( (self.d1 * self.d2) / self.vol)
        ultima = ((vega * ( (self.d1 * self.d2) / self.vol)) * (1/self.vol) * (self.d1*self.d2 - (self.d1/self.d2) - (self.d2/self.d1) - 1))
        vega_bleed = (vega * ((self.rate - self.b) + ((self.b*self.d1)/(self.vol*torch.sqrt(self.time))) - ((1 + self.d1*self.d2)/(2*self.time)))) / 36500
        DdeltaDvar = -self.spot * torch.exp((self.b - self.rate)*self.time) * self.pdf(self.d1) * (self.d2/(2 * (self.vol*self.vol)))
        
        # Put Theta (one day time decay)
        theta = (((-self.spot*torch.exp((self.b-self.rate)*self.time) * self.pdf(self.d1) * self.vol) / (2*torch.sqrt(self.time)))  +  ((self.b-self.rate) * self.spot*torch.exp((self.b-self.rate)*self.time)*self.cdf(-self.d1))  +  self.rate*self.strike*torch.exp(-self.rate*self.time)*self.cdf(-self.d2)) / 365
        Bleed_Offset_Volatility = theta/vega
        
        
        epsilon = self.dividend.grad
        strike_greek = self.strike.grad
        return [self.strike, delta, vanna/100, charm, DvannaDvol, Elasticity, gamma, gammaP, zommaP, speedP, colorP, vega/100, vegaP, volga/10000, vommaP, ultima, vega_bleed, DdeltaDvar, theta, Bleed_Offset_Volatility]


















def greeks(df, rate = 0.0058):
    
    q = 1
    spot = df.loc[:, 'active_underlying_price']
    strike = df.loc[:, 'strike']
    vol = df.loc[:, 'implied_volatility']
    cboe_gamma = df.loc[:, 'gamma']
    cboe_vega = df.loc[:, 'vega']
    maturity = df.loc[:, 'dte'].dt.days

    """ask, bid, change, contractSymbol, expiration,vol, inTheMoney, lastPrice, lastTradeDate,
    openInterest, percentChange, strike, volume,maturity, BA_Spread, DaysSinceLastTrade,
    spot, Call_Put_Flag = list(column_list)"""
    d1 = (np.log(spot / strike) + (maturity / 365) * (rate - q + (vol ** 2) / 2)) / (vol * np.sqrt(maturity / 365))
    d2 = d1 - vol * np.sqrt(maturity / 365)
    gamma = (((np.exp(-q * maturity / 365) / (spot * vol * np.sqrt(maturity / 365))) * 1 / np.sqrt(2 * np.pi)) * np.exp(
        (-d1 ** 2) / 2)).round(4)
    vega = ((((spot * np.exp(-q * maturity / 365)) * np.sqrt(maturity / 365)) / 100) * 1 / np.sqrt(2 * 3.14159) * np.exp(
        (-d1 ** 2) / 2)).round(4)
    vanna = (np.exp(-q * maturity / 365) * np.sqrt(maturity / 365) * (d2 / vol) * np.exp(-(d1 ** 2) / 2) / (2 * np.pi)).round(4)

    volga = ((spot * (np.exp(q * maturity / 365)) * np.sqrt(maturity / 365) * (np.exp(-(d1 ** 2) / 2) * d1 * d2) / (
        np.sqrt(2 * np.pi)) * vol)).round(4)
    ultima = (- cboe_vega / (vol * vol) * (d1 * d2 * (1 - d1 * d2) + d1 * d1 + d2 * d2)).round(4)
    color = (- np.exp(-q * maturity / 365) * 1 / np.sqrt(2 * 3.14159) * np.exp((-d1 ** 2) / 2) * 1 / (
                2 * spot * (maturity / 365) * vol * np.sqrt(maturity / 365)) * (2 * q * (maturity / 365) + 1 + (
                2 * (rate - q) * (maturity / 365) - d2 * vol * np.sqrt(maturity / 365) / (
                    vol * np.sqrt(maturity / 365)) * d1))).round(4)
    
    zomma = (cboe_gamma * ((d1 * d2 - 1) / vol)).round(4)
    speed = (- cboe_gamma / spot * (d1 / (vol * np.sqrt(maturity / 365)) + 1)).round(4)
    veta = (- spot * np.exp(-q * maturity / 365) * 1 / np.sqrt(2 * 3.14159) * np.exp((-d1 ** 2) / 2) * np.sqrt(
        maturity / 365) * (
                       q + ((rate - q) * d1) / (vol * np.sqrt(maturity / 365) - (1 + d1 * d2) / (2 * (maturity / 365))))).round(4)
    

    
    C = df[(df.loc[:, 'option_type']=='C')]
    cspot = C.loc[:, 'active_underlying_price']
    cstrike = C.loc[:, 'strike']    
    cmaturity = C.loc[:, 'dte'].dt.days
    cvol = C.loc[:, 'implied_volatility']
    d1_c = (np.log(cspot / cstrike) + (cmaturity / 365) * (rate - q + (cvol ** 2) / 2)) / (cvol * np.sqrt(cmaturity / 365))
    d2_c = d1_c - cvol * np.sqrt(cmaturity / 365)
    Cpart1 = -q * (np.exp(-q * cmaturity / 365)) * norm.cdf(d1_c)
    Cpart2 = np.exp(-q * cmaturity / 365) * norm.cdf(d1_c) * (
        (2 * (rate - q) * cmaturity / 365) - (d2_c * cvol * np.sqrt(cmaturity / 365))) / (
        2 * (cmaturity / 365) * cvol * np.sqrt(cmaturity / 365))
        
    chrm_c = Cpart1 + Cpart2
    chrmidx_c = (C.loc[:, 'option_type'].index, chrm_c)

    
    P = df[(df.loc[:, 'option_type']=='P')]
    pspot = P.loc[:, 'active_underlying_price']
    pstrike = P.loc[:, 'strike']
    pmaturity = P.loc[:, 'dte'].dt.days
    pvol = P.loc[:, 'implied_volatility']
    d1_p = (np.log(pspot / pstrike) + (pmaturity / 365) * (rate - q + (pvol ** 2) / 2)) / (pvol * np.sqrt(pmaturity / 365))
    d2_p = d1_p - pvol * np.sqrt(pmaturity / 365)
    Ppart1 = q * (np.exp(-q * pmaturity / 365)) * norm.cdf(-d1_p)
    Ppart2 = np.exp(-q * pmaturity / 365) * norm.cdf(d1_p) * (
        (2 * (rate - q) * pmaturity / 365) - (d2_p * pvol * np.sqrt(pmaturity / 365))) / (
        2 * (pmaturity / 365) * pvol * np.sqrt(pmaturity / 365))

    chrm_p = Ppart1 + Ppart2
    chrmidx_p = (P.loc[:, 'option_type'].index, chrm_p)        
    charm = [chrmidx_c, chrmidx_p]
    
    
    Charm = pd.concat([pd.DataFrame(charm[0][1], index = charm[0][0], columns=['charm']),pd.DataFrame(charm[1][1], index = charm[1][0], columns=['charm'])])  
    data = pd.DataFrame([gamma, vega, vanna, volga, ultima, color, zomma, speed, veta]).T
    data.columns = ['gamma', 'vega', 'vanna', 'volga', 'ultima', 'color', 'zomma', 'speed', 'veta']
    data = data.merge(Charm, left_index=True, right_index=True)
    return data
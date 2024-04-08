import numpy as np
import pandas as pd
from sklearn import linear_model
import scipy.stats
from sklearn.isotonic import IsotonicRegression
import scipy.integrate as integrate

def preparewealthsurveydata(country,survey,df_dd,df_d_cols,mean=False):
    if mean==False: #standard case for implicate
        df_d=df_dd.copy()
        df_d=df_d.loc[:,df_d_cols.values()].copy()
        df_d.columns=df_d_cols.keys()
        df_d=df_d[df_d["Country"]==country]
        df_d=df_d.fillna(0) #asi nepotřebujem
    else: #create a mean across df_d implicates
        df_d=list(range(5))
        for i in range(5):
            df_d[i]=df_dd[i].copy()
            df_d[i]=df_d[i].loc[:,df_d_cols.values()].copy()
            df_d[i].columns=df_d_cols.keys()
            df_d[i]=df_d[i][df_d[i]["Country"]==country]
            df_d[i]=df_d[i].fillna(0) #asi nepotřebujem
            df_d[i].Wealth=[float(w) for w in df_d[i].Wealth]
        ##get the mean
        df_d_mean=df_d[0].copy()
        for col in df_d[0].columns:
            if col not in ["Country"]:
                #get the desired column in each implicate
                aux=[np.array(df_d[i].loc[:,col]) for i in range(5)]
                #replace original value by the mean
                df_d_mean.loc[:,col]=np.mean(aux,axis=0)
        #overwrite df_d with the desired data frame
        df_d=df_d_mean.copy()
    try:
        survey_aux=survey.loc[:,["HID","NewWeight1","NewWeightIter"]].groupby("HID").min() #min or max doesn't matter
    except: #if we dont do iterated weights
        survey_aux=survey.loc[:,["HID","NewWeight1"]].groupby("HID").min()
    df_d=df_d.merge(survey_aux,left_index=True,right_index=True,how="left").copy()
    df_d.Wealth=[float(w) for w in df_d.Wealth] #somehow in 2014 data wealth is coded as string
    survey_aux2=survey.loc[:,["HID","PInc"]].groupby("HID").sum()
    df_d=df_d.merge(survey_aux2,left_index=True,right_index=True,how="left").copy()
    surveyw=df_d.rename(columns={"PInc": "Income","Weight":"OrigWeight"}).copy() #we rename "weight" to "origweight"
    return surveyw

def KS_test(treshold,surveyw,forbes,weightcol,richlist=True,avgrank=True,pseudoml=True):
    """
    Insert data frames, specify treshold and name of weightcolumn.
    Returns the value of K-S test (we want to minimize this)
    """ 
    df=surveyw[surveyw.Wealth>=treshold].copy() #in df we need wealth and weight column
    if richlist==True:
        aux=forbes["Wealth"]
        aux=pd.DataFrame(aux)
        c=len(aux)/len(df)
        df.loc[:,weightcol]=df[weightcol]-c
        df=df.append(aux).copy()
        df=df.fillna(1)
    else:
        True
    
    #get rank and avg rank, then get empirical ccdf
    df=df.sort_values("Wealth", axis=0, ascending=False).reset_index(drop=True)
    df["Rank"]=df[weightcol].cumsum()
    df["AvgRank"]=df["Rank"].copy() #(for avgrank, the data must be sorted by wealth and index reset)
    df.loc[0,"AvgRank"]=df.loc[0,"Rank"].copy()/2 #the wealthiest observaion
    first=list(df["Rank"][1:])
    second=list(df["Rank"][:-1])
    df.loc[1:,"AvgRank"]=[(x + y)/2 for x, y in zip(first, second)] #the rest
    if avgrank==True:
        df["Empiricalccdf"]=df["AvgRank"]/(df[weightcol].sum())
    else:
        df["Empiricalccdf"]=df["Rank"]/(df[weightcol].sum())

    #estimate Pareto coeff, get Pareto ccdf
    if pseudoml==False: #linear regression
        reg = linear_model.LinearRegression() #fit_intercept=False
        x=np.log(np.array(df["Wealth"]).reshape(-1, 1))
        if avgrank==True:
            y=np.log(np.array(df["AvgRank"]))
        else:
            y=np.log(np.array(df["Rank"]))
        reg.fit(x,y)
        coeff=np.absolute(reg.coef_[0])
    elif pseudoml==True: #pseudo maximum likelihood
        N=df[weightcol].sum()
        coeff=1/np.sum((df[weightcol]/N)*np.log(df.Wealth/treshold))
    else:
        raise Error("pseudo max. likelihood must be either true or false")

    #pareto ccdf - from formula:
    ccdfpareto = lambda x: (treshold/x)**coeff
    ccdfpareto=np.vectorize(ccdfpareto)
    df["Paretoccdf"]=ccdfpareto(df["Wealth"])
    

    #what we want: K-S test (done on ccdf based on AvgRank, but if we define cdf as 1-ccdf the result is the same)
    ks=np.abs(df.Empiricalccdf-df.Paretoccdf).max()
    return ks
KS_test=np.vectorize(KS_test,excluded=[1,2,3,4,5,6])

def log_KS_test(treshold,surveyw,forbes,weightcol,richlist=True,avgrank=True,pseudoml=True):
    """
    Insert data frames, specify treshold and name of weightcolumn.
    Returns the value of LOG K-S test (we want to minimize this)
    """ 
    df=surveyw[surveyw.Wealth>=treshold].copy() #in df we need wealth and weight column
    if richlist==True:
        aux=forbes["Wealth"]
        aux=pd.DataFrame(aux)
        c=len(aux)/len(df)
        df.loc[:,weightcol]=df[weightcol]-c
        df=df.append(aux).copy()
        df=df.fillna(1)
    else:
        True
    
    #get rank and avg rank, then get empirical ccdf
    df=df.sort_values("Wealth", axis=0, ascending=False).reset_index(drop=True)
    df["Rank"]=df[weightcol].cumsum()
    df["AvgRank"]=df["Rank"].copy() #(for avgrank, the data must be sorted by wealth and index reset)
    df.loc[0,"AvgRank"]=df.loc[0,"Rank"].copy()/2 #the wealthiest observaion
    first=list(df["Rank"][1:])
    second=list(df["Rank"][:-1])
    df.loc[1:,"AvgRank"]=[(x + y)/2 for x, y in zip(first, second)] #the rest
    if avgrank==True:
        df["Empiricalccdf"]=df["AvgRank"]/(df[weightcol].sum())
    else:
        df["Empiricalccdf"]=df["Rank"]/(df[weightcol].sum())

    #estimate Pareto coeff, get Pareto ccdf
    if pseudoml==False: #linear regression
        reg = linear_model.LinearRegression() #fit_intercept=False
        x=np.log(np.array(df["Wealth"]).reshape(-1, 1))
        if avgrank==True:
            y=np.log(np.array(df["AvgRank"]))
        else:
            y=np.log(np.array(df["Rank"]))
        reg.fit(x,y)
        coeff=np.absolute(reg.coef_[0])
    elif pseudoml==True: #pseudo maximum likelihood
        N=df[weightcol].sum()
        coeff=1/np.sum((df[weightcol]/N)*np.log(df.Wealth/treshold))
    else:
        raise Error("pseudo max. likelihood must be either true or false")
    #pareto ccdf - from formula:
    ccdfpareto = lambda x: (treshold/x)**coeff
    ccdfpareto=np.vectorize(ccdfpareto)
    df["Paretoccdf"]=ccdfpareto(df["Wealth"])

    #what we want: K-S test (done on ccdf based on AvgRank, but if we define cdf as 1-ccdf the result is the same)
    ks=np.abs(np.log(df.Empiricalccdf)-np.log(df.Paretoccdf)).max()
    return ks
log_KS_test=np.vectorize(log_KS_test,excluded=[1,2,3,4,5,6])

def KS_test_2011(treshold,surveyw,forbes,weightcol,richlist=True,avgrank=True,pseudoml=True):
    """
    Insert data frames, specify treshold and name of weightcolumn.
    Returns the data frame with kstest values
    """ 
    df=surveyw[surveyw.Wealth>=treshold].copy() #in df we need wealth and weight column
    if richlist==True:
        aux=forbes["Wealth"]
        aux=pd.DataFrame(aux)
        c=len(aux)/len(df)
        df.loc[:,weightcol]=df[weightcol]-c
        df=df.append(aux).copy()
        df=df.fillna(1)
    else:
        True
    
    #get rank and avg rank, then get empirical ccdf
    df=df.sort_values("Wealth", axis=0, ascending=False).reset_index(drop=True)
    df["Rank"]=df[weightcol].cumsum()
    df["AvgRank"]=df["Rank"].copy() #(for avgrank, the data must be sorted by wealth and index reset)
    df.loc[0,"AvgRank"]=df.loc[0,"Rank"].copy()/2 #the wealthiest observaion
    first=list(df["Rank"][1:])
    second=list(df["Rank"][:-1])
    df.loc[1:,"AvgRank"]=[(x + y)/2 for x, y in zip(first, second)] #the rest
    if avgrank==True:
        df["Empiricalccdf"]=df["AvgRank"]/(df[weightcol].sum())
    else:
        df["Empiricalccdf"]=df["Rank"]/(df[weightcol].sum())

    #estimate Pareto coeff, get Pareto ccdf
    if pseudoml==False: #linear regression
        reg = linear_model.LinearRegression() #fit_intercept=False
        x=np.log(np.array(df["Wealth"]).reshape(-1, 1))
        if avgrank==True:
            y=np.log(np.array(df["AvgRank"]))
        else:
            y=np.log(np.array(df["Rank"]))
        reg.fit(x,y)
        coeff=np.absolute(reg.coef_[0])
    elif pseudoml==True: #pseudo maximum likelihood
        N=df[weightcol].sum()
        coeff=1/np.sum((df[weightcol]/N)*np.log(df.Wealth/treshold))
    else:
        raise Error("pseudo max. likelihood must be either true or false")
    #pareto ccdf - from formula:
    ccdfpareto = lambda x: (treshold/x)**coeff
    ccdfpareto=np.vectorize(ccdfpareto)
    df.loc[:,"Paretoccdf"]=ccdfpareto(df["Wealth"])

    #what we want: K-S test (done on ccdf based on AvgRank, but if we define cdf as 1-ccdf the result is the same)
    df.loc[:,"KS"]=np.abs(df.Empiricalccdf-df.Paretoccdf)
    return df

def topshare(df,top,inccol="Wealth",weightcol="Weight"):
    '''
    Insert dataframe (can be pseudosample or raw), the top share we want and the weight column.
    Returns the top share
    
    Code double-checked; the "borderline" survey observation is taken into account well.
    '''
    df=df.sort_values(inccol, axis=0, ascending=False).reset_index(drop=True)
    df["Rank"]=df[weightcol].cumsum() #no. of people with wealth higher or equal than the observation
    pop=df[weightcol].sum()
    toptail=top*pop
    a=df["Rank"]<toptail #dummy for toptail rows
    
    #now get the "borderline" row
    b=sum(a)
    border_wealth=df.iloc[b,:][inccol]
    resid_weight=toptail-(df[a][weightcol].sum())
    borderwealth=border_wealth*resid_weight
    num=borderwealth+(df[a][inccol]*df[a][weightcol]).sum()
    denom=(df[inccol]*df[weightcol]).sum()
    return np.round(num/denom, 4)

def ratio_above_treshold(df,thr,inccol="Wealth",weightcol="Weight"):
    '''
    Insert dataframe (can be pseudosample or raw), the treshold and the weight column.
    Returns the *weighted* ratio of individuals above that treshold
    
    '''
    df=df.sort_values(inccol, axis=0, ascending=False).reset_index(drop=True)
    df["Rank"]=df[weightcol].cumsum() #no. of people with wealth higher or equal than the observation
    dftop=df[df[inccol]>thr]
    return np.round(dftop["Rank"].max()/df[weightcol].sum(),3)

def topshare_thr(df,top,inccol="Wealth",weightcol="Weight",adult=False):
    '''
    Insert dataframe (can be pseudosample or raw), the top population we care about and the weight column.
    Returns the borderline income (or wealth) value of the top
    '''
    if adult==True:
        df=df[df.Age>=20]
    else:
        True
    df=df.sort_values(inccol, axis=0, ascending=False).reset_index(drop=True)
    df["Rank"]=df[weightcol].cumsum() #no. of people with wealth higher or equal than the observation
    pop=df[weightcol].sum()
    toptail=top*pop
    a=df["Rank"]<toptail #dummy for toptail rows
    
    #now get the "borderline" row
    b=sum(a)
    border_wealth=df.iloc[b,:][inccol]
    return border_wealth

def n_above_treshold(df,thr,inccol="Wealth"):
    '''
    Insert dataframe (can be pseudosample or raw), the treshold and the weight column.
    Returns the NUMBER of survey observations above the treshold
    
    '''
    df=df.sort_values(inccol, axis=0, ascending=False).reset_index(drop=True)
    dftop=df[df[inccol]>=thr-0.0001]
    return len(dftop)

def n_strictly_above_treshold(df,thr,inccol="Wealth"):
    '''
    Insert dataframe (can be pseudosample or raw), the treshold and the weight column.
    Returns the NUMBER of survey observations above the treshold
    
    '''
    df=df.sort_values(inccol, axis=0, ascending=False).reset_index(drop=True)
    dftop=df[df[inccol]>thr+0.0001]
    return len(dftop)

def n_below_treshold(df,thr,inccol="Wealth"):
    '''
    Insert dataframe (can be pseudosample or raw), the treshold and the weight column.
    Returns the NUMBER of survey observations below the treshold
    
    '''
    df=df.sort_values(inccol, axis=0, ascending=False).reset_index(drop=True)
    dfbot=df[df[inccol]<thr-0.0001]
    return len(dfbot)

def gini(x, w=None):
    # The rest of the code requires numpy arrays.
    x = np.asarray(x)
    if w is not None:
        w = np.asarray(w)
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = w[sorted_indices]
        # Force float dtype to avoid overflows
        cumw = np.cumsum(sorted_w, dtype=float)
        cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
        return (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) / 
                (cumxw[-1] * cumw[-1]))
    else:
        sorted_x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(sorted_x, dtype=float)
        # The above formula, with all weights equal to 1 simplifies to:
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

def pareto_pseudosample(df,forbes,weightcol,treshold,richlist=True,pseudoml=False,avgrank=True,replace_with_forbes=True,fit_intercept=False,pseudoml_n_adj=True):
    '''
    Insert two data frames, plus the name of the weight column plus w_min
    Returns a new (complete) data frame with a pseudosample in the top tail
    
    Code double-checked, the only problematic thing may be that we include intercept in the Pareto regression (update: We DON'T)
    (but I think Vermeulen 2018 does it like this too - but perhaps not Vermeulen 2014)
    Plus we might be introducing some bias by taking logs of variables etc... 
    But the KS_test leads to a 0 result with the pseudosample - that's great.
    '''
    df2=df.copy()
    if richlist==True: #append richlist with weight 1, decrease all other weights
        aux=forbes["Wealth"]
        aux=pd.DataFrame(aux)
        c=len(aux)/len(df2)
        df2.loc[:,weightcol]=df2[weightcol]-c
        df2=df2.append(aux).copy()
        df2=df2.fillna(1) #we only use weights and wealth in the subsequent code
    else:
        True
    
    #get rank and avgrank
    df2=df2.sort_values("Wealth", axis=0, ascending=False).reset_index(drop=True)
    df2["Rank"]=df2[weightcol].cumsum() #no. of people with wealth higher or equal
    df2["AvgRank"]=df2["Rank"].copy() #(for avgrank, the data must be sorted by wealth and index reset)
    df2.loc[0,"AvgRank"]=df2.loc[0,"Rank"].copy()/2 #the wealthiest observaion
    first=list(df2["Rank"][1:])
    second=list(df2["Rank"][:-1])
    df2.loc[1:,"AvgRank"]=[(x + y)/2 for x, y in zip(first, second)] #the rest
    
    #use treshold to get the top tail to replace (there are no "border" scenarios)
    toptail_cond=df2.Wealth>=treshold
    dftop=df2[toptail_cond].copy()

    #estimate the coefficient
    if pseudoml==False: #linear regression
        reg = linear_model.LinearRegression(fit_intercept=fit_intercept)
        #fit_intercept=False --- well this makes a fucking visual difference
        x=np.log(np.array(dftop["Wealth"]/treshold).reshape(-1, 1))
        if avgrank==True:
            y=np.log(np.array(dftop["AvgRank"]/dftop[weightcol].sum()))
        else:
            y=np.log(np.array(dftop["Rank"]/dftop[weightcol].sum()))
        reg.fit(x,y)
        coeff=np.absolute(reg.coef_[0])
    elif pseudoml==True: #pseudo maximum likelihood
        N=dftop[weightcol].sum()
        coeff=1/np.sum((dftop[weightcol]/N)*np.log(dftop.Wealth/treshold))
        if pseudoml_n_adj==True: #adjustment like in Vermeulen 2018
            n=len(dftop)
            coeff=(n-1)/n*coeff
        else:
            True
    else:
        raise Error("pseudo max. likelihood must be either true or false")

    #get pseudosample
    n=dftop[weightcol].sum() #no. of observations to replace
    wmin=treshold
    ps=np.arange(n).astype(np.float64)
    if avgrank==True:
        f = lambda i: ((i+0.5)/n)**(-1/coeff)*wmin
    else:
        f = lambda i: ((i+1)/n)**(-1/coeff)*wmin
    ps=f(ps)
    ps=pd.DataFrame(ps,columns=["Wealth"])
    ps[weightcol]=1

    #merge survey below treshold + pseudosample, do graphs..
    bottomtail=df2.Wealth<treshold
    dfnew=df2[bottomtail].copy()
    dfnew=dfnew.append(ps)
    dfnew=dfnew.sort_values("Wealth", axis=0, ascending=False).reset_index(drop=True)
    dfnew=dfnew.rename(columns={weightcol: "Weight"}).copy() #maybe this can be optional?
    dfnew["Rank"]=dfnew["Weight"].cumsum()
    dfnew["AvgRank"]=dfnew["Rank"].copy()
    dfnew.loc[0,"AvgRank"]=dfnew.loc[0,"Rank"].copy()/2
    first=list(dfnew["Rank"][1:])
    second=list(dfnew["Rank"][:-1])
    dfnew.loc[1:,"AvgRank"]=[(x + y)/2 for x, y in zip(first, second)] #the rest
    dfnew["Coeff"]=coeff
    dfnew=dfnew[["Wealth","Weight","Rank","AvgRank","Coeff"]]
    
    #replace the top observations with forbes (note: The first non-replaced observation may have higher wealth than the last replaced one)
    if richlist==True and replace_with_forbes==True:
        dfnew.loc[0:(len(forbes)-1),"Wealth"]=np.array(forbes["Wealth"])    
    else:
        True
    return dfnew

def topshare_EV(df,forbes,top,treshold,weightcol="Weight",richlist=True,pseudoml=False,avgrank=True,fit_intercept=False,pseudoml_n_adj=True):
    '''
    Computes top share based on EV of Pareto distribution. The code first computes the Pareto distribution,
    it therefore combines the two steps of *pareto_pseudosample* and *topshare*.
    Returns the top share plus the Pareto coefficient
    '''
    ### compute Pareto distribution - code copied from pareto_pseudosample
    df2=df.copy()
    if richlist==True: #append richlist with weight 1
        aux=forbes["Wealth"]
        aux=pd.DataFrame(aux)
        c=len(aux)/len(df2)
        df2.loc[:,weightcol]=df2[weightcol]-c
        df2=df2.append(aux).copy()
        df2=df2.fillna(1) #we only use weights and wealth in the subsequent code
    else:
        True

    #get rank and avgrank
    df2=df2.sort_values("Wealth", axis=0, ascending=False).reset_index(drop=True)
    df2["Rank"]=df2[weightcol].cumsum() #no. of people with wealth higher or equal
    df2["AvgRank"]=df2["Rank"].copy() #(for avgrank, the data must be sorted by wealth and index reset)
    df2.loc[0,"AvgRank"]=df2.loc[0,"Rank"].copy()/2 #the wealthiest observaion
    first=list(df2["Rank"][1:])
    second=list(df2["Rank"][:-1])
    df2.loc[1:,"AvgRank"]=[(x + y)/2 for x, y in zip(first, second)] #the rest
    
    #use treshold to get the top tail and bottom tail
    toptail_cond=df2.Wealth>=treshold
    bottomtail_cond=df2.Wealth<treshold
    dftop=df2[toptail_cond].copy()
    dfbottom=df2[bottomtail_cond].copy()

    #estimate the coefficient
    if pseudoml==False: #linear regression
        reg = linear_model.LinearRegression(fit_intercept=fit_intercept)
        #fit_intercept=False --- well this makes a fucking visual difference
        x=np.log(np.array(dftop["Wealth"]/treshold).reshape(-1, 1))
        if avgrank==True:
            y=np.log(np.array(dftop["AvgRank"]/dftop[weightcol].sum()))
        else:
            y=np.log(np.array(dftop["Rank"]/dftop[weightcol].sum()))
        reg.fit(x,y)
        coeff=np.absolute(reg.coef_[0])
    elif pseudoml==True: #pseudo maximum likelihood
        N=dftop[weightcol].sum()
        coeff=1/np.sum((dftop[weightcol]/N)*np.log(dftop.Wealth/treshold))
        if pseudoml_n_adj==True: #adjustment like in Vermeulen 2018
            n=len(dftop)
            coeff=(n-1)/n*coeff
        else:
            True
    else:
        raise Error("pseudo max. likelihood must be either true or false")


    ###two cases: Either the Pareto tail is entirely part of the top share or not
    pop=df2[weightcol].sum()
    topshare_n=top*pop
    paretotail_n=dftop[weightcol].sum()
    if topshare_n>paretotail_n: #we include the entire Pareto tail plus part of the survey data
        wealth_Pareto_EV=treshold*coeff/(coeff-1)*paretotail_n #EV*N
        ##now get the "top" tail from dfbottom - copied from *topshare*
        toptail_bottom=topshare_n-paretotail_n #this is the number of people that are part of the top share
        dfbottom=dfbottom.sort_values("Wealth", axis=0, ascending=False).reset_index(drop=True)
        dfbottom["Rank"]=dfbottom[weightcol].cumsum()
        pop_bottom=dfbottom[weightcol].sum()
        a=dfbottom["Rank"]<toptail_bottom #dummy for the top tail rows
        #get the "borderline" row
        b=sum(a)
        border_wealth=dfbottom.iloc[b,:]["Wealth"]
        resid_weight=toptail_bottom-(dfbottom[a][weightcol].sum())
        borderwealth=border_wealth*resid_weight
        num1=borderwealth+(dfbottom[a]["Wealth"]*dfbottom[a][weightcol]).sum()
        num2=wealth_Pareto_EV
        totalwealth=wealth_Pareto_EV+(dfbottom["Wealth"]*dfbottom[weightcol]).sum()
        return [np.round((num1+num2)/totalwealth,4),coeff]
    else: #the top tail is a subset of the Pareto tail
        qf= lambda p: treshold*((1-p)**(-1/coeff))  #quantile function
        perc=topshare_n/paretotail_n #smaller than or equal to one; effectively the top share in the Pareto tail that we want
        wealth_top_EV=integrate.quad(qf,1-perc,1)[0]*paretotail_n #formula derived from Blanchet2016, pp 68
        wealth_Pareto_EV=treshold*coeff/(coeff-1)*paretotail_n #EV*N; can be also obtained using integration
        totalwealth=wealth_Pareto_EV+(dfbottom["Wealth"]*dfbottom[weightcol]).sum()
        return [np.round(wealth_top_EV/totalwealth,4),coeff]

def vanderwijk(df,wmin,weightcol="Weight"):
    wealthcol="Wealth"
    cond_above=df[wealthcol]>=wmin
    num=(df[cond_above][wealthcol]*df[cond_above][weightcol]).sum()
    denom=df[cond_above][weightcol].sum()
    avgwealth=num/denom
    return avgwealth/wmin
vanderwijk=np.vectorize(vanderwijk,excluded=[0,2])

def bootstrap_variance(l,n=5):
    '''
    Insert list of lists (output of bootstrap code)
    Returns the final estimate + its variance
    '''
    within_variance=[]
    for i in range(n):
        ll=l[i]
        estimates=np.array(ll[1:]) #excluding the first entry based on standard weights
        replicate_mean=estimates.mean()
        Um=1/(len(estimates)-1)* np.sum((estimates-replicate_mean)*(estimates-replicate_mean))
        within_variance.append(Um)
    M=len(within_variance)
    W=np.sum(within_variance)/M

    estimates_of_interest=np.array([l[i][0] for i in range(n)])
    final_estimate=np.mean(estimates_of_interest)
    Q=1/(M-1)*np.sum((estimates_of_interest-final_estimate)* (estimates_of_interest-final_estimate)  )
    total_variance=W+(1+1/M)*Q
    return [final_estimate,total_variance]


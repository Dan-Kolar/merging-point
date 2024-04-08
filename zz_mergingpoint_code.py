import numpy as np
import pandas as pd
from sklearn import linear_model
import scipy.stats
from sklearn import linear_model
from sklearn.isotonic import IsotonicRegression
import scipy.integrate as integrate
#####functions for merging point

kernelG = lambda u: 1/np.sqrt(2*3.14159)*np.exp((-1/2)*u*u) #must have correctly defined n,h,s
kernelG = np.vectorize(kernelG)

def pdf(y,s,n,h):
    return 1/(h*n)*np.sum(kernelG((y-s)/h))
pdf=np.vectorize(pdf,excluded=[1,2,3])

def h_opt(s,n):
    """
    Returns optimal bandwidth - Silverman's rule
    we must have our sample defined: s,n
    """
    sdev=s.std()
    q3=np.percentile(s,75)
    q1=np.percentile(s,25)
    return 0.9*np.minimum(sdev,(q3-q1)/1.349)*np.power(n,-1/5)

def pdf_adaptive(y,pdfs,s,n,h):
    """
    Adaptive Kernel estimator - returns density at point y
    we must have our sample defined: s,n,h (optimally h_opt()); We also need functions pdf and kernelG
    pdfs=pdf(s) - we need this too, outside the function
    """
    theta=1/2
    g=scipy.stats.mstats.gmean(pdfs)
    lambda_i= (g/pdfs)**theta
    r=1/n*np.sum(1/(h*lambda_i)*kernelG((y-s)/(h*lambda_i)))
    return r
pdf_adaptive=np.vectorize(pdf_adaptive,excluded=[1,2,3,4])

def pdf_w(y,s,w,n_w,h):
    return 1/(h*n_w)*np.sum(kernelG((y-s)/h)*w)
pdf_w=np.vectorize(pdf_w,excluded=[1,2,3,4])

def pdf_w_adaptive(y,pdfs,s,w,n_w,h):
    """
    Adaptive Kernel estimator for complex weights - returns density at point y
    we must have our sample defined: s,w,n,n_w,h (optimally h(_w)_opt()); We also need function kernelG
    pdfs=pdf_w(s) - we need this too, outside the function
    """
    theta=1/2
    g=scipy.stats.mstats.gmean(pdfs)
    lambda_i= (g/pdfs)**theta
    r=1/n_w*np.sum(1/(h*lambda_i)*kernelG((y-s)/(h*lambda_i))*w)
    return r
pdf_w_adaptive=np.vectorize(pdf_w_adaptive,excluded=[1,2,3,4,5])

def h_w_opt(s,w):
    """
    Returns optimal bandwidth for weighted observations - Silverman's rule
    we must have our sample defined: s,w
    
    Imho doesn't work well
    """
    #here we use pseudosample (simpler)
    ss=np.repeat(s,w)
    nn=len(ss)
    sdev=ss.std()
    q3=np.percentile(ss,75)
    q1=np.percentile(ss,25)
    return 0.9*np.minimum(sdev,(q3-q1)/1.349)*(nn**(-1/5))

def h_w_opt_neff(s,w):
    """
    Returns optimal bandwidth for weighted observations - Silverman's rule
    we must have our sample defined: s,w
    also uses "Kish's approximation for the effective sample size (neff)."
    
    Imho doesn't work well
    """
    neff=sum(w)**2 / sum(w**2)
    #here we use pseudosample (simpler)
    ss=np.repeat(s,w)
    sdev=ss.std()
    q3=np.percentile(ss,75)
    q1=np.percentile(ss,25)
    return 0.9*np.minimum(sdev,(q3-q1)/1.349)*(neff**(-1/5))

def getgridformp(taxgp,surveyad,adultpop1,trustablespan,TU=False,neff=False):
    grid_df=taxgp.loc[:,["p","thr"]].copy()
    grid_df=grid_df[grid_df.p>=trustablespan-0.0001].reset_index(drop=True).copy()
    grid_df.columns=["Rank","Income"]
    a=np.asarray(grid_df["Income"][1:])
    b=np.array([np.inf])
    grid_df["IncomeUpper"]=np.concatenate((a,b))

    #get the density in the tax
    l=[]
    for i in grid_df.index:
        if i<len(grid_df)-1:
            a=(grid_df.Rank[i+1]-grid_df.Rank[i])*adultpop1 #adultpop musí být stejnej jako populace v tax tabulce pro gpinter -- je
            l.append(a)
        else:
            a=(1-grid_df.Rank[i])*adultpop1
            l.append(a)
    grid_df["DensityTax"]=l
    
    #TU adjustment
    if TU==True:
        aux1=surveyad.loc[:,["Weight","TUID"]].groupby("TUID").mean()
        aux2=surveyad.loc[:,["PInc","TUID"]].groupby("TUID").sum()
        surveyad=aux1.merge(aux2,left_index=True,right_index=True)
    
    #get the (histogram) density in surveys
    f=lambda lower, upper: ( surveyad.Weight[(surveyad.PInc >= lower) & (surveyad.PInc < upper)].sum() )
    f=np.vectorize(f)
    grid_df["DensitySurvey"]=f(grid_df.Income,grid_df.IncomeUpper)

    #kernel survey density - weighted,adaptive
    s=surveyad.PInc #we work with surveyad to make the bandwidths more relevant (hopefully)
    w=surveyad.Weight
    n=len(surveyad.PInc)
    n_w=surveyad.Weight.sum()
    if neff==True:
        h=h_w_opt_neff(s,w)
    else:
        h=h_opt(s,n)
    pdfs=pdf_w(s,s,w,n_w,h)
    grid_df["DensitySurveyKernel"]=[integrate.quad(pdf_w_adaptive,grid_df.Income[i],grid_df.IncomeUpper[i],args=(pdfs,s,w,n_w,h))[0] 
                                    for i in range(len(grid_df))]
    grid_df["DensitySurveyKernel"]=np.array(grid_df["DensitySurveyKernel"])*n_w #adultpop
    return grid_df



def evaluatemp(mp,grid_df,granularity=False):
    """
    Insert mp (max. 0.99 inclusive), the function evaluates to what extent we preserve the continuity of the density function
    It returns the difference of frequencies at the mp, plus an array with new densities. We would like the difference to be low
    """
    grid_dff=grid_df.copy()
    #total size of adjustment above:
    cond_above=grid_dff.Rank>=mp-0.00001
    above=grid_dff[cond_above].copy()
    tot_adj_above=(above["DensityTax"]-above["DensitySurveyKernel"]).sum()

    #adjustment below - per quantile:
    if granularity==False:
        adj_below=tot_adj_above/(mp*100)
    else:
        denom1=tot_adj_above/(mp*100)
        denom2=(1-granularity)*(1/granularity)*100
        adj_below=tot_adj_above/(denom1+denom2)
    #new density:
    grid_dff["DensityNew"]=grid_dff["DensitySurveyKernel"]-adj_below
    grid_dff.loc[cond_above,"DensityNew"]=grid_dff.loc[cond_above,"DensityTax"].copy()

    #return the difference of the frequencies at the border (plus the new frequencies)
    if np.round(mp,2)==0.99:
        i=len(grid_dff)-len(above)
        diff=np.abs(grid_dff.DensityNew[i-1]-grid_dff.DensityNew[i:].sum())
    else:    
        i=len(grid_dff)-len(above)
        diff=np.abs(grid_dff.DensityNew[i-1]-grid_dff.DensityNew[i])
    return [diff,grid_dff["DensityNew"]]

def getdfforoptimalmp(grid_df,granularity=False):
    l0=np.arange(grid_df.iloc[1,0],1,0.01)
    l0=l0[l0<0.9999] #np arrange has sometimes inconsistent output
    l0=list(grid_df.iloc[1:len(grid_df)-27,0]) #should be the same as above but without numpy breaking the values
    l=[[i,evaluatemp(i,grid_df,granularity)[0]] for i in l0]
    df=pd.DataFrame(l)
    df.columns=["Rank","Dist"]
    df=df.sort_values("Dist")
    return df


def getcandidatemps(df_mp,adultpop1,thr,dist=None):
    """
    Returns candidate merging points: Those with distance below treshold, plus removing neighboring points.
    Optionally we can add a condition that we also keep points with a distance close to the lowest one - this way
    we will never have no treshold.
    Recommended treshold probably 0.05? Or 0.03..
    """
    if dist==None:
        df=df_mp[df_mp.Dist<=adultpop1*0.01*thr]
    else:
        cond1=df_mp.Dist<=adultpop1*0.01*thr
        m=df_mp.Dist.min()
        cond2=df_mp.Dist<=m*(1+dist)
        df=df=df_mp[(cond1) | (cond2)]
    if len(df)==0: #no candidate mp
        return df
    else:
        ##remove neighboring candidates
        #group candidates into neighboring groups
        df=df.sort_values("Rank")
        abcd=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","AA","BA","CA","DA","EA","FA","GA"]
        l=[]
        i=0
        ii=0
        for rank in df.Rank:
            aa=len(df)-1
            if i==0:
                l.append(abcd[ii])
                i+=1
            else:
                if np.abs(df.iloc[i,0]-df.iloc[i-1,0])<0.01001:
                    l.append(abcd[ii])
                    i+=1
                else:
                    ii+=1
                    l.append(abcd[ii])
                    i+=1
        df["Group"]=l

        #get the minimum value in each group
        aux=df.groupby("Group")["Dist"].min()
        l=[]
        for i in range(len(df)):
            if df.iloc[i,1] in list(aux):
                l.append(True)
            else:
                l.append(False)
        df_final=df[l].sort_values("Dist").reset_index(drop=True)
        return df_final
    
def getoptimalmpv2(df_mp,adultpop0,threshold,distance):
    '''
    Computes the merging point from the average of test statistics over the 5 datasets
    '''
    for i in range(5):
        if i==0:
            df_mp[i]=df_mp[i].rename(columns={"Dist":f"Dist_{i}"})
            df_aux=df_mp[i].copy()
        else:
            df_mp[i]=df_mp[i].rename(columns={"Dist":f"Dist_{i}"})
            df_aux=df_aux.merge(df_mp[i],on="Rank")
    df_aux["Dist"]=df_aux.iloc[:,1:6].mean(axis=1)
    df_aux=df_aux.sort_values(by="Dist")

    df_aux_c=getcandidatemps(df_aux.loc[:,["Rank","Dist"]],adultpop0,thr=threshold,dist=distance)
    optimalmp_v2=np.round(df_aux_c.Rank.max(),2)
    return df_aux_c, optimalmp_v2


def mergegridabove99th(grid_dfj,gpinter=False):
    if gpinter==False:
        rankcol="Rank"
        thrcol="Income"
    else:
        rankcol="p"
        thrcol="thr"
        grid_copy=grid_dfj.copy()
    cond=grid_dfj[rankcol]>0.989
    grid99=grid_dfj[cond].sum()
    grid99[rankcol]=0.99
    grid99[thrcol]=grid_dfj[cond][thrcol].min()
    grid_dfj=grid_dfj[~cond]
    grid_dfj=grid_dfj.append(grid99,ignore_index=True)
    if gpinter==False:
        True
    else: #we change the bracket average for the top entry (it's not just the sum)
        cond=grid_dfj[rankcol]>0.989
        cond2=grid_copy[rankcol]>0.989
        grid_dfj.loc[cond,"bracketavg"]=grid_copy.loc[cond2,"topavg"].min()
        grid_dfj=grid_dfj.drop(labels="topavg",axis=1)
    return grid_dfj

def mergegridabove99th(grid_dfj,gpinter=False,country=False):
    '''
    Creates the same grid but with the 99 percentile as one.
    Special code for gpinter=True and country=France, because French tax data are not an output of gpinter
    '''
    if gpinter==False:
        rankcol="Rank"
        thrcol="Income"
    else:
        rankcol="p"
        thrcol="thr"
        grid_copy=grid_dfj.copy()
    cond=grid_dfj[rankcol]>0.9899
    if country=="FR": #special case for France
        grid99=grid_dfj[cond].min()

    else: 
        grid99=grid_dfj[cond].sum()
        grid99[rankcol]=0.99
        grid99[thrcol]=grid_dfj[cond][thrcol].min()        
    grid_dfj=grid_dfj[~cond]
    grid_dfj=grid_dfj.append(grid99,ignore_index=True)
    if gpinter==False:
        True
    elif country=="FR":
        True
    else:
        #we change the bracket average for the top entry (it's not just the sum)
        cond=grid_dfj[rankcol]>0.989
        cond2=grid_copy[rankcol]>0.989
        grid_dfj.loc[cond,"bracketavg"]=grid_copy.loc[cond2,"topavg"].min()
        grid_dfj=grid_dfj.drop(labels="topavg",axis=1)
    return grid_dfj

def getgranulartaxgp(taxgp,granularity,country=False):
    '''
    Instert tax gp (with the 99th percentile merged). Returns tax gp that can enter gpinter to get more granular brackets.
    Special code for France where the tax data is not an output of gpinter.
    '''
    aux=taxgp.copy()
    
    aux=aux[aux.p>1-granularity-0.0001].reset_index(drop=True)
    for i in range(len(aux))[::-1]:
        c=aux.loc[i,"p"]
        b=1-c
        a=b/granularity
        rank=1-a
        aux.loc[i,"p"]=(np.round(rank,4))
    if country=="FR":
        aux=aux.rename(columns={"Average income E(y | y≥y(p))":"topavg"})
        aux["average"]=aux.topavg.min()
    else:
        aux["average"]=aux.bracketavg.mean()
    return aux


def gettruerank(rank,granularity):
    '''
    From the rank recorded in grid_df (output of gpinter), we calculate the true corresponding rank given the granularity
    
    '''
    a=1-rank
    b=a*granularity
    b
    c=1-b
    d=np.floor(np.round(c,3)*100)/100
    return d
gettruerank=np.vectorize(gettruerank,excluded=[1])
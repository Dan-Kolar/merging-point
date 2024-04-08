import numpy as np
import pandas as pd
import numpy as np
import random
from sklearn import linear_model
import scipy.stats

##### functions for calibration
def preparesurveydata(country,df_d,df_p,df_h,df_d_cols,df_p_cols,df_h_cols,OECD_AT=False,year=None,TU=False,kidsinTU=True):
    """
    Returns a survey df with income properly sorted for calibration, based on our income variables of choice
    Special code for deducting social security contributions in AT
    Special code for creating tax units. By default, dependent children are assigned to their parents as tax units.
    """
    df_d=df_d.loc[:,df_d_cols.values()].copy()
    df_d.columns=df_d_cols.keys()
    
    if df_h_cols=={}:
        True
    else:
        df_h=df_h.loc[:,df_h_cols.values()].copy()
        df_h.columns=df_h_cols.keys()
        df_h["TotalHInc"]=df_h.sum(axis=1)
    
    df_p=df_p.loc[:,df_p_cols.values()].copy()
    df_p.columns=df_p_cols.keys()
    
    #####what if someone did not state their age or gender? (happens occasionally in eastern europe countries
    aux=df_p[df_p["Country"]==country]
    if aux["Age"].isna().sum()>0:
        print("Caution: We have people with 0 age. Their age will be assigned semi-randomly (make sure you set seed for replication)")
        cond=(df_p["Age"].isna() ) & (~df_p["Pensions"].isna() ) #pensioners
        df_p.loc[cond,"Age"]=np.random.choice(np.arange(65,76),size=cond.sum())
        cond=(df_p["Age"].isna() ) & (~df_p["Wage"].isna() ) #now without pensioners
        df_p.loc[cond,"Age"]=np.random.choice(np.arange(18,65),size=cond.sum())
        cond=(df_p["Country"].isna() ) & (~df_p["SelfEmp"].isna() ) #now without pensioners, laborers
        df_p.loc[cond,"Age"]=np.random.choice(np.arange(18,65),size=cond.sum())
        cond=df_p["Age"].isna() #the rest - can be also children
        df_p.loc[cond,"Age"]=np.random.choice(np.arange(65),size=cond.sum())
    else:
        True
    if aux["Gender"].isna().sum()>0:
        print("Caution: We have people who didn't respond to the gender. We assign it randomly (make sure you set seed for replication)")
        cond=df_p["Gender"].isna() 
        df_p.loc[cond,"Gender"]=np.random.choice([1,2],size=cond.sum())
    else:
        True
    #####end of gender/age adjustment
    
    df_p=df_p.fillna(0)
    #add household weight from df2_d
    df_d_aux=df_d.loc[:,"Weight"].copy()
    df_p=df_p.merge(df_d_aux,left_on="HID",right_index=True,how="left").copy()
    
    ######OECD AT ADJUSTMENT:

    if OECD_AT==True:
        if year==2011:
            #employees - per month
            hi=3.95
            pi=10.25
            ui1=1128
            ui2=1230
            ui3=1384
            upper_limit=4020
            lower_limit=357.7

            #self-emp
            se=23.7
            limit_se=13310/12 #always larger than for labor

            #pensions
            hi_pens=5.1
        elif year==2014:
            #employees - per month
            hi=3.95
            pi=10.25
            ui1=1219
            ui2=1330
            ui3=1497
            upper_limit=4440
            lower_limit=386.8

            #self-emp
            se=26.2
            limit_se=16254.8/12

            #pensions
            hi_pens=5.1
        elif year==2017: #averages for data between 2015 and 2017
            #employees - per month
            hi=3.87
            pi=10.25
            ui1=1325
            ui2=1445
            ui3=1626.5
            upper_limit=4920 
            lower_limit=420

            #self-emp
            se=26.2
            limit_se=17792.5/12

            #pensions
            hi_pens=5.1 
        else:
            raise ValueError("Wrong year")
        df_p2=df_p.copy()
        ###deduct employees
        #take into account ceiling
        cond_upper=df_p2["Wage"]>upper_limit*12
        df_p2.loc[cond_upper,"Wage"]=upper_limit*12
        cond_lower=df_p2["Wage"]<lower_limit*12
        df_p2.loc[cond_lower,"Wage"]=0
        
        df_p2["WageSI"]=df_p2["Wage"].copy()
        #first bracket
        cond1=df_p2.Wage<=(ui1*12)
        df_p2.loc[cond1,"WageSI"]=df_p2.loc[cond1,"Wage"]*(hi/100+pi/100)
        #second bracket
        cond2aux=df_p2.Wage<=(ui2*12)
        cond2=(~cond1 & cond2aux)
        df_p2.loc[cond2,"WageSI"]=df_p2.loc[cond2,"Wage"]*(hi/100+pi/100) + (df_p2.loc[cond2,"Wage"]-ui1*12)*0.01
        #third bracket
        cond3aux=df_p2.Wage<=(ui3*12)
        cond3=(~cond2aux & cond3aux)
        df_p2.loc[cond3,"WageSI"]=df_p2.loc[cond3,"Wage"]*(hi/100+pi/100) + (df_p2.loc[cond3,"Wage"]-ui2*12)*0.02 + (ui2*12-ui1*12)*0.01
        #forth bracket
        cond4=~cond3aux
        df_p2.loc[cond4,"WageSI"]=df_p2.loc[cond4,"Wage"]*(hi/100+pi/100) + (df_p2.loc[cond4,"Wage"]-ui3*12)*0.03 + (ui3*12-ui2*12)*0.02+ (ui2*12-ui1*12)*0.01


        ###deduct pensioneers
        df_p2["PensionsSI"]=df_p2["Pensions"]*hi_pens/100

        ###deduct self employment
        df_p2["SelfEmpSI"]=df_p2["SelfEmp"]*se/100
        #limit -- we include in this the SI for wages! Otherwise it would be extremely unfair to high labor+self-employment earners
        cond_lim_se=df_p2["SelfEmpSI"]>limit_se*12
        df_p2.loc[cond_lim_se,"SelfEmpSI"]=limit_se*12
        df_p2["SI"]=df_p2["PensionsSI"]+df_p2["WageSI"]+df_p2["SelfEmpSI"]
        cond_lim_total=df_p2["SI"]>limit_se*12
        df_p2.loc[cond_lim_total,"SI"]=limit_se*12
        ### put it back to df_p - we can deduct if from the wage, for example, it should not matter - we will be summing it up later.
        df_p.loc[:,"Wage"]=df_p.loc[:,"Wage"]-df_p2.loc[:,"SI"]
    else:
        True
        
    ##### END OF OECD ADJUSTMENT
    

    
    ###split household income into equal-split adults => get PInc
    #get no. of adults in a household
    a=df_p.Age>=20
    aux=df_p[["HID","Age"]][a].groupby('HID').count()
    aux.columns=["NoOfAdults"]
    aux=aux
    df_p=df_p.merge(aux,left_on="HID",right_index=True,how="left").copy()
    #get no.of people in a household
    aux=df_p[["HID","Age"]].groupby('HID').count()
    aux.columns=["NoOfPeople"]
    df_p=df_p.merge(aux,left_on="HID",right_index=True,how="left").copy()
    #get household income
    if df_h_cols=={}:
        True
    else:
        df_p=df_p.merge(df_h["TotalHInc"],left_on="HID",right_index=True,how="left").copy()
        #split household income between adults
        aux=df_p[a]
        aux2=(aux["TotalHInc"]/aux["NoOfAdults"])
        aux2=aux2.rename("HInc")
        df_p=df_p.merge(aux2,left_index=True,right_index=True,how="left").copy()
    ###get PInc: Total personal income based on our variables of choice
    #we choose which rows of df_p to sum
    keys_aux=np.array(list(df_p_cols.keys()))
    cond=[not("HID" in l or "Age" in l or "Country" in l or "Gender" in l or "RelToRef" in l or "Married" in l) for l in keys_aux] ######## attention, this peace of code must be changed if we add new variables from the individual-level data
    keys_aux2=list(keys_aux[cond])
    if df_h_cols=={}:
        True
    else:   
        keys_aux2.append("HInc")
    df_p["PInc"]=df_p[keys_aux2].sum(axis=1)
    survey=df_p[df_p["Country"]==country]
    
    
    ######TU ADJUSTMENT (using French data & algorithm in Yonzan (by default extended to include children).)
    if TU==True:
        survey.loc[:,"TUID"]=survey.index.copy()
        if kidsinTU==True:    
            for index in survey.index:
                #1. Head of household, spouse (if they are married) and their children
                if survey.loc[index,"RelToRef"]==1 or (survey.loc[index,"RelToRef"]==2 and survey.loc[index,"Married"]==2) or \
                (survey.loc[index,"RelToRef"]==3 and survey.loc[index,"Age"]<18):
                    survey.loc[index,"TUID"]=survey.loc[index,"HID"]+"_1"
                #2. Other married couples as additional tax units. Any other dependent children will be assumed to be children of this other married couple 
                elif survey.loc[index,"Married"]==2 or survey.loc[index,"Age"]<18:
                    survey.loc[index,"TUID"]=survey.loc[index,"HID"]+"_2"
                #3. Everybody else (non-married with age>18): Their TUID is their personal ID.
                else:
                    True

        elif kidsinTU==False:
             for index in survey.index:
                #1. Head of household, spouse (if they are married)
                if survey.loc[index,"RelToRef"]==1 or (survey.loc[index,"RelToRef"]==2 and survey.loc[index,"Married"]==2):
                    survey.loc[index,"TUID"]=survey.loc[index,"HID"]+"_1"
                #2. Other married couples as additional tax units.
                elif survey.loc[index,"Married"]==2:
                    survey.loc[index,"TUID"]=survey.loc[index,"HID"]+"_2"
                #3. Everybody else (non-married and children): Their TUID is their personal ID.
                else:
                    True    
        else:
            True

        ##additional adjustment: random split in case more than two "other married" people are in a household. Not great, we are assuming that all children belong to the first household. But also not terrible
        aux=survey.loc[survey.Married==2,["HID","Married"]].groupby("HID").count() #counts the number of married individuals in each household
        aux.max()
        if aux.max()[0]>4:
            for hid in survey.HID:
                tuid2=hid+"_2"
                cond=(survey.TUID==tuid2) & (survey.Age>=18)
                length=len(survey.loc[cond,:])
                if length==3:
                    survey.loc[cond,"TUID"]=[tuid2 ,tuid2+"_two",tuid2 ]
                elif length==4:
                    print(f"Household with 4 adults in group 2: {hid}")
                    coin=random.randint(0,1)
                    if coin==0:   
                        survey.loc[cond,"TUID"]=[tuid2 ,tuid2+"_two",tuid2 ,tuid2+"_two"]
                    else:
                        survey.loc[cond,"TUID"]=[tuid2 ,tuid2 ,tuid2+"_two",tuid2+"_two"]
                elif length==5:
                    survey.loc[cond,"TUID"]=[tuid2 ,tuid2+"_two",tuid2 ,tuid2+"_two",tuid2+"_three"]        
                elif length==6:
                    survey.loc[cond,"TUID"]=[tuid2 ,tuid2+"_two",tuid2 ,tuid2+"_two",tuid2+"_three",tuid2+"_three"]
                elif length==7:
                    survey.loc[cond,"TUID"]=[tuid2 ,tuid2+"_two",tuid2 ,tuid2+"_two",tuid2+"_three",tuid2+"_three",tuid2+"_four"]        
                elif length==8:
                    survey.loc[cond,"TUID"]=[tuid2 ,tuid2+"_two",tuid2 ,tuid2+"_two",tuid2+"_three",tuid2+"_three",tuid2+"_four",tuid2+"_four"]
                elif length>8: #absolute borderline case, where we split the remaining "other married" into two groups randomly.
                    l=length
                    aux=[tuid2 ,tuid2+"_two",tuid2 ,tuid2+"_two",tuid2+"_three",tuid2+"_three",tuid2+"_four",tuid2+"_four"]        
                    aux2=(np.random.binomial(l-8,0.5)+np.random.randint(5,1000,1)).astype("str") #this is an "ugly" way to do the random split
                    survey.loc[cond,"TUID"]=aux+aux2
                else: 
                    True
    
    ##### END OF TU ADJUSTMENT  
    
    
    return survey

def addsurveymean(survey):
    '''
    Code to append an average over 5 implicates to the survey
    '''
    ##test that we indeed have 5 implicates
    n=len(survey)
    if n==5:
        True
    else:
        raise ValueError('number of implicates is not 5')
    ##get the mean, with indexes of the first implicate
    survey_oneyear=survey[0].copy()
    for col in survey[0].columns:
        if col not in ["ID","HID","TUID","Gender","Country"]:
            #get the desired column in each implicate
            aux=[np.array(survey[i].loc[:,col]) for i in range(n)]
            #replace original value by the mean
            survey_oneyear.loc[:,col]=np.mean(aux,axis=0)
    return survey_oneyear

def prepareeusilcdata(eusilc_p,eusilc_h,eusilc_p_cols,eusilc_h_cols,OECD_AT=False,year=None):
    """
    Returns a survey df with income properly sorted for grid_df, based on our income variables of choice
    """
    
    eusilc_p=eusilc_p.loc[:,eusilc_p_cols.values()].copy()
    eusilc_p.columns=eusilc_p_cols.keys()
    eusilc_p["Age"]=year-eusilc_p["YearOfBirth"]
    eusilc_p["HID"]=[int(str(eusilc_p["ID"][i])[:-2]) for i in range(len(eusilc_p))] #HID is ID without last two digits
    
    if len(eusilc_h_cols)==1: #nothing to merge
        True
    else:
        eusilc_h=eusilc_h.loc[:,eusilc_h_cols.values()].copy()
        eusilc_h.columns=eusilc_h_cols.keys()
        eusilc_h["TotalHInc"]=eusilc_h.iloc[:,1:].sum(axis=1) #exclude first column, which is household ID
    

    #Pensions: suma toho co endswith pensions
    if eusilc_p.columns.str.endswith("Pensions").sum()==0:
        True
    else:
        cond=eusilc_p.columns.str.endswith("Pensions")
        pensions=eusilc_p.iloc[:,cond].sum(axis=1)
        eusilc_p=eusilc_p.iloc[:,~cond] #drop pensions we are summing up
        eusilc_p["Pensions"]=pensions.copy()

    
    ######OECD AT ADJUSTMENT:

    if OECD_AT==True:
        if year==2011:
            #employees - per month
            hi=3.95
            pi=10.25
            ui1=1128
            ui2=1230
            ui3=1384
            upper_limit=4020
            lower_limit=357.7

            #self-emp
            se=23.7
            limit_se=13310/12 #always larger than for labor

            #pensions
            hi_pens=5.1
        elif year==2014:
            #employees - per month
            hi=3.95
            pi=10.25
            ui1=1219
            ui2=1330
            ui3=1497
            upper_limit=4440
            lower_limit=386.8

            #self-emp
            se=26.2
            limit_se=16254.8/12

            #pensions
            hi_pens=5.1
        elif year==2017: #averages for data between 2015 and 2017
            #employees - per month
            hi=3.87
            pi=10.25
            ui1=1325
            ui2=1445
            ui3=1626.5
            upper_limit=4920 
            lower_limit=420

            #self-emp
            se=26.2
            limit_se=17792.5/12

            #pensions
            hi_pens=5.1 
        else:
            raise ValueError("Wrong year")
        eusilc_p2=eusilc_p.copy()
        ###deduct employees
        #take into account ceiling
        cond_upper=eusilc_p2["Wage"]>upper_limit*12
        eusilc_p2.loc[cond_upper,"Wage"]=upper_limit*12
        cond_lower=eusilc_p2["Wage"]<lower_limit*12
        eusilc_p2.loc[cond_lower,"Wage"]=0
        
        eusilc_p2["WageSI"]=eusilc_p2["Wage"].copy()
        #first bracket
        cond1=eusilc_p2.Wage<=(ui1*12)
        eusilc_p2.loc[cond1,"WageSI"]=eusilc_p2.loc[cond1,"Wage"]*(hi/100+pi/100)
        #second bracket
        cond2aux=eusilc_p2.Wage<=(ui2*12)
        cond2=(~cond1 & cond2aux)
        eusilc_p2.loc[cond2,"WageSI"]=eusilc_p2.loc[cond2,"Wage"]*(hi/100+pi/100) + (eusilc_p2.loc[cond2,"Wage"]-ui1*12)*0.01
        #third bracket
        cond3aux=eusilc_p2.Wage<=(ui3*12)
        cond3=(~cond2aux & cond3aux)
        eusilc_p2.loc[cond3,"WageSI"]=eusilc_p2.loc[cond3,"Wage"]*(hi/100+pi/100) + (eusilc_p2.loc[cond3,"Wage"]-ui2*12)*0.02 + (ui2*12-ui1*12)*0.01
        #forth bracket
        cond4=~cond3aux
        eusilc_p2.loc[cond4,"WageSI"]=eusilc_p2.loc[cond4,"Wage"]*(hi/100+pi/100) + (eusilc_p2.loc[cond4,"Wage"]-ui3*12)*0.03 + (ui3*12-ui2*12)*0.02+ (ui2*12-ui1*12)*0.01


        ###deduct pensioneers
        eusilc_p2["PensionsSI"]=eusilc_p2["Pensions"]*hi_pens/100

        ###deduct self employment
        eusilc_p2["SelfEmpSI"]=eusilc_p2["SelfEmp"]*se/100
        #limit -- we include in this the SI for wages! Otherwise it would be extremely unfair to high labor+self-employment earners
        cond_lim_se=eusilc_p2["SelfEmpSI"]>limit_se*12
        eusilc_p2.loc[cond_lim_se,"SelfEmpSI"]=limit_se*12
        eusilc_p2["SI"]=eusilc_p2["PensionsSI"]+eusilc_p2["WageSI"]+eusilc_p2["SelfEmpSI"]
        cond_lim_total=eusilc_p2["SI"]>limit_se*12
        eusilc_p2.loc[cond_lim_total,"SI"]=limit_se*12
        ### put it back to eusilc_p - we can deduct if from the wage, for example, it should not matter - we will be summing it up later.
        eusilc_p.loc[:,"Wage"]=eusilc_p.loc[:,"Wage"]-eusilc_p2.loc[:,"SI"]
    else:
        True
        
    ##### END OF OECD ADJUSTMENT
    
    
    
    ###split household income into equal-split adults => get PInc
    #get no. of adults in a household
    a=eusilc_p.Age>=20
    aux=eusilc_p[["HID","Age"]][a].groupby('HID').count()
    aux.columns=["NoOfAdults"]
    eusilc_p=eusilc_p.merge(aux,left_on="HID",right_index=True,how="left").copy()
    #get no.of people in a household
    aux=eusilc_p[["HID","Age"]].groupby('HID').count()
    aux.columns=["NoOfPeople"]
    eusilc_p=eusilc_p.merge(aux,left_on="HID",right_index=True,how="left").copy()
    #get household income
    if len(eusilc_h_cols)==1:
        True
    else:
        eusilc_p=eusilc_p.merge(eusilc_h[["HID","TotalHInc"]],left_on="HID",right_on="HID",how="left").copy()
        #split household income between adults
        aux=eusilc_p[a]
        aux2=(aux["TotalHInc"]/aux["NoOfAdults"])
        aux2=aux2.rename("HInc")
        eusilc_p=eusilc_p.merge(aux2,left_index=True,right_index=True,how="left").copy()
    ##get PInc: Total personal income based on our variables of choice
    #we choose which rows of eusilc_p to sum
    keys_aux=np.array(list(eusilc_p.columns))
    cond=[not("ID" in l or 'Weight' in l or "YearOfBirth" in l or 'Age' in l
             or 'HID' in l or 'NoOfAdults' in l or 'NoOfPeople' in l or 'TotalHInc' in l) for l in keys_aux] 
    keys_aux2=list(keys_aux[cond])
    eusilc_p["PInc"]=eusilc_p[keys_aux2].sum(axis=1)
    survey=eusilc_p.copy()
    return survey


def getgridforcalibration(grid_df,survey,adultpop1,optimalmp,merge_brackets=True,TU=False):
    """
    Returns grid for calibration
    If TUs are used, adultpop1 should reflect that.
    ORIGINAL Version: This is the one based on which I have all my results in the paper. I merge brackets with <5 obs.
    """
    #we work with surveyad to ensure everything is consistent
    try:
        surveyad=survey[survey.Age>=20]
    except: #if we have MC
        surveyad=survey.copy()
    
    
    #TU adjustment
    if TU==True:
        aux1=surveyad.loc[:,["Weight","TUID"]].groupby("TUID").mean()
        aux2=surveyad.loc[:,["PInc","TUID"]].groupby("TUID").sum()
        surveyad=aux1.merge(aux2,left_index=True,right_index=True)
        
    grid_df=grid_df[["Rank","Income","IncomeUpper"]]
    grid_df=grid_df[grid_df.Rank>=optimalmp-0.000001].reset_index(drop=True)
    #add the number of observations in surveys in bracket
    f=lambda lower, upper: ( (surveyad.PInc >= lower) & (surveyad.PInc < upper) ).sum()
    f=np.vectorize(f)
    grid_df["Observations"]=f(grid_df.Income,grid_df.IncomeUpper)

    ##merge brackets with less than 5 observations, start with the highest bracket
    while grid_df.Observations.min() < 5:
        if len(grid_df)<=1: #special case when there is nothing to merge
            break
        else:
            for i in range(len(grid_df))[::-1]:
                if grid_df.Observations[i]<5:
                    if i==(len(grid_df)-1): #highest bracket - we can only merge below
                        grid_df.loc[i-1,"Observations"]=(grid_df.loc[i,"Observations"].copy()+grid_df.loc[i-1,"Observations"].copy())
                        grid_df.loc[i-1,"IncomeUpper"]=grid_df.loc[i,"IncomeUpper"].copy()
                        grid_df=grid_df.drop(i).copy() #drop the row we don't want
                        grid_df=grid_df.reset_index(drop=True).copy()
                    else: #we must decide where to merge - we merge to where no. of observations is lower
                        try:
                            obslower=grid_df.loc[i-1,"Observations"]
                            obsupper=grid_df.loc[i+1,"Observations"]
                        except: #this happens when we have the lowest bracket - we can only merge with a higher bracket
                            obslower=2
                            obsupper=1
                        if obslower>obsupper: #we merge with a higher bracket
                            grid_df.loc[i+1,"Observations"]=(grid_df.loc[i,"Observations"].copy()+grid_df.loc[i+1,"Observations"].copy())
                            grid_df.loc[i+1,"Income"]=grid_df.loc[i,"Income"].copy()
                            grid_df.loc[i+1,"Rank"]=grid_df.loc[i,"Rank"].copy()
                            grid_df=grid_df.drop(i).copy() #drop the row we don't want
                            grid_df=grid_df.reset_index(drop=True).copy()
                        else: #we merge with a lower bracket
                            grid_df.loc[i-1,"Observations"]=(grid_df.loc[i,"Observations"].copy()+grid_df.loc[i-1,"Observations"].copy())
                            grid_df.loc[i-1,"IncomeUpper"]=grid_df.loc[i,"IncomeUpper"].copy()
                            grid_df=grid_df.drop(i).copy() #drop the row we don't want
                            grid_df=grid_df.reset_index(drop=True).copy()
                    break #so we always adjust only one row at a time
                else:
                    True

    #now get the desired number of people in each bracket (based on adultpop1)
    a=np.asarray(grid_df.Rank[1:])
    b=np.array([1])
    aux=np.concatenate((a,b))
    grid_df["N"]=(aux-grid_df.Rank)*adultpop1

    #get the original number of people in each bracket
    f2=lambda lower, upper: ( surveyad.Weight[(surveyad.PInc >= lower) & (surveyad.PInc < upper)].sum() )
    f2=np.vectorize(f2)
    grid_df["N_original"]=f2(grid_df.Income,grid_df.IncomeUpper)
    
    #get theta
    grid_df["theta"]=grid_df["N_original"]/grid_df["N"]
    
    if merge_brackets==True:
    ##merge brackets with theta >5 or <0.2, starting at the highest bracket
        while grid_df.theta.min() <0.2 or grid_df.theta.min() >5:
            if len(grid_df)<=1: #special case when there is nothing to merge - we break the while loop
                break
            else:
                for i in range(len(grid_df))[::-1]:
                    if grid_df.theta[i]<0.2 or grid_df.theta[i]>5:
                        if i==(len(grid_df)-1): #highest bracket - we only have one choice where to merge - below
                            grid_df.loc[i-1,"Observations"]=(grid_df.loc[i,"Observations"].copy()+grid_df.loc[i-1,"Observations"].copy())
                            grid_df.loc[i-1,"N"]=(grid_df.loc[i,"N"].copy()+grid_df.loc[i-1,"N"].copy())
                            grid_df.loc[i-1,"N_original"]=(grid_df.loc[i,"N_original"].copy()+grid_df.loc[i-1,"N_original"].copy())
                            grid_df.loc[i-1,"IncomeUpper"]=grid_df.loc[i,"IncomeUpper"].copy()
                            grid_df=grid_df.drop(i).copy() #drop the row we don't want
                            grid_df["theta"]=grid_df["N_original"]/grid_df["N"]
                            grid_df=grid_df.reset_index(drop=True).copy()
                        else: #we must decide where to merge - we merge to where theta is lower (not good if theta is higher than 5,
                                #but that shouldnt happen anyways)
                            try:
                                thetalower=grid_df.loc[i-1,"theta"]
                                thetaupper=grid_df.loc[i+1,"theta"]
                            except: #this happens when we have the lowest bracket - we can only merge with a higher bracket
                                thetalower=2
                                thetaupper=1
                            if thetalower>thetaupper: #we merge with a higher bracket
                                grid_df.loc[i+1,"Observations"]=(grid_df.loc[i,"Observations"].copy()+grid_df.loc[i+1,"Observations"].copy())
                                grid_df.loc[i+1,"Income"]=grid_df.loc[i,"Income"].copy()
                                grid_df.loc[i+1,"Rank"]=grid_df.loc[i,"Rank"].copy()
                                grid_df.loc[i+1,"N"]=(grid_df.loc[i,"N"].copy()+grid_df.loc[i+1,"N"].copy())
                                grid_df.loc[i+1,"N_original"]=(grid_df.loc[i,"N_original"].copy()+grid_df.loc[i+1,"N_original"].copy())
                                grid_df=grid_df.drop(i).copy() #drop the row we don't want
                                grid_df["theta"]=grid_df["N_original"]/grid_df["N"]
                                grid_df=grid_df.reset_index(drop=True).copy()       
                            else: #we merge with a lower bracket
                                grid_df.loc[i-1,"Observations"]=(grid_df.loc[i,"Observations"].copy()+grid_df.loc[i-1,"Observations"].copy())
                                grid_df.loc[i-1,"N"]=(grid_df.loc[i,"N"].copy()+grid_df.loc[i-1,"N"].copy())
                                grid_df.loc[i-1,"N_original"]=(grid_df.loc[i,"N_original"].copy()+grid_df.loc[i-1,"N_original"].copy())
                                grid_df.loc[i-1,"IncomeUpper"]=grid_df.loc[i,"IncomeUpper"].copy()
                                grid_df=grid_df.drop(i).copy() #drop the row we don't want
                                grid_df["theta"]=grid_df["N_original"]/grid_df["N"]
                                grid_df=grid_df.reset_index(drop=True).copy()
                        break #so we always adjust only one row at a time
                    else:
                        True   
    else:
        True
    return grid_df


def getgridforcalibrationV2(grid_df,survey,adultpop1,optimalmp,merge_brackets=True,TU=False):
    """
    Returns grid for calibration
    If TUs are used, adultpop1 should reflect that.
    NEW VERSION: We merge brackets with less than 10 obs.
    """
    #we work with surveyad to ensure everything is consistent
    try:
        surveyad=survey[survey.Age>=20]
    except: #if we have MC
        surveyad=survey.copy()
    
    
    #TU adjustment
    if TU==True:
        aux1=surveyad.loc[:,["Weight","TUID"]].groupby("TUID").mean()
        aux2=surveyad.loc[:,["PInc","TUID"]].groupby("TUID").sum()
        surveyad=aux1.merge(aux2,left_index=True,right_index=True)
        
    grid_df=grid_df[["Rank","Income","IncomeUpper"]]
    grid_df=grid_df[grid_df.Rank>=optimalmp-0.000001].reset_index(drop=True)
    #add the number of observations in surveys in bracket
    f=lambda lower, upper: ( (surveyad.PInc >= lower) & (surveyad.PInc < upper) ).sum()
    f=np.vectorize(f)
    grid_df["Observations"]=f(grid_df.Income,grid_df.IncomeUpper)

    ##merge brackets with less than 5 observations, start with the highest bracket
    #first get location of columns
    observations_i=list(grid_df.columns).index("Observations")
    income_i=list(grid_df.columns).index("Income")
    incomeupper_i=list(grid_df.columns).index("IncomeUpper")
    rank_i=list(grid_df.columns).index("Rank")
    j=0
    while grid_df.Observations.min() < 10:
        length=len(grid_df)
        if length<=1: #special case when there is nothing to merge
            break
        else:
            for i in range(length): #start from the lowest bracket
                if grid_df.loc[i,"Observations"]<10:
                    #get location of the relevant line
                    i_i=list(grid_df.index).index(i)
                    if i==(length-1): #highest bracket - we can only merge below
                        if j<0: #we can wait with the highest (not yet)
                            j+=1
                        else:
                            grid_df.iloc[i_i-1,observations_i]=(grid_df.iloc[i_i,observations_i].copy()+grid_df.iloc[i_i-1,observations_i].copy())
                            grid_df.iloc[i_i-1,incomeupper_i]=grid_df.iloc[i_i,incomeupper_i].copy()
                            grid_df=grid_df.drop(i).copy() #drop the row we don't want
                    else:
                        #we merge with a higher bracket
                        grid_df.iloc[i_i+1,observations_i]=(grid_df.iloc[i_i,observations_i].copy()+grid_df.iloc[i_i+1,observations_i].copy())
                        grid_df.iloc[i_i+1,income_i]=grid_df.iloc[i_i,income_i].copy()
                        grid_df.iloc[i_i+1,rank_i]=grid_df.iloc[i_i,rank_i].copy()
                        grid_df=grid_df.drop(i).copy() #drop the row we don't want
                        
                
                else:
                    True
            grid_df=grid_df.reset_index(drop=True).copy()

    #now get the desired number of people in each bracket (based on adultpop1)
    a=np.asarray(grid_df.Rank[1:])
    b=np.array([1])
    aux=np.concatenate((a,b))
    grid_df["N"]=(aux-grid_df.Rank)*adultpop1

    #get the original number of people in each bracket
    f2=lambda lower, upper: ( surveyad.Weight[(surveyad.PInc >= lower) & (surveyad.PInc < upper)].sum() )
    f2=np.vectorize(f2)
    grid_df["N_original"]=f2(grid_df.Income,grid_df.IncomeUpper)
    
    #get theta
    grid_df["theta"]=grid_df["N_original"]/grid_df["N"]
    
    if merge_brackets==True:
    ##merge brackets with theta >5 or <0.2, starting at the highest bracket
        while grid_df.theta.min() <0.2 or grid_df.theta.min() >5:
            if len(grid_df)<=1: #special case when there is nothing to merge - we break the while loop
                break
            else:
                for i in range(len(grid_df))[::-1]:
                    if grid_df.theta[i]<0.2 or grid_df.theta[i]>5:
                        if i==(len(grid_df)-1): #highest bracket - we only have one choice where to merge - below
                            grid_df.loc[i-1,"Observations"]=(grid_df.loc[i,"Observations"].copy()+grid_df.loc[i-1,"Observations"].copy())
                            grid_df.loc[i-1,"N"]=(grid_df.loc[i,"N"].copy()+grid_df.loc[i-1,"N"].copy())
                            grid_df.loc[i-1,"N_original"]=(grid_df.loc[i,"N_original"].copy()+grid_df.loc[i-1,"N_original"].copy())
                            grid_df.loc[i-1,"IncomeUpper"]=grid_df.loc[i,"IncomeUpper"].copy()
                            grid_df=grid_df.drop(i).copy() #drop the row we don't want
                            grid_df["theta"]=grid_df["N_original"]/grid_df["N"]
                            grid_df=grid_df.reset_index(drop=True).copy()
                        else: #we must decide where to merge - we merge to where theta is lower (not good if theta is higher than 5,
                                #but that shouldnt happen anyways)
                            try: #actually here I try to only merge with a higher bracket - this is similar to what we do with the first merging
                                thetalower=2
                                thetaupper=1
                                #thetalower=grid_df.loc[i-1,"theta"]
                                #thetaupper=grid_df.loc[i+1,"theta"]
                            except: #this happens when we have the lowest bracket - we can only merge with a higher bracket
                                thetalower=2
                                thetaupper=1
                            if thetalower>thetaupper: #we merge with a higher bracket
                                grid_df.loc[i+1,"Observations"]=(grid_df.loc[i,"Observations"].copy()+grid_df.loc[i+1,"Observations"].copy())
                                grid_df.loc[i+1,"Income"]=grid_df.loc[i,"Income"].copy()
                                grid_df.loc[i+1,"Rank"]=grid_df.loc[i,"Rank"].copy()
                                grid_df.loc[i+1,"N"]=(grid_df.loc[i,"N"].copy()+grid_df.loc[i+1,"N"].copy())
                                grid_df.loc[i+1,"N_original"]=(grid_df.loc[i,"N_original"].copy()+grid_df.loc[i+1,"N_original"].copy())
                                grid_df=grid_df.drop(i).copy() #drop the row we don't want
                                grid_df["theta"]=grid_df["N_original"]/grid_df["N"]
                                grid_df=grid_df.reset_index(drop=True).copy()       
                            else: #we merge with a lower bracket
                                grid_df.loc[i-1,"Observations"]=(grid_df.loc[i,"Observations"].copy()+grid_df.loc[i-1,"Observations"].copy())
                                grid_df.loc[i-1,"N"]=(grid_df.loc[i,"N"].copy()+grid_df.loc[i-1,"N"].copy())
                                grid_df.loc[i-1,"N_original"]=(grid_df.loc[i,"N_original"].copy()+grid_df.loc[i-1,"N_original"].copy())
                                grid_df.loc[i-1,"IncomeUpper"]=grid_df.loc[i,"IncomeUpper"].copy()
                                grid_df=grid_df.drop(i).copy() #drop the row we don't want
                                grid_df["theta"]=grid_df["N_original"]/grid_df["N"]
                                grid_df=grid_df.reset_index(drop=True).copy()
                        break #so we always adjust only one row at a time
                    else:
                        True   
    else:
        True
    return grid_df



def getdtx(grid_df,survey,hard=True,mc=False,age_brackets=[0,20,30,40,50,65,999],TU=False):
    """
    Returns d,t,x from grid_df
    hard=True includes also calibration based on age and gender
    we can change the age brackets if we wish, but they should start with 0 and end with 999
    """
    
    d=np.array(survey.Weight).reshape(-1,1) #d is the same always
    
    if TU==True:
        #####TU adjustment
        surveyad=survey[survey.Age>=20].copy()
        aux1=surveyad[["TUID","PInc"]].groupby("TUID").count()
        aux1=aux1.rename(columns={"PInc":"TUsize"})
        surveyad=surveyad.merge(aux1,left_on="TUID",right_index=True,how="left")
        aux2=surveyad[["TUID","PInc"]].groupby("TUID").sum()
        aux2=aux2.rename(columns={"PInc":"TUInc"})
        surveyad=surveyad.merge(aux2,left_on="TUID",right_index=True,how="left")
        survey=survey.merge(surveyad.loc[:,["TUsize","TUInc"]],left_index=True,right_index=True,how="left")
        survey.loc[:,"TUsize"]=survey.loc[:,"TUsize"].fillna(value=1)
        #get x and t for our top incomes, using TUInc and TUsize  
        l=[]
        for i in grid_df.index:
            lower=grid_df.loc[i,"Income"]
            upper=grid_df.loc[i,"IncomeUpper"]        
            a=(survey.TUInc >= lower) & (survey.TUInc < upper) #no need to worry about young people, non-adults don't have TUInc
            aa=np.array(a)/survey.TUsize
            l.append(aa) 
        x=np.array(l) #jediný problém může být, že tady máme floats a ne ints (protože x už nejsou jen dummy vars)
        x=np.round(x,6) #třeba toto sníží výpočetní náročnost?
        x=x.astype('float32')
        t=np.array(grid_df.N).reshape(-1,1)
        ####end of TU adjustment
        
    else:
        #get x and t for our top incomes, we work with surveyad to ensure everything is consistent  
        l=[]
        for i in grid_df.index:
            lower=grid_df.loc[i,"Income"]
            upper=grid_df.loc[i,"IncomeUpper"]        
            try:
                a=(survey.PInc >= lower) & (survey.PInc < upper) & (survey.Age>=20) #we're manually getting surveyad
            except: #if we have mc
                a=(survey.PInc >= lower) & (survey.PInc < upper)
            l.append(a)
        x=np.array(l)
        x=np.multiply(x,1)
        t=np.array(grid_df.N).reshape(-1,1)

    if mc==True:
        #get x and t so that total population size is unchanged
        #this is all we need if we have MC simulation
        x_3=np.ones(len(survey)).reshape(-1,1).transpose()
        t_3=np.array([survey.Weight.sum()]).reshape(-1,1)
        x=np.append(x,x_3,axis=0)
        t=np.append(t,t_3,axis=0)

    else:
        #get x and t for household sizes
        rang=survey.NoOfPeople.unique()[:-1]   #we exclude one to prevent perfect collinearity -- probably not necessary though

        l=[]
        l2=[]
        for r in rang:
            a=np.array(survey.NoOfPeople==r)
            l.append(a)

            b=survey[survey.NoOfPeople==r]["Weight"].sum()
            l2.append(b)

        x_2=np.array(l)
        x_2=np.where(x_2==True,1,0)
        t_2=np.array(l2).reshape(-1,1)
        x=np.append(x,x_2,axis=0)
        t=np.append(t,t_2,axis=0)

        #get x and t so that total population size (plus adults) is unchanged
        x_3=np.ones(len(survey)).reshape(-1,1).transpose()
        t_3=np.array([survey.Weight.sum()]).reshape(-1,1)
        x=np.append(x,x_3,axis=0)
        t=np.append(t,t_3,axis=0)
        #we dont need to calibrate the total number of adults, this condition is specified by the age brackets
        #x_4=np.array(survey.Age>=20)
        #x_4=np.where(x_4==True,1,0).reshape(-1,1).transpose()
        #t_4=np.array(survey[survey.Age>=20].Weight.sum()).reshape(-1,1)
        #x=np.append(x,x_4,axis=0)
        #t=np.append(t,t_4,axis=0)

        if hard==True:
            ### add hard conditions for each family size
            survey_hard=survey.copy()
            survey_hard=survey_hard.reset_index()
            l=[]
            z=np.zeros(len(survey_hard))
            for i in survey_hard.index[:-1]:
                if survey_hard.loc[i,"NoOfPeople"]==1:
                    True
                else:
                    if survey_hard.loc[i,"HID"]==survey_hard.loc[i+1,"HID"]:
                        zz=z.copy()
                        zz[i]=1
                        zz[i+1]=-1
                        l.append(zz)
                    else:
                        True

            x_5=np.array(l)
            t_5=np.zeros(len(l)).reshape(-1,1)
            x=np.append(x,x_5,axis=0)
            t=np.append(t,t_5,axis=0)
        else:
            True
        #gender
        x_6=np.array(survey.Gender==1).astype('float32')
        x_6=np.where(x_6==True,1,0).reshape(-1,1).transpose()
        t_6=np.array(survey[survey.Gender==1].Weight.sum()).reshape(-1,1)
        x=x.astype('float32')
        x=np.append(x,x_6,axis=0)
        t=np.append(t,t_6,axis=0)
        #age
        l=[]
        l2=[]
        z=np.zeros(len(survey))
        for i in range(len(age_brackets)):
            if i==0 or i==1: #exclude the lowest bracket
                True 
            else:
                cond = (survey.Age>=age_brackets[i-1]) & (survey.Age<age_brackets[i]) 
                zz=z.copy()
                zz[cond]=1
                l.append(zz)
                l2.append(survey[cond].Weight.sum())

        x_7=np.array(l).astype('float32')
        t_7=np.array(l2).reshape(-1,1)
        x=np.append(x,x_7,axis=0)
        t=np.append(t,t_7,axis=0)
    return d,t,x

def calibration(d,t,x,float32=False):
    '''
    Now we calibrate:
    d: original survey weights - as a column vector (use reshape)
    t,x such that sum of w*x=t
    returns a list of new weights
    '''
    if float32==True:
        d=d.astype('float32')
        t=t.astype('float32')
        x=x.astype('float32')
    else:
        True
    #Matrix T
    l=0
    n=t.shape[0]
    array=np.zeros([n,n]).astype('float32')
    for row in x.T:
        r=row.reshape(-1,1)
        if float32==True:
            r=r.astype('float32')
        else:
            True
        z=np.matmul(r,r.T)
        z=z*d[l]
        array=array+z
        l+=1
    T=array.copy()
    if float32==True:
        Tinv=np.linalg.inv(T).astype('float32')
    else:
        Tinv=np.linalg.inv(T)

    #Beta
    v=d*(x.T)
    vv=np.sum(v,0).reshape(-1,1)
    Beta=np.matmul(Tinv,(t-vv)).reshape(-1,1)

    #new weights
    l=0
    weights=[]
    for row in x.T:
        r=row.reshape(-1,1)
        z=1+np.matmul(Beta.T,r)
        z=z*d[l]
        weights.append(z[0][0])
        l+=1
    
    #correct the negative weights:
    if np.array(weights).min()<0.999:
        weights_first_iter=np.array(weights)
        cond=np.array(weights)<0.999
        print(f"We have {cond.sum()} negative weights. We fix it with (my) simple method. The lowest negative weight is {np.round(np.array(weights)[cond].max(),2)} and the highest {np.round(np.array(weights)[cond].min(),2)}")
        #alternative would be the iterative method of Singh and Mohl (1996, method 5), which I dont apply
        indexes=np.where(cond==True)
        l1=[]
        l2=[]
        for i in indexes[0]:
            z1=np.zeros(len(d))
            z1[i]=1
            l1.append(z1)
            l2.append(1)
        
        x=np.append(x,np.array(l1),axis=0)
        t=np.append(t,np.array(l2).reshape(-1,1),axis=0)    
        #now we run the calibration again, with our new conditions being added
        if float32==True:
            t=t.astype('float32')
            x=x.astype('float32')
        else:
            True
        #Matrix T
        l=0
        n=t.shape[0]
        array=np.zeros([n,n]).astype('float32')
        for row in x.T:
            r=row.reshape(-1,1)
            if float32==True:
                r=r.astype('float32')
            else:
                True
            z=np.matmul(r,r.T)
            z=z*d[l]
            array=array+z
            l+=1
        T=array.copy()
        if float32==True:
            Tinv=np.linalg.inv(T).astype('float32')
        else:
            Tinv=np.linalg.inv(T)

        #Beta
        v=d*(x.T)
        vv=np.sum(v,0).reshape(-1,1)
        Beta=np.matmul(Tinv,(t-vv)).reshape(-1,1)

        #new weights
        l=0
        weights=[]
        for row in x.T:
            r=row.reshape(-1,1)
            z=1+np.matmul(Beta.T,r)
            z=z*d[l]
            weights.append(z[0][0])
            l+=1
            
    else:
        True
    
    #what if the problem remains:
    jj=0
    if np.array(weights).min()<0.999:
        weights=weights_first_iter.copy()
    else: 
        True
    while np.array(weights).min()<0.999:
        print(f"We have not solved the non-negativity problem in one iteration, we use weights from the original iteration and correct them in proportion to all other observations.")     
        cond=weights<1.0001 #in case the while loop loops more than once
        totaladj=(1-weights[cond]).sum() #how much adjustment we are making: We increase each weights lower than one to one.
        length=len(weights[~cond]) #the number of observations that are OK
        adj=totaladj/length #adjustment per observation that is OK
        weights[~cond]=weights[~cond]-adj
        weights[cond]=1
        jj+=1
        if jj>20:
            global weights_error #global seems to only work if we copy-paste the function, not import it
            weights_error=weights
            raise ValueError("Error - cannot solve non-negative things. See weights_error in the environment")
        
    return np.array(weights)
    
def iterativecalibration(survey_iter,grid_df,adultpop0,optimalmp,TU=False):
    '''
    Calibration code for iterative stuff
    '''
    #perform calibration,without imposing that households' weights are the same
    grid_df_cal=getgridforcalibration(grid_df,survey_iter,adultpop0,optimalmp,TU=TU)
    d,t,x=getdtx(grid_df_cal,survey_iter,hard=False,TU=TU)
    newweights=calibration(d.astype('float32'),t.astype('float32'),x.astype('float32'))
    #average the weights in each household
    newweightsavg=avgweights(survey_iter,survey_iter.index,newweights)
    return newweightsavg
    
def weightdiff(old,new,condition=None):
    '''
    insert old and new weights,
    returns the absolute sum of the weight differences. This is what is minimized in the calibration process
    '''
    if condition==None:
        return np.abs(old-new).sum()
    else:
        return np.abs(old[condition]-new[condition]).sum()

def griddiff(grid_df,income,weights):
    '''
    insert grid df, array of survey income and of new weights
    returns the absolute difference in target weights of top brackets
    should be zero if we do the exact estimation; should hopefully converge to zero with our iteration
    '''
    f2=lambda lower, upper: ( weights[(income >= lower) & (income < upper)].sum() )
    f2=np.vectorize(f2)
    #get the resulting weights for each bracket
    N_new=f2(grid_df.Income,grid_df.IncomeUpper)
    return (np.abs(N_new-grid_df.N)).sum()

def avgweights(survey,ID,newweights):
    '''
    Insert ID and new weights 
    Average out the weights of each family
    '''
    hid=survey.loc[ID,"HID"]
    return newweights[survey.HID==hid].mean()
avgweights=np.vectorize(avgweights,excluded=[0,2])

def addspacestoint(integer):
    i=str(integer)
    j=len(i)
    if j>=7:
        i_cut2=i[len(i)-3:]
        i_cut1=i[len(i)-6:len(i)-3]
        i_start=i[:len(i)-6]
        return i_start +" "+ i_cut1 + " "+ i_cut2

    elif j>=5:
        i_cut1=i[len(i)-3:]
        i_start=i[:len(i)-3]
        return i_start +" "+ i_cut1

    else:
        return i

def addcommastoint(integer):
    i=str(integer)
    j=len(i)
    if j>=7:
        i_cut2=i[len(i)-3:]
        i_cut1=i[len(i)-6:len(i)-3]
        i_start=i[:len(i)-6]
        return i_start +","+ i_cut1 + ","+ i_cut2

    elif j>=5:
        i_cut1=i[len(i)-3:]
        i_start=i[:len(i)-3]
        return i_start +","+ i_cut1

    else:
        return i
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, f1_score, classification_report
import xgboost as xgb
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import time
import seaborn as sns
import scipy as Sp

#Setting directories
print (os.getcwd())
os.chdir('Desktop\ml-ddi-code')
my_path = os.path.abspath(os.getcwd())

#Importing Datasets
input_data=pd.read_csv('Data Files/Input Data.csv')
in_vitro_data=pd.read_excel('Data Files/Substrates.xlsx',sheet_name='Perpr Interactions')
timeseries_features=pd.read_csv('Data Files/Timeseries features.csv',index_col=['Unnamed: 0'])
oeselma=pd.read_csv('Data Files/oeselma_output.csv')
humanpk=pd.read_excel('Data Files/Substrates.xlsx',sheet_name='Human PK values')
#Merging Replacing rat pk values with human pk values (timeseries_features is the main featureset)
humanpk.drop(columns=humanpk.columns[0:6],inplace=True)
timeseries_features=pd.merge(timeseries_features,humanpk,left_index=True,right_index=True)
timeseries_features=timeseries_features[timeseries_features.columns.drop(list(timeseries_features.filter(regex='Rat')))]
sns.set_theme()

#Instantiating models
elasticnet=ElasticNet()
xgb_regressor=xgb.XGBRegressor()
rf=RandomForestRegressor()
svr=SVR()

#Creating different featuresets for comparison
#non-oeselma features
nonoeselma_features=timeseries_features.copy()
drop_perp=oeselma.columns
drop_sub=oeselma.columns
drop_perp = [f'{col} Perp' for col in drop_perp]
exceptions_perp=['Name Perp', 'Ion class Perp', 'Min. dist DD Perp', 'Min. dist DA Perp', 'Min. dist AA Perp']
exceptions_sub=['Name Sub', 'Ion class Sub', 'Min. dist DD Sub', 'Min. dist DA Sub', 'Min. dist AA Sub']
drop_perp = [i for i in drop_perp if i not in exceptions_perp]
drop_sub =[f'{col} Sub' for col in drop_sub]
drop_sub=[i for i in drop_sub if i not in exceptions_sub] 
#oeselma
oeselma_features=pd.merge(timeseries_features[drop_sub],timeseries_features[drop_perp],left_index=True,right_index=True)
#cyp and %fm features
cyp_data=timeseries_features.filter(regex=' T ')
fm_data=timeseries_features.filter(regex=r' %fm ')
cyp_features=pd.merge(cyp_data,fm_data,left_index=True,right_index=True)
#in vitro data
dfs=[]
for col in in_vitro_data.columns[2:10]:
    dfs.append(timeseries_features.filter(regex=f'{col}'))
in_vitro_features=pd.concat(dfs, axis=1)
#ECFP4 data
ecfp4=timeseries_features.filter(regex='ECFP4')
#Combined Chemical Feautures
chem=ecfp4.join(oeselma_features)

#Creating an optional function that removes correlated features and can be used with the fit and transform methods of an Sklearn Pipeline
#This was not used in the final model
class remove_correlated_features_function(BaseEstimator,TransformerMixin):
    def __init__(self):
        return None
    
    def fit(self,x,y=None):
        cor_matrix=x.corr().abs()
        upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
        self.to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        return self
        
    def transform(self,x,y=None):
        new_features=x.drop(self.to_drop, axis=1)
        print('Decorrelator transformed')
        return new_features
    
#Classifier for DDI categories based on FDA guidance, with either all 7 classes, or combining them into 3 classes
def classifier(model, class_number):
    output=pd.read_csv(f'{model} output.csv',index_col='Unnamed: 0')
    #7 Classes based on FDA guidance
    if class_number == 7:
        Pred_conditions=[(np.log10(1.25)<=output['Predicted AUCR'])&(output['Predicted AUCR']<np.log10(2)),
                    (np.log10(2)<=output['Predicted AUCR'])&(output['Predicted AUCR']<np.log10(5)),
                    (output['Predicted AUCR']>=np.log10(5)),
                    (np.log10(0.8)>=output['Predicted AUCR'])&(output['Predicted AUCR']>np.log10(0.5)),
                    (np.log10(0.5)>=output['Predicted AUCR'])&(output['Predicted AUCR']>np.log10(0.2)),
                    (np.log10(0.2)>=output['Predicted AUCR']),
                    (np.log10(0.8)<output['Predicted AUCR'])&(output['Predicted AUCR']<np.log10(1.25))
                    ]
        Obs_conditions=[(np.log10(1.25)<=output['Observed AUCR'])&(output['Observed AUCR']<np.log10(2)),
                    (np.log10(2)<=output['Observed AUCR'])&(output['Observed AUCR']<np.log10(5)),
                    (output['Observed AUCR']>=np.log10(5)),
                    (np.log10(0.8)>=output['Observed AUCR'])&(output['Observed AUCR']>np.log10(0.5)),
                    (np.log10(0.5)>=output['Observed AUCR'])&(output['Observed AUCR']>np.log10(0.2)),
                    (np.log10(0.2)>=output['Observed AUCR']),
                    (np.log10(0.8)<output['Observed AUCR'])&(output['Observed AUCR']<np.log10(1.25))
                    ]
        allocations=['Weak Inhibition','Moderate Inhibition','Strong Inhibition',
                    'Weak Induction','Moderate Induction','Strong Induction','No Interaction']
        allocations_order=['No Interaction','Weak Inhibition','Weak Induction','Moderate Inhibition','Moderate Induction',
                                   'Strong Inhibition','Strong Induction']
        report_order=['No Interaction','Weak Inhibition','Weak Induction','Moderate Inhibition','Moderate Induction',
                                   'Strong Inhibition','Strong Induction']
        output['Pred Category'] = np.select(Pred_conditions, allocations)
        output['Obs Category'] = np.select(Obs_conditions, allocations)
        magnitude_conditions=[(output['Obs Category']=='Strong Inhibition')|(output['Obs Category']=='Strong Induction'),
                              (output['Obs Category']=='Moderate Inhibition')|(output['Obs Category']=='Moderate Induction'),
                              (output['Obs Category']=='Weak Inhibition')|(output['Obs Category']=='Weak Induction'),
                              (output['Obs Category']=='No Interaction')]
        magnitude_allocations=['Strong Interaction', 'Moderate Interaction', 'Weak Interaction', 'No Interaction']
        rotation=30
    #3 Classes based on merging strong and moderate effects, and weak and non-interactions
    elif class_number == 3:
        Pred_conditions=[(output['Predicted AUCR']>=np.log10(2)),
                        (np.log10(0.5)<output['Predicted AUCR'])&(output['Predicted AUCR']<np.log10(2)),
                        (np.log10(0.5)>=output['Predicted AUCR'])
                    ]
        Obs_conditions=[(output['Observed AUCR']>=np.log10(2)),
                        (np.log10(0.5)<output['Observed AUCR'])&(output['Observed AUCR']<np.log10(2)),
                        (np.log10(0.5)>=output['Observed AUCR'])
                    ]
        allocations=['Moderate-Strong Inhibition','No/Weak Interaction','Moderate-Strong Induction']
        allocations_order=['No/Weak Interaction','Moderate-Strong Inhibition','Moderate-Strong Induction']
        report_order=['No/Weak Interaction','Moderate-Strong Inhibition','Moderate-Strong Induction']
        output['Pred Category'] = np.select(Pred_conditions, allocations)
        output['Obs Category'] = np.select(Obs_conditions, allocations)
        magnitude_conditions=[(output['Obs Category']=='Moderate-Strong Inhibition')|(output['Obs Category']=='Moderate-Strong Induction'),
                              (output['Obs Category']=='No/Weak Interaction')]
        magnitude_allocations=['Moderate/Strong Interaction','No/Weak Interaction']
        rotation=0
    output['DDI Magnitude']=np.select(magnitude_conditions, magnitude_allocations)
    print(output)
    output.to_csv(my_path+fr'\{model} output complete.csv')
    #Crosstable between observed and predicted classes
    ct=pd.crosstab(output['Pred Category'],output['Obs Category'],dropna=False)
    ct=ct.reindex(allocations_order,axis="columns")
    ct=ct.reindex(allocations_order,axis="rows")
    for allocation in allocations:
        ct[allocation]=(ct[allocation]/np.sum(ct[allocation]))*100 #Converting to %
    #Creating datafrane for calculating classification metrics
    report=classification_report(output['Obs Category'],output['Pred Category'],output_dict=True)
    report=pd.DataFrame.from_dict(report)
    report=report.transpose()
    report.drop(columns=['support'],inplace=True)
    report.drop('macro avg',inplace=True)
    report.drop('accuracy',inplace=True)
    report.drop('weighted avg',inplace=True)
    report.drop(columns=['f1-score'],inplace=True)
    print(report.columns)
    report=report.reindex(report_order,axis="rows")
    #Plotting crosstable heatmap and scoring bar chart
    fig, ax = plt.subplots()
    ax=sns.heatmap(ct,annot=True,cmap='Blues')
    ax.xaxis.set_tick_params(rotation=30)
    plt.show()
    report.plot(kind='bar')
    plt.xticks(rotation=rotation)
    plt.show()
    f1=f1_score(output['Obs Category'],output['Pred Category'],average=None)
    print(f'F1:{f1}, Mean F1:{np.mean(f1)}')
    return(output['DDI Magnitude'])

classifier('rf',7)
    
#Main model with shuffle cross validation
#Not used for SVR due to SFM incompatability - SVR model is below
def model_test_main(model,x,y,class_level,decorrelation=False,cmax=False):
    start_time = time.time()
    sns.set_theme()
    #Hyperparameter dictionaries
    if model == elasticnet:
        param_grid={'elasticnet__alpha':[1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0,500,1000],
                    'elasticnet__l1_ratio':[0.001,0.01,0.1,0,1]}
    elif model == xgb_regressor:
        param_grid={'xgbregressor__n_estimators'     : np.array([20, 50, 100, 500, 1000]),
                    'xgbregressor__learning_rate'    : np.array([1e-3, 1e-2, 1e-1, 1, 10]),
                    'xgbregressor__max_depth'        : np.array([3, 5, 8, 10,12])}
    elif model == rf:
        param_grid={'randomforestregressor__n_estimators'   : np.array([50, 100, 500, 1000]),
                    'randomforestregressor__min_samples_leaf'   : np.array([1, 5, 10, 50]), #100, 200, 500]),
                    'randomforestregressor__max_features' : np.array(['auto', 'sqrt'])}
    elif model == svr:
        param_grid={'svr__C':[1e-2, 1e-1, 1.0, 10.0, 100.0,500],
                    'svr__epsilon':[0.01,0.05,0.1,0.5,1,10],
                    'svr__kernel':['linear','poly','rbf','sigmoid','precomputed']}
    param_grid['selectfrommodel__max_features']=[20,40,60,80,100,120]
    #Creating CV folds stratified so that train and test folds have similar label distributions
    outer=KFold(n_splits=2,shuffle=True,random_state=0)
    #Instantiating lists to collect observed and predicted labels, and metrics for each CV fold
    rmse_results=[]
    r2_results=[]
    twofolderror_results=[]
    all_y_pred=[]
    all_y_test=[]
    #Instantiating pipeline
    if decorrelation == False:
        pipeline=make_pipeline(StandardScaler(), SelectFromModel(model),model)
    elif decorrelation == True:
        pipeline=make_pipeline(remove_correlated_features_function(),StandardScaler(),SelectFromModel(model),model)      
    print(param_grid)
    for train_ix, test_ix in outer.split(x,y):
        print(test_ix)
        #Iterating through indices for each fold for outer CV
        x_train, x_test = x.iloc[train_ix], x.iloc[test_ix]
        y_train, y_test=y.iloc[train_ix], y.iloc[test_ix]
        inner=KFold(n_splits=5,shuffle=True,random_state=0)
        #Inner CV for hyperparameter tuning
        search_results=GridSearchCV(pipeline,param_grid,cv=inner,n_jobs=-1,scoring='neg_root_mean_squared_error').fit(x_train,y_train)
        #Selecting optimal hyperparameters
        optimal_params=search_results.best_params_
        #Renaming dictionary keys for hyperparameter->model allocation
        max_features=optimal_params['selectfrommodel__max_features']
        if model == rf:
            optimal_params['n_estimators']=optimal_params.pop('randomforestregressor__n_estimators')
            optimal_params['min_samples_leaf']=optimal_params.pop('randomforestregressor__min_samples_leaf')
            optimal_params['max_features']=optimal_params.pop('randomforestregressor__max_features')
        elif model == elasticnet:
            optimal_params['alpha']=optimal_params.pop('elasticnet__alpha')
            optimal_params['l1_ratio']=optimal_params.pop('elasticnet__l1_ratio')
        elif model == xgb_regressor:
            optimal_params['n_estimators']=optimal_params.pop('xgbregressor__n_estimators')
            optimal_params['learning_rate']=optimal_params.pop('xgbregressor__learning_rate')
            optimal_params['max_depth']=optimal_params.pop('xgbregressor__max_depth')
        elif model == svr:
            optimal_params['C']=optimal_params.pop('svr__C')
            optimal_params['epsilon']=optimal_params.pop('svr__epsilon')
        print(optimal_params)
        optimal_params.pop('selectfrommodel__max_features')
        model_name=((str(model)).partition('('))[0]
        #Re-instantiating models with optimal hyperparameters
        if model==elasticnet:
            model=ElasticNet(**optimal_params)
        elif model==xgb_regressor:
            model=xgb.XGBRegressor(**optimal_params)
        elif model == rf:
            model =RandomForestRegressor(**optimal_params)
        #Re-instantiating pipeline with optimised models
        if decorrelation == False:
            pipeline=make_pipeline(StandardScaler(), SelectFromModel(model,max_features=max_features),model)
        elif decorrelation == True:
            pipeline=make_pipeline(remove_correlated_features_function(),StandardScaler(),SelectFromModel(model,max_features=max_features),model)        
        #Fitting pipeline scaler, dimensionality reduction and model to training data fold
        pipeline.fit(x_train,y_train)
        #Predicting using fitted model on test fold
        y_pred=pipeline.predict(x_test)
        print('PREDICTIONS DONE')
        all_y_pred.extend(y_pred)
        all_y_test.extend(y_test)
        rmse_results.append(mean_squared_error(y_test,y_pred,squared=False))
        r2_results.append(pipeline.score(x_test,y_test))
        twofolderror_results.append((np.count_nonzero(np.logical_and(y_pred < y_test+np.log10(2), y_pred> y_test-np.log10(2)))/len(y_pred))*100)
        print('FOLD COMPLETE')
    print(f'RMSEs={rmse_results}, r2s={r2_results},2-fold error% ={twofolderror_results} mean RMSE={np.mean(rmse_results)}, mean r2 ={np.mean(r2_results)},mean 2fold error={np.mean(twofolderror_results)}')
    time_taken=(time.time() - start_time)
    print(f'{time_taken} seconds or {time_taken/60} mins or {time_taken/(60*60)} hrs')
    #Fitting regression line between observed labels and predicted labels
    output=pd.DataFrame({'Observed AUCR': all_y_test,
                         'Predicted AUCR': all_y_pred})
    output.to_csv(f'{model_name} output.csv')
    all_y_test=np.array(all_y_test)
    all_y_pred=np.array(all_y_pred)
    reg=LinearRegression().fit(all_y_test.reshape(-1,1),all_y_pred.reshape(-1,1))
    fig, ax = plt.subplots()
    #Plotting predictions
    palette = {"Strong Interaction":"indianred","Moderate Interaction":"tab:orange", "Weak Interaction":"tab:blue",
           "No Interaction":"tab:grey","Moderate/Strong Interaction":"maroon","No/Weak Interaction":"tab:blue",
           }
    if cmax==False:
        plot=sns.scatterplot(all_y_test,all_y_pred,hue=classifier(model_name,class_level),palette=palette)
    elif cmax==True:
        plot=sns.scatterplot(all_y_test,all_y_pred)
    plot.axes.set_ylabel('Log10 Predicted AUC Ratio')
    plot.axes.set_xlabel('Log10 Observed AUC Ratio')
    #Computing 2 fold margins
    x_ideal=np.copy(all_y_test)  
    upper_x=x_ideal+np.log10(2)
    lower_x=x_ideal-np.log10(2)
    #twofolderror=(np.count_nonzero(np.logical_and(all_y_pred < upper_x, all_y_pred > lower_x))/len(all_y_pred))*100
    #Plotting unit line and legend
    plt.xlim([-2,2])
    plt.ylim([-2,2])
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()]),]
    plt.axline((-1,-1),(1,1), color="black", lw=1,linestyle='--',label=f'R2 = {np.round(np.mean(r2_results),3)} RMSE = {np.round(np.mean(rmse_results),3)} <2 fold error {np.round(np.mean(twofolderror_results),1)}%')
    plot.axes.legend()
    #Plotting regression line and 2 fold error margins
    #plot.axes.plot(all_y_test.reshape(-1,1),reg.predict(all_y_test.reshape(-1,1)),color='black')
    plt.axline((upper_x[1],all_y_test[1]),(upper_x[2],all_y_test[2]),color='black',lw=0.2)
    plt.axline((lower_x[1],all_y_test[1]),(lower_x[2],all_y_test[2]),color='black',lw=0.2)
    plt.axhline(0, color='grey')
    plt.axvline(0, color='grey')
    plt.title(f'Performance of {model_name}')
    plt.xlabel('Observed Log AUC Ratio')
    plt.ylabel('Predicted Log AUC Ratio')
    plt.gca().set_aspect('equal')
    fig = plt.gcf()
    fig.set_size_inches(18, 12)
    print(model_name)
    #fig.savefig(my_path + fr'\Figures\Model Performance\{model_name} SelectFromModel CV10and5.png')
    plt.show()
    #Residuals Plot
    residuals=all_y_test-all_y_pred
    plt.scatter(all_y_test, residuals)
    plt.axhline(0, color='grey')
    plt.axvline(0, color='grey')
    plt.xlabel('Predicted Log AUC Ratio')
    plt.ylabel('Residual')
    plt.title(f'Distribution of Residuals for {model_name}')
    fig = plt.gcf()
    fig.set_size_inches(18, 12)
    #fig.savefig(my_path + fr'\Figures\Model Performance\{model_name} SelectFromModel CV10and5 Residuals.png')
    plt.show()
    pipeline.fit(x,y)
    #feature_names = np.array(x.columns.values)
    filename = f'{model_name}SFM .sav'
    #pickle.dump(pipeline, open(filename, 'wb'))

#Note - svr used model_test_main_svr code at bottom of file
model_test_main(rf,timeseries_features,input_data['log AUC Ratio'],7,decorrelation=False,cmax=False) 

#Classification metrics
def class_metrics(output_filename,n_folds):
    results=pd.read_csv(f'{output_filename}.csv',index_col='Unnamed: 0')
    if n_folds==3:
        report_fold1=classification_report(results['Obs Category'][:39],results['Pred Category'][:39])
        report_fold2=classification_report(results['Obs Category'][40:79],results['Pred Category'][40:79])
        report_fold3=classification_report(results['Obs Category'][80:119],results['Pred Category'][80:119])
        report_fold4='N/A'
        report_fold5='N/A'
    if n_folds==5:
        report_fold1=classification_report(results['Obs Category'][:23],results['Pred Category'][:23])
        report_fold2=classification_report(results['Obs Category'][24:47],results['Pred Category'][24:47])
        report_fold3=classification_report(results['Obs Category'][48:71],results['Pred Category'][48:71])
        report_fold4=classification_report(results['Obs Category'][72:95],results['Pred Category'][72:95])
        report_fold5=classification_report(results['Obs Category'][96:119],results['Pred Category'][96:119])
    print(f'Fold 1:{report_fold1},Fold 2:{report_fold2},Fold 3:{report_fold3},Fold 4:{report_fold4},Fold 5:{report_fold5}')
class_metrics('RandomForestRegressor output complete',5)

#Histogram and scatter plot of residuals 
def residual_func(output_filename):
    results=pd.read_csv(f'{output_filename}.csv',index_col='Unnamed: 0')
    results['Residuals']=results['Predicted AUCR']-results['Observed AUCR']
    sns.scatterplot(results['Observed AUCR'],results['Residuals'])
    plt.axhline(0, color='grey')
    plt.axvline(0, color='grey')
    plt.xlabel('Observed Log AUC Ratio')
    plt.ylabel('Residual')
    fig = plt.gcf()
    fig.set_size_inches(18, 12)
    plt.show()
    plt.hist(results['Residuals'],color='#669bbc')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.show()
    print(f"Residual distribution normal? {Sp.stats.kstest(results['Residuals'],'norm')}")
residual_func('SVR output complete')
    
    
#normality test
def normality_test(model):
    output=pd.read_csv(f'{model} output complete analysis.csv',index_col='Unnamed: 0')
    exp_observed=np.power(10,output['Observed AUCR'])
    reg=LinearRegression().fit(np.array(output['Observed AUCR']).reshape(-1,1),np.array(output['Predicted AUCR']).reshape(-1,1))
    residuals=reg.predict(np.array(output['Observed AUCR']).reshape(-1,1))-np.array(output['Predicted AUCR']).reshape(-1,1)
    plt.scatter(np.array(output['Observed AUCR']).reshape(-1,1), residuals)
    plt.show()
    print(f"Observed AUCRs Uniform?: {Sp.stats.kstest(output['Observed AUCR'],'uniform')}")
    print(f"Observed AUCRs Normal?: {Sp.stats.kstest(output['Observed AUCR'],'norm')}")
    print(f"Predicted AUCRs Normal?: {Sp.stats.kstest(output['Predicted AUCR'],'norm')}")
    print(f"Residuals: {Sp.stats.kstest(residuals,'norm')}")
    print(f"Against each other: {Sp.stats.wilcoxon(output['Observed AUCR'],output['Predicted AUCR'])}")
    #print(Sp.stats.kruskal(output['Observed AUCR'],output['Predicted AUCR']))
    
print(f"Observed AUCRs: {Sp.stats.kstest(input_data['log AUC Ratio'],'norm')}")
print(input_data[(input_data['log AUC Ratio']>np.log10(0.8))&(input_data['log AUC Ratio']<np.log10(1.25))]['log AUC Ratio'])
print(len(input_data[(input_data['log AUC Ratio']<np.log10(0.8))]['log AUC Ratio']))
normality_test('SVR')

#Modelling minimum performance by predicting mean of training label
def min_model(x,y):
    start_time = time.time()
    sns.set_theme()
    #Creating CV folds stratified so that train and test folds have similar label distributions
    outer=KFold(n_splits=5,shuffle=True,random_state=0)
    #Instantiating lists to collect observed and predicted labels, and metrics for each CV fold
    rmse_results=[]
    r2_results=[]
    twofolderror_results=[]
    all_y_pred=[]
    all_y_test=[]
    #Instantiating pipeline
    pipeline=make_pipeline(StandardScaler(),DummyRegressor())
    for train_ix, test_ix in outer.split(x,y):
        print(test_ix)
        #Iterating through indices for each fold for outer CV
        x_train, x_test = x.iloc[train_ix], x.iloc[test_ix]
        y_train, y_test=y.iloc[train_ix], y.iloc[test_ix]
        #Fitting pipeline scaler, dimensionality reduction and model to training data fold
        pipeline.fit(x_train,y_train)
        #Predicting using fitted model on test fold
        y_pred=pipeline.predict(x_test)
        print('PREDICTIONS DONE')
        all_y_pred.extend(y_pred)
        all_y_test.extend(y_test)
        rmse_results.append(mean_squared_error(y_test,y_pred,squared=False))
        r2_results.append(pipeline.score(x_test,y_test))
        twofolderror_results.append((np.count_nonzero(np.logical_and(y_pred < y_test+np.log10(2), y_pred> y_test-np.log10(2)))/len(y_pred))*100)
        print('FOLD COMPLETE')
    print(f'RMSEs={rmse_results}, r2s={r2_results},2-fold error% ={twofolderror_results} mean RMSE={np.mean(rmse_results)}, mean r2 ={np.mean(r2_results)},mean 2fold error={np.mean(twofolderror_results)}')
    output=pd.DataFrame({'Observed AUCR': all_y_test,
                         'Predicted AUCR': all_y_pred})
    output.to_csv(my_path+r'\Dummy output.csv')
    classifier('Dummy',7)
    #Fitting regression line between observed labels and predicted labels
    all_y_test=np.array(all_y_test)
    all_y_pred=np.array(all_y_pred)
    reg=LinearRegression().fit(all_y_test.reshape(-1,1),all_y_pred.reshape(-1,1))
    fig, ax = plt.subplots()
    #Plotting predictions
    ax.scatter(all_y_test,all_y_pred)
    ax.set_ylabel('Log10 Predicted AUC Ratio')
    ax.set_xlabel('Log10 Observed AUC Ratio')
    #Computing 2 fold margins
    x_ideal=np.copy(all_y_test)  
    upper_x=x_ideal+np.log10(2)
    lower_x=x_ideal-np.log10(2)
    #twofolderror=(np.count_nonzero(np.logical_and(all_y_pred < upper_x, all_y_pred > lower_x))/len(all_y_pred))*100
    #Plotting unit line and legend
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=1,label=f'R2 = {np.round(np.mean(r2_results),3)} RMSE = {np.round(np.mean(rmse_results),3)} <2 fold error {np.round(np.mean(twofolderror_results),1)}%')
    ax.legend()
    #Plotting regression line and 2 fold error margins
    ax.plot(all_y_test.reshape(-1,1),reg.predict(all_y_test.reshape(-1,1)))
    ax.plot(upper_x,all_y_test,color='black',lw=0.2)
    ax.plot(lower_x,all_y_test,color='black',lw=0.2)
    plt.axhline(0, color='grey')
    plt.axvline(0, color='grey')
    plt.xlim([-2,2])
    plt.ylim([-2,2])
    plt.title(f'Performance of dummy regressor')
    plt.xlabel('Observed Log AUC Ratio')
    plt.ylabel('Predicted Log AUC Ratio')
    fig = plt.gcf()
    fig.set_size_inches(18, 12)
    #fig.savefig(my_path + fr'\Figures\Model Performance\{model_name} SelectFromModel CV10and5.png')
    plt.show()
    #Residuals Plot
    residuals=all_y_test-all_y_pred
    plt.scatter(all_y_test, residuals)
    plt.axhline(0, color='grey')
    plt.axvline(0, color='grey')
    plt.xlabel('Predicted Log AUC Ratio')
    plt.ylabel('Residual')
    plt.title(f'Distribution of Residuals for dummy regressor')
    fig = plt.gcf()
    fig.set_size_inches(18, 12)
    #fig.savefig(my_path + fr'\Figures\Model Performance\{model_name} SelectFromModel CV10and5 Residuals.png')
    time_taken=(time.time() - start_time)
    print(f'{time_taken} seconds or {time_taken/60} mins or {time_taken/(60*60)} hrs')
    plt.show()

min_model(timeseries_features,input_data['log AUC Ratio']) 

#Main model for SVR (no feature selection)
def model_test_main_svr(model,x,y,class_level,decorrelation=False,cmax=False):
    start_time = time.time()
    sns.set_theme()
    #Hyperparameter dictionaries
    if model == elasticnet:
        param_grid={'elasticnet__alpha':[1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0,500,1000],
                    'elasticnet__l1_ratio':[0.001,0.01,0.1,0,1]}
    elif model == xgb_regressor:
        param_grid={'xgbregressor__n_estimators'     : np.array([20, 50, 100, 500, 1000]),
                    'xgbregressor__learning_rate'    : np.array([1e-3, 1e-2, 1e-1, 1, 10]),
                    'xgbregressor__max_depth'        : np.array([3, 5, 8, 10,12])}
    elif model == rf:
        param_grid={'randomforestregressor__n_estimators'   : np.array([50, 100, 500, 1000]),
                    'randomforestregressor__min_samples_leaf'   : np.array([1, 5, 10, 50, 100, 200, 500]),
                    'randomforestregressor__max_features' : np.array(['auto', 'sqrt'])}
    elif model == svr:
        param_grid={'svr__C':[1e-2, 1e-1, 1.0, 10.0, 100.0,500],
                    'svr__epsilon':[0.01,0.05,0.1,0.5,1,10],
                    'svr__kernel':['linear','poly','rbf','sigmoid','precomputed']}
    #Creating CV folds stratified so that train and test folds have similar label distributions
    outer=KFold(n_splits=5,shuffle=True,random_state=0)
    #Instantiating lists to collect observed and predicted labels, and metrics for each CV fold
    rmse_results=[]
    r2_results=[]
    twofolderror_results=[]
    all_y_pred=[]
    all_y_test=[]
    #Instantiating pipeline
    if decorrelation == False:
        pipeline=make_pipeline(StandardScaler(),model)
    elif decorrelation == True:
        pipeline=make_pipeline(remove_correlated_features_function(),StandardScaler(),model)      
    print(param_grid)
    for train_ix, test_ix in outer.split(x,y):
        print(test_ix)
        #Iterating through indices for each fold for outer CV
        x_train, x_test = x.iloc[train_ix], x.iloc[test_ix]
        y_train, y_test=y.iloc[train_ix], y.iloc[test_ix]
        inner=KFold(n_splits=5,shuffle=True,random_state=0)
        #Inner CV for hyperparameter tuning
        search_results=GridSearchCV(pipeline,param_grid,cv=inner,n_jobs=-1,scoring='neg_root_mean_squared_error').fit(x_train,y_train)
        #Selecting optimal hyperparameters
        optimal_params=search_results.best_params_
        #Renaming dictionary keys for hyperparameter->model allocation
        if model == rf:
            optimal_params['n_estimators']=optimal_params.pop('randomforestregressor__n_estimators')
            optimal_params['min_samples_leaf']=optimal_params.pop('randomforestregressor__min_samples_leaf')
            optimal_params['max_features']=optimal_params.pop('randomforestregressor__max_features')
        elif model == elasticnet:
            optimal_params['alpha']=optimal_params.pop('elasticnet__alpha')
            optimal_params['l1_ratio']=optimal_params.pop('elasticnet__l1_ratio')
        elif model == xgb_regressor:
            optimal_params['n_estimators']=optimal_params.pop('xgbregressor__n_estimators')
            optimal_params['learning_rate']=optimal_params.pop('xgbregressor__learning_rate')
            optimal_params['max_depth']=optimal_params.pop('xgbregressor__max_depth')
        elif model == svr:
            optimal_params['C']=optimal_params.pop('svr__C')
            optimal_params['epsilon']=optimal_params.pop('svr__epsilon')
        print(optimal_params)
        model_name=((str(model)).partition('('))[0]
        #Re-instantiating models with optimal hyperparameters
        if model==elasticnet:
            model=ElasticNet(**optimal_params)
        elif model==xgb_regressor:
            model=xgb.XGBRegressor(**optimal_params)
        elif model == rf:
            model =RandomForestRegressor(**optimal_params)
        #Re-instantiating pipeline with optimised models
        if decorrelation == False:
            pipeline=make_pipeline(StandardScaler(),model)
        elif decorrelation == True:
            pipeline=make_pipeline(remove_correlated_features_function(),StandardScaler(),model)        
        #Fitting pipeline scaler, dimensionality reduction and model to training data fold
        pipeline.fit(x_train,y_train)
        #Predicting using fitted model on test fold
        y_pred=pipeline.predict(x_test)
        print('PREDICTIONS DONE')
        all_y_pred.extend(y_pred)
        all_y_test.extend(y_test)
        rmse_results.append(mean_squared_error(y_test,y_pred,squared=False))
        r2_results.append(pipeline.score(x_test,y_test))
        twofolderror_results.append((np.count_nonzero(np.logical_and(y_pred < y_test+np.log10(2), y_pred> y_test-np.log10(2)))/len(y_pred))*100)
        print('FOLD COMPLETE')
    print(f'RMSEs={rmse_results}, r2s={r2_results},2-fold error% ={twofolderror_results} mean RMSE={np.mean(rmse_results)}, mean r2 ={np.mean(r2_results)},mean 2fold error={np.mean(twofolderror_results)}')
    time_taken=(time.time() - start_time)
    print(f'{time_taken} seconds or {time_taken/60} mins or {time_taken/(60*60)} hrs')
    #Fitting regression line between observed labels and predicted labels
    output=pd.DataFrame({'Observed AUCR': all_y_test,
                         'Predicted AUCR': all_y_pred})
    output.to_csv(f'{model_name} output.csv')
    all_y_test=np.array(all_y_test)
    all_y_pred=np.array(all_y_pred)
    reg=LinearRegression().fit(all_y_test.reshape(-1,1),all_y_pred.reshape(-1,1))
    fig, ax = plt.subplots()
    #Plotting predictions
    palette = {"Strong Interaction":"indianred","Moderate Interaction":"tab:orange", "Weak Interaction":"tab:blue",
           "No Interaction":"tab:grey","Moderate/Strong Interaction":"maroon","No/Weak Interaction":"tab:blue",
           }
    if cmax==False:
        plot=sns.scatterplot(all_y_test,all_y_pred,hue=classifier(model_name,class_level),palette=palette)
    elif cmax==True:
        plot=sns.scatterplot(all_y_test,all_y_pred)
    plot.axes.set_ylabel('Log10 Predicted AUC Ratio')
    plot.axes.set_xlabel('Log10 Observed AUC Ratio')
    #Computing 2 fold margins
    x_ideal=np.copy(all_y_test)  
    upper_x=x_ideal+np.log10(2)
    lower_x=x_ideal-np.log10(2)
    #twofolderror=(np.count_nonzero(np.logical_and(all_y_pred < upper_x, all_y_pred > lower_x))/len(all_y_pred))*100
    #Plotting unit line and legend
    plt.xlim([-2,2])
    plt.ylim([-2,2])
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()]),]
    plt.axline((-1,-1),(1,1), color="black", lw=1,linestyle='--',label=f'R2 = {np.round(np.mean(r2_results),3)} RMSE = {np.round(np.mean(rmse_results),3)} <2 fold error {np.round(np.mean(twofolderror_results),1)}%')
    plot.axes.legend()
    #Plotting regression line and 2 fold error margins
    #plot.axes.plot(all_y_test.reshape(-1,1),reg.predict(all_y_test.reshape(-1,1)),color='black')
    plt.axline((upper_x[1],all_y_test[1]),(upper_x[2],all_y_test[2]),color='black',lw=0.2)
    plt.axline((lower_x[1],all_y_test[1]),(lower_x[2],all_y_test[2]),color='black',lw=0.2)
    plt.axhline(0, color='grey')
    plt.axvline(0, color='grey')
    plt.title(f'Performance of {model_name}')
    plt.xlabel('Observed Log AUC Ratio')
    plt.ylabel('Predicted Log AUC Ratio')
    plt.gca().set_aspect('equal')
    fig = plt.gcf()
    fig.set_size_inches(18, 12)
    print(model_name)
    #fig.savefig(my_path + fr'\Figures\Model Performance\{model_name} SelectFromModel CV10and5.png')
    plt.show()
    #Residuals Plot
    residuals=all_y_test-all_y_pred
    plt.scatter(all_y_test, residuals)
    plt.axhline(0, color='grey')
    plt.axvline(0, color='grey')
    plt.xlabel('Predicted Log AUC Ratio')
    plt.ylabel('Residual')
    plt.title(f'Distribution of Residuals for {model_name}')
    fig = plt.gcf()
    fig.set_size_inches(18, 12)
    #fig.savefig(my_path + fr'\Figures\Model Performance\{model_name} SelectFromModel CV10and5 Residuals.png')
    plt.show()
    #In final model below fitting will have separate GridSearchCV for hyperparameter tuning
    pipeline.fit(x,y)
    filename = f'{model_name}.sav'
    pickle.dump(pipeline, open(filename, 'wb'))

model_test_main_svr(svr,timeseries_features,input_data['log AUC Ratio'],7,decorrelation=False,cmax=False) 

def scatter_plot(filename):
    results=pd.read_csv(f'{filename}.csv',index_col='Unnamed: 0')
    plot=sns.scatterplot(results['Observed AUCR'],results['Predicted AUCR'])
    plot.axes.set_ylabel('Log10 Predicted AUC Ratio')
    plot.axes.set_xlabel('Log10 Observed AUC Ratio')
    #Computing 2 fold margins
    x_ideal=np.copy(results['Observed AUCR'])  
    upper_x=x_ideal+np.log10(2)
    lower_x=x_ideal-np.log10(2)
    #twofolderror=(np.count_nonzero(np.logical_and(all_y_pred < upper_x, all_y_pred > lower_x))/len(all_y_pred))*100
    #Plotting unit line and legend
    plt.xlim([-2,2])
    plt.ylim([-2,2])
    plt.axline((-1,-1),(1,1), color="black", lw=1,linestyle='--')
    plt.axline((upper_x[1],results['Observed AUCR'][1]),(upper_x[2],results['Observed AUCR'][2]),color='black',lw=0.2)
    plt.axline((lower_x[1],results['Observed AUCR'][1]),(lower_x[2],results['Observed AUCR'][2]),color='black',lw=0.2)
    plt.axhline(0, color='grey')
    plt.axvline(0, color='grey')
    plt.xlabel('Observed Log AUC Ratio')
    plt.ylabel('Predicted Log AUC Ratio')
    plt.gca().set_aspect('equal')
    fig = plt.gcf()
    fig.set_size_inches(18, 12)
    plt.show()
scatter_plot('SVR output complete')
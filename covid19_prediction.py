# -*- coding: utf-8 -*-




import pandas as pd
import numpy as np


#import dei  modelli
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from pgmpy.estimators import HillClimbSearch, BicScore,  MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianModel

#import di metrics per calcolare l'accuratezza del modello
from sklearn import metrics
from sklearn.model_selection import KFold

#import del dataset
dataSet = pd.read_csv('both_test_balanced (1).csv')
print(dataSet.head())


#K-FOLD e classificazione

kf = KFold(n_splits=10, shuffle=True, random_state= 100)

X_evaluation = dataSet.to_numpy()
y_evaluation = dataSet["class"].to_numpy()

knn_metrics_score = {'averagePrecision_list': [], 
                    'precision_list' : [],
                    'recall_list' : [],
                    'f1_list' : [],
}

rfc_metrics_score = {'averagePrecision_list': [],
                    'precision_list' : [],
                    'recall_list' : [],
                    'f1_list' : [],
}

dct_metrics_score = {'averagePrecision_list': [],
                    'precision_list' : [],
                    'recall_list' : [],
                    'f1_list' : [],
}

svm_metrics_score = {'averagePrecision_list': [],
                    'precision_list' : [],
                    'recall_list' : [],
                    'f1_list' : [],
}

net_metrics_score = {'averagePrecision_list': [],
                    'precision_list' : [],
                    'recall_list' : [],
                    'f1_list' : [],
}



knn_cl = KNeighborsClassifier()
rfc_cl = RandomForestClassifier()
dct_cl = DecisionTreeClassifier()
svm_cl = svm.SVC()





for train_index, test_index in kf.split(X_evaluation, y_evaluation):

  
    training_set, test_set = X_evaluation[train_index], X_evaluation[test_index]

    #Dati di train
    data_train = pd.DataFrame(training_set, columns=dataSet.columns)
    X_train = data_train.drop("class", axis=1)
    y_train = data_train['class']

    #Dati di test
    data_test = pd.DataFrame(test_set, columns=dataSet.columns)
    y_test = data_test['class'] #expected
    X_test = data_test.drop("class", axis=1)

    #fit del classificatore knn
    knn_cl.fit(X_train, y_train)
    y_pred_knn = knn_cl.predict(X_test)

    #fit del classificatore decisionTree
    dct_cl.fit(X_train, y_train)
    y_pred_dct= dct_cl.predict(X_test)

    #fit del classificatore RandomForest
    rfc_cl.fit(X_train, y_train)
    y_pred_rfc = rfc_cl.predict(X_test)#predicted

    #fit del classificatore SVM
    svm_cl.fit(X_train, y_train)
    y_pred_rfc = svm_cl.predict(X_test)#predicted


    #inizializzazione delle funzioni di score per la migliore struttura della BN (DAG)
    bic = BicScore(data_train)
    #inizializzazione della ricerca della migliore struttura della BN
    hc_bic = HillClimbSearch(data_train, bic)
    #stima della migliore struttura della BN (DAG)
    bic_model = hc_bic.estimate(scoring_method=bic)
    #FASE DI VALIDAZIONE
    


    net = BayesianModel(ebunch=bic_model.edges())
    net.fit(data=data_train, estimator=MaximumLikelihoodEstimator)
    y_pred_net = net.predict(X_test)
    
    

    
          
    
    







    








    #Salvo le metriche del fold attuale nel dizionario per il knn
    knn_metrics_score['averagePrecision_list'].append(metrics.average_precision_score(y_test,y_pred_knn))
    knn_metrics_score['precision_list'].append(metrics.precision_score(y_test,y_pred_knn))
    knn_metrics_score['recall_list'].append(metrics.recall_score(y_test,y_pred_knn))
    knn_metrics_score['f1_list'].append(metrics.f1_score(y_test,y_pred_knn))

    #Salvo le metriche del fold attuale nel dizionario per il RandomForest
    dct_metrics_score['averagePrecision_list'].append(metrics.average_precision_score(y_test,y_pred_dct))
    dct_metrics_score['precision_list'].append(metrics.precision_score(y_test,y_pred_dct))
    dct_metrics_score['recall_list'].append(metrics.recall_score(y_test,y_pred_dct))
    dct_metrics_score['f1_list'].append(metrics.f1_score(y_test,y_pred_dct))

    #Salvo le metriche del fold attuale nel dizionario per il RandomForest
    rfc_metrics_score['averagePrecision_list'].append(metrics.average_precision_score(y_test,y_pred_rfc))
    rfc_metrics_score['precision_list'].append(metrics.precision_score(y_test,y_pred_rfc))
    rfc_metrics_score['recall_list'].append(metrics.recall_score(y_test,y_pred_rfc))
    rfc_metrics_score['f1_list'].append(metrics.f1_score(y_test,y_pred_rfc))

    #Salvo le metriche del fold attuale nel dizionario per l' SVM
    svm_metrics_score['averagePrecision_list'].append(metrics.average_precision_score(y_test,y_pred_rfc))
    svm_metrics_score['precision_list'].append(metrics.precision_score(y_test,y_pred_rfc))
    svm_metrics_score['recall_list'].append(metrics.recall_score(y_test,y_pred_rfc))
    svm_metrics_score['f1_list'].append(metrics.f1_score(y_test,y_pred_rfc))

    

    net_metrics_score['averagePrecision_list'].append(metrics.average_precision_score(y_test,y_pred_net))
    net_metrics_score['precision_list'].append(metrics.precision_score(y_test,y_pred_net))
    net_metrics_score['recall_list'].append(metrics.recall_score(y_test,y_pred_net))
    net_metrics_score['f1_list'].append(metrics.f1_score(y_test,y_pred_net))



   

#Media delle metriche RandomForest
print("\nMedia delle metriche del RandomForest")
print("Media  Average Precision: %f" % (np.mean(rfc_metrics_score['averagePrecision_list'])))
print("Media Precision: %f" % (np.mean(rfc_metrics_score['precision_list'])))
print("Media Recall: %f" % (np.mean(rfc_metrics_score['recall_list'])))
print("Media f1: %f" % (np.mean(rfc_metrics_score['f1_list'])))

#Media delle metriche del knn
print("\nMedia delle metriche del KNN")
print("Media  Average Precision: %f" % (np.mean(knn_metrics_score['averagePrecision_list'])))
print("Media Precision: %f" % (np.mean(knn_metrics_score['precision_list'])))
print("Media Recall: %f" % (np.mean(knn_metrics_score['recall_list'])))
print("Media f1: %f" % (np.mean(knn_metrics_score['f1_list'])))


#Media delle metriche  del DecisionTree
print("\nMedia delle metriche del DecisionTree")
print("Media  Average Precision: %f" % (np.mean(dct_metrics_score['averagePrecision_list'])))
print("Media Precision: %f" % (np.mean(dct_metrics_score['precision_list'])))
print("Media Recall: %f" % (np.mean(dct_metrics_score['recall_list'])))
print("Media f1: %f" % (np.mean(dct_metrics_score['f1_list'])))

#Media delle metriche del SVM
print("\nMedia delle metriche del SVM")
print("Media Average Precision: %f" % (np.mean(svm_metrics_score['averagePrecision_list'])))
print("Media Precision: %f" % (np.mean(svm_metrics_score['precision_list'])))
print("Media Recall: %f" % (np.mean(svm_metrics_score['recall_list'])))
print("Media f1: %f" % (np.mean(svm_metrics_score['f1_list'])))


#Media delle metriche della BN
print("\nMedia delle metriche della BN")
print("Media Accuracy: %f" % (np.mean(net_metrics_score['averagePrecision_list'])))
print("Media Precision: %f" % (np.mean(net_metrics_score['precision_list'])))
print("Media Recall: %f" % (np.mean(net_metrics_score['recall_list'])))
print("Media f1: %f" % (np.mean(net_metrics_score['f1_list'])))

bic = BicScore(dataSet)
hc_bic = HillClimbSearch(dataSet, bic)
bic_model = hc_bic.estimate(scoring_method=bic)
best_model = BayesianModel(ebunch=bic_model.edges())
best_model.fit(data=dataSet, estimator=MaximumLikelihoodEstimator)
#print(best_model.edges())
    #in particolare esegue VE per rispondere alla query probabilistica e ritorna P(variabili query | evidenze)
def exactInference(model : BayesianModel, evidence : dict , variables : list):
        inference = VariableElimination(model)
        return inference.query(evidence=evidence, variables=variables, show_progress=False)

    #evidenze da dare in input ai metodi di inferenza
states_list = ["throat_pain","dyspnea","fever","cough","headache","taste_disorders","olfactory_disorders","coryza","gender","health_professional","class"]
print("[+]Write variables you can choose between(you can select multiple, but separate them with a space):")
for s in states_list:
        print(f"\t\u251C{s}")
_input = input(" > ")
var_list = _input.split(" ")
for v in var_list:
      if v not in states_list:
            raise Exception("[-]Variable selected not in features list")
for x in var_list:
        states_list.remove(x)
print("[+]Select evidence from the list specifying value (ie. gender:1 taste_disorders:0)")
for s in states_list:
        print(f"\t\u251C{s}")
_input = input(" > ")
evidence_string_list = _input.split(" ")
tokens = []
for e in evidence_string_list:
        tokens += e.split(":")
       
evidence_dict = {}
evidence = []
if evidence_string_list != ['']:
      
      i = 0
      for t in tokens:
            if i % 2 == 0:
                if t not in states_list:
                    raise Exception("[-]Variable selected not in features list")
            else:
                if int(t) > 1 or int(t) < 0:
                    raise Exception("[-]Value selected not in domain")
            i += 1
it = iter(tokens)
for x in it:
    evidence_dict.update({ x : int(next(it))})

        
print(exactInference(model=best_model,evidence=evidence_dict,variables=var_list))
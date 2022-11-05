import numpy as np
import pandas as pd

df = pd.read_csv('weather_data.txt',delimiter = '\t')

x = df.drop([df.columns[-1]],axis=1)
y = df[df.columns[-1]]

features = list(x.columns)
print(features)

x_train = x
y_train = y

train_size = x.shape[0]
num_feats = x.shape[1]

likelihoods = {}
pred_prior = {}
class_prior = {}

for feature in features:
likelihoods[feature] = {}
pred_prior[feature] = {}
for feat_val in x_train[feature]:
pred_prior[feature].update({feat_val:0})
for outcome in np.unique(y_train):
likelihoods[feature].update({feat_val+'_'+outcome:0})
class_prior.update({outcome:0})

print(likelihoods)
print(class_prior)
print(pred_prior)

for outcome in np.unique(y_train):
outcome_count = sum(y_train==outcome)
class_prior[outcome] = outcome_count/train_size

for feature in features:
feat_vals = x_train[feature].value_counts().to_dict()
print(feat_vals)
for feat_val,count in feat_vals.items():
pred_prior[feature][feat_val] = count/train_size

for feature in features:
for outcome in np.unique(y_train):
outcome_count = sum(y_train==outcome)
feat_likelihoods = x_train[feature][y_train[y_train==outcome].index.values.tolist()].value_counts().to_dict()
for feat_val,count in feat_likelihoods.items():
likelihoods[feature][feat_val+'_'+outcome] = count/outcome_count

print(likelihoods)
print(class_prior)
print(pred_prior)

qu = np.array([['Rainy','Hot','Normal','t']])

qu = np.array(qu)

for query in qu:
prob = {}
for outcome in np.unique(y_train):
prior = class_prior[outcome]
likelihood = 1
evidence = 1

    for feat,feat_val in zip(features,query):
        likelihood *= likelihoods[feat][feat_val+'_'+outcome]
        evidence *= pred_prior[feat][feat_val]
        
        posterior = (likelihood*prior)/evidence
        prob[outcome] = posterior
print(prob)
results = []
result = max(prob,key = lambda x: prob[x])
results.append(result)
print(results)

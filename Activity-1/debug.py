from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd


df = pd.read_pickle('../data/df_social_data_train.pkl')

# pre-processar e extrair caracteristicas dos dados para construir a tabela atributo-valor
from sentence_transformers import SentenceTransformer

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

df = df.dropna()

df['features'] = list(model.encode(df['content'].tolist(), show_progress_bar=True))

df['engagement'] = df['engagement'].map({'low': 0, 'high': 1})

labels = df['engagement'].to_numpy()
data = np.array(df['features'].tolist())

kf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

def applyKfold(data, label, classifier):

    accVet = []
    f1Vet = []
    f1MacroVet = []

    for train_index, test_index in kf.split(data, label):
        # particição dos dados em conjunto de treino / teste
        data_train, data_test = data[train_index], data[test_index]
        label_train, label_test = label[train_index], label[test_index]


        # classificador
        clf = classifier
        clf.fit(data_train, label_train)

        # predição
        y_pred = clf.predict(data_test)

        # avaliação
        acc = accuracy_score(label_test, y_pred)
        f1 = f1_score(label_test, y_pred)
        f1_macro = f1_score(label_test, y_pred, average='macro')

        accVet.append(acc)
        f1Vet.append(f1)
        f1MacroVet.append(f1_macro)


    return (accVet, f1Vet, f1MacroVet)




acc, f1, f1Macro = applyKfold(data, labels, GaussianNB())
print("Accuracy Score: ", round(np.mean(acc), 4))
print("F1 Score: ", round(np.mean(f1), 4))
print("F1 Macro Score: ", round(np.mean(f1Macro), 4))

acc, f1, f1Macro = applyKfold(data, labels, DecisionTreeClassifier(criterion='entropy'))
print("Accuracy Score: ", round(np.mean(acc), 4))
print("F1 Score: ", round(np.mean(f1), 4))
print("F1 Macro Score: ", round(np.mean(f1Macro), 4))

mlp = Sequential()
mlp.add(Dense(32, input_shape=(data.shape[1],), activation='tanh'))
mlp.add(Dense(16, activation='tanh'))
mlp.add(Dense(1, activation='sigmoid'))  # ← 1 saída binária

mlp.compile(loss='binary_crossentropy', metrics=['accuracy'])
mlp.summary()

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

mlp.fit(X_train,y_train, epochs=1000)
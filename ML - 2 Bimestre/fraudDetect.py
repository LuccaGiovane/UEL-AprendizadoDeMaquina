
from google.colab import drive
drive.mount('/content/drive')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import keras
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
import time

df = pd.read_csv("./dataset-fraude.csv")
df.columns

df.head()



df.dtypes

df.duplicated().any()


df = df.dropna()

plt.style.use("seaborn")

plt.rcParams['figure.figsize']= (22,11)

title = "Mapa de Calor de Correlação"

plt.title(title,fontsize=18, weight= 'bold')

sns.heatmap(df.corr(), cmap="BuPu", annot=True)

plt.show()

random_state = 42
features = [col for col in df.columns if col not in ["id", "Class"]]
X_train, X_test, y_train, y_test =train_test_split(df[features], df['Class'],
                                                   test_size=0.2,
                                                   random_state=random_state,
                                                   stratify=df['Class'],
                                                   shuffle=True)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)


def show_model_results(model):
  pred = model.predict(X_test)
  pred = np.where(pred > 0.5, 1, 0)
  print(classification_report(y_test, pred))
  cm = confusion_matrix(y_test, pred)
  disp = ConfusionMatrixDisplay(confusion_matrix = cm)
  disp.plot()
  plt.show()

def plot_history(history):
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']
    plt.plot(training_loss, label='Training Loss')
    plt.plot(validation_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

callbacks = [
    keras.callbacks.EarlyStopping(

        # Para o treino quando val_loss nao esta melhorando
        monitor="val_loss",
        min_delta=1e-2,
        # nao melhorando por 2 epochs
        patience=2,
        verbose=1,
    )
]

start_time = time.time()
logisticRegression = LogisticRegression()

teste1 = logisticRegression.fit(X_train, y_train)

print(f'tempo de treinamento: {round(time.time()-start_time, 2)} segundos')
show_model_results(teste1)
y_pred = logisticRegression.predict(X_test)
print(classification_report(y_test, y_pred))

def createModelSequential(neurons):
    model = Sequential()
    # adicionando uma camada densa a rede neural
    # em uma camada densa cada neuronio é conectado a todos os neuronios da
    # camada anterior
    # usa a função de ativação relu
    model.add(Dense(neurons, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # adiciona outra camada densa con um unico neuronio usando a funcao de ativacao
    # sigmoid
    # camada de saida
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # compila o modelo utilizando o algoritmo de otimizacao adam, funcao de perda
    # binary_crossentropy pois é um problema de classificação binária
    # usando accuracy de métrica de avaliação do modelo para o treinamento.
    return model

start_time = time.time()
sequentialModel = createModelSequential(len(features))
# Estanciando o modelo da rede neural sequencial com o tanto de neuronios(features)
history = sequentialModel.fit(X_train, y_train, epochs=5, batch_size=1024, verbose=1, validation_split=0.3) #callbacks= callbacks)
# divide em 10 epocas lotes de 512, deixa 20% para validação durante o treinamento e utiliza
# o callbacks definido anteriormente para parar o treinamento se nao houver melhorias por mais de 2 epocas
print(f'tempo de treinamento: {round(time.time()-start_time, 2)} segundos')

plot_history(history)
# plota o grafico do treinamento que mostra como o modelo foi melhorando conforme as epocas

sequentialModel.summary()

show_model_results(sequentialModel)

# plota as métricas resultantes do modelo


def createCnnModel(input_length):
  model = keras.Sequential([
      #definindo as camadas do modelo
        Conv1D(filters=5, #cria uma camada de convolução de 1 dimensão
               kernel_size=10, #kernel size
               activation="relu", #função de ativação
               input_shape=(input_length, 1), #forma da entrada, (tamanho, 1 dimensão)
               strides=2),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  return model

start_time = time.time()
cnnModel = createCnnModel(len(features))
history = cnnModel.fit(X_train, y_train, epochs=20, batch_size=512, verbose=1, validation_split=0.2, callbacks=callbacks)
print(f'tempo de treinamento: {round(time.time()-start_time, 2)} segundos')
plot_history(history)

show_model_results(cnnModel)
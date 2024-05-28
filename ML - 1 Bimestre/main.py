import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Carregar o dataset
#
dataset = pd.read_csv("Steam_Game_Popularity.csv")

# Mapear o nome do mês para o número correspondente
meses = {
    'janeiro': 1, 'fevereiro': 2, 'março': 3, 'abril': 4, 'maio': 5, 'junho': 6,
    'julho': 7, 'agosto': 8, 'setembro': 9, 'outubro': 10, 'novembro': 11, 'dezembro': 12
}

# Pré-processamento dos dados
# Extrair mês e ano da data
dataset['Mes'] = dataset['Data'].str.split('-').str[0].str.lower().map(meses)
dataset['Ano'] = dataset['Data'].str.split('-').str[1].astype(int)

# Seleciona as características relevantes (média de jogadores, mês e ano)
X = dataset[['Media de Jogadores', 'Mes', 'Ano']]
# Variável alvo são as tags
y = dataset['Tags']

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinamento do modelo de regressão logística
model = LogisticRegression(max_iter=5000, solver='saga')
model.fit(X_train, y_train)

# Avaliação do modelo
accuracy = model.score(X_test, y_test)
print(f'Acurácia do modelo: {accuracy}')

# Previsões para o futuro
# Substitua estas linhas com as datas desejadas para as quais você deseja fazer previsões
datas_futuras = ['outubro-21']
futuro_X = pd.DataFrame({
    'Media de Jogadores': [1000] * len(datas_futuras),  # Substitua 1000 pela média de jogadores esperada
    'Mes': [meses[data.split('-')[0]] for data in datas_futuras],
    'Ano': [int(data.split('-')[1]) for data in datas_futuras]
})

# Fazendo previsões para o futuro
predicoes = model.predict(futuro_X)

# Exibindo as previsões
for data, predicao in zip(datas_futuras, predicoes):
    print(f'{data}: {predicao}')
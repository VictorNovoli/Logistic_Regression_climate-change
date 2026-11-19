import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

#Dados de entrada (Features do modelo)
# Lê o arquivo. O sep=';' indica como as colunas estão divididas.
#df => DataFrame
#a função pd.read_csv lê um .csv e retorna um df, assim eu posso selecionar colunas pelo nome
#por exemplo: df['temperatura'] // tomar cuidado com espaços em branco no meu csv, como por ex: ano;temp;anomalia ;urbano
#eu também posso filtrar linhas baseadas em condições, ex: df[df['ano'] > 2015]
df = pd.read_csv('dados_clima.csv', sep=';')

def passar_dados():
    print("COMEÇANDO A TREINAR O MODELO")
    modelo = LinearRegression()
    # 1. Você extrai a matriz X (todas as linhas e 3 colunas) do DataFrame inteiro
    X = df[['G_Anomalia', 'R_ENSO', 'U_Urbano']]

    # 2. Você extrai o vetor y (todas as linhas e 1 coluna) do DataFrame inteiro
    y = df['Y_Temp_SP'] 
    # Note que usamos 'df', não 'linha'! Isso pega todos os 8 anos de uma vez.

    # 3. Você entrega tudo junto para o modelo aprender de uma tacada só
    modelo.fit(X, y)
    return modelo

#def passar_dados(): #laço que já extrai os dados e treina o modelo
#   modelo = LinearRegression()
#    for index, linha in df.iterrows():
#        y = linha['Y_Temp_SP']
#        gt = linha['G_Anomalia']
#        rt = linha['R_ENSO']
#        ut = linha['U_Urbano']
#        X = [y, gt, rt, ut]
#        modelo.fit(X, ) #aqui eu vou chamar a função de treinar várias vezes, isso esta certo?
#        print(f"Bateria {index} de treino concluida")
#    return modelo

def prever_futuro(modelo, cenario_futuro):
    previsao = modelo.predict(cenario_futuro) 
    return previsao

if __name__ == "__main__":
    print("----INICIANDO O TREINAMENTO DO MODELO----")
    modelo = passar_dados()
    print("----TREINAMENTO FINALIZADO----")
    print("----PARTINDO PARA AS PREDIÇÕES----")
    cenario_2020 = pd.DataFrame(
    [[1.15, -0.5, 9]], 
    columns=['G_Anomalia', 'R_ENSO', 'U_Urbano']
    )
    temperatura = prever_futuro(modelo, cenario_2020)
    print("----PREVISÃO FINALIZADA----")
    print(f"Previsão para 2020 de: {temperatura}")
# # Projeto de Análise de Churn em uma Operadora de Telecomunicações

# ## Contexto

# Uma operadora de telecomunicações que enfrenta um aumento significativo na taxa de cancelamento de clientes, também conhecida como **churn**. 
# 
# Para abordar esse desafio, este projeto visa analisar os dados disponíveis e identificar padrões que possam fornecer insights sobre os motivos do cancelamento. Além disso, o projeto se propõe a desenvolver estratégias de retenção com base nos resultados da análise.

# ## Objetivos

# 1. Identificar padrões e características associadas ao churn de clientes.
# 2. Realizar análise exploratória de dados (EDA) para compreender a distribuição e correlação entre variáveis.
# 3. Desenvolver estratégias de retenção com base nos insights obtidos.
# 4. Avaliar a viabilidade de construir um modelo de machine learning para prever o churn.

# ## Conjunto de Dados

#  O conjunto de dados utilizado neste projeto consiste em informações de clientes da empresa, incluindo detalhes demográficos, serviços contratados, avaliações de satisfação, e a variável de destino "Churn", indicando se o cliente cancelou ou não o serviço.

# # Importação das bibliotecas necessárias

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import plotly.express as px
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind, mannwhitneyu


# # Importação e visualização dos dados


df = pd.read_csv('telecom_churn.csv')
print(df.head())


# # Verifica se há colunas com valores nulos

df.isna().sum()


# # Verificar se existem colunas com valores vazios


colunas_com_valores_vazios = df.isnull().any()

# Exibir as colunas com valores vazios, se houver
colunas_com_valores_vazios = colunas_com_valores_vazios[colunas_com_valores_vazios]
if not colunas_com_valores_vazios.empty:
    print("Colunas com valores vazios:")
    print(colunas_com_valores_vazios)
else:
    print("Não há colunas com valores vazios.")


# # Verificar se existem valores vazios (espaços em branco) em todo o DataFrame


valores_vazios = df.applymap(lambda x: x.isspace() if isinstance(x, str) else False)
colunas_com_valores_vazios = valores_vazios.any()

# Exibir as colunas com valores vazios, se houver
colunas_com_valores_vazios = colunas_com_valores_vazios[colunas_com_valores_vazios]
if not colunas_com_valores_vazios.empty:
    print("Colunas com valores vazios:")
    print(colunas_com_valores_vazios)
else:
    print("Não há colunas com valores vazios.")


# # Visualizar as linhas com célas da coluna "TotalCharges" vazias

df[df.TotalCharges==' ']

# # Tratar as células da coluna "TotalCharges" com espaços vazios


df['TotalCharges'] = df['TotalCharges'].replace(' ', 0)


# # Analizando os de conteúdo das colunas

# value count of unique variables
for i in df.columns:
    if df[i].dtype=='object':
        print(df[i].value_counts())
        print('#========================= \n')


# # Analisando tipos de dados das colunas

print(df.info())


# # Excluir a coluna 'customerID'

df = df.drop('customerID', axis=1)

# # Converte a coluna 'TotalCharges' para tipo o numérico

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# # Visualização da distribuição de churn

qtd_categorias = df['Churn'].value_counts()
print(qtd_categorias)

qtd_categorias_perc = df['Churn'].value_counts(normalize=True)
print(qtd_categorias_perc)


# # Teste de Qui-Quadrado para Variáveis Categóricas:

# ### Realize o teste qui-quadrado para avaliar a dependência entre variáveis categóricas.

cat_columns = df.select_dtypes(include='object').columns
for col in cat_columns:
    contingency_table = pd.crosstab(df[col], df['Churn'])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    print(f'Chi-squared test for {col}: p-value = {p:.4f}')


# ### É possível observar que as variáveis 'gender" e "PhoneService"  obtiveram um valor acima do 0,05 ( Nível de Significância) , sinalizando uma possível **NÃO** relação entre a variavel "Churn"

# # Gráfico para Análise de Churn nas variáveis "Gender" e "PhoneService"

# Criar subplots com 1 linha e 2 colunas
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

# Gráfico 1: Contagem de gênero
sns.countplot(x='gender', hue='Churn', data=df, ax=axes[0])
axes[0].set_title('Churn Count by Gender')

# Gráfico 2: Contagem de PhoneService
sns.countplot(x='PhoneService', hue='Churn', data=df, ax=axes[1])
axes[1].set_title('Churn Count by PhoneService')

# Ajustes de layout
plt.tight_layout()
plt.show()


# ### Nós dois gráficos podemos confirmar que estas variáveis não influenciam no Churn, pois parece ser proporcional as duas opções da categoria

# # testes t e Mann-Whitney U para as variáveis numéricas

numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_columns:
    churn_values = df[df['Churn']=='Yes'][col]
    no_churn_values = df[df['Churn']=='No'][col]
    t_stat, p_value_t = ttest_ind(churn_values, no_churn_values)
    mwu_stat, p_value_mwu = mannwhitneyu(churn_values, no_churn_values)
    print(f'Test for {col}: t-Test p-value = {p_value_t:.4f}, Mann-Whitney U p-value = {p_value_mwu:.4f}')


# # Visualização da distribuição de 'tenure'

sns.histplot(df['tenure'], bins=30, kde=False)
plt.title('Distribution of Tenure')
plt.show()

# # Visualização da distribuição de 'MonthlyCharges'

grafico = px.histogram(df, x='MonthlyCharges', color='Churn', nbins=30, histnorm='percent')
grafico.show()

# ### Há um percentual maior de churn nas faixas de 'MonthlyCharges' entre 70 e 100, diminuindo em 110 até 120.

# # Contagem de serviços de internet

sns.countplot(x='InternetService', hue='Churn', data=df)
plt.title('Churn Count by Internet Service')
plt.show()


# ###  'Fiber Optic' tem um percentual de churn significativamente mais alto em comparação com os outros tipos de serviço de internet.

# # Contagem de contratos

sns.countplot(x='Contract', hue='Churn', data=df)
plt.title('Churn Count by Contract Type')
plt.show()

# ### 'Month-to-month' tem um percentual bem mais alto de churn em comparação com os outros tipos de contrato.

# # Scatter plot entre 'MonthlyCharges' e 'TotalCharges'

sns.scatterplot(x='MonthlyCharges', y='TotalCharges', hue='Churn', data=df)
plt.title('Scatter Plot of MonthlyCharges vs. TotalCharges')
plt.show()

# ### Há uma concentração de churn no lado inferior direito do gráfico, indicando uma relação entre despesas mensais mais altas e menor total de gastos.

# # Boxplot para 'tenure' agrupado por 'Churn'

sns.boxplot(x='Churn', y='tenure', data=df)
plt.title('Tenure Distribution by Churn')
plt.show()

# # Gráfico de barras empilhadas para 'Partner' e 'Dependents'

df_partner_dependents = df.groupby(['Partner', 'Dependents', 'Churn']).size().unstack(fill_value=0)
df_partner_dependents.plot(kind='bar', stacked=True)
plt.title('Churn Count by Partner and Dependents')
plt.show()


# # Contagem de método de pagamento

sns.countplot(x='PaymentMethod', hue='Churn', data=df)
plt.title('Churn Count by Payment Method')
plt.xticks(rotation=45)
plt.show()


# ### 'Electronic Check' tem um percentual de churn extremamente maior do que os outros métodos de pagamento.

# # Distribuição de 'MonthlyCharges' por 'Churn'

sns.kdeplot(df[df['Churn']=='No']['MonthlyCharges'], label='No Churn', fill=True)
sns.kdeplot(df[df['Churn']=='Yes']['MonthlyCharges'], label='Churn', fill=True)
plt.title('Monthly Charges Distribution by Churn')
plt.show()


# ### Há um volume maior de churn no lado direito do gráfico, na faixa de 65 a 110, superando a densidade do churn em outras faixas.

# # Gráfico de barras empilhadas para serviços de streaming


df_streaming = df.groupby(['StreamingTV', 'StreamingMovies', 'Churn']).size().unstack(fill_value=0)
df_streaming.plot(kind='bar', stacked=True)
plt.title('Churn Count by Streaming Services')
plt.show()


# ### Serviços como segurança online, streaming de TV, backup online, suporte técnico sem conexão com a Internet estão negativamente relacionados ao churn.

# # Distribuição de 'TotalCharges' por 'Churn'

sns.kdeplot(df[df['Churn']=='No']['TotalCharges'].astype(float), label='No Churn', fill=True)
sns.kdeplot(df[df['Churn']=='Yes']['TotalCharges'].astype(float), label='Churn', fill=True)
plt.title('Total Charges Distribution by Churn')
plt.show()


# # Scatter plot entre 'MonthlyCharges' e 'tenure' com cores representando 'Churn'

sns.scatterplot(x='MonthlyCharges', y='tenure', hue='Churn', data=df)
plt.title('Scatter Plot of MonthlyCharges vs. Tenure')
plt.show()

# ### Há um agrupamento maior de churn no lado inferior direito na faixa de 60 a 110

# # Conclusões:

# 1. Clientes com contratos 'Month-to-month' e serviços de 'Fiber Optic' têm maior probabilidade de churn.
# 2. A forma como o pagamento é feito, especialmente com 'Electronic Check', está fortemente associada ao churn.
# 3. Existe uma relação entre 'MonthlyCharges', 'TotalCharges', e churn, indicando que clientes com despesas mensais mais altas podem ter maior probabilidade de churn.
# 4. Serviços de streaming e a combinação 'Partner'/'Dependents' também têm impacto no churn.
# 5. Contratos mais longos (dois anos) estão associados a uma menor rotatividade.

# # Recomendações:

# 1. Considerar estratégias de retenção específicas para clientes com contratos 'Month-to-month' e serviços de 'Fiber Optic'.
# 
# 2. Explorar opções de incentivo para alterar o método de pagamento, especialmente para reduzir o uso de 'Electronic Check'.
# 
# 3. Desenvolver ofertas personalizadas para clientes com despesas mensais mais altas.
# 
# 4. Avaliar melhorias nos serviços de streaming e considerar promoções para clientes sem parceiro ou dependentes.
# 
# 5. Monitorar de perto clientes com contratos 'Month-to-month' e altos gastos mensais para antecipar ações de retenção.
# 
# Essas recomendações podem ajudar a reduzir a taxa de churn e melhorar a satisfação do cliente.

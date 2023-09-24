import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

dataset = pd.read_csv("files/credit_simple.csv", sep=";")

y = dataset["CLASSE"]
X = dataset.iloc[:,:-1]

print(X.isnull().sum()) #sum of null attributes

median = X["SALDO_ATUAL"].median()

X["SALDO_ATUAL"].fillna(median, inplace=True) #replacing the null attributes for the median
X.isnull().sum()

grouped_estadocivil = X.groupby(["ESTADOCIVIL"]).size()
print(grouped_estadocivil)

X["ESTADOCIVIL"].fillna("masculino solteiro", inplace=True)

print(X.isnull().sum()) #now there are no null attributes

sd = X["SALDO_ATUAL"].std()

print(X.loc[X["SALDO_ATUAL"] >= 2 * sd, "SALDO_ATUAL"]) #finding outliers

X.loc[X["SALDO_ATUAL"] >= 2 * sd, "SALDO_ATUAL"] = median #replacing outliers by the median

grouped_proposito = X.groupby(["PROPOSITO"]).size()
print(grouped_proposito)

X.loc[X["PROPOSITO"] == "Eletrodomésticos", "PROPOSITO"] = "outros"
X.loc[X["PROPOSITO"] == "qualificação", "PROPOSITO"] = "outros"

grouped_proposito = X.groupby(["PROPOSITO"]).size()
print(grouped_proposito)

X["DATA"] = pd.to_datetime(X["DATA"], format="%d/%m/%Y")
print(X["DATA"])

#creating 3 columns
X["ANO"] = X["DATA"].dt.year
X["MES"] = X["DATA"].dt.month
X["DIASEMANA"] = X["DATA"].dt.day_name()

print(X["ESTADOCIVIL"].unique())
print(X["PROPOSITO"].unique())
print(X["DIASEMANA"].unique())

labelencoder1 = LabelEncoder()
X["ESTADOCIVIL"] = labelencoder1.fit_transform(X["ESTADOCIVIL"])
X["PROPOSITO"] = labelencoder1.fit_transform(X["PROPOSITO"])
X["DIASEMANA"] = labelencoder1.fit_transform(X["DIASEMANA"])

print(X)

outros = X["OUTROSPLANOSPGTO"].unique()
print(outros)

#one hot encoding
z = pd.get_dummies(X["OUTROSPLANOSPGTO"], prefix="OUTROS") #it creates a different dataframe
print(z)

#A padronização é um passo comum em muitos algoritmos de aprendizado de máquina, pois ajuda a garantir que os 
# atributos estejam na mesma escala, o que pode ser importante para o desempenho de alguns modelos.
sc = StandardScaler()
m = sc.fit_transform(X.iloc[:,0:3]) #getting all the lines and the first 3 columns
print(m)

#now we need to add z and m to the original dataframe
#m is an object of numpy, that's why we need to transform it into a dataframe
X = pd.concat([X, z, pd.DataFrame(m, columns=["SALDO_ATUAL_N", "RESIDENCIADESDE_N", "IDADE_N"])], axis=1)
print(X)

#excluding the colums we don't need anymore
X.drop(columns=["SALDO_ATUAL", "RESIDENCIADESDE", "IDADE", "OUTROSPLANOSPGTO", "DATA", "OUTROS_banco"], inplace=True)

print(X)

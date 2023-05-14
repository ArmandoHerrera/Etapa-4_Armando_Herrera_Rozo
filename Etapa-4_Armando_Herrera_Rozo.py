#!/usr/bin/env python
# coding: utf-8

# ##Presentado: Armando Herrera Rozo
# ##Grupo: 67
# ##Análisis de datos_202016908
# ##Tutora GLORIA ALEJANDRA RUBIO
# 
# ## Desarrolla el ejercicio propuesto empleando el Arboles de decisiones

# In[58]:


##Importar las bibliotecas
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


# In[36]:


##Cargamos el archivo "covid.csv"
pd.read_csv("covid.csv")
df = pd.read_csv("covid.csv")
display(df.head(5))


# In[37]:


##Cargamos el archivo en un DataFrame
num_filas = df.shape[0]
print("El DataFrame tiene", num_filas, "filas.")


# In[38]:


##Eliminar las columnas que no necesitas
columnas_a_eliminar = ["id","patient_type","entry_date","date_symptoms","date_died","other_disease","icu"]
df.drop(columnas_a_eliminar, axis=1, inplace=True)
display(df.head(5))


# In[39]:


##Filtrar las filas y eliminar aquellas que tengan el valor 3 em la columna covid_res
df = df[df["covid_res"]!= 3] 
display(df.head(5))


# In[40]:


##Filtrar las filas y eliminar aquellas que tengan el valor 99 em la columna contact_other_covid
df = df[df["contact_other_covid"]!= 99] 
display(df.head(5))


# In[41]:


##Filtrar las filas y eliminar aquellas que tengan el valor 98 em la columna tobacco
df = df[df["tobacco"]!= 98] 
display(df.head(5))


# In[42]:


##Filtrar las filas y eliminar aquellas que tengan el valor 98 em la columna renal_chroni
df = df[df["renal_chronic"]!= 98] 
display(df.head(5))


# In[43]:


##Filtrar las filas y eliminar aquellas que tengan el valor 98 em la columna cardiovascular
df = df[df["cardiovascular"]!= 98] 
display(df.head(5))


# In[44]:


##Filtrar las filas y eliminar aquellas que tengan el valor 98 em la columna hypertension
df = df[df["hypertension"]!= 98] 
display(df.head(5))


# In[45]:


##Filtrar las filas y eliminar aquellas que tengan el valor 98 em la columna inmsupr
df = df[df["inmsupr"]!= 98] 
display(df.head(5))


# In[46]:


##Filtrar las filas y eliminar aquellas que tengan el valor 98 em la columna asthma
df = df[df["asthma"]!= 98] 
display(df.head(5))


# In[47]:


##Filtrar las filas y eliminar aquellas que tengan el valor 98 em la columna copd
df = df[df["copd"]!= 98] 
display(df.head(5))


# In[48]:


##Filtrar las filas y eliminar aquellas que tengan el valor 98 em la columna diabetes
df = df[df["diabetes"]!= 98] 
display(df.head(5))


# In[49]:


##Filtrar las filas y eliminar aquellas que tengan el valor 97 em la columna pregnancy
df = df[df["pregnancy"]!= 97] 
display(df.head(5))


# In[50]:


##Filtrar las filas y eliminar aquellas que tengan el valor 99 em la columna pneumonia
df = df[df["pneumonia"]!= 99] 
display(df.head(5))


# In[51]:


#Filtrar las filas y eliminar aquellas que tengan el valor 97 em la columna intubed
df = df[df["intubed"]!= 97] 
display(df.head(5))


# In[52]:


#Mostrar total de filas
num_filas = df.shape[0]
print("El DataFrame tiene", num_filas, "filas.")


# In[19]:


#Separar las caracteristicas de la variable objetivo
X = df.drop("covid_res", axis=1)
y = df["covid_res"]


# In[20]:


#Dividir el cojunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# In[21]:


#crear una instancia del modelo de arbol de decision
arbol_decision = DecisionTreeClassifier(random_state=1) 


# In[22]:


#Entrenar el modelo con el conjunto de entrenamiento
arbol_decision.fit(X_train, y_train)


# In[23]:


#Utilizar el modelo para Hacer prediciones en el conjunto de prueba
y_pred = arbol_decision.predict(X_test)


# In[24]:


#Evaluar el desempeño del modelo
accuracy = accuracy_score(y_test, y_pred)
print("La Predicion del modelo es:", accuracy)


# In[25]:


pip install graphviz


# In[26]:


conda install python-graphviz


# In[27]:


# simplificar el arbol a 10 aperturas
arbol_decision = DecisionTreeClassifier(max_depth=3, min_samples_split=10) 
arbol_decision.fit(X_train, y_train)


# In[28]:


from sklearn.tree import export_graphviz
import graphviz

#Exportar el arbol de decisiones a un archivo .dot
export_graphviz(arbol_decision, out_file='arbol_decision.dot',
               feature_names=X.columns.values, filled=True, rounded=True, special_characters=True)

#Convertir el archivo .dot a un objeto graphviz
with open('arbol_decision.dot') as f:
    dot_graph = f.read()
    graph = graphviz.Source(dot_graph)
    
# Mostrar el arbol de decisiones
    graph


# In[29]:


from IPython.display import Image
from graphviz import render

render('dot', 'png', 'arbol_decision.dot')


# In[30]:


df['covid_res']=df['covid_res'].map({
        0: 'No', 1: 'Yes'
    })
df.head()


# In[53]:


##Visualizamos algunas características de los datos Ej: se crea un gráfico de varas de edad de los pacientes
df["age"].hist(bins=20)
plt.xlabel("age")
plt.ylabel("count")
plt.show()


# In[54]:


df.info() 


# In[55]:


df.groupby(['diabetes', 'obesity'])['obesity'].count()


# In[60]:


ax=sns.countplot(x='diabetes', hue='obesity', palette='Set1', data=df)
ax.set(title='Total de Pacientes con Obesidad en funcion de precencia de diabetes', 
       xlabel='diabetes', ylabel='Total')
plt.show()


# ## Desarrolla el ejercicio propuesto empleando el KNN

# In[61]:


from sklearn.neighbors import KNeighborsClassifier


# In[62]:


#Definir las caracteristicas de la variable objetivo
X = df.drop("covid_res", axis=1)
y = df["covid_res"]


# In[63]:


#Dividir los datos en un conjonto de entrenamiento y un conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)


# In[64]:


#Crear el modelo de vesinos mas cercanos
knn = KNeighborsClassifier(n_neighbors=5)


# In[65]:


#Entrenar el modelo con los datos de entrenamiento
knn.fit(X_train, y_train)


# In[66]:


# Realizar prediciones con los datos de prueba
y_pred = knn.predict(X_test)


# In[67]:


#calcular la precision del modelo

accuracy = accuracy_score(y_test, y_pred)
print("La Predicion del modelo es:", accuracy * 100)


# In[68]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

#definir las columnas que se utilizan como carateristicas
features = ['sex','intubed','pneumonia','age','pregnancy','diabetes','copd','asthma','inmsupr','hypertension'
            ,'cardiovascular','obesity','renal_chronic','tobacco','contact_other_covid','covid_res']

#Definir la columna que se utilizara como objetivo
target = 'covid_res'

#Definir el conjunto de datos en carateristicas (x) y objetivo (y)
X = df[features]
y = df[target]

#Dividir los datos en un conjonto de entrenamiento y un conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

#Crear el modelo de knn con un valor de 5
knn = KNeighborsClassifier(n_neighbors=5)

#Entrenar el modelo con los datos de entrenamiento
knn.fit(X_train, y_train)


# Realizar prediciones con los datos de prueba
y_pred = knn.predict(X_test)

#crear la matriz de confusion
conf_mat = confusion_matrix(y_test, y_pred)

#Imprimir la matriz de confusion
print(conf_mat)


# In[69]:


import matplotlib.pyplot as plt
import seaborn as sns

#Crear una figura para la matriz de confusion
fig, ax = plt.subplots(figsize=(6,4))

#Crear la matriz de confusion utilizando la funcion heatmap de seaborn
sns.heatmap(conf_mat, annot=True, cmap="Greens",fmt="d", xticklabels=["Negativo", "Positivo"], yticklabels=["Negativo", "Positivo"] )

#Configurar las etiquetas de los ejes
ax.set_xlabel('Prediccion')
ax.set_ylabel('Valor Real')
ax.set_title('Matriz de confusiòn')

#Mostrar la figura
plt.show()


# ## Desarrolla el ejercicio propuesto empleando el Naive Bayes

# In[70]:


from sklearn.naive_bayes import GaussianNB


# In[71]:


#Crear un objeto GaussianNB
naive_bayes = GaussianNB()

#Definir las caracteristicas de la variable objetivo
X = df.drop("covid_res", axis=1)
y = df["covid_res"]

#Dividir los datos en un conjonto de entrenamiento y un conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

#Entrenar el modelo 
naive_bayes.fit(X_train, y_train)

#calcular la precision del modelo
accuracy = accuracy_score(y_test, y_pred)


# In[72]:


#Mostrar el porcentaje de precision del modelo
print("porcentaje de precision del modelo: {:.2f}%".format(accuracy * 100))


# In[73]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

#Obtener las predicciones en el conjunto de prueba
y_pred = naive_bayes.predict(X_test)

#Crear la matriz de confusion
cm = confusion_matrix(y_test, y_pred)

#Graficar la matriz de confusion
sns.heatmap(cm, annot=True, fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Matriz de Predicciones')

#Mostrar la figura
plt.show()


# In[ ]:





# In[74]:


#Crear una fila con los datos del usuario
usuario = pd.DataFrame({
    'sex': 2,
    'sex': 2,
    'intubed': 2,
    'pneumonia': 2,
    'age': 35,
    'pregnancy': 2,
    'diabetes': 2,
    'copd': 2,
    'asthma': 2,
    'inmsupr': 2,
    'hypertension': 2,
    'cardiovascular': 2,
    'obesity': 2,
    'renal_chronic': 2,
    'tobacco': 2,
    'contact_other_covid': 2
}, index=[0])

#Obtener las predicciones para el usuario
probabilidad = naive_bayes.predict_proba(usuario)[0][1]

#Mostrar la probabilidad de que el usuario tenga COVID-19
print("Probabilidad de COVID-19: {:.2f}%".format(probabilidad * 100))


# In[ ]:





# In[ ]:





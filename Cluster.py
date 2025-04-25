# Paso 1: Importar librerías necesarias
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Paso 2: Cargar el archivo limpio
df = pd.read_csv("/content/Premier_League_Stats_Limpio.txt", sep='\t')
print(" Archivo cargado correctamente. Shape:", df.shape)

# Paso 3: Filtrar solo las columnas numéricas para el escalado
df_numeric = df.select_dtypes(include=[np.number])  # Selecciona solo columnas numéricas
print(" Columnas numéricas seleccionadas:", df_numeric.columns.tolist())

# Paso 4: Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)  # Escalar solo las columnas numéricas
print(" Datos escalados correctamente.")

# Paso 5: Determinar número óptimo de clusters (método del codo)
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Visualizar el gráfico del codo
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Método del Codo')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Inercia')
plt.grid(True)
plt.show()

# Paso 6: Elegir el valor de K (según el codo del gráfico)
k = 4  # Cambia este valor si ves otro mejor en el gráfico
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Agregar los clusters al DataFrame original
df['Cluster'] = clusters
print("Clusters asignados correctamente.")
print(df[['Cluster']].value_counts())

# Paso 7: Mostrar ejemplos de jugadores por cluster
for i in range(k):
    print(f"\n🔹 Cluster {i}")
    print(df[df['Cluster'] == i].head())

# Paso 8: Mostrar promedios por cluster (análisis)
print("\n Promedios por cluster:")
print(df.groupby('Cluster').mean(numeric_only=True))

# Paso 9: Visualizar los clusters en 2D con PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='Set2', s=100)
plt.title("Visualización de Clusters de Jugadores (PCA)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.grid(True)
plt.show()

# Paso 10: Guardar el DataFrame con clusters
# Guardar el DataFrame con los clusters
output_file = "Premier_League_Stats_Clusterizado.txt"

try:
    df.to_csv(output_file, index=False, sep='\t')
    print(f" Archivo guardado exitosamente como '{output_file}'")

    # Verificamos que realmente existe y mostramos las primeras filas
    df_verificado = pd.read_csv(output_file, sep='\t')
    print(" Verificación: archivo cargado exitosamente.")
    print(df_verificado.head())
except Exception as e:
    print(" Error al guardar el archivo:", e)


<p align="center">
  <img src="https://raw.githubusercontent.com/github/explore/main/topics/python/python.png" width="90">
</p>

<h1 align="center">
  Clustering para Identificación de Perfiles de Pobreza Multidimensional
</h1>

<p align="center">
  <b>Machine Learning · Análisis de Datos · Economía Aplicada</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white" alt="Jupyter">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white" alt="scikit-learn">
</p>

---

## 📑 Tabla de Contenidos
- [Acerca del Proyecto](#acerca-del-proyecto)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Metodología](#metodología)
- [Algoritmos Implementados](#algoritmos-implementados)
- [Requisitos](#requisitos)
- [Cómo Ejecutar](#cómo-ejecutar)
- [Resultados Principales](#resultados-principales)
- [Conclusiones](#conclusiones)
- [Próximos Pasos](#próximos-pasos)
- [Licencia](#licencia)
- [Autor](#autor)

---

## Acerca del Proyecto

Este proyecto aplica técnicas avanzadas de **Machine Learning no supervisado** para identificar perfiles diferenciados de hogares según sus privaciones multidimensionales, utilizando datos de la **Encuesta de Calidad de Vida (ECV) 2024** de Colombia.

### Objetivos principales:

1. **Identificar perfiles diferenciados** de hogares según sus privaciones multidimensionales
2. **Validar la consistencia** de los clusters obtenidos con el Índice de Pobreza Multidimensional (IPM) oficial del DANE
3. **Caracterizar territorialmente** los grupos identificados para entender patrones geográficos
4. **Generar insumos** para el diseño de políticas públicas focalizadas y diferenciadas

### Contexto

El proyecto va más allá de la clasificación binaria tradicional (pobre/no pobre) del IPM oficial, buscando identificar matices y patrones intermedios que permitan comprender mejor la heterogeneidad de la pobreza en Colombia. Esto es fundamental para el diseño de intervenciones más efectivas y focalizadas.

---

## Estructura del Repositorio
```
Repositorio_Clustering/
│
├── clustering_pobreza.ipynb    # Notebook principal con análisis completo
├── hogares_ML.csv              # Base de datos de hogares (53,103 registros)
├── h                           # Archivo auxiliar
└── README.md                   # Este archivo
```

### Descripción de archivos:

- **clustering_pobreza.ipynb**: Contiene el análisis completo desde la carga de datos hasta la visualización de resultados
- **hogares_ML.csv**: Dataset con 53,103 hogares y 30 variables incluyendo las 15 privaciones del IPM
- **README.md**: Documentación del proyecto

---

## Metodología

### 1. Variables utilizadas

El análisis utiliza **17 variables** agrupadas en tres categorías:

**Variables de privación del IPM (15 variables):**

*Educación:*
- Bajo logro educativo (LOGRO_2)
- Analfabetismo (ALFA_2)
- Inasistencia escolar (ASISTE_2)
- Rezago escolar (REZAGO_2)
- Barreras de acceso a servicios de cuidado infantil (A_INTEGRAL_2)

*Trabajo:*
- Trabajo infantil (TRABAJOINF_2)
- Desempleo de larga duración (DES_DURA_2)
- Empleo informal (EFORMAL_2)

*Salud:*
- Sin aseguramiento en salud (SEGURO_SALUD_2)
- Barreras de acceso a servicios de salud (SALUD_NEC_2)

*Vivienda y servicios:*
- Sin acceso a acueducto (ACUEDUCTO)
- Sin acceso a alcantarillado (ALCANTARILLADO)
- Pisos inadecuados (PISOS)
- Paredes inadecuadas (PAREDES)
- Hacinamiento crítico (HACINAMIENTO)

**Variables territoriales (2 variables):**
- Zona rural (ZONA_RURAL)
- Centro poblado (ZONA_CENTRO_POBLADO)

### 2. Preprocesamiento

- **Estandarización**: Uso de StandardScaler y RobustScaler para normalizar variables
- **Reducción dimensional**: Aplicación de PCA para visualización
- **Tratamiento de outliers**: Manejo robusto de valores atípicos

### 3. Proceso de clustering

El análisis sigue un enfoque sistemático:

1. **Selección del número óptimo de clusters**: Uso del método del codo y análisis de silueta
2. **Aplicación de múltiples algoritmos** para comparación y validación
3. **Evaluación cuantitativa**: Métricas de Silhouette, Calinski-Harabasz y Davies-Bouldin
4. **Caracterización de perfiles**: Análisis estadístico de cada cluster identificado
5. **Validación externa**: Comparación con el IPM oficial del DANE

---

## Algoritmos Implementados

### 1. K-Means
Algoritmo de partición que agrupa hogares minimizando la varianza intra-cluster.

**Ventajas:**
- Rápido y eficiente
- Resultados interpretables
- Bien documentado

**Limitaciones:**
- Asume clusters esféricos
- Sensible a outliers

### 2. DBSCAN (Density-Based Spatial Clustering)
Algoritmo basado en densidad que identifica clusters de forma arbitraria.

**Ventajas:**
- Detecta outliers automáticamente
- No requiere especificar número de clusters
- Encuentra formas complejas

**Limitaciones:**
- Sensible a parámetros (eps, min_samples)
- Menos eficiente en alta dimensionalidad

### 3. Clustering Jerárquico (Agglomerative)
Construye una jerarquía de clusters mediante agregación sucesiva.

**Ventajas:**
- Produce dendrograma interpretable
- No requiere número de clusters a priori
- Robusto a diferentes formas

**Limitaciones:**
- Computacionalmente costoso
- No permite reasignación de puntos

### 4. Gaussian Mixture Models (GMM)
Modelo probabilístico que asume distribuciones gaussianas superpuestas.

**Ventajas:**
- Proporciona probabilidades de pertenencia
- Flexible en formas de clusters
- Fundamento estadístico sólido

**Limitaciones:**
- Más complejo de interpretar
- Puede sobreajustar con pocos datos

---

## Requisitos

### Software necesario

- **Python** ≥ 3.7
- **Jupyter Notebook** o **Google Colab**

### Librerías de Python
```python
# Análisis de datos
import pandas as pd
import numpy as np

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Machine Learning
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA

# Métricas de evaluación
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples
)
```

### Instalación de dependencias
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
```

---

## Cómo Ejecutar

### Opción 1: Local (Jupyter Notebook)
```bash
# 1. Clonar el repositorio
git clone https://github.com/jmeza-data/Repositorio_Clustering.git
cd Repositorio_Clustering

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Iniciar Jupyter Notebook
jupyter notebook

# 4. Abrir clustering_pobreza.ipynb y ejecutar las celdas secuencialmente
```

### Opción 2: Google Colab (Recomendado)

1. Abrir [Google Colab](https://colab.research.google.com/)
2. File → Upload notebook → Seleccionar `clustering_pobreza.ipynb`
3. Subir el archivo `hogares_ML.csv` a la sesión
4. Ejecutar todas las celdas: Runtime → Run all

### Estructura de ejecución

El notebook está organizado en secciones secuenciales:

1. **Configuración inicial**: Importación de librerías y configuración de visualización
2. **Carga de datos**: Lectura del CSV y verificación de variables críticas
3. **Selección de variables**: Preparación del conjunto de features para clustering
4. **Análisis exploratorio**: Correlaciones y estadísticas descriptivas
5. **Determinación de K óptimo**: Método del codo y análisis de silueta
6. **Aplicación de algoritmos**: Ejecución de K-Means, DBSCAN, Jerárquico y GMM
7. **Evaluación y comparación**: Métricas de calidad de clustering
8. **Caracterización de perfiles**: Análisis estadístico de cada cluster
9. **Visualización espacial**: Mapas tipo isoyetas con interpolación territorial

---

## Resultados Principales

### Perfiles identificados

El análisis reveló la existencia de **grupos diferenciados** de hogares según sus patrones de privación, que van más allá de la clasificación binaria tradicional del IPM.

### Validación con IPM oficial

Los clusters muestran **coherencia estadística** con el Índice de Pobreza Multidimensional oficial del DANE, validando la robustez del análisis.

### Heterogeneidad territorial

Se identificó una **variabilidad geográfica significativa** en la distribución de los perfiles de pobreza, con patrones diferenciados entre zonas urbanas, rurales y centros poblados.

### Visualizaciones generadas

El proyecto genera múltiples visualizaciones incluyendo:
- Gráfico del método del codo
- Análisis de silueta por cluster
- Dendrogramas jerárquicos
- Mapas tipo isoyetas con interpolación espacial
- Perfiles de privación por cluster
- Análisis de componentes principales (PCA)

---

## Conclusiones

### Hallazgos principales:

1. **Perfiles diferenciados**: Se identificaron exitosamente grupos de hogares con patrones de privación característicos que permiten una comprensión más matizada de la pobreza multidimensional

2. **Coherencia metodológica**: Los clusters obtenidos muestran consistencia con el IPM oficial del DANE, validando tanto la metodología como los resultados

3. **Dimensión territorial**: Existe una heterogeneidad territorial significativa que debe ser considerada en el diseño de políticas públicas focalizadas

4. **Variables críticas**: El bajo logro educativo, analfabetismo y condiciones de vivienda emergieron como las variables más correlacionadas con el IPM

### Implicaciones para política pública:

- Los perfiles identificados permiten diseñar **intervenciones diferenciadas** según las necesidades específicas de cada grupo
- La caracterización territorial facilita la **focalización geográfica** de programas sociales
- Los resultados proporcionan **evidencia cuantitativa** para la toma de decisiones en política social

---

## Próximos Pasos

### Extensiones propuestas:

1. **Modelado predictivo**: Utilizar los clusters identificados como variable categórica en modelos supervisados (XGBoost, Random Forest) para predecir trayectorias de pobreza

2. **Validación cualitativa**: Contrastar los perfiles identificados con expertos en política social y representantes de comunidades para validación externa

3. **Análisis temporal**: Incorporar datos de encuestas anteriores para estudiar la evolución de los perfiles de pobreza a lo largo del tiempo

4. **Diseño de intervenciones**: Desarrollar propuestas específicas de políticas públicas diferenciadas por perfil identificado

5. **Ampliación territorial**: Realizar análisis desagregados a nivel municipal para identificar patrones locales más específicos

---

## Licencia

Este proyecto es de **uso académico libre**. Desarrollado como parte del trabajo de investigación aplicada en el programa de Economía de la Universidad Nacional de Colombia.

### Datos

Los microdatos utilizados provienen de la **Encuesta de Calidad de Vida (ECV) 2024** del DANE y están sujetos a las políticas de uso establecidas por esta entidad.

### Citación sugerida
```
Meza García, J. S. (2024). Clustering para Identificación de Perfiles de Pobreza 
Multidimensional en Colombia. GitHub. https://github.com/jmeza-data/Repositorio_Clustering
```

---

## Autor

**Jhoan Sebastián Meza García**  
Estudiante de Economía  
Universidad Nacional de Colombia

**Áreas de investigación:**
- Pobreza multidimensional y desigualdad
- Machine Learning aplicado a ciencias sociales
- Análisis de políticas públicas
- Econometría y estadística aplicada

**Contacto:**  
📧 GitHub: [jmeza-data](https://github.com/jmeza-data)

---

<p align="center">
  <i>Machine Learning al servicio de la comprensión y el combate a la pobreza<br>
  Universidad Nacional de Colombia · 2024</i>
</p>

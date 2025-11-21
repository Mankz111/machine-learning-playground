import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import PCA


st.set_page_config(page_title="Machine Learning", layout="wide")


@st.cache_data
def carregar_dados_portugal():
    try:
        df = pd.read_csv('portugal_real_estate.csv', low_memory=False)
        cols = ['TotalArea', 'NumberOfBathrooms', 'Price', 'District', 'Type']
        
        # 1. Converter para números
        for col in cols:
            if col in df.columns and col not in ['District', 'Type']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'Type' in df.columns:
            df = df[df['Type'] == 'Apartment']
            
        
        if 'NumberOfBathrooms' in df.columns:
            df['NumberOfBathrooms'] = df['NumberOfBathrooms'].fillna(1)     
        cols_check = [c for c in ['TotalArea', 'Price', 'District'] if c in df.columns]
        df = df.dropna(subset=cols_check)
        df = df[(df['Price'] < 1500000) & (df['TotalArea'] < 400)]
        return df
    except FileNotFoundError:
        return None


df_pt = carregar_dados_portugal()
st.sidebar.title("Navegação")
menu = st.sidebar.radio(
    "Módulos:",
    [
        "1. Regressão (Imobiliário PT)", 
        "2. Árvores (Iris Dataset)", 
        "3. K-Means (Documentos)", 
        "4. Comparação Final (O Vencedor)"
    ]
)
st.sidebar.info("Trabalho de Machine Learning.")

# ==============================================================================
# 1. REGRESSÃO LINEAR
# ==============================================================================
if menu == "1. Regressão (Imobiliário PT)":
    st.title("🏠 Previsão Imobiliária (Portugal)")
    
    if df_pt is not None and not df_pt.empty:
        distritos = df_pt['District'].value_counts()
        lista = distritos[distritos > 10].index.sort_values()
        c1, c2 = st.columns([1, 2])
        distrito = c1.selectbox("Distrito:", lista)
        area = c1.slider("Área (m²)", 30, 300, 90)
        wc = c1.slider("WC", 1, 5, 1)
        
        dados = df_pt[df_pt['District'] == distrito].copy()
        
        if len(dados) > 5: 
            # Treino com nomes de colunas
            modelo = LinearRegression().fit(dados[['TotalArea', 'NumberOfBathrooms']], dados['Price'])
            input_pred = pd.DataFrame([[area, wc]], columns=['TotalArea', 'NumberOfBathrooms'])
            pred = modelo.predict(input_pred)[0] 
            c2.subheader(f"Preço em {distrito}")
            c2.metric("Valor Estimado", f"€ {pred:,.0f}")
            fig, ax = plt.subplots(figsize=(6, 3)) 
            sns.scatterplot(data=dados, x='TotalArea', y='Price', alpha=0.3, ax=ax)
            ax.scatter([area], [pred], color='red', s=200, marker='X', label='Simulação')
            ax.legend()
            c2.pyplot(fig)
        else:
            st.warning(f"Ainda assim, há poucos dados para {distrito} ({len(dados)} casas).")
    else:
        st.error("Ficheiro CSV não encontrado.")

# ==============================================================================
# 2. ÁRVORES (IRIS)
# ==============================================================================
elif menu == "2. Árvores (Iris Dataset)":
    st.title("🌸 Classificação de Flores (Iris)")
    iris = load_iris()
    X, y = iris.data, iris.target
    modelo = DecisionTreeClassifier(max_depth=3).fit(X, y)
    
    c1, c2 = st.columns(2)
    s_c = c1.slider("Sépala Comp.", 4.0, 8.0, 5.0)
    s_l = c1.slider("Sépala Larg.", 2.0, 4.5, 3.5)
    p_c = c1.slider("Pétala Comp.", 1.0, 7.0, 1.4)
    p_l = c1.slider("Pétala Larg.", 0.1, 2.5, 0.2)
    
    pred = modelo.predict([[s_c, s_l, p_c, p_l]])[0]
    c1.success(f"Espécie: **{iris.target_names[pred].upper()}**")
    
    fig, ax = plt.subplots()
    ax.scatter(X[:, 2], X[:, 3], c=y, cmap='viridis')
    ax.scatter([p_c], [p_l], c='red', s=150, marker='X')
    c2.pyplot(fig)

# ==============================================================================
# 3. K-MEANS 
# ==============================================================================
elif menu == "3. K-Means (Documentos)":
    st.title("📚 Agrupamento de Notícias (Clustering)")
    st.write("Usando o dataset '20 Newsgroups' para agrupar textos por tema automaticamente.")

    # 1. Carregar o Dataset (Usamos cache para não baixar sempre que mexe no slider)
    @st.cache_resource
    def carregar_dados():
        # Vamos carregar apenas 4 categorias para ser mais rápido e visual
        categorias = ['sci.med', 'sci.space', 'comp.graphics', 'rec.sport.baseball']
        dataset = fetch_20newsgroups(subset='train', categories=categorias, 
                                     remove=('headers', 'footers', 'quotes'))
        return dataset

    dataset = carregar_dados()
    
    # 2. Transformar Texto em Números (TF-IDF)
    # max_features=1000 limita o vocabulário às 1000 palavras mais importantes (para ser leve)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X_texto = vectorizer.fit_transform(dataset.data)

    # Configuração do K-Means
    c1, c2 = st.columns([1, 2])
    k = c1.slider("Número de Grupos (K)", 2, 6, 4)
    
    # 3. Aplicar K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_texto)

    # 4. Mostrar as palavras principais e ADIVINHAR o tema
    c1.subheader("Análise dos Grupos")
    termos = vectorizer.get_feature_names_out()
    centroides = kmeans.cluster_centers_
    
    
    def adivinhar_tema(palavras_encontradas):
        texto = " ".join(palavras_encontradas)
        if any(x in texto for x in ['space', 'nasa', 'orbit', 'moon']):
            return "🪐 Astronomia"
        elif any(x in texto for x in ['game', 'team', 'baseball', 'play']):
            return "⚾ Desporto"
        elif any(x in texto for x in ['graphics', 'image', 'file', 'format']):
            return "💻 Computadores"
        elif any(x in texto for x in ['med', 'doctor', 'health', 'disease']):
            return "🏥 Medicina"
        else:
            return "❓ Misto / Indefinido"

    for i in range(k):
        indices_top = centroides[i].argsort()[-10:][::-1]
        palavras = [termos[ind] for ind in indices_top]
        nome_tema = adivinhar_tema(palavras)
        with c1.expander(f"Grupo {i}: {nome_tema}"):
            st.write(f"**Palavras-chave:** {', '.join(palavras[:5])}")
            st.info("O algoritmo agrupou estes textos baseando-se nestas palavras.")


    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_texto.toarray())
    
    fig, ax = plt.subplots(figsize=(6, 4))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=clusters, cmap='viridis', s=10, alpha=0.5)
    

    ax.set_title("Mapa de Similaridade dos Textos")
    ax.set_xlabel("Componente PCA 1")
    ax.set_ylabel("Componente PCA 2")
    

    legend1 = ax.legend(*scatter.legend_elements(), title="Grupos")
    ax.add_artist(legend1)
    
    c2.pyplot(fig)

# ==============================================================================
# 4. BATALHA FINAL (O VENCEDOR)
# ==============================================================================
elif menu == "4. Comparação Final (O Vencedor)":
    st.title("⚔️ A Batalha dos Algoritmos")
    st.markdown("Quem tem a melhor performance em dados difíceis (curvas)?")

    # 1. GERAR DADOS
    np.random.seed(42)
    X = np.sort(5 * np.random.rand(100, 1), axis=0)
    y = np.sin(X).ravel()
    y[::5] += 3 * (0.5 - np.random.rand(20))
    
    # 2. CONFIGURAÇÃO
    c_conf, c_viz = st.columns([1, 3])
    with c_conf:
        st.subheader("Controlos")
        depth = st.slider("Profundidade Árvore", 1, 15, 5)
        k_clusters = st.slider("K-Means (Grupos)", 2, 10, 4)

    # 3. TREINAR
    lin_reg = LinearRegression().fit(X, y)
    tree_reg = DecisionTreeRegressor(max_depth=depth).fit(X, y)
    dados_combinados = np.column_stack((X, y))
    kmeans = KMeans(n_clusters=k_clusters, random_state=42).fit(dados_combinados)

    # 4. MÉTRICAS DE ERRO
    mse_lin = mean_squared_error(y, lin_reg.predict(X))
    mse_tree = mean_squared_error(y, tree_reg.predict(X))

    # 5. GRÁFICO
    with c_viz:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(X, y, c=kmeans.labels_, cmap='viridis', s=50, alpha=0.4, label='Dados')
        ax.plot(X, lin_reg.predict(X), color='blue', linewidth=2, linestyle='--', label='Regressão (Reta)')
        X_grid = np.arange(0, 5, 0.01)[:, np.newaxis]
        ax.plot(X_grid, tree_reg.predict(X_grid), color='green', linewidth=3, label='Árvore (Curva)')
        ax.legend()
        st.pyplot(fig)

    # 6. DECLARAÇÃO DO VENCEDOR
    st.divider()
    st.header("🏆 O Veredicto Oficial")

    col_v1, col_v2 = st.columns([1, 3])
    
    with col_v1:
        st.metric("Erro Regressão", f"{mse_lin:.3f}")
        st.metric("Erro Árvore", f"{mse_tree:.3f}")

    with col_v2:
        if mse_tree < mse_lin:
            st.success(f"### VENCEDOR: Árvore de Decisão! 🌲")
            st.write(f"A Árvore adaptou-se melhor à curva dos dados.")
        else:
            st.info("### VENCEDOR: Regressão Linear! 📉")
            st.write("A linha reta foi suficiente.")
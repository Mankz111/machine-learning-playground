import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris, fetch_20newsgroups
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

st.set_page_config(page_title="Machine Learning Portfolio", layout="wide")

@st.cache_data
def carregar_dados_portugal():
    try:
        df = pd.read_csv('portugal_real_estate.csv', low_memory=False)
        cols = ['TotalArea', 'NumberOfBathrooms', 'Price', 'District', 'Type']
        
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
        "4. Comparação Final (O Vencedor)",
        "5. Conclusões e Análise Crítica",
    ]
)

if menu == "1. Regressão (Imobiliário PT)":
    st.title("Previsão Imobiliária (Portugal)")
    
    if df_pt is not None and not df_pt.empty:
        distritos = df_pt['District'].value_counts()
        lista = distritos[distritos > 10].index.sort_values()
        
        c1, c2 = st.columns([1, 2])
        distrito = c1.selectbox("Distrito:", lista)
        area = c1.slider("Área (m²)", 30, 300, 90)
        wc = c1.slider("WC", 1, 5, 1)
        
        dados = df_pt[df_pt['District'] == distrito].copy()
        
        if len(dados) > 5: 
            modelo = LinearRegression().fit(dados[['TotalArea', 'NumberOfBathrooms']], dados['Price'])
            input_pred = pd.DataFrame([[area, wc]], columns=['TotalArea', 'NumberOfBathrooms'])
            pred = modelo.predict(input_pred)[0] 
            
            c2.subheader(f"Preço estimado em {distrito}")
            c2.metric("Valor de Mercado", f"€ {pred:,.0f}")
            
            fig, ax = plt.subplots(figsize=(6, 3)) 
            sns.scatterplot(data=dados, x='TotalArea', y='Price', alpha=0.3, ax=ax)
            ax.scatter([area], [pred], color='red', s=200, marker='X', label='Simulação')
            ax.legend()
            c2.pyplot(fig)
        else:
            st.warning(f"Dados insuficientes para {distrito}.")
    else:
        st.error("Ficheiro CSV não encontrado.")

elif menu == "2. Árvores (Iris Dataset)":
    st.title("Classificação de Flores (Iris)")
    iris = load_iris()
    X, y = iris.data, iris.target
    
    profundidade = st.sidebar.slider("Complexidade da Árvore (Profundidade)", 1, 5, 3)
    modelo = DecisionTreeClassifier(max_depth=profundidade, random_state=42).fit(X, y)
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Simulador")
        s_c = st.slider("Sépala Comp.", 4.0, 8.0, 5.0)
        s_l = st.slider("Sépala Larg.", 2.0, 4.5, 3.5)
        p_c = st.slider("Pétala Comp.", 1.0, 7.0, 1.4)
        p_l = st.slider("Pétala Larg.", 0.1, 2.5, 0.2)
        
        pred = modelo.predict([[s_c, s_l, p_c, p_l]])[0]
        st.success(f"Classificação: **{iris.target_names[pred].upper()}**")

    with c2:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(X[:, 2], X[:, 3], c=y, cmap='viridis', alpha=0.6)
        ax.scatter([p_c], [p_l], c='red', s=150, marker='X', label='Amostra')
        ax.set_xlabel("Pétala Comprimento")
        ax.set_ylabel("Pétala Largura")
        ax.legend()
        st.pyplot(fig)

    st.divider()
    st.subheader("Estrutura de Decisão do Algoritmo")
    fig_tree, ax_tree = plt.subplots(figsize=(12, 6))
    plot_tree(modelo, filled=True, feature_names=iris.feature_names, 
              class_names=iris.target_names, rounded=True, ax=ax_tree)
    st.pyplot(fig_tree)

elif menu == "3. K-Means (Documentos)":
    st.title("Agrupamento de Notícias (Clustering)")
    
    @st.cache_resource
    def carregar_dados_news():
        categorias = ['sci.med', 'sci.space', 'comp.graphics', 'rec.sport.baseball']
        dataset = fetch_20newsgroups(subset='train', categories=categorias, 
                                     remove=('headers', 'footers', 'quotes'))
        return dataset

    dataset = carregar_dados_news()
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X_texto = vectorizer.fit_transform(dataset.data)
    
    c1, c2 = st.columns([1, 2])
    k = c1.slider("Número de Clusters (K)", 2, 6, 4)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_texto)

    c1.subheader("Análise Semântica")
    termos = vectorizer.get_feature_names_out()
    centroides = kmeans.cluster_centers_
    
    def identificar_topico(palavras):
        texto = " ".join(palavras)
        if any(x in texto for x in ['space', 'nasa', 'orbit']): return "Astronomia"
        if any(x in texto for x in ['game', 'team', 'baseball']): return "Desporto"
        if any(x in texto for x in ['graphics', 'image', 'file']): return "Tecnologia"
        if any(x in texto for x in ['med', 'doctor', 'health']): return "Medicina"
        return "Tópico Geral"

    for i in range(k):
        indices_top = centroides[i].argsort()[-10:][::-1]
        palavras = [termos[ind] for ind in indices_top]
        topico = identificar_topico(palavras)
        with c1.expander(f"Grupo {i+1}: {topico}"):
            st.write(f"**Keywords:** {', '.join(palavras[:5])}")

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_texto.toarray())
    
    fig, ax = plt.subplots(figsize=(6, 4))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=clusters, cmap='viridis', s=10, alpha=0.5)
    ax.set_title("Projeção Vetorial dos Documentos")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    c2.pyplot(fig)

elif menu == "4. Comparação Final (O Vencedor)":
    st.title("Comparação de Performance")
    st.markdown("### Análise de Ajuste em Dados Não-Lineares")

    np.random.seed(42)
    X = np.sort(5 * np.random.rand(100, 1), axis=0)
    y = np.sin(X).ravel()
    y[::5] += 3 * (0.5 - np.random.rand(20))
    
    c_conf, c_text = st.columns([1, 2])
    with c_conf:
        st.markdown("**Configuração de Hiperparâmetros**")
        depth = st.slider("Profundidade (Árvore)", 1, 15, 5)
        k_clusters = st.slider("Centróides (K-Means)", 2, 20, 8)

    lin_reg = LinearRegression().fit(X, y)
    y_pred_lin = lin_reg.predict(X)

    tree_reg = DecisionTreeRegressor(max_depth=depth).fit(X, y)
    y_pred_tree = tree_reg.predict(X)
    
    dados_combinados = np.column_stack((X, y))
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10).fit(dados_combinados)
    y_pred_kmeans = np.zeros_like(y)
    
    for i in range(k_clusters):
        mask = (kmeans.labels_ == i)
        if np.any(mask):
            y_pred_kmeans[mask] = kmeans.cluster_centers_[i, 1]

    score_lin = r2_score(y, y_pred_lin)
    score_tree = r2_score(y, y_pred_tree)
    score_kmeans = r2_score(y, y_pred_kmeans)

    st.subheader("1. Visualização do Ajuste")
    fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.scatter(X, y, c='gray', s=30, alpha=0.5, label='Dados Observados')
    ax.plot(X, y_pred_lin, color='blue', linewidth=2, linestyle='--', label=f'Linear (R²: {score_lin:.2f})')
    
    X_grid = np.arange(0, 5, 0.01)[:, np.newaxis]
    y_grid_tree = tree_reg.predict(X_grid)
    ax.plot(X_grid, y_grid_tree, color='green', linewidth=2, label=f'Árvore (R²: {score_tree:.2f})')
    ax.scatter(X, y_pred_kmeans, color='red', marker='x', s=50, label=f'K-Means (R²: {score_kmeans:.2f})')

    ax.legend()
    st.pyplot(fig)

    st.subheader("2. Ranking de Precisão (R² Score)")
    
    col_graf, col_exp = st.columns([2, 1])
    
    with col_graf:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        algoritmos = ['Regressão Linear', 'Árvore de Decisão', 'K-Means']
        scores = [score_lin, score_tree, score_kmeans]
        cores = ['blue', 'green', 'red']
        
        barras = ax2.bar(algoritmos, scores, color=cores, alpha=0.8)
        
        ax2.set_ylabel("R² Score (Precisão)")
        ax2.set_ylim(0, 1.1)
        
        for barra in barras:
            height = barra.get_height()
            if height < 0: height = 0
            ax2.text(barra.get_x() + barra.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
            
        st.pyplot(fig2)

    with col_exp:
        st.markdown("**Interpretação da Métrica**")
        st.write("O **R² Score** varia de 0 a 1.")
        st.write("- **Próximo de 1:** O modelo capturou a tendência.")
        st.write("- **Próximo de 0:** O modelo falhou na generalização.")
        
        if score_tree > score_lin:
            st.success("Vencedor: Árvore de Decisão")
        else:
            st.info("Vencedor: Regressão Linear")

elif menu == "5. Conclusões e Análise Crítica":
    st.title("Relatório de Conclusões e Análise Técnica")
    
    st.markdown("""
    Esta secção apresenta a síntese técnica comparativa entre as abordagens de **Aprendizagem Supervisionada** e **Não Supervisionada** exploradas neste estudo.
    """)
    
    st.divider()

    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("#### Regressão Linear")
        st.markdown("**Definição**")
        st.write("Modelo paramétrico que estima relações entre variáveis através de uma função linear de primeiro grau.")
        
        st.markdown("**Análise de Desempenho**")
        st.write("- **Eficiência:** Alta capacidade de interpretação em tendências macroscópicas.")
        st.write("- **Limitação:** Apresenta elevado viés (underfitting) quando confrontado com dados de morfologia não-linear ou complexa.")

    with c2:
        st.markdown("#### Árvore de Decisão")
        st.markdown("**Definição**")
        st.write("Modelo não-paramétrico que segmenta o espaço de dados através de regras condicionais hierárquicas.")
        
        st.markdown("**Análise de Desempenho**")
        st.write("- **Flexibilidade:** Excelente adaptação a fronteiras de decisão irregulares e curvas.")
        st.write("- **Risco:** Tendência a alta variância (overfitting) em profundidades elevadas, comprometendo a generalização.")

    with c3:
        st.markdown("#### Clustering (K-Means)")
        st.markdown("**Definição**")
        st.write("Algoritmo não supervisionado de quantização vetorial que particiona dados em grupos baseados em centróides.")
        
        st.markdown("**Análise de Aplicação**")
        st.write("- **Utilidade:** Fundamental para prospeção de dados não rotulados e redução de dimensionalidade.")
        st.write("- **Restrição:** Dependência crítica da inicialização e da definição prévia do hiperparâmetro K.")

    st.markdown("---")
    
    st.subheader("Veredicto Técnico")
    
    st.markdown("""
    A experimentação comprovou que não existe um algoritmo universalmente superior (teorema "No Free Lunch"). A seleção da arquitetura deve alinhar-se com a natureza dos dados:
    
    1.  **Linearidade:** Para relações diretas e necessidade de explicabilidade causal, a **Regressão Linear** permanece o padrão da indústria.
    2.  **Complexidade:** Para capturar nuances e padrões não-lineares, as **Árvores de Decisão** oferecem desempenho superior, exigindo controlo rigoroso de profundidade.
    3.  **Exploração:** Em cenários de "Cold Start" sem rótulos definidos, o **K-Means** é a ferramenta primária para inferência de estrutura.
    """)
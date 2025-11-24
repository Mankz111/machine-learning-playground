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
        "4. Comparação Final (O Vencedor)",
        "5. Conclusões e Análise Crítica",
    ]
)
st.sidebar.info("Trabalho de Machine Learning.")

# ==============================================================================
# 1. REGRESSÃO LINEAR
# ==============================================================================
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
    st.title("Classificação de Flores (Iris)")
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
    st.title("Agrupamento de Notícias (Clustering)")
    st.write("Usando o dataset '20 Newsgroups' para agrupar textos por tema automaticamente.")

    # 1. Carregar o Dataset (Usamos cache para não baixar sempre que mexe no slider)
    @st.cache_resource
    def carregar_dados():
        
        categorias = ['sci.med', 'sci.space', 'comp.graphics', 'rec.sport.baseball']
        dataset = fetch_20newsgroups(subset='train', categories=categorias, 
                                     remove=('headers', 'footers', 'quotes'))
        return dataset

    dataset = carregar_dados()
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X_texto = vectorizer.fit_transform(dataset.data)
    c1, c2 = st.columns([1, 2])
    k = c1.slider("Número de Grupos (K)", 2, 6, 4)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_texto)


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
# 4. COMPARAÇÃO
# ==============================================================================
elif menu == "4. Comparação Final (O Vencedor)":
    st.title("Comparação do Algoritmos")
    st.markdown("### Quem vence em dados complexos (Curvas)?")

    
    np.random.seed(42)
    X = np.sort(5 * np.random.rand(100, 1), axis=0)
    y = np.sin(X).ravel()
    y[::5] += 3 * (0.5 - np.random.rand(20)) # Adiciona ruído
    
    
    c_conf, c_text = st.columns([1, 2])
    with c_conf:
        st.caption("Parâmetros dos Modelos")
        depth = st.slider("Profundidade da Árvore", 1, 15, 5)
        k_clusters = st.slider("K-Means (K)", 2, 20, 8)


    # A. Regressão Linear
    lin_reg = LinearRegression().fit(X, y)
    y_pred_lin = lin_reg.predict(X)

    # B. Árvore de Decisão
    tree_reg = DecisionTreeRegressor(max_depth=depth).fit(X, y)
    y_pred_tree = tree_reg.predict(X)
    dados_combinados = np.column_stack((X, y))
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10).fit(dados_combinados)
    y_pred_kmeans = np.zeros_like(y)
    for i in range(k_clusters):
        mask = (kmeans.labels_ == i)
        if np.any(mask):
            y_pred_kmeans[mask] = kmeans.cluster_centers_[i, 1]


    mse_lin = mean_squared_error(y, y_pred_lin)
    mse_tree = mean_squared_error(y, y_pred_tree)
    mse_kmeans = mean_squared_error(y, y_pred_kmeans)

    st.subheader("1. Visualização do Comportamento")
    fig, ax = plt.subplots(figsize=(10, 4))
    
    
    ax.scatter(X, y, c='gray', s=30, alpha=0.5, label='Dados Reais')
    
    
    ax.plot(X, y_pred_lin, color='blue', linewidth=2, linestyle='--', label='Regressão Linear (Reta)')
    
    
    X_grid = np.arange(0, 5, 0.01)[:, np.newaxis]
    y_grid_tree = tree_reg.predict(X_grid)
    ax.plot(X_grid, y_grid_tree, color='green', linewidth=2, label='Árvore de Decisão (Degraus)')
    
    
    ax.scatter(X, y_pred_kmeans, color='red', marker='x', s=50, label='K-Means (Centróides)')

    ax.set_title("Como cada algoritmo tenta 'desenhar' a curva")
    ax.legend()
    st.pyplot(fig)

    
    st.subheader("2. Ranking de Erro (Quem falhou menos?)")
    
    col_graf, col_exp = st.columns([2, 1])
    
    with col_graf:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        algoritmos = ['Regressão Linear', 'Árvore de Decisão', 'K-Means']
        erros = [mse_lin, mse_tree, mse_kmeans]
        cores = ['blue', 'green', 'red']
        
        barras = ax2.bar(algoritmos, erros, color=cores, alpha=0.7)
        ax2.set_ylabel("Erro Médio Quadrático (MSE)")
        ax2.set_title("Comparação de Erro (Menor é Melhor)")
        
        # Adicionar o valor em cima da barra
        for barra in barras:
            height = barra.get_height()
            ax2.text(barra.get_x() + barra.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
            
        st.pyplot(fig2)

    with col_exp:
        st.info("**Interpretação**")
        if mse_tree < mse_lin:
            st.write("**A Árvore venceu!**")
            st.caption("Como os dados são curvos, a árvore consegue adaptar-se melhor que a reta rígida da regressão.")
        else:
            st.write("**A Regressão venceu!**")
        
        st.write("---")
        st.write(f"**Sobre o K-Means:**")
        st.caption(f"Com {k_clusters} grupos, o erro é {mse_kmeans:.2f}. Se aumentares o K, o erro diminui porque ele cobre melhor os pontos.")

    
    st.divider()
    st.subheader("Entender as Diferenças")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.subheader("1. Análise de Regressão")
        st.markdown("##### Linearidade vs. Complexidade")
        st.write("""
        * **Desempenho:** A Regressão Linear demonstrou eficácia na identificação de tendências macroscópicas e correlações diretas.
        * **Restrição:** O modelo evidenciou *underfitting* ao ser confrontado com distribuições de dados não-lineares, falhando na captura de nuances locais e curvaturas acentuadas.
        """)

    with c2:
        st.subheader("2. Clustering (K-Means)")
        st.markdown("##### Padrões Latentes em Dados Não Estruturados")
        st.write("""
        * **Eficácia:** O algoritmo segregou documentos temáticos com sucesso sem necessidade de rotulagem prévia, validando o uso de métricas de distância vetorial.
        * **Sensibilidade:** A performance mostrou-se dependente da definição prévia do hiperparâmetro *K* (número de clusters) e da inicialização dos centróides.
        """)
        
    st.markdown("---")
    
    st.subheader("3. Comparação de Performance em Dados Não-Lineares")
    
    col_text, col_viz = st.columns([3, 1])
    
    with col_text:
        st.write("""
        Na simulação com dados sinusoidais, a análise comparativa entre **Regressão Linear** e **Árvores de Decisão** permitiu concluir:
        
        1.  **Adaptabilidade do Modelo:** A Árvore de Decisão superou a Regressão Linear ao segmentar o espaço de decisão, ajustando-se à morfologia não-linear dos dados.
        2.  **Compromisso Viés-Variância:** Embora a Árvore apresente menor erro no conjunto de treino, profundidades excessivas levam à memorização de ruído (*overfitting*), prejudicando a capacidade de generalização do modelo.
        """)
    
    with col_viz:
        x = np.linspace(0, 10, 100)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(x, np.sin(x), 'g-', label='Modelo Flexível (Árvore)')
        ax.plot(x, x*0, 'b--', label='Modelo Rígido (Linear)')
        ax.set_title("Comparativo de Ajuste", fontsize=10)
        ax.set_yticks([])
        ax.legend(fontsize=8)
        st.pyplot(fig)

    st.markdown("""
    ### Considerações Finais
    Conclui-se que a seleção do algoritmo deve ser orientada pela natureza dos dados e pelo objetivo do negócio:
    * **Regressão Linear:** Recomendada para inferências causais simples e ambientes onde a interpretabilidade é prioritária.
    * **Árvores de Decisão:** Superiores em cenários com dados complexos e fronteiras de decisão não-lineares.
    * **K-Means:** Essencial para análise exploratória e descoberta de padrões em conjuntos de dados não rotulados.
    """)
    

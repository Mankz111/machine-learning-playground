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

# --- CONFIGURAÇÃO INICIAL ---
st.set_page_config(page_title="Trabalho Final Machine Learning", layout="wide")

# --- FUNÇÃO PARA CARREGAR DADOS REAIS (PORTUGAL) ---
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
            
        # --- CORREÇÃO AVEIRO: Se não tiver nº de WC, assumimos que é 1 ---
        if 'NumberOfBathrooms' in df.columns:
            df['NumberOfBathrooms'] = df['NumberOfBathrooms'].fillna(1)
            
        # Agora só apagamos se faltar PREÇO ou ÁREA (que são obrigatórios)
        cols_check = [c for c in ['TotalArea', 'Price', 'District'] if c in df.columns]
        df = df.dropna(subset=cols_check)
        
        # Remover outliers
        df = df[(df['Price'] < 1500000) & (df['TotalArea'] < 400)]
        return df
    except FileNotFoundError:
        return None

# --- A LINHA QUE FALTAVA ESTÁ AQUI EM BAIXO ---
# Executar a função e guardar os dados na variável 'df_pt'
df_pt = carregar_dados_portugal()

# --- BARRA LATERAL (NAVEGAÇÃO) ---
st.sidebar.title("Navegação")
menu = st.sidebar.radio(
    "Módulos:",
    [
        "1. Regressão (Imobiliário PT)", 
        "2. Árvores (Iris Dataset)", 
        "3. K-Means (Clientes)", 
        "4. ⚔️ Batalha Final (O Vencedor)"
    ]
)
st.sidebar.info("Trabalho de Machine Learning.")

# ==============================================================================
# 1. REGRESSÃO LINEAR (DADOS REAIS)
# ==============================================================================
if menu == "1. Regressão (Imobiliário PT)":
    st.title("🏠 Previsão Imobiliária (Portugal)")
    
    if df_pt is not None and not df_pt.empty:
        distritos = df_pt['District'].value_counts()
        # Mostrar distritos com mais de 10 casas (agora Aveiro vai aparecer!)
        lista = distritos[distritos > 10].index.sort_values()
        
        c1, c2 = st.columns([1, 2])
        distrito = c1.selectbox("Distrito:", lista)
        area = c1.slider("Área (m²)", 30, 300, 90)
        wc = c1.slider("WC", 1, 5, 1)
        
        dados = df_pt[df_pt['District'] == distrito].copy()
        
        if len(dados) > 5: # Baixei o limite mínimo para garantir que funciona
            modelo = LinearRegression().fit(dados[['TotalArea', 'NumberOfBathrooms']], dados['Price'])
            pred = modelo.predict([[area, wc]])[0]
            
            c2.subheader(f"Preço em {distrito}")
            c2.metric("Valor Estimado", f"€ {pred:,.0f}")
            
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.scatterplot(data=dados, x='TotalArea', y='Price', alpha=0.3, ax=ax)
            ax.scatter([area], [pred], color='red', s=200, marker='X', label='Simulação')
            ax.legend()
            st.pyplot(fig)
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
# 3. K-MEANS (CLIENTES)
# ==============================================================================
elif menu == "3. K-Means (Clientes)":
    st.title("🛍️ Segmentação de Clientes")
    X = np.array([[200, 5], [150, 3], [400, 10], [100, 2], [350, 8], [450, 9], [120, 4]])
    
    c1, c2 = st.columns(2)
    k = c1.slider("K (Grupos)", 2, 4, 2)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    
    c1.dataframe(pd.DataFrame(kmeans.cluster_centers_, columns=['Gastos', 'Compras']))
    
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', s=100)
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, marker='X')
    c2.pyplot(fig)

# ==============================================================================
# 4. BATALHA FINAL (O VENCEDOR)
# ==============================================================================
elif menu == "4. ⚔️ Batalha Final (O Vencedor)":
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
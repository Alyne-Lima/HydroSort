import streamlit as st
import pandas as pd
from io import BytesIO
from modelo_cpp_roc_sort import executar_modelo

st.set_page_config(
    page_title="HYDROSORT",
    page_icon="📊",
    layout="wide"
)

st.title("🌊 HYDROSORT")
st.caption("Hydrological Multicriteria Decision Support System under Uncertainty.")
st.sidebar.header("Configurações")

epsilon = st.sidebar.slider(
    "Limiar de indiferença ε",
    min_value=0.00,
    max_value=0.30,
    value=0.10,
    step=0.01
)

categorias = []

st.sidebar.markdown("---")
st.sidebar.info("Carregue o arquivo de entrada do HYDROSORT ou utilize a planilha de demonstração."
)

abas = st.tabs([
    "1. Entrada de Dados",
    "2. Pesos ROC",
    "3. Resultados",
    "4. Detalhamento",
    "5. Exportação"
])

with abas[0]:

    st.subheader("Arquivo de entrada")

    arquivo_excel = st.file_uploader(
        "📂 Selecione o arquivo HYDROSORT_Input_Completo.xlsx",
        type=["xlsx"]
    )

    st.info(
        "Carregue a planilha única do HYDROSORT."
    )


def read_hydrosort(uploaded):

    if uploaded is None:
        return None

    excel = pd.ExcelFile(uploaded)

    dados = {}

    dados["configuracoes"] = pd.read_excel(
        excel,
        "Configuracoes"
    )

    dados["alternativas"] = pd.read_excel(
        excel,
        "Alternativas"
    )

    dados["criterios"] = pd.read_excel(
        excel,
        "Criterios"
    )

    dados["categorias"] = pd.read_excel(
        excel,
        "Categorias"
    )

    dados["decisores"] = pd.read_excel(
        excel,
        "Decisores"
    )

    dados["avaliacoes"] = pd.read_excel(
        excel,
        "Avaliacoes"
    )

    dados["rankings"] = pd.read_excel(
        excel,
        "Rankings_ROC"
    )

    dados["perfis"] = pd.read_excel(
        excel,
        "Perfis"
    )

    dados["variancias"] = pd.read_excel(
        excel,
        "Variancias"
    )

    dados["pesos"] = pd.read_excel(
        excel,
        "Pesos_Decisores"
    )

    return dados    
   

avaliacoes_df = None
rankings_df = None
perfis_df = None
variancias_df = None
pesos_decisores_df = None

dados = read_hydrosort(arquivo_excel)

if dados is not None:

    avaliacoes_df = dados["avaliacoes"]
    rankings_df = dados["rankings"]
    perfis_df = dados["perfis"]
    variancias_df = dados["variancias"]
    pesos_decisores_df = dados["pesos"]

    categorias = dados["categorias"]["Categoria"].tolist()

if all(x is not None for x in [avaliacoes_df, rankings_df, perfis_df, variancias_df]):
    try:
        saidas = executar_modelo(
            avaliacoes_df=avaliacoes_df,
            rankings_df=rankings_df,
            perfis_df=perfis_df,
            variancias_df=variancias_df,
            pesos_decisores_df=pesos_decisores_df,
            categorias=categorias,
            epsilon=epsilon
        )

        with abas[1]:
            st.subheader("Pesos ROC por decisor")
            st.dataframe(saidas["pesos_roc"], use_container_width=True)

            st.markdown("Verificação: soma dos pesos por decisor")
            pesos_check = saidas["pesos_roc"].set_index("Decisor").drop(columns=[], errors="ignore")
            st.dataframe(pesos_check.assign(Soma=pesos_check.sum(axis=1)).reset_index(), use_container_width=True)

        with abas[2]:
            st.subheader("Classificação final")
            st.dataframe(saidas["classificacao"], use_container_width=True)

            st.subheader("Índice probabilístico grupal δ")
            delta_pivot = saidas["agregacao_grupo"].pivot(
                index="Alternativa",
                columns="Perfil",
                values="Delta_grupo"
            )
            st.dataframe(delta_pivot, use_container_width=True)

            st.bar_chart(delta_pivot)

        with abas[3]:
            st.subheader("Avaliações normalizadas")
            st.dataframe(saidas["avaliacoes_normalizadas"], use_container_width=True)

            st.subheader("Probabilidades de superação")
            st.dataframe(saidas["probabilidades"], use_container_width=True)

            st.subheader("Agregação individual")
            st.dataframe(saidas["agregacao_individual"], use_container_width=True)

            st.subheader("Agregação em grupo")
            st.dataframe(saidas["agregacao_grupo"], use_container_width=True)

        with abas[4]:
            st.subheader("Exportar resultados")

            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                for nome, df in saidas.items():
                    if isinstance(df, pd.DataFrame):
                        df.to_excel(writer, sheet_name=nome[:31], index=False)

            st.download_button(
                label="📥 Baixar resultados em Excel",
                data=output.getvalue(),
                file_name="HYDROSORT_Results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error("Ocorreu um erro ao executar o modelo.")
        st.exception(e)

else:
    with abas[2]:
        st.warning("Carregue o arquivo HYDROSORT_Input_Completo.xlsx.")

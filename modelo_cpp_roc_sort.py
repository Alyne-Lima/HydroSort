import numpy as np
import pandas as pd
from scipy.stats import norm


def calcular_pesos_roc(rankings_df):
    """
    rankings_df columns: Decisor, Criterio, Ranking
    Returns DataFrame with rows Decisor and columns Criterio.
    """
    criterios = sorted(
    rankings_df["Criterio"].unique(),
    key=lambda x: int(x.replace("C", ""))
)
    m = len(criterios)

    pesos = []
    for decisor, grupo in rankings_df.groupby("Decisor"):
        linha = {"Decisor": decisor}
        mapa = dict(zip(grupo["Criterio"], grupo["Ranking"]))
        for criterio in criterios:
            r = int(mapa[criterio])
            linha[criterio] = sum(1 / j for j in range(r, m + 1)) / m
        pesos.append(linha)

    return pd.DataFrame(pesos).set_index("Decisor")


def normalizar_avaliacoes(avaliacoes_df):
    """
    avaliacoes_df columns: Decisor, Alternativa, Criterio, Valor
    Valor must be 1 to 5.
    """
    df = avaliacoes_df.copy()
    df["Valor_Normalizado"] = (df["Valor"] - 1) / 4
    return df


def preparar_perfis(perfis_df):
    """
    perfis_df columns: Perfil, Criterio, Valor
    """
    return perfis_df.pivot(index="Perfil", columns="Criterio", values="Valor")


def preparar_variancias(variancias_df):
    """
    variancias_df columns: Criterio, Variancia
    """
    return dict(zip(variancias_df["Criterio"], variancias_df["Variancia"]))


def preparar_pesos_decisores(pesos_decisores_df, decisores):
    """
    pesos_decisores_df columns: Decisor, Peso
    If None or empty, returns equal weights.
    """
    if pesos_decisores_df is None or pesos_decisores_df.empty:
        return {d: 1 / len(decisores) for d in decisores}

    pesos = dict(zip(pesos_decisores_df["Decisor"], pesos_decisores_df["Peso"]))
    soma = sum(pesos.values())
    if soma == 0:
        return {d: 1 / len(decisores) for d in decisores}

    return {d: pesos.get(d, 0) / soma for d in decisores}


def calcular_probabilidades(avaliacoes_norm_df, perfis_df, variancias):
    """
    Returns long DataFrame:
    Decisor, Alternativa, Perfil, Criterio, P_superacao
    """
    linhas = []

    for _, row in avaliacoes_norm_df.iterrows():
        decisor = row["Decisor"]
        alternativa = row["Alternativa"]
        criterio = row["Criterio"]
        x = row["Valor_Normalizado"]

        for perfil in perfis_df.index:
            b = perfis_df.loc[perfil, criterio]
            V = variancias[criterio]
            z = (x - b) / np.sqrt(2 * V)
            p_superacao = norm.cdf(z)

            linhas.append({
                "Decisor": decisor,
                "Alternativa": alternativa,
                "Perfil": perfil,
                "Criterio": criterio,
                "P_superacao": p_superacao
            })

    return pd.DataFrame(linhas)


def agregar_por_decisor(prob_df, pesos_roc_df):
    """
    Aggregates P_superacao by ROC weights.
    Returns DataFrame: Decisor, Alternativa, Perfil, A_mais, Delta_individual
    """
    linhas = []

    for (decisor, alternativa, perfil), grupo in prob_df.groupby(["Decisor", "Alternativa", "Perfil"]):
        soma = 0
        for _, row in grupo.iterrows():
            criterio = row["Criterio"]
            peso = pesos_roc_df.loc[decisor, criterio]
            soma += peso * row["P_superacao"]

        linhas.append({
            "Decisor": decisor,
            "Alternativa": alternativa,
            "Perfil": perfil,
            "A_mais_individual": soma,
            "Delta_individual": 2 * soma - 1
        })

    return pd.DataFrame(linhas)


def agregar_grupo(agreg_ind_df, pesos_decisores):
    """
    Aggregates individual A+ using decision-maker weights.
    Returns DataFrame: Alternativa, Perfil, A_mais_grupo, Delta_grupo
    """
    linhas = []

    for (alternativa, perfil), grupo in agreg_ind_df.groupby(["Alternativa", "Perfil"]):
        soma = 0
        for _, row in grupo.iterrows():
            decisor = row["Decisor"]
            soma += pesos_decisores[decisor] * row["A_mais_individual"]

        linhas.append({
            "Alternativa": alternativa,
            "Perfil": perfil,
            "A_mais_grupo": soma,
            "Delta_grupo": 2 * soma - 1
        })

    return pd.DataFrame(linhas)


def classificar_pontual(delta_grupo_df, categorias):
    """
    categorias: list of category names. Number of profiles must be len(categorias)-1.
    """
    classificacoes = []

    for alternativa, grupo in delta_grupo_df.groupby("Alternativa"):
        grupo = grupo.sort_values("Perfil")
        perfis = list(grupo["Perfil"])
        deltas = dict(zip(grupo["Perfil"], grupo["Delta_grupo"]))

        if deltas[perfis[0]] < 0:
            classe = categorias[0]
        elif deltas[perfis[-1]] >= 0:
            classe = categorias[-1]
        else:
            classe = categorias[0]
            for i in range(len(perfis) - 1):
                if deltas[perfis[i]] >= 0 and deltas[perfis[i + 1]] < 0:
                    classe = categorias[i + 1]
                    break

        classificacoes.append({
            "Alternativa": alternativa,
            "Classificacao_Pontual": classe
        })

    return pd.DataFrame(classificacoes)


def classificar_final(delta_grupo_df, categorias, epsilon):

    pontual_df = classificar_pontual(delta_grupo_df, categorias)
    pontual = dict(
        zip(
            pontual_df["Alternativa"],
            pontual_df["Classificacao_Pontual"]
        )
    )

    linhas = []

    for alternativa, grupo in delta_grupo_df.groupby("Alternativa"):

        grupo = grupo.sort_values("Perfil")
        intervalos = []

        for i, (_, row) in enumerate(grupo.iterrows()):

            if i >= len(categorias) - 1:
                continue

            delta = row["Delta_grupo"]

            if abs(delta) <= epsilon:
                intervalos.append(
                    f"[{categorias[i]}, {categorias[i+1]}]"
                )

        if intervalos:
            intervalar = " / ".join(intervalos)
            final = intervalar
        else:
            intervalar = "-"
            final = pontual[alternativa]

        linhas.append({
            "Alternativa": alternativa,
            "Classificacao_Pontual": pontual[alternativa],
            "Classificacao_Intervalar": intervalar,
            "Classificacao_Final": final
        })

    return pd.DataFrame(linhas)

def executar_modelo(avaliacoes_df, rankings_df, perfis_df, variancias_df, pesos_decisores_df=None, categorias=None, epsilon=0.10):
    if categorias is None:
        n_perfis = perfis_df["Perfil"].nunique()
        categorias = [f"C{i}" for i in range(1, n_perfis + 2)]

    pesos_roc = calcular_pesos_roc(rankings_df)
    aval_norm = normalizar_avaliacoes(avaliacoes_df)
    perfis_pivot = preparar_perfis(perfis_df)
    variancias = preparar_variancias(variancias_df)
    decisores = sorted(avaliacoes_df["Decisor"].unique())
    pesos_dec = preparar_pesos_decisores(pesos_decisores_df, decisores)

    prob = calcular_probabilidades(aval_norm, perfis_pivot, variancias)
    agreg_ind = agregar_por_decisor(prob, pesos_roc)
    agreg_grupo = agregar_grupo(agreg_ind, pesos_dec)
    classificacao = classificar_final(agreg_grupo, categorias, epsilon)

    resultados = agreg_grupo.merge(classificacao, on="Alternativa", how="left")

    return {
        "pesos_roc": pesos_roc.reset_index(),
        "avaliacoes_normalizadas": aval_norm,
        "probabilidades": prob,
        "agregacao_individual": agreg_ind,
        "agregacao_grupo": agreg_grupo,
        "classificacao": classificacao,
        "resultados": resultados
    }

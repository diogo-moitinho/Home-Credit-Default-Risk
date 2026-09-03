"""
Home Credit Default Risk — carga, integração e feature engineering.

Fluxo principal:
    load_and_merge_home_credit_data()
        -> creating_table() para train e test
            -> agregações por SK_ID_CURR de cada tabela auxiliar
            -> feature engineering da tabela principal
            -> persistência em BRONZE_DIR (parquet)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW_DIR, BRONZE_DIR

ID_COLS = {'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV'}

DEFAULT_AGGS = ('mean', 'sum', 'max', 'count')


def _numeric_cols_without_ids(df: pd.DataFrame) -> list[str]:
    """Colunas numéricas do DataFrame, excluindo as colunas de ID."""
    return [c for c in df.select_dtypes(include='number').columns if c not in ID_COLS]


def aggregate_frame(tbl: pd.DataFrame,
                    prefix: str,
                    agg_funcs: tuple[str, ...] = DEFAULT_AGGS,
                    id_col: str = 'SK_ID_CURR') -> pd.DataFrame:
    """
    Agrega todas as colunas numéricas de `tbl` por `id_col` e achata o MultiIndex,
    prefixando os nomes para evitar colisão entre tabelas.
    """
    if id_col not in tbl.columns:
        raise KeyError(f"'{id_col}' ausente na tabela com prefixo '{prefix}'")

    num_cols = _numeric_cols_without_ids(tbl) + [id_col]
    agg = tbl[num_cols].groupby(id_col).agg(list(agg_funcs)).reset_index()

    agg.columns = [
        id_col if col[0] == id_col else f'{prefix}_' + '_'.join(col).strip('_')
        for col in agg.columns.values
    ]
    return agg


def aggregate_table(path: Path,
                    filename: str,
                    prefix: str,
                    agg_funcs: tuple[str, ...] = DEFAULT_AGGS) -> pd.DataFrame | None:
    """Lê um CSV auxiliar e devolve a agregação por SK_ID_CURR. None se o arquivo não existe."""
    fp = path / filename
    if not fp.exists():
        print(f"  [skip] {filename} não encontrado em {path}")
        return None

    print(f"Processando {filename}...")
    return aggregate_frame(pd.read_csv(fp), prefix, agg_funcs)


# ---------------------------------------------------------------------------
# Bureau (+ bureau_balance)
# ---------------------------------------------------------------------------

def agg_bureau_balance(bureau_bal: pd.DataFrame) -> pd.DataFrame:
    """Resume o histórico mensal de cada contrato (SK_ID_BUREAU) a partir do STATUS."""
    bureau_bal = bureau_bal.copy()

    bureau_bal['is_closed'] = (bureau_bal['STATUS'] == 'C').astype(int)
    bureau_bal['is_dpd_zero'] = (bureau_bal['STATUS'] == '0').astype(int)
    bureau_bal['is_dpd_bad'] = bureau_bal['STATUS'].isin(['1', '2', '3', '4', '5']).astype(int)
    bureau_bal['is_unknown'] = (bureau_bal['STATUS'] == 'X').astype(int)

    agg_dict = {
        # min = idade do contrato; count = total de meses observados
        'MONTHS_BALANCE': ['min', 'max', 'count'],
        'is_closed': ['sum', 'mean'],
        'is_dpd_zero': ['sum', 'mean'],
        # max indica se houve atraso grave em algum momento
        'is_dpd_bad': ['sum', 'max', 'mean'],
        'is_unknown': ['sum', 'mean'],
    }

    bureau_bal_agg = bureau_bal.groupby('SK_ID_BUREAU').agg(agg_dict)
    bureau_bal_agg.columns = ['bb_' + '_'.join(c).strip('_') for c in bureau_bal_agg.columns.values]
    bureau_bal_agg.reset_index(inplace=True)

    return bureau_bal_agg


def build_bureau_agg(path: Path) -> pd.DataFrame | None:
    """
    bureau não usa aggregate_table direto porque precisa do merge com
    bureau_balance (chave SK_ID_BUREAU) antes de agregar por SK_ID_CURR.
    """
    bureau_fp = path / "bureau.csv"
    if not bureau_fp.exists():
        print(f"  [skip] bureau.csv não encontrado em {path}")
        return None

    print("Processando bureau e bureau_balance...")
    bureau = pd.read_csv(bureau_fp)

    bal_fp = path / "bureau_balance.csv"
    if bal_fp.exists():
        bureau = bureau.merge(
            agg_bureau_balance(pd.read_csv(bal_fp)),
            on='SK_ID_BUREAU', how='left'
        )
    else:
        print("  [skip] bureau_balance.csv não encontrado")

    return aggregate_frame(bureau, 'bureau', agg_funcs=('mean', 'sum', 'max', 'min', 'count'))


# ---------------------------------------------------------------------------
# Previous application
# ---------------------------------------------------------------------------

def agg_previous_application(prev: pd.DataFrame) -> pd.DataFrame:
    """One-hot das categóricas relevantes + agregação por SK_ID_CURR."""
    cat_cols = ['NAME_CONTRACT_STATUS', 'CODE_REJECT_REASON', 'NAME_CASH_LOAN_PURPOSE']
    cat_cols = [c for c in cat_cols if c in prev.columns]

    prev_encoded = pd.get_dummies(prev, columns=cat_cols, dummy_na=True)

    # get_dummies devolve colunas bool, que select_dtypes(include='number')
    # NÃO captura. Sem esse cast o one-hot seria silenciosamente descartado.
    bool_cols = prev_encoded.select_dtypes(include='bool').columns
    if len(bool_cols) > 0:
        prev_encoded[bool_cols] = prev_encoded[bool_cols].astype('uint8')

    num_cols = _numeric_cols_without_ids(prev_encoded) + ['SK_ID_CURR']
    agg_dict = {c: ['mean', 'sum', 'max'] for c in num_cols if c != 'SK_ID_CURR'}

    prev_agg = prev_encoded[num_cols].groupby('SK_ID_CURR').agg(agg_dict)
    prev_agg.columns = ['prev_' + '_'.join(c).strip('_') for c in prev_agg.columns.values]
    prev_agg.reset_index(inplace=True)

    return prev_agg


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def feature_engineering_application(df: pd.DataFrame) -> pd.DataFrame:
    """Transformações financeiras e demográficas na tabela principal."""
    df = df.copy()

    # 1. Ajustes básicos
    df['AGE_YEARS'] = (df['DAYS_BIRTH'] / -365).astype(int)

    # 365243 é o sentinela de "não empregado" — vira NaN, não 1000 anos de casa
    if 'DAYS_EMPLOYED' in df.columns:
        df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)

    # 2. Indicadores financeiros (denominador zerado -> NaN, não inf)
    df['CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY'].replace(0, np.nan)
    df['CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE'].replace(0, np.nan)
    df['DOWN_PAYMENT_AMOUNT'] = df['AMT_GOODS_PRICE'] - df['AMT_CREDIT']
    df['INCOME_TO_CREDIT_RATIO'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT'].replace(0, np.nan)
    df['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'].replace(0, np.nan)
    df['EMPLOYED_TO_AGE_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH'].replace(0, np.nan)

    # 3. EXT_SOURCES: resumos são mais estáveis do que divisões diretas
    ext_cols = [c for c in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'] if c in df.columns]
    if ext_cols:
        df['EXT_SOURCES_MEAN'] = df[ext_cols].mean(axis=1)
        df['EXT_SOURCES_STD'] = df[ext_cols].std(axis=1)
        df['EXT_SOURCES_MIN'] = df[ext_cols].min(axis=1)
        df['EXT_SOURCES_MAX'] = df[ext_cols].max(axis=1)
        df['EXT_SOURCES_NAN_COUNT'] = df[ext_cols].isna().sum(axis=1)

    if 'EXT_SOURCE_3' in df.columns:
        # EXT_SOURCE fica em [0, 1]: sem o clip, valores perto de zero
        # explodem o rácio em outliers de ordem 1e6.
        ext3_safe = df['EXT_SOURCE_3'].clip(lower=0.01)
        for col in ['AMT_CREDIT', 'AMT_ANNUITY', 'AGE_YEARS', 'DAYS_EMPLOYED']:
            if col in df.columns:
                df[f'{col}_DIV_EXT3'] = df[col] / ext3_safe

    return df


def feature_engineering_installments(installments: pd.DataFrame) -> pd.DataFrame:
    """
    Comportamento de atraso em janela recente (180 dias), agregado por SK_ID_CURR.
    DAYS_INSTALMENT é negativo, então "recente" é >= -180.
    """
    installments = installments.copy()

    # positivo = pagou depois do vencimento
    installments['DAYS_DELAY'] = installments['DAYS_ENTRY_PAYMENT'] - installments['DAYS_INSTALMENT']
    installments['IS_LATE'] = (installments['DAYS_DELAY'] > 0).astype(int)

    inst_180d = installments[installments['DAYS_INSTALMENT'] >= -180]

    if inst_180d.empty:
        return pd.DataFrame(columns=['SK_ID_CURR'])

    agg_180d = inst_180d.groupby('SK_ID_CURR').agg({
        'IS_LATE': ['mean', 'sum'],
        'DAYS_DELAY': ['max', 'mean'],
        'AMT_PAYMENT': ['sum', 'mean'],
        'AMT_INSTALMENT': ['sum', 'mean'],
    })
    agg_180d.columns = [f'INST_180D_{c[0]}_{c[1].upper()}' for c in agg_180d.columns]
    agg_180d.reset_index(inplace=True)

    agg_180d['INST_180D_PAYMENT_RATIO'] = (
        agg_180d['INST_180D_AMT_PAYMENT_SUM']
        / agg_180d['INST_180D_AMT_INSTALMENT_SUM'].replace(0, np.nan)
    )

    return agg_180d


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------

def creating_table(path: Path, df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """Junta todas as tabelas auxiliares na base principal e persiste o resultado."""
    n_cols_inicial = df.shape[1]

    # 1. bureau (+ bureau_balance)
    bureau_agg = build_bureau_agg(path)
    if bureau_agg is not None:
        df = df.merge(bureau_agg, on='SK_ID_CURR', how='left')

    # 2. previous_application (one-hot antes de agregar)
    prev_fp = path / "previous_application.csv"
    if prev_fp.exists():
        print("Processando previous_application.csv...")
        df = df.merge(agg_previous_application(pd.read_csv(prev_fp)),
                      on='SK_ID_CURR', how='left')
    else:
        print(f"  [skip] previous_application.csv não encontrado em {path}")

    # 3. tabelas com agregação genérica
    for src, prefix in [("POS_CASH_balance.csv", "pos"),
                        ("credit_card_balance.csv", "cc")]:
        agg = aggregate_table(path, src, prefix)
        if agg is not None:
            df = df.merge(agg, on='SK_ID_CURR', how='left')

    # 4. installments: um único read para as duas agregações
    inst_fp = path / "installments_payments.csv"
    if inst_fp.exists():
        print("Processando installments_payments.csv...")
        inst = pd.read_csv(inst_fp)
        df = df.merge(aggregate_frame(inst, 'inst'), on='SK_ID_CURR', how='left')
        df = df.merge(feature_engineering_installments(inst), on='SK_ID_CURR', how='left')
    else:
        print(f"  [skip] installments_payments.csv não encontrado em {path}")

    # 5. feature engineering da tabela principal
    df = feature_engineering_application(df)

    print(f"Colunas: {n_cols_inicial} -> {df.shape[1]} | linhas: {len(df)}")

    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    output_file = BRONZE_DIR / filename
    print(f"Salvando em: {output_file}")
    df.to_parquet(output_file, index=False)

    return df


def load_and_merge_home_credit_data(path: Path = RAW_DIR,
                                    force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carrega e integra as tabelas do Home Credit para treino e teste.

    force=False reaproveita os parquets já gravados em BRONZE_DIR;
    force=True refaz o pipeline do zero.
    """
    train_file = BRONZE_DIR / "train_merged.parquet"
    test_file = BRONZE_DIR / "test_merged.parquet"

    if not force and train_file.exists() and test_file.exists():
        print(f"Lendo tabelas já processadas de {BRONZE_DIR} (use force=True para refazer)")
        return pd.read_parquet(train_file), pd.read_parquet(test_file)

    print(f"Carregando application_train e application_test de {path}...")
    df_train = pd.read_csv(path / "application_train.csv")
    df_test = pd.read_csv(path / "application_test.csv")

    print("\n=== Base de TREINO ===")
    df_train = creating_table(path, df_train, train_file.name)

    print("\n=== Base de TESTE ===")
    df_test = creating_table(path, df_test, test_file.name)

    return df_train, df_test


def process_categorical_application(df_train: pd.DataFrame,
                                    df_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Converte colunas object para 'category' com o MESMO mapeamento em train e test."""
    df_train, df_test = df_train.copy(), df_test.copy()

    cat_cols = [
        'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
        'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
        'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE',
        'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE', 'FONDKAPREMONT_MODE',
        'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE'
    ]
    cat_cols = [c for c in cat_cols if c in df_train.columns and c in df_test.columns]

    for col in cat_cols:
        cats = sorted(set(df_train[col].dropna().unique()) | set(df_test[col].dropna().unique()))
        dtype = pd.CategoricalDtype(categories=cats)
        df_train[col] = df_train[col].astype(dtype)
        df_test[col] = df_test[col].astype(dtype)

    print(f"{len(cat_cols)} colunas convertidas para 'category' com categorias alinhadas.")
    return df_train, df_test
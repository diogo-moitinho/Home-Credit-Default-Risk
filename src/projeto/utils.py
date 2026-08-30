import pandas as pd
from pathlib import Path


path = Path("../data/raw")
output_dir = Path('/data/PROJETOS/GIT/HONE-CREDIT/data/bronze')


def load_and_merge_home_credit_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carrega e integra as principais tabelas do Home Credit para treino e teste.
    """    
    print("Carregando application_train e application_test...")
    df_train = pd.read_csv(path / "application_train.csv")
    df_test = pd.read_csv(path / "application_test.csv")
    
    print("\nProcessando base de TREINO:")
    df_train = creating_table(path, df_train, "train_merged.csv")
    
    print("\nProcessando base de TESTE:")
    df_test = creating_table(path, df_test, "test_merged.csv")
    
    return df_train, df_test


def creating_table(path, df, filename) -> pd.DataFrame:
    df = join_bureau(path, df)
    df = join_previous_application(path, df)
    df = join_pos_cash(path, df) #
    df = join_installments_payments(path, df)
    df = join_credit_card_balance(path, df)

    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / filename
    print(f"Salvando em: {output_file}")
    df.to_csv(output_file, index=False)
    
    return df


def join_bureau(path: Path, df: pd.DataFrame) -> pd.DataFrame:
    if (path / "bureau.csv").exists() and (path / "bureau_balance.csv").exists():
        print("Processando bureau e bureau_balance...")

        # BUREAU BALANCE
        bureau_bal = pd.read_csv(path / "bureau_balance.csv")
        bureau_bal_agg = agg_bureau_balance(bureau_bal)
        
        #BUREAU
        bureau = pd.read_csv(path / "bureau.csv")
        bureau = bureau.merge(bureau_bal_agg, on='SK_ID_BUREAU', how='left')
        
        num_cols_bur = bureau.select_dtypes(include='number').columns.tolist()
        if 'SK_ID_CURR' not in num_cols_bur:
            num_cols_bur.append('SK_ID_CURR')
            
        bureau_agg = bureau[num_cols_bur].groupby('SK_ID_CURR').agg(['mean', 'sum', 'max', 'min', 'count']).reset_index()
        bureau_agg.columns = ['_'.join(col).strip('_') for col in bureau_agg.columns.values]
        df = df.merge(bureau_agg, on='SK_ID_CURR', how='left')
    return df


def agg_bureau_balance(bureau_bal: pd.DataFrame) -> pd.DataFrame:
    bureau_bal['is_closed'] = (bureau_bal['STATUS'] == 'C').astype(int)
    bureau_bal['is_dpd_zero'] = (bureau_bal['STATUS'] == '0').astype(int)
    bureau_bal['is_dpd_bad'] = bureau_bal['STATUS'].isin(['1', '2', '3', '4', '5']).astype(int)
    bureau_bal['is_unknown'] = (bureau_bal['STATUS'] == 'X').astype(int)
    
    agg_dict = {
        'MONTHS_BALANCE': ['min', 'max', 'count'],  # min é a idade do empréstimo, count é o total de meses
        'is_closed': ['sum', 'mean'],
        'is_dpd_zero': ['sum', 'mean'],
        'is_dpd_bad': ['sum', 'max', 'mean'],       # max diz se em algum momento ele teve atraso grave
        'is_unknown': ['sum', 'mean']
    }
    
    bureau_bal_agg = bureau_bal.groupby('SK_ID_BUREAU').agg(agg_dict)
    
    # Achata os nomes das colunas multi-index
    bureau_bal_agg.columns = ['bb_' + '_'.join(col).strip('_') for col in bureau_bal_agg.columns.values]
    bureau_bal_agg.reset_index(inplace=True)
    
    return bureau_bal_agg


def join_previous_application(path: Path, df: pd.DataFrame) -> pd.DataFrame:
    if (path / "previous_application.csv").exists():
        print("Processando previous_application...")
        prev = pd.read_csv(path / "previous_application.csv")
        prev_agg = agg_previous_application(prev)
        
        df = df.merge(prev_agg, on='SK_ID_CURR', how='left')
    return df


def agg_previous_application(prev: pd.DataFrame) -> pd.DataFrame:
    cat_cols = ['NAME_CONTRACT_STATUS', 'CODE_REJECT_REASON', 'NAME_CASH_LOAN_PURPOSE']
    cat_cols = [col for col in cat_cols if col in prev.columns]
    
    prev_encoded = pd.get_dummies(prev, columns=cat_cols, dummy_na=True)
    
    num_cols = prev_encoded.select_dtypes(include='number').columns.tolist()
    if 'SK_ID_CURR' not in num_cols:
        num_cols.append('SK_ID_CURR')
        
    # Remove SK_ID_CURR do dicionário de agregações comuns para tratá-lo separadamente se necessário, 
    # ou simplesmente agrupa direto:
    agg_dict = {col: ['mean', 'sum', 'max'] for col in num_cols if col != 'SK_ID_CURR'}
    
    prev_agg = prev_encoded[num_cols].groupby('SK_ID_CURR').agg(agg_dict)
    
    # Achata as colunas mantendo o SK_ID_CURR seguro no índice
    prev_agg.columns = ['prev_' + '_'.join(col).strip('_') for col in prev_agg.columns.values]
    prev_agg.reset_index(inplace=True) # Aqui o SK_ID_CURR volta a ser coluna limpa apenas uma vez
    
    return prev_agg


def join_pos_cash(path: Path, df: pd.DataFrame) -> pd.DataFrame:
    if (path / "POS_CASH_balance.csv").exists():
        print("Processando POS_CASH_balance...")
        pos = pd.read_csv(path / "POS_CASH_balance.csv")
        num_cols = pos.select_dtypes(include='number').columns.tolist()
        if 'SK_ID_CURR' not in num_cols:
            num_cols.append('SK_ID_CURR')
            
        pos_agg = pos[num_cols].groupby('SK_ID_CURR').agg(['mean', 'sum', 'max', 'count']).reset_index()
        pos_agg.columns = ['_'.join(col).strip('_') for col in pos_agg.columns.values]
        df = df.merge(pos_agg, on='SK_ID_CURR', how='left')
    return df


def join_installments_payments(path: Path, df: pd.DataFrame) -> pd.DataFrame:
    if (path / "installments_payments.csv").exists():
        print("Processando installments_payments...")
        inst = pd.read_csv(path / "installments_payments.csv")
        num_cols = inst.select_dtypes(include='number').columns.tolist()
        if 'SK_ID_CURR' not in num_cols:
            num_cols.append('SK_ID_CURR')
            
        inst_agg = inst[num_cols].groupby('SK_ID_CURR').agg(['mean', 'sum', 'max', 'count']).reset_index()
        inst_agg.columns = ['_'.join(col).strip('_') for col in inst_agg.columns.values]
        df = df.merge(inst_agg, on='SK_ID_CURR', how='left')
    return df


def join_credit_card_balance(path: Path, df: pd.DataFrame) -> pd.DataFrame:
    if (path / "credit_card_balance.csv").exists():
        print("Processando credit_card_balance...")
        cc = pd.read_csv(path / "credit_card_balance.csv")
        num_cols = cc.select_dtypes(include='number').columns.tolist()
        if 'SK_ID_CURR' not in num_cols:
            num_cols.append('SK_ID_CURR')
            
        cc_agg = cc[num_cols].groupby('SK_ID_CURR').agg(['mean', 'sum', 'max', 'count']).reset_index()
        cc_agg.columns = ['_'.join(col).strip('_') for col in cc_agg.columns.values]
        df = df.merge(cc_agg, on='SK_ID_CURR', how='left')
    return df
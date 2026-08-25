"""
Visualization module using SEABORN and MATPLOTLIB.
Optimized for large datasets and customized visual storytelling.
"""
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import probplot
from scipy import stats


sns.set_theme(style="whitegrid", palette="muted")


def _get_random_color() -> str:
    """Returns a random hex color from a standard professional palette."""
    palette = sns.color_palette("tab10").as_hex()
    return random.choice(palette)


def _get_statistical_summary_text(df: pd.DataFrame, feature: str) -> str:
    """Generates the statistical summary string for the lateral text box."""
    data = df[feature].dropna()
        
    mean_val = data.mean()
    std_dev = data.std()
    
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    total_outliers_upper = (data > upper_bound).sum()
    total_outliers_lower = (data < lower_bound).sum()
    
    null_count = df[feature].isna().sum()
    median_val = data.median()
    min_val = data.min()
    max_val = data.max()
    unique_vals = data.nunique()
    percentage_of_outliers = (total_outliers_lower + total_outliers_upper) / data.count() * 100

    return (
        f"{feature.upper()}\n\n"
        f"--- Descriptive Statistics ---\n"
        f"Mean: {mean_val:.2f}\n"
        f"Median: {median_val:.2f}\n"
        f"Std Dev: {std_dev:.2f}\n\n"
        f"--- IQR Boundaries ---\n"
        f"Upper Bound: {upper_bound:.2f}\n"
        f"Lower Bound: {lower_bound:.2f}\n"
        f"Min / Max: {min_val:.2f} | {max_val:.2f}\n\n"
        f"--- Data Quality ---\n"
        f"Null Values: {null_count} ({(null_count/len(df))*100:.1f}%)\n"
        f"Unique Values: {unique_vals}\n"
        f'Percentage of Outliers {round(percentage_of_outliers,2)}%' 
    )


def plot_boxplot(df: pd.DataFrame, feature: str, ax=None, orientation='h', hue=None, color_map=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        
    x_val = feature if orientation == 'h' else None
    y_val = feature if orientation == 'v' else None
    
    if hue is None:
        plot_color = _get_random_color()
        sns.boxplot(data=df, x=x_val, y=y_val, ax=ax, color=plot_color)
    else:
        palette_choice = color_map if color_map else "Set2"
        sns.boxplot(data=df, x=x_val, y=y_val, hue=hue, ax=ax, palette=palette_choice)
    
    if orientation == 'h':
        ax.set_xscale('symlog')
    else:
        ax.set_yscale('symlog')
        
    ax.set_title(f'BOXPLOT: {feature.upper()}', fontweight='bold')
    return ax


def plot_feature_density(df: pd.DataFrame, feature: str, ax=None, color=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        
    plot_color = color if color else _get_random_color()
    
    # OTIMIZAÇÃO: Usa o numpy e o Matplotlib nativo com log=True
    data = df[feature].dropna().to_numpy()
    
    ax.hist(data, bins=100, color=plot_color, edgecolor='black', linewidth=0.5, log=True)
    
    ax.set_title(f'DISTRIBUTION (LOG): {feature.upper()}', fontweight='bold')
    ax.set_xlabel(feature)
    ax.set_ylabel('Frequency (Log)')
    return ax


def plot_zscore_density(df: pd.DataFrame, feature: str, ax=None, color=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        
    plot_color = color if color else _get_random_color()
    
    # OTIMIZAÇÃO: Z-score super rápido via numpy array
    data = df[feature].dropna().to_numpy()
    z_scores = stats.zscore(data)
    
    ax.hist(z_scores, bins=100, color=plot_color, edgecolor='black', linewidth=0.5, log=True)
    
    ax.set_title(f'Z-SCORE (LOG): {feature.upper()}', fontweight='bold')
    ax.set_xlabel(f'{feature} (Z-Score)')
    ax.set_ylabel('Frequency (Log)')
    return ax


def plot_qq_normality(df: pd.DataFrame, feature: str, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        
    data_clean = pd.to_numeric(df[feature], errors='coerce').dropna()
    
    if data_clean.empty:
        ax.set_title(f"QQ-Plot: {feature.upper()} (No Numeric Data)")
        return ax

    # TRAVA DE SEGURANÇA: Se a base for absurdamente gigante, o QQ-Plot 
    # usa internamente uma amostra máxima de 50 mil pontos para não explodir a RAM.
    if len(data_clean) > 50000:
        data_clean = data_clean.sample(50000, random_state=42)

    (osm, osr), (slope, inter, _) = probplot(data_clean, dist="norm")
    
    # Ajustei o tamanho do ponto (s=2) para ficar mais limpo em bases grandes
    ax.scatter(osm, osr, color='red', s=2, alpha=0.5, label='Sample')
    ax.plot(osm, slope * osm + inter, color='black', linewidth=2, label='Theoretical Normal')
    
    ax.set_title(f"QQ-PLOT: {feature.upper()}", fontweight='bold')
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Values")
    return ax


def _convert_to_numeric(df: pd.DataFrame, include=()):
    """Converts categorical variables to numeric mappings for correlation."""
    num = df.select_dtypes("number").copy()
    for c in df.columns:
        s = df[c]
        if isinstance(s.dtype, pd.CategoricalDtype):
            if s.dtype.ordered or s.nunique(dropna=True) == 2:
                num[c] = s.cat.codes.replace(-1, np.nan)
    for c in include:
        if c not in num.columns:
            num[c] = pd.factorize(df[c])[0].astype(float)
            num.loc[df[c].isna(), c] = np.nan
    return num


def plot_correlation_heatmaps(df: pd.DataFrame, include=()):
    num = _convert_to_numeric(df, include=include)
    
    if num.shape[1] < 2:
        print("Insufficient numeric columns for correlation.")
        return None

    cp = num.corr(method="pearson")
    cs = num.corr(method="spearman")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    sns.heatmap(cp, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, 
                ax=axes[0], square=True, linewidths=.5, cbar_kws={"shrink": .8})
    axes[0].set_title("Pearson (Linear Correlation)", fontweight='bold')
    
    sns.heatmap(cs, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, 
                ax=axes[1], square=True, linewidths=.5, cbar_kws={"shrink": .8})
    axes[1].set_title("Spearman (Monotonic Correlation)", fontweight='bold')
    
    fig.suptitle("Variable Correlation Analysis", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def generate_quantitative_panel(df: pd.DataFrame, features: list, hue=None, color_map=None):
    """Generates a complete panel (Z-Score, Density, Boxplot, QQ-Plot) for numeric features."""
    for feature in features:
        plot_color = _get_random_color()
        summary_text = _get_statistical_summary_text(df, feature)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Univariate distributions use the single color
        plot_zscore_density(df, feature=feature, ax=axes[0, 0], color=plot_color)
        plot_feature_density(df, feature=feature, ax=axes[0, 1], color=plot_color)
        
        # Boxplot leverages hue and color map if provided to separate categories (e.g. Fraud vs Legitimate)
        plot_boxplot(df, feature=feature, ax=axes[1, 0], hue=hue, color_map=color_map)
        
        plot_qq_normality(df, feature=feature, ax=axes[1, 1])
        
        fig.suptitle(f"Quantitative Analysis: {feature.upper()}", fontsize=20, fontweight='bold')
        
        plt.subplots_adjust(right=0.78, hspace=0.3, wspace=0.2)
        
        fig.text(0.80, 0.5, summary_text, fontsize=11, family='monospace', va='center',
                 bbox=dict(boxstyle="round,pad=1", facecolor="white", edgecolor="black", linewidth=1.5))
        
        plt.show()


def _get_categorical_summary_text(df: pd.DataFrame, feature: str) -> str:
    """Generates the statistical summary string for categorical features."""
    data = df[feature]
    total = len(data)
    missing = data.isna().sum()
    unique = data.nunique()
    
    top_cat = data.mode()[0] if not data.mode().empty else "N/A"
    top_freq = data.value_counts().iloc[0] if not data.empty else 0

    return (
        f"{feature.upper()}\n\n"
        f"--- Overview ---\n"
        f"Total Records: {total}\n"
        f"Missing Values: {missing} ({(missing/total)*100:.1f}%)\n"
        f"Unique Categories: {unique}\n\n"
        f"--- Top Category ---\n"
        f"Value: {top_cat}\n"
        f"Count: {top_freq}\n"
        f"Share: {(top_freq/total)*100:.1f}%\n"
    )


def plot_category_by_target(df: pd.DataFrame, feature: str, target: str, ax=None, color_map=None):
    """Plots absolute counts separated by the target variable using Log scale."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    palette_choice = color_map if color_map else "Set2"
    
    # Usando hue para separar as classes do alvo
    sns.countplot(data=df, x=feature, hue=target, palette=palette_choice, ax=ax, edgecolor='black')

    # Escala Logarítmica obrigatória, senão os 0.1% de fraudes ficarão invisíveis
    ax.set_yscale('symlog')
    
    ax.set_title(f"COUNT BY TARGET (LOG): {feature.upper()}", fontweight='bold')
    ax.set_ylabel("Count (Log Scale)")
    ax.set_xlabel(feature)
    
    ax.tick_params(axis='x', rotation=90)
    
    return ax


def plot_target_rate(df: pd.DataFrame, feature: str, target: str, ax=None, color=None):
    """Calculates and plots the percentage of the target=1 within each category."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Matemática Ninja do Pandas: A média de uma coluna [0, 1] é exatamente a % de 1s.
    rates = df.groupby(feature)[target].mean().mul(100).sort_values(ascending=False).reset_index()

    plot_color = color if color else '#c0392b' # Vermelho escuro padrão para risco
    sns.barplot(data=rates, x=feature, y=target, color=plot_color, ax=ax, edgecolor='black')

    # Adicionando os rótulos de porcentagem em cima das barras
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.2f}%', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', fontsize=10, xytext=(0, 5), 
                        textcoords='offset points')

    ax.set_title(f"FRAUD RATE (%): {feature.upper()}", fontweight='bold')
    ax.set_ylabel(f"{target.upper()} Rate (%)")
    ax.set_xlabel(feature)
    ax.tick_params(axis='x', rotation=90)
    return ax


def plot_relative_frequency(df: pd.DataFrame, feature: str, ax=None, color=None, color_map=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        
    freq_df = (df[feature].value_counts(normalize=True) * 100).reset_index()
    freq_df.columns = [feature, 'percentage']
    
    if color_map:
        # Modern Seaborn requires hue when using a specific palette mapping
        sns.barplot(data=freq_df, x=feature, y='percentage', hue=feature, palette=color_map, ax=ax, legend=False)
    else:
        plot_color = color if color else _get_random_color()
        sns.barplot(data=freq_df, x=feature, y='percentage', color=plot_color, ax=ax)
    
    # Add percentage labels on top of the bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}%', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', fontsize=10, xytext=(0, 5), 
                        textcoords='offset points')
        
    ax.set_title(f"RELATIVE FREQUENCY: {feature.upper()}", fontweight='bold')
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel(feature)
    ax.tick_params(axis='x', rotation=90)
    return ax


def generate_categorical_panel(df: pd.DataFrame, features: list, target=None, color_map=None):
    """Generates a complete panel for categorical features, optionally crossing with a target."""
    for feature in features:
        summary_text = _get_categorical_summary_text(df, feature)

        # Se o usuário mandou uma variável alvo (isFraud), gera o painel triplo
        if target:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            plot_relative_frequency(df, feature=feature, ax=axes[0])
            plot_category_by_target(df, feature=feature, target=target, ax=axes[1], color_map=color_map)
            plot_target_rate(df, feature=feature, target=target, ax=axes[2])
            
            # Abre espaço extra na direita para a caixa de texto
            plt.subplots_adjust(right=0.82, wspace=0.3)
            
        # Se não tem alvo, gera um painel duplo mais simples
        else:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            plot_relative_frequency(df, feature=feature, ax=axes[0])
            sns.countplot(data=df, x=feature, ax=axes[1], color=_get_random_color(), edgecolor='black')
            axes[1].set_title(f"ABSOLUTE COUNT: {feature.upper()}", fontweight='bold')
            
            plt.subplots_adjust(right=0.78, wspace=0.3)

        fig.suptitle(f"Categorical Analysis: {feature.upper()}", fontsize=20, fontweight='bold', y=1.05)

        # Caixa de texto padronizada
        fig.text(0.84 if target else 0.80, 0.5, summary_text, fontsize=11, family='monospace', va='center',
                 bbox=dict(boxstyle="round,pad=1", facecolor="white", edgecolor="black", linewidth=1.5))

        plt.show()
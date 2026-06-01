import os
import sys
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.db_manager import init_db, clear_cache_files
from backend.ml_engine import (
    load_data,
    preprocess_data,
    split_data,
    engineer_features,
    tune_kmeans,
    final_train_and_pca,
)

st.set_page_config(
    page_title="Kmeans++ Clustering",
    page_icon="🧠",
    layout="wide",
)

HIDE_STREAMLIT_UI = """
<style>
    /* main menu (hamburger) */
    #MainMenu { visibility: hidden !important; }
    /* header bar with rerun/share */
    header { visibility: hidden !important; }
    /* footer */
    footer { visibility: hidden !important; }
    /* explicit toolbar buttons (best-effort selectors) */
    button[title="Rerun"] { display: none !important; }
    div[aria-label="Main menu"] { display: none !important; }
    div[title="Share this app"] { display: none !important; }
    /* also hide the top-right toolbar container if present */
    [data-testid="stToolbar"] { display: none !important; }
</style>
"""

st.markdown(HIDE_STREAMLIT_UI, unsafe_allow_html=True)

CHART_HEIGHT = 520

BASE_FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"

LIGHT_THEME = {
    'background': '#F8FAFC',
    'surface': '#FFFFFF',
    'surface_alt': '#F1F5F9',
    'text': '#0F172A',
    'muted': '#64748B',
    'border': '#E2E8F0',
    'accent': '#6366F1',
    'accent_alt': '#2563EB',
}

DARK_THEME = {
    'background': '#0F172A',
    'surface': '#111827',
    'surface_alt': '#1E293B',
    'text': '#E2E8F0',
    'muted': '#94A3B8',
    'border': '#334155',
    'accent': '#818CF8',
    'accent_alt': '#6366F1',
}


def get_theme_css(theme_name: str) -> str:
    theme = LIGHT_THEME if theme_name == 'Light Mode' else DARK_THEME
    return f"""
    <style>
        :root {{
            color-scheme: {'light' if theme_name == 'Light Mode' else 'dark'};
        }}

        .stApp {{
            background: {theme['background']};
            color: {theme['text']};
            font-family: {BASE_FONT_FAMILY};
        }}

        .block-container {{
            padding: 2rem !important;
            background: transparent;
        }}

        .dashboard-card {{
            background: {theme['surface']};
            border: 1px solid {theme['border']};
            border-radius: 18px;
            padding: 1.5rem 1.6rem;
            margin-bottom: 1.5rem;
            color: {theme['text']};
        }}

        .section-header {{
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.75rem;
            color: {theme['text']};
        }}

        .body-copy {{
            color: {theme['muted']};
            line-height: 1.7;
            margin: 0;
        }}

        .stMetric > div {{
            border-radius: 16px !important;
            background: {theme['surface_alt']} !important;
            border: 1px solid {theme['border']} !important;
            padding: 1rem !important;
            color: {theme['text']} !important;
        }}

        .stMetric span, .stMetric div {{
            color: {theme['text']} !important;
            opacity: 1 !important;
        }}

        .stMetric .stMetricLabel, .stMetric .stMetricDelta {{
            color: {theme['muted']} !important;
        }}

        .stButton > button {{
            border-radius: 999px !important;
            padding: 0.85rem 1.4rem !important;
            background: {theme['accent']} !important;
            color: white !important;
            border: none !important;
        }}

        .stSidebar {{
            background: {theme['surface']} !important;
            color: {theme['text']} !important;
        }}

        .stSidebar .css-1d391kg, .stSidebar .css-18e3th9, .stSidebar .css-ffhzg2,
        .stSidebar p, .stSidebar label, .stSidebar div {{
            color: {theme['text']} !important;
        }}

        .stExpander {{
            background: {theme['surface_alt']} !important;
            border: 1px solid {theme['border']} !important;
            border-radius: 16px !important;
            color: {theme['text']} !important;
        }}

        .css-1d391kg, .css-18e3th9, .css-ffhzg2 {{
            background: transparent !important;
        }}
    </style>
    """


@st.cache_data
def load_data_cached() -> pd.DataFrame:
    return load_data()


@st.cache_data
def preprocess_data_cached(dataframe: pd.DataFrame) -> pd.DataFrame:
    return preprocess_data(dataframe)


@st.cache_data
def split_data_cached(dataframe: pd.DataFrame):
    return split_data(dataframe)


@st.cache_data
def engineer_features_cached(X_train, X_val, X_test):
    return engineer_features(X_train, X_val, X_test)


def style_plotly(fig: go.Figure, theme_name: str) -> go.Figure:
    theme = LIGHT_THEME if theme_name == 'Light Mode' else DARK_THEME
    fig.update_layout(
        template='plotly_white' if theme_name == 'Light Mode' else 'plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=theme['text'], family=BASE_FONT_FAMILY),
        title_font=dict(color=theme['text'], size=16, family=BASE_FONT_FAMILY),
        xaxis_title_font=dict(color=theme['text'], size=14, family=BASE_FONT_FAMILY),
        yaxis_title_font=dict(color=theme['text'], size=14, family=BASE_FONT_FAMILY),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=theme['text'], family=BASE_FONT_FAMILY)),
        margin=dict(l=18, r=18, t=50, b=28),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=theme['border'],
        tickfont=dict(color=theme['text']),
        title_font=dict(color=theme['text']),
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        linecolor=theme['border'],
        tickfont=dict(color=theme['text']),
        title_font=dict(color=theme['text']),
    )
    return fig


def render_card(title: str, content: str, description: str = None) -> None:
    description_html = f"<p class='body-copy'>{description}</p>" if description else ''
    st.markdown(
        f"""
        <div class='dashboard-card'>
            <div class='section-header'>{title}</div>
            <div class='body-copy'>{content}</div>
            {description_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    init_db()

    st.sidebar.markdown("## Controls")
    st.sidebar.markdown("Fine tune behavior and switch visual style without cluttering the canvas.")
    if st.sidebar.button("🧹 Clear cache & retrain"):
        clear_cache_files()
        st.cache_data.clear()
        st.sidebar.success("Cache cleared — refresh to rerun.")
        time.sleep(1)
        st.experimental_rerun()

    theme_choice = st.sidebar.radio("Theme", ['Dark Mode', 'Light Mode'], index=0)
    show_raw = st.sidebar.checkbox("Show raw sample", value=False)
    show_clusters = st.sidebar.checkbox("Enable cluster views", value=True)

    st.markdown(get_theme_css(theme_choice), unsafe_allow_html=True)

    st.markdown("# Kmeans++ Clustering")
    st.markdown("**A premium analytics workspace for clean cluster exploration.**")
    st.markdown("---")

    with st.spinner("Loading dataset…"):
        df = load_data_cached()

    df_new = preprocess_data_cached(df)
    X_train, X_val, X_test = split_data_cached(df_new)
    X_train_scaled, X_val_scaled, X_test_scaled, all_feature_names, _ = engineer_features_cached(
        X_train,
        X_val,
        X_test,
    )

    with st.spinner("Running K-fold tuning and training the final model…"):
        tuning_results, k_values, silhouette_avgs, fold_scores, best_k, tuning_cached = tune_kmeans(X_train_scaled)
        (
            train_score,
            val_score,
            test_score,
            train_pca_df,
            val_pca_df,
            test_pca_df,
            variance,
            pca_loadings,
            top_features_df,
            cluster_summary_test,
            final_cached,
        ) = final_train_and_pca(
            X_train_scaled,
            X_val_scaled,
            X_test_scaled,
            best_k,
            all_feature_names,
        )

    overview_tab, data_tab, tuning_tab, insights_tab = st.tabs([
        "Overview",
        "Data",
        "Model Tuning",
        "Cluster Insights",
    ])

    with overview_tab:
        st.markdown("### Summary")
        st.write("A premium data workspace that keeps key metrics clean, stable, and easy to scan.")

        row_a, row_b = st.columns([2, 1], gap='large')
        with row_a:
            metric_col1, metric_col2, metric_col3 = st.columns(3, gap='large')
            metric_col1.metric("Best clusters", f"k = {best_k}")
            metric_col2.metric("Train silhouette", f"{train_score:.3f}")
            metric_col3.metric("Test silhouette", f"{test_score:.3f}")

            render_card(
                "Model snapshot",
                f"{df.shape[0]} rows · {df_new.shape[1]} features · {variance:.1f}% explained variance.",
                "This view lets you understand the current dataset size and model quality at a glance.",
            )

        with row_b:
            st.markdown(
                f"""
                <div class='dashboard-card'>
                    <div class='section-header'>Cache status</div>
                    <p class='body-copy'>
                        K-fold tuning loaded from cache: <strong>{'Yes' if tuning_cached else 'No'}</strong><br>
                        Final training loaded from cache: <strong>{'Yes' if final_cached else 'No'}</strong>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with data_tab:
        st.markdown("### Data workflow")
        st.write("Clean feature preparation, split diagnostics, and a polished feature distribution chart.")

        with st.container():
            left, right = st.columns([2, 1], gap='large')
            with left:
                st.markdown("<div class='section-header'>Raw dataset preview</div>", unsafe_allow_html=True)
                st.dataframe(df.head(6), use_container_width=True)
            with right:
                st.markdown("<div class='section-header'>Split sizes</div>", unsafe_allow_html=True)
                st.metric("Train", f"{X_train.shape[0]}")
                st.metric("Validation", f"{X_val.shape[0]}")
                st.metric("Test", f"{X_test.shape[0]}")

        numeric_columns = df_new.select_dtypes(include='number').columns.tolist()
        st.markdown("#### Feature distributions")
        st.write("Explore the full numeric feature set to understand value ranges and shape.")

        num_cols = 2
        for idx, feature in enumerate(numeric_columns):
            if idx % num_cols == 0:
                cols = st.columns(num_cols, gap='large')

            hist_fig = px.histogram(
                df_new,
                x=feature,
                nbins=22,
                title=f'Distribution: {feature}',
                color_discrete_sequence=[LIGHT_THEME['accent'] if theme_choice == 'Light Mode' else DARK_THEME['accent']],
            )
            hist_fig.update_traces(marker_line_width=0, opacity=0.92)
            style_plotly(hist_fig, theme_choice)
            hist_fig.update_layout(height=CHART_HEIGHT // 2)
            cols[idx % num_cols].plotly_chart(hist_fig, use_container_width=True)

        corr_fig = px.imshow(
            df_new.corr(numeric_only=True),
            text_auto='.2f',
            color_continuous_scale='Blues' if theme_choice == 'Light Mode' else 'Viridis',
            title='Feature correlation matrix',
        )
        style_plotly(corr_fig, theme_choice)
        corr_fig.update_layout(height=CHART_HEIGHT)
        st.plotly_chart(corr_fig, use_container_width=True)

        if show_raw:
            with st.expander("Show the raw sample table"):
                st.dataframe(df.sample(min(8, len(df))), use_container_width=True)

    with tuning_tab:
        st.markdown("### Model tuning")
        st.write("Evaluate cluster stability across K and review fold-level silhouette behavior.")

        if tuning_cached:
            st.success("K-fold tuning data loaded from persistence.")

        st.dataframe(tuning_results, use_container_width=True)

        avg_fig = go.Figure()
        avg_fig.add_trace(
            go.Scatter(
                x=k_values,
                y=silhouette_avgs,
                mode='lines+markers',
                marker=dict(size=10, color=LIGHT_THEME['accent'] if theme_choice == 'Light Mode' else DARK_THEME['accent']),
                line=dict(width=3),
                name='Avg silhouette',
            )
        )
        avg_fig.add_vline(
            x=best_k,
            line=dict(color=LIGHT_THEME['accent_alt'] if theme_choice == 'Light Mode' else DARK_THEME['accent_alt'], dash='dash'),
            annotation_text=f"Best k = {best_k}",
            annotation_position='top right',
        )
        avg_fig.update_layout(title='Average silhouette by k', height=CHART_HEIGHT)
        style_plotly(avg_fig, theme_choice)
        st.plotly_chart(avg_fig, use_container_width=True)

        if fold_scores and len(fold_scores[0]) > 0:
            fold_fig = go.Figure()
            for fold_index in range(len(fold_scores[0])):
                fold_fig.add_trace(
                    go.Scatter(
                        x=k_values,
                        y=[scores[fold_index] for scores in fold_scores],
                        mode='lines+markers',
                        marker=dict(size=8),
                        line=dict(width=2),
                        name=f'Fold {fold_index + 1}',
                    )
                )
            fold_fig.update_layout(title='K-fold silhouette curves', height=CHART_HEIGHT)
            style_plotly(fold_fig, theme_choice)
            st.plotly_chart(fold_fig, use_container_width=True)

    with insights_tab:
        st.markdown("### Cluster insights")
        st.write("A polished view of split performance, PCA structure, and cluster balances.")

        insight_cols = st.columns(3, gap='large')
        insight_cols[0].metric("Train silhouette", f"{train_score:.3f}")
        insight_cols[1].metric("Val silhouette", f"{val_score:.3f}")
        insight_cols[2].metric("Test silhouette", f"{test_score:.3f}")

        split_fig = px.bar(
            pd.DataFrame({
                'Split': ['Train', 'Validation', 'Test'],
                'Silhouette': [train_score, val_score, test_score],
            }),
            x='Split',
            y='Silhouette',
            color='Split',
            text='Silhouette',
            color_discrete_sequence=['#6366F1', '#4F46E5', '#2563EB'],
            title='Silhouette by split',
        )
        split_fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        style_plotly(split_fig, theme_choice)
        split_fig.update_layout(height=CHART_HEIGHT)
        st.plotly_chart(split_fig, use_container_width=True)

        if show_clusters:
            pca_cols = st.columns(2, gap='large')
            with pca_cols[0]:
                pca_train = px.scatter(
                    train_pca_df,
                    x='PC1',
                    y='PC2',
                    color='Cluster',
                    title='Train PCA 2D',
                    opacity=0.85,
                    color_discrete_sequence=px.colors.qualitative.Bold,
                )
                style_plotly(pca_train, theme_choice)
                pca_train.update_layout(height=CHART_HEIGHT)
                st.plotly_chart(pca_train, use_container_width=True)

            with pca_cols[1]:
                pca_val = px.scatter(
                    val_pca_df,
                    x='PC1',
                    y='PC2',
                    color='Cluster',
                    title='Validation PCA 2D',
                    opacity=0.85,
                    color_discrete_sequence=px.colors.qualitative.Bold,
                )
                style_plotly(pca_val, theme_choice)
                pca_val.update_layout(height=CHART_HEIGHT)
                st.plotly_chart(pca_val, use_container_width=True)

            pca_3d = px.scatter_3d(
                test_pca_df,
                x='PC1',
                y='PC2',
                z='PC3',
                color='Cluster',
                opacity=0.85,
                title='Test PCA 3D',
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            pca_3d.update_traces(marker=dict(size=5, line=dict(width=0)))
            style_plotly(pca_3d, theme_choice)
            pca_3d.update_layout(height=CHART_HEIGHT)
            st.plotly_chart(pca_3d, use_container_width=True)

        cluster_balance = test_pca_df['Cluster'].value_counts().reset_index()
        cluster_balance.columns = ['Cluster', 'Count']
        cluster_balance_fig = px.bar(
            cluster_balance,
            x='Cluster',
            y='Count',
            text='Count',
            title='Test cluster balance',
            color='Cluster',
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        cluster_balance_fig.update_traces(textposition='outside')
        style_plotly(cluster_balance_fig, theme_choice)
        cluster_balance_fig.update_layout(height=CHART_HEIGHT)
        st.plotly_chart(cluster_balance_fig, use_container_width=True)

        loading_fig = px.bar(
            pca_loadings.abs().nlargest(10, 'PC1').reset_index().rename(columns={'index': 'Feature'}),
            x='Feature',
            y='PC1',
            title='Top PCA loadings',
            color='PC1',
            color_continuous_scale='Blues',
        )
        style_plotly(loading_fig, theme_choice)
        loading_fig.update_layout(height=CHART_HEIGHT)
        st.plotly_chart(loading_fig, use_container_width=True)

        with st.expander('Cluster summary table'):
            st.dataframe(cluster_summary_test, use_container_width=True)

    st.markdown('---')
    st.write('Built as a high-end Streamlit App by [Aditya Pratap Singh]')


if __name__ == '__main__':
    main()

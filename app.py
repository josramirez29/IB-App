import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import sqlite3
import os
import json
import chardet

# ----- Configuración inicial -----
def setup_page():
    st.set_page_config(page_title="Data Quality Dashboard", layout="wide", page_icon="📊")
    
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 30px;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown('<h1 class="main-header">Dashboard de Calidad de Datos</h1>', unsafe_allow_html=True)

# ----- Funciones para manejar cualquier tipo de CSV -----
def detect_encoding(file_content):
    result = chardet.detect(file_content)
    return result['encoding']

def read_csv_with_autodetection(uploaded_file):
    try:
        file_content = uploaded_file.read()
        uploaded_file.seek(0)  
        encoding = detect_encoding(file_content)
        delimiters = [',', ';', '\t', '|']
        
        for delimiter in delimiters:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=encoding, delimiter=delimiter, 
                                 low_memory=False, on_bad_lines='skip')
                if df.shape[1] > 1:
                    st.success(f"Archivo leído con encoding '{encoding}' y delimitador '{delimiter}'")
                    return df
            except Exception:
                continue
        
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding=encoding, low_memory=False, on_bad_lines='skip')
        st.success(f"Archivo leído con encoding '{encoding}'")
        return df
    except Exception as e:
        st.error(f"Error al procesar el archivo: {str(e)}")
        return None

def clean_column_names(df):
    if df is None or df.empty:
        return df
    clean_df = df.copy()
    new_columns = []
    for i, col in enumerate(clean_df.columns):
        if not isinstance(col, str) or not col.strip():
            new_columns.append(f"col_{i}")
        else:
            clean_name = col.strip().replace(' ', '_').replace('.', '_').replace('-', '_').lower()
            new_columns.append(clean_name)
    clean_df.columns = new_columns
    return clean_df

def safe_division(numerator, denominator):
    if denominator == 0 or denominator is None or numerator is None:
        return 0
    try:
        return float(numerator) / float(denominator)
    except (ValueError, TypeError):
        return 0

# ----- Procesamiento de datos -----
def analyze_data_quality(df, dataset_name):
    if df is None or df.empty:
        return None
    try:
        df_clean = clean_column_names(df)
        n_rows, n_cols = df_clean.shape
        total_cells = n_rows * n_cols
        null_counts = df_clean.isnull().sum()
        total_nulls = null_counts.sum()
        null_percentage = safe_division(total_nulls, total_cells) * 100
        duplicate_rows = df_clean.duplicated().sum()
        duplicate_percentage = safe_division(duplicate_rows, n_rows) * 100
        data_types_count = df_clean.dtypes.astype(str).value_counts().reset_index()
        data_types_count.columns = ['tipo', 'cantidad']
        
        column_stats = []
        for col in df_clean.columns:
            col_type = str(df_clean[col].dtype)
            null_count = null_counts[col]
            null_pct = safe_division(null_count, n_rows) * 100
            unique_count = df_clean[col].nunique()
            unique_pct = safe_division(unique_count, n_rows) * 100
            min_val = max_val = mean_val = std_val = None
            try:
                numeric_series = pd.to_numeric(df_clean[col], errors='coerce')
                if not numeric_series.isna().all():
                    min_val = float(numeric_series.min())
                    max_val = float(numeric_series.max())
                    mean_val = float(numeric_series.mean())
                    std_val = float(numeric_series.std())
            except Exception:
                pass
            column_stats.append({
                'columna': col,
                'tipo': col_type,
                'valores_nulos': int(null_count),
                'porcentaje_nulos': float(null_pct),
                'valores_unicos': int(unique_count),
                'porcentaje_unicos': float(unique_pct),
                'min': min_val,
                'max': max_val,
                'media': mean_val,
                'desviacion_estandar': std_val
            })
        
        return {
            'dataset_name': dataset_name,
            'n_rows': int(n_rows),
            'n_cols': int(n_cols),
            'total_cells': int(total_cells),
            'total_nulls': int(total_nulls),
            'null_percentage': float(null_percentage),
            'duplicate_rows': int(duplicate_rows),
            'duplicate_percentage': float(duplicate_percentage),
            'data_types': json.dumps(data_types_count.to_dict('records')),
            'null_counts': json.dumps({str(k): int(v) for k, v in null_counts.to_dict().items()}),
            'column_stats': json.dumps(column_stats),
            'df_clean': df_clean
        }
    except Exception as e:
        st.error(f"Error en analyze_data_quality: {str(e)}")
        return None

def save_to_database(df, dataset_name):
    if df is None or df.empty:
        return None
    conn = sqlite3.connect('data_quality.db')
    try:
        quality_data = analyze_data_quality(df, dataset_name)
        if quality_data is None:
            return None
        df_clean = quality_data['df_clean']
        df_clean.to_sql(dataset_name, conn, if_exists='replace', index=False)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS dataset_quality (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_name TEXT UNIQUE,
            n_rows INTEGER,
            n_cols INTEGER,
            total_cells INTEGER,
            total_nulls INTEGER,
            null_percentage REAL,
            duplicate_rows INTEGER,
            duplicate_percentage REAL,
            data_types TEXT,
            null_counts TEXT,
            column_stats TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        conn.execute("""
        INSERT OR REPLACE INTO dataset_quality 
        (dataset_name, n_rows, n_cols, total_cells, total_nulls, null_percentage, 
         duplicate_rows, duplicate_percentage, data_types, null_counts, column_stats)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            quality_data['dataset_name'],
            quality_data['n_rows'],
            quality_data['n_cols'],
            quality_data['total_cells'],
            quality_data['total_nulls'],
            quality_data['null_percentage'],
            quality_data['duplicate_rows'],
            quality_data['duplicate_percentage'],
            quality_data['data_types'],
            quality_data['null_counts'],
            quality_data['column_stats']
        ))
        conn.commit()
        return get_quality_metrics_from_db(dataset_name)
    except Exception as e:
        st.error(f"Error al guardar en base de datos: {str(e)}")
        return None
    finally:
        conn.close()

def get_quality_metrics_from_db(dataset_name):
    conn = sqlite3.connect('data_quality.db')
    try:
        query = f"SELECT * FROM dataset_quality WHERE dataset_name = '{dataset_name}'"
        metrics_df = pd.read_sql_query(query, conn)
        if not metrics_df.empty:
            metrics = metrics_df.iloc[0].to_dict()
            metrics['data_types'] = pd.DataFrame(json.loads(metrics['data_types']))
            metrics['null_counts'] = pd.Series(json.loads(metrics['null_counts']))
            metrics['column_stats'] = pd.DataFrame(json.loads(metrics['column_stats']))
            df_clean = pd.read_sql_query(f"SELECT * FROM {dataset_name}", conn)
            metrics['df_clean'] = df_clean
            return metrics
        return None
    except Exception as e:
        st.error(f"Error al obtener métricas: {str(e)}")
        return None
    finally:
        conn.close()

# ----- Gráficas estadísticas -----
def get_column_data(dataset_name, column):
    conn = sqlite3.connect("data_quality.db")
    try:
        query = f"SELECT \"{column}\" FROM {dataset_name} WHERE \"{column}\" IS NOT NULL"
        df = pd.read_sql_query(query, conn)
        return df[column]
    except Exception as e:
        st.error(f"Error obteniendo datos de la columna {column}: {e}")
        return pd.Series([])
    finally:
        conn.close()

def create_histogram(dataset_name, column):
    data = get_column_data(dataset_name, column)
    if data.empty:
        return None
    try:
        fig = px.histogram(data, x=column, nbins=30, title=f"Histograma de {column}")
        return fig
    except Exception as e:
        st.error(f"Error en histograma: {e}")
        return None

def create_boxplot(dataset_name, column):
    data = get_column_data(dataset_name, column)
    if data.empty:
        return None
    try:
        fig = px.box(data, y=column, title=f"Boxplot de {column}")
        return fig
    except Exception as e:
        st.error(f"Error en boxplot: {e}")
        return None

def create_frequency_plot(dataset_name, column):
    data = get_column_data(dataset_name, column)
    if data.empty:
        return None
    try:
        freq = data.value_counts().reset_index()
        freq.columns = [column, "frecuencia"]
        fig = px.bar(freq, x=column, y="frecuencia", title=f"Frecuencia de valores en {column}")
        return fig
    except Exception as e:
        st.error(f"Error en frecuencia: {e}")
        return None

def create_stat_summary_chart(dataset_name, column):
    data = get_column_data(dataset_name, column)
    if data.empty:
        return None
    try:
        numeric_data = pd.to_numeric(data, errors="coerce").dropna()
        if numeric_data.empty:
            return None
        stats = {
            "media": numeric_data.mean(),
            "mediana": numeric_data.median(),
            "moda": numeric_data.mode().iloc[0] if not numeric_data.mode().empty else None,
            "varianza": numeric_data.var(),
            "desviación estándar": numeric_data.std()
        }
        stats_df = pd.DataFrame(list(stats.items()), columns=["Métrica", "Valor"])
        fig = px.bar(stats_df, x="Métrica", y="Valor", title=f"Estadísticas de {column}")
        return fig
    except Exception as e:
        st.error(f"Error en resumen estadístico: {e}")
        return None

# ----- Visualización -----
def create_completeness_chart(null_counts, n_rows):
    if null_counts is None or n_rows == 0:
        return None
    try:
        completeness = [(1 - safe_division(null_counts[col], n_rows)) * 100 for col in null_counts.index]
        fig = px.bar(
            x=null_counts.index, 
            y=completeness,
            title="Porcentaje de Completitud por Columna",
            labels={'x': 'Columnas', 'y': 'Porcentaje de Completitud (%)'}
        )
        fig.update_layout(yaxis_range=[0, 100])
        return fig
    except Exception as e:
        st.error(f"Error al crear gráfico de completitud: {str(e)}")
        return None

def create_data_types_chart(data_types):
    if data_types is None or data_types.empty:
        return None
    try:
        fig = px.pie(data_types, values='cantidad', names='tipo', title="Distribución de Tipos de Datos")
        return fig
    except Exception as e:
        st.error(f"Error al crear gráfico de tipos: {str(e)}")
        return None

def create_top_nulls_chart(null_counts, n_rows):
    if null_counts is None or n_rows == 0:
        return None
    try:
        null_pct = (null_counts / n_rows * 100).sort_values(ascending=False).head(10)
        fig = px.bar(
            x=null_pct.index,
            y=null_pct.values,
            title="Top 10 Columnas con más Valores Nulos (%)",
            labels={'x': 'Columnas', 'y': 'Porcentaje de Nulos (%)'}
        )
        fig.update_layout(yaxis_range=[0, 100])
        return fig
    except Exception as e:
        st.error(f"Error al crear gráfico de nulos: {str(e)}")
        return None

def create_cardinality_chart(df):
    if df is None or df.empty:
        return None
    try:
        cardinality = df.nunique().sort_values(ascending=False).head(10)
        fig = px.bar(
            x=cardinality.index,
            y=cardinality.values,
            title="Top 10 Columnas con más Valores Únicos",
            labels={'x': 'Columnas', 'y': 'Cantidad de Valores Únicos'}
        )
        return fig
    except Exception as e:
        st.error(f"Error al crear gráfico de cardinalidad: {str(e)}")
        return None

def display_quality_dashboard(quality_data, df_original):
    if quality_data is None:
        st.error("No hay datos de calidad para mostrar")
        return
    df_clean = quality_data.get('df_clean')
    if df_clean is None:
        st.error("No hay datos limpios para mostrar")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de Registros", f"{quality_data['n_rows']:,}")
    with col2:
        st.metric("Total de Columnas", quality_data['n_cols'])
    with col3:
        st.metric("Valores Nulos", f"{quality_data['total_nulls']} ({quality_data['null_percentage']:.2f}%)")
    with col4:
        st.metric("Filas Duplicadas", f"{quality_data['duplicate_rows']} ({quality_data['duplicate_percentage']:.2f}%)")

    st.subheader("Visualizaciones de Calidad de Datos")
    tab1, tab2, tab3, tab4 = st.tabs(["Completitud", "Tipos de Datos", "Top Nulos", "Cardinalidad"])
    
    with tab1:
        fig_completeness = create_completeness_chart(quality_data.get('null_counts'), quality_data.get('n_rows', 0))
        if fig_completeness:
            st.plotly_chart(fig_completeness, use_container_width=True)
    with tab2:
        data_types_df = quality_data.get('data_types')
        if data_types_df is not None and not data_types_df.empty:
            fig_datatypes = create_data_types_chart(data_types_df)
            if fig_datatypes:
                st.plotly_chart(fig_datatypes, use_container_width=True)
    with tab3:
        fig_top_nulls = create_top_nulls_chart(quality_data.get('null_counts'), quality_data.get('n_rows', 0))
        if fig_top_nulls:
            st.plotly_chart(fig_top_nulls, use_container_width=True)
    with tab4:
        fig_cardinality = create_cardinality_chart(df_clean)
        if fig_cardinality:
            st.plotly_chart(fig_cardinality, use_container_width=True)

    st.subheader("Estadísticas Detalladas por Columna")
    column_stats = quality_data.get('column_stats')
    if column_stats is not None and not column_stats.empty:
        st.dataframe(column_stats, use_container_width=True)

    st.subheader("Análisis Estadístico por Columna")
    selected_col = st.selectbox("Selecciona una columna para analizar:", df_clean.columns)
    if selected_col:
        tab_a, tab_b, tab_c, tab_d = st.tabs(["Histograma", "Boxplot", "Frecuencia", "Medidas estadísticas"])
        with tab_a:
            fig_hist = create_histogram(quality_data['dataset_name'], selected_col)
            if fig_hist: st.plotly_chart(fig_hist, use_container_width=True)
        with tab_b:
            fig_box = create_boxplot(quality_data['dataset_name'], selected_col)
            if fig_box: st.plotly_chart(fig_box, use_container_width=True)
        with tab_c:
            fig_freq = create_frequency_plot(quality_data['dataset_name'], selected_col)
            if fig_freq: st.plotly_chart(fig_freq, use_container_width=True)
        with tab_d:
            fig_stats = create_stat_summary_chart(quality_data['dataset_name'], selected_col)
            if fig_stats: st.plotly_chart(fig_stats, use_container_width=True)

    st.subheader("Consultas SQL Personalizadas")
    query = st.text_area("Escribe tu consulta SQL:", f"SELECT * FROM {quality_data['dataset_name']} LIMIT 5")
    if st.button("Ejecutar Consulta"):
        try:
            conn = sqlite3.connect('data_quality.db')
            result_df = pd.read_sql_query(query, conn)
            st.dataframe(result_df, use_container_width=True)
            conn.close()
        except Exception as e:
            st.error(f"Error en la consulta: {str(e)}")

# ----- Main -----
def main():
    setup_page()
    with st.sidebar:
        st.header("📂 Cargar Dataset")
        uploaded_file = st.file_uploader("Sube un archivo CSV", type="csv")
        st.header("⚙️ Opciones")
        show_raw_data = st.checkbox("Mostrar datos crudos")
        st.header("ℹ️ Información")
        st.info("""
        Este dashboard analiza la calidad de tus datos y muestra:
        - Valores nulos y duplicados
        - Distribución de tipos de datos
        - Completitud por columna
        - Correlaciones entre variables
        - Histogramas, boxplots, frecuencias y medidas estadísticas
        """)

    if uploaded_file is not None:
        try:
            dataset_name = os.path.splitext(uploaded_file.name)[0]
            df = read_csv_with_autodetection(uploaded_file)
            if df is not None and not df.empty:
                with st.spinner("Analizando calidad de datos..."):
                    quality_data = save_to_database(df, dataset_name)
                if quality_data is not None:
                    if show_raw_data:
                        st.subheader("Datos Crudos")
                        st.dataframe(df, use_container_width=True)
                    display_quality_dashboard(quality_data, df)
        except Exception as e:
            st.error(f"Error al procesar el archivo: {str(e)}")
    else:
        st.info("👈 Sube un archivo CSV desde la barra lateral para comenzar el análisis.")

if __name__ == "__main__":
    main()

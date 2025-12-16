import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re

st.set_page_config(
    page_title="Animal Disease News Dashboard",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: bold;
    }
    div[data-testid="stMetricLabel"] {
        color: #ffffff !important;
        font-weight: bold;
    }
    .stMetric {
        background-color: #e74c3c;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: none;
    }
    h1, h2, h3 {
        color: #1f2c39;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🐾 Animal Disease News Analytics")
st.markdown("### Comprehensive analysis of animal disease news coverage")
st.markdown("---")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('dataset.csv')
        if 'Nb Mots' in df.columns:
            df['Nb Mots'] = pd.to_numeric(df['Nb Mots'], errors='coerce').fillna(0)
        if 'Nb Caractères' in df.columns:
            df['Nb Caractères'] = pd.to_numeric(df['Nb Caractères'], errors='coerce').fillna(0)
        if 'Date Publication' in df.columns:
            df['Date_Obj'] = pd.to_datetime(df['Date Publication'], format='%d-%m-%Y', errors='coerce')
        return df
    except FileNotFoundError:
        return None

df = load_data()

if df is not None:
    st.sidebar.header("🔍 Filters")
    
    if 'Date_Obj' in df.columns:
        min_date = df['Date_Obj'].min()
        max_date = df['Date_Obj'].max()
        if pd.notnull(min_date) and pd.notnull(max_date):
            filter_option = st.sidebar.radio("Quick Date Filter", ["All Time", "Last 30 Days", "Custom Range"])
            
            if filter_option == "Last 30 Days":
                start_date = max_date - pd.Timedelta(days=30)
                end_date = max_date
                df = df[(df['Date_Obj'] >= start_date) & (df['Date_Obj'] <= end_date)]
            elif filter_option == "Custom Range":
                start_date, end_date = st.sidebar.date_input(
                    "Select Date Range",
                    [min_date, max_date],
                    min_value=min_date,
                    max_value=max_date
                )
                df = df[(df['Date_Obj'] >= pd.to_datetime(start_date)) & (df['Date_Obj'] <= pd.to_datetime(end_date))]

    if 'Langue' in df.columns:
        languages = df['Langue'].unique().tolist()
        selected_langs = st.sidebar.multiselect("Select Language", languages, default=languages)
        if selected_langs:
            df = df[df['Langue'].isin(selected_langs)]

    if 'Domaine' in df.columns:
        domains = df['Domaine'].unique().tolist()
        selected_domains = st.sidebar.multiselect("Select Domain", domains)
        if selected_domains:
            df = df[df['Domaine'].isin(selected_domains)]

    st.subheader("📊 Key Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Articles", len(df))
    with col2:
        avg_words = int(df['Nb Mots'].mean()) if 'Nb Mots' in df.columns else 0
        st.metric("Avg Word Count", f"{avg_words} words")
    with col3:
        max_words = int(df['Nb Mots'].max()) if 'Nb Mots' in df.columns else 0
        st.metric("Max Word Count", f"{max_words} words")
    with col4:
        unique_sources = df['Domaine'].nunique() if 'Domaine' in df.columns else 0
        st.metric("Unique Domains", unique_sources)

    st.markdown("---")

    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("📈 Publication Trend")
        if 'Date_Obj' in df.columns:
            df_time = df.dropna(subset=['Date_Obj']).groupby('Date_Obj').size().reset_index(name='Count')
            fig_time = px.line(df_time, x='Date_Obj', y='Count', 
                               title="News Volume Over Time",
                               labels={'Date_Obj': 'Date', 'Count': 'Number of Articles'},
                               markers=True,
                               line_shape='spline',
                               color_discrete_sequence=['#2980b9'])
            fig_time.update_layout(xaxis_rangeslider_visible=True, plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info("No date information available.")

    with c2:
        st.subheader("🌐 Language Share")
        if 'Langue' in df.columns:
            fig_lang = px.pie(df, names='Langue', hole=0.4, 
                              title="Languages Used",
                              color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_lang, use_container_width=True)

    c3, c4 = st.columns(2)
    
    with c3:
        st.subheader("📰 Top News Sources (Domains)")
        if 'Domaine' in df.columns:
            domain_counts = df['Domaine'].value_counts().head(15).reset_index()
            domain_counts.columns = ['Domaine', 'Count']
            fig_domain = px.bar(domain_counts, x='Count', y='Domaine', orientation='h',
                                title="Top 15 Domains",
                                color='Count', color_continuous_scale='Blues')
            fig_domain.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_domain, use_container_width=True)

    with c4:
        st.subheader("📏 Article Length Stats")
        if 'Nb Mots' in df.columns:
            fig_words = px.box(df, y='Nb Mots', points="all",
                               title="Word Count Distribution (Min/Max/Avg)",
                               color_discrete_sequence=['#e74c3c'])
            st.plotly_chart(fig_words, use_container_width=True)

    st.subheader("🔠 Content Analysis (Top Keywords)")
    if 'Contenu' in df.columns:
        all_text = " ".join(df['Contenu'].dropna().astype(str))
        stopwords = set(['le', 'la', 'les', 'de', 'des', 'du', 'et', 'en', 'un', 'une', 'est', 'a', 'dans', 'pour', 'sur', 'au', 'aux', 'par', 'ce', 'qui', 'que', 'the', 'and', 'of', 'to', 'in', 'is', 'for', 'on', 'with', 'as', 'at', 'by', 'il', 'elle', 'sont', 'pas', 'plus', 'cette', 'avec', 'nous', 'vous', 'ils', 'elles', 'mais', 'ou', 'si', 'ne', 'se', 'son', 'sa', 'ses', 'leur', 'leurs', 'y', 'aussi', 'très', 'tout', 'tous', 'toute', 'toutes', 'sans', 'dont', 'où', 'ni', 'car', 'donc', 'or', 'ni', 'car', 'après', 'avant', 'depuis', 'pendant', 'vers', 'chez', 'entre', 'sous', 'contre', 'vers', 'dès', 'hors', 'selon', 'sauf', 'malgré', 'parmi', 'afin', 'ainsi', 'alors', 'après', 'assez', 'aucun', 'aussi', 'autre', 'avant', 'avec', 'beaucoup', 'car', 'ceci', 'cela', 'celle', 'celui', 'cent', 'cependant', 'chacun', 'chaque', 'combien', 'comme', 'comment', 'dans', 'déjà', 'depuis', 'devant', 'donc', 'dont', 'durant', 'elle', 'elles', 'en', 'encore', 'entre', 'envers', 'est', 'et', 'eu', 'eux', 'hormis', 'ici', 'il', 'ils', 'je', 'jusqu', 'la', 'laquelle', 'le', 'lequel', 'les', 'lesquelles', 'lesquels', 'leur', 'leurs', 'lors', 'lui', 'maintenant', 'mais', 'malgré', 'me', 'même', 'mes', 'mien', 'mienne', 'miennes', 'miens', 'moi', 'moins', 'mon', 'moyennant', 'ne', 'néanmoins', 'ni', 'non', 'nos', 'notre', 'nôtre', 'nôtres', 'nous', 'on', 'ou', 'où', 'outre', 'par', 'parmi', 'partant', 'pas', 'pendant', 'peu', 'plus', 'plusieurs', 'pour', 'pourquoi', 'près', 'puis', 'qu', 'quand', 'que', 'quel', 'quelle', 'quelles', 'quels', 'qui', 'quiconque', 'quoi', 'quoique', 'rien', 'sa', 'sans', 'sauf', 'se', 'selon', 'ses', 'si', 'sien', 'sienne', 'siennes', 'siens', 'sinon', 'soi', 'son', 'sont', 'sous', 'soyez', 'sur', 'ta', 'tandis', 'te', 'tel', 'telle', 'telles', 'tels', 'tes', 'tien', 'tienne', 'tiennes', 'tiens', 'toi', 'ton', 'tous', 'tout', 'toute', 'toutes', 'tu', 'un', 'une', 'vôtre', 'vôtres', 'vous', 'vu'])
        
        words = re.findall(r'\w+', all_text.lower())
        filtered_words = [w for w in words if w not in stopwords and len(w) > 3]
        word_counts = Counter(filtered_words).most_common(20)
        
        df_words = pd.DataFrame(word_counts, columns=['Word', 'Count'])
        fig_words_bar = px.bar(df_words, x='Word', y='Count', 
                               title="Top 20 Frequent Words (Content)",
                               color='Count', color_continuous_scale='Reds')
        st.plotly_chart(fig_words_bar, use_container_width=True)

    c5, c6 = st.columns(2)
    
    with c5:
        st.subheader("🦠 Top Diseases Mentioned")
        if 'Maladie' in df.columns:
            df_disease = df[df['Maladie'] != 'Unknown']
            if not df_disease.empty:
                disease_counts = df_disease['Maladie'].value_counts().head(10).reset_index()
                disease_counts.columns = ['Maladie', 'Count']
                fig_disease = px.bar(disease_counts, x='Count', y='Maladie', orientation='h',
                                     title="Most Frequent Diseases (Known)",
                                     color='Count', color_continuous_scale='Viridis')
                st.plotly_chart(fig_disease, use_container_width=True)
            else:
                st.info("No specific diseases identified yet.")

    with c6:
        st.subheader("📍 Geographic Focus")
        if 'Lieu' in df.columns:
            df_loc = df[df['Lieu'] != 'Unknown']
            if not df_loc.empty:
                loc_counts = df_loc['Lieu'].value_counts().head(10).reset_index()
                loc_counts.columns = ['Lieu', 'Count']
                fig_loc = px.bar(loc_counts, x='Lieu', y='Count',
                                 title="Top Locations (Known)",
                                 color_discrete_sequence=['#9b59b6'])
                st.plotly_chart(fig_loc, use_container_width=True)
            else:
                st.info("No specific locations identified yet.")

    st.markdown("---")
    st.subheader("🗃️ Data Explorer")
    # Data Explorer with improved UX
    st.subheader("🗃️ Data Explorer with Summaries")
    
    # Select columns to show by default
    default_cols = ['Titre', 'Maladie', 'Lieu', 'Date Publication', 'Source']
    if 'Résumé 100' in df.columns:
        default_cols.append('Résumé 100')
    
    # Allow user to pick an article to view details
    selected_idx = st.selectbox("Select an article to view summary:", options=df.index, format_func=lambda x: f"{df.loc[x, 'Titre'][:80]}..." if 'Titre' in df.columns else f"Article {x}")
    
    if selected_idx is not None:
        row = df.loc[selected_idx]
        with st.container():
            st.markdown(f"### {row['Titre'] if 'Titre' in df.columns else 'Untitled'}")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.info(f"**Disease:** {row['Maladie'] if 'Maladie' in df.columns else 'N/A'}")
                st.write(f"**Date:** {row['Date Publication'] if 'Date Publication' in df.columns else 'N/A'}")
            with col_b:
                st.success(f"**Location:** {row['Lieu'] if 'Lieu' in df.columns else 'N/A'}")
                st.write(f"**Source:** {row['Source'] if 'Source' in df.columns else 'N/A'}")
            
            st.markdown("#### 📝 Summary")
            if 'Résumé 100' in df.columns:
                st.markdown(f"> {row['Résumé 100']}")
            else:
                st.warning("No summary available.")
                
            with st.expander("View Full Content"):
                st.write(row['Contenu'] if 'Contenu' in df.columns else "No content.")
                if 'URL' in df.columns:
                    st.markdown(f"[Read Original Article]({row['URL']})")

    with st.expander("View Raw Data Table"):
        st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Dataset CSV",
        data=csv,
        file_name='animal_disease_news_enhanced.csv',
        mime='text/csv',
    )

else:
    st.error("Dataset not found. Please run the extraction script first.")
    st.code("python extract_data.py", language="bash")

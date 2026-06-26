"""
=========================================================================================
🔥 THE UNIVERSAL DATA-AGNOSTIC BIAS & ML SUITE WITH INTEGRATED CHATBOX PREDICTOR 🔥
=========================================================================================
Features:
- 100% Structural Fluidity: Automatically adjusts titles, columns, and text logic.
- Intelligent Target Discovery: Self-detects your prediction goals (Attrition, Churn, Status).
- Dynamic Theming: Strips light text elements to avoid contrast bugs on Streamlit Cloud.
- Embedded Model Assistant Chatbox: Run textual risk checks and get real-time charts.
"""

import warnings
warnings.filterwarnings("ignore")
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Forces headless plot rendering to prevent server thread locks
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# ---------------------------------------------------------------------------------------
# I. GLOBAL THEME DESIGN & BASE LAYOUT INITIALIZATION
# ---------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Universal Business Intelligence & Super-Learning Suite",
    layout="wide",
    page_icon="⚖️",
    initial_sidebar_state="expanded"
)

# Custom premium styling matrix maximizing visibility (No white-text bugs)
st.markdown("""
    <style>
    .reportview-container { background-color: #F8FAFC; }
    .dynamic-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%);
        padding: 30px;
        border-radius: 12px;
        color: #FFFFFF !important;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 10px rgba(15, 23, 42, 0.15);
    }
    .dynamic-header h1, .dynamic-header p { color: #FFFFFF !important; }
    .kpi-card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 10px;
        border-top: 5px solid #2563EB;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        text-align: center;
    }
    .dynamic-insight-card {
        background-color: #FAFAFA;
        border-left: 6px solid #2563EB;
        padding: 20px;
        border-radius: 8px;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .dynamic-insight-card h4, .dynamic-insight-card p, .dynamic-insight-card li { color: #1E293B !important; }
    .anomaly-alert-card {
        background-color: #FEF2F2;
        border-left: 6px solid #DC2626;
        padding: 22px;
        border-radius: 8px;
        margin-top: 15px;
        margin-bottom: 20px;
        color: #7F1D1D !important;
    }
    .anomaly-alert-card h4, .anomaly-alert-card p, .anomaly-alert-card li { color: #7F1D1D !important; }
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: white !important;
        font-weight: bold;
        border: none;
        padding: 14px 35px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------------------
# II. INTELLIGENT SCHEMA DETECTION ENGINE
# ---------------------------------------------------------------------------------------
def auto_discover_schema(dataframe):
    """
    Analyzes generic tabular headers and content types to discover target vectors,
    demographics, geographic segments, and continuous variables dynamically.
    """
    schema = {
        'target_col': None,
        'age_col': None,
        'income_col': None,
        'group_col': None,
        'numeric_features': [],
        'categorical_features': []
    }
    
    cols = dataframe.columns.tolist()
    
    # 1. Isolate the target classification dimension
    for c in cols:
        c_upper = str(c).upper().strip()
        if any(keyword in c_upper for keyword in ['STATUS', 'ATTRITION', 'CHURN', 'TARGET', 'OUTPUT', 'DEFAULT']):
            schema['target_col'] = c
            break
    if not schema['target_col']:
        schema['target_col'] = cols[-1]
        
    # 2. Map structural evaluation vectors
    for c in cols:
        if c == schema['target_col']: continue
        c_upper = str(c).upper().strip()
        
        if any(k in c_upper for k in ['AGE', 'BIRTH']):
            schema['age_col'] = c
        elif any(k in c_upper for k in ['INCOME', 'SALARY', 'EARNING', 'RATE', 'MONTHLYINCOME', 'DAILYRATE', 'HOURLYRATE']):
            schema['income_col'] = c
        elif any(k in c_upper for k in ['ZONE', 'DEPARTMENT', 'TEAM', 'BRANCH', 'OFFICE', 'LOCATION', 'STATE', 'JOBROLE']):
            schema['group_col'] = c
            
    # 3. Separate continuous vs categorical variables for machine learning models
    for c in cols:
        if c == schema['target_col']: continue
        if pd.api.types.is_numeric_dtype(dataframe[c]):
            schema['numeric_features'].append(c)
        else:
            schema['categorical_features'].append(c)
            
    # Fallback group bindings if schema patterns yield nothing
    if not schema['age_col'] and schema['numeric_features']: schema['age_col'] = schema['numeric_features'][0]
    if not schema['income_col'] and len(schema['numeric_features']) > 1: schema['income_col'] = schema['numeric_features'][1]
    if not schema['group_col'] and schema['categorical_features']: schema['group_col'] = schema['categorical_features'][0]
    
    return schema

# ---------------------------------------------------------------------------------------
# III. DATA UTILITY CLEANING LAYER
# ---------------------------------------------------------------------------------------
@st.cache_data
def universal_cleanse_pipeline(file_source):
    if isinstance(file_source, str):
        try:
            raw_df = pd.read_csv(file_source)
        except FileNotFoundError:
            return None, None
    else:
        raw_df = pd.read_csv(file_source)
        
    df_clean = raw_df.copy()
    
    # Strip random special notations from numeric elements globally
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            sample_stripped = df_clean[col].astype(str).str.replace(r'[,\s\$₹]', '', regex=True)
            if sample_stripped.str.isnumeric().sum() / len(df_clean) > 0.8:
                df_clean[col] = pd.to_numeric(sample_stripped, errors='coerce')
                
    # Fill remaining text empty records
    for col in df_clean.select_dtypes(include=['object', 'category']).columns:
        df_clean[col] = df_clean[col].fillna('UNKNOWN').astype(str).str.strip().str.upper()
        
    schema = auto_discover_schema(df_clean)
    
    return df_clean, schema

# ---------------------------------------------------------------------------------------
# IV. WORKSPACE SETUP & RUNTIME SIDEBAR CONFIGURATIONS
# ---------------------------------------------------------------------------------------
st.sidebar.markdown("### 📁 Universal Data Loader")
uploaded_file = st.sidebar.file_uploader("Upload Any Business Database (CSV Format)", type=["csv"])

if uploaded_file is not None:
    df, schema = universal_cleanse_pipeline(uploaded_file)
else:
    # Auto-load sequence based on file priority
    df, schema = universal_cleanse_pipeline("EA.csv")
    if df is None:
        df, schema = universal_cleanse_pipeline("Insurance.csv")

if df is None:
    st.error("❌ Base configuration template ledger missing! Please upload a data CSV via the sidebar container.")
    st.stop()

# Extract discovered tracking coordinates
target_var = schema['target_col']
age_var = schema['age_col']
income_var = schema['income_col']
group_var = schema['group_col']

# Compute dynamic label contexts automatically
unique_target_classes = df[target_var].unique().tolist()
risk_class_counts = df[target_var].value_counts()
risk_anchor_target = risk_class_counts.index[-1] if len(risk_class_counts) > 1 else unique_target_classes[0]
safe_anchor_target = risk_class_counts.index[0] if len(risk_class_counts) > 1 else unique_target_classes[0]

# Dynamic header rendering based on parsed context
clean_domain_name = str(target_var).replace('_', ' ').title()
st.markdown(f'<div class="dynamic-header"><h1>⚖️ Universal Enterprise {clean_domain_name} & Risk Auditing Engine</h1><p>Data-agnostic analytical pipeline running dynamic behavioral auditing against target metric: <b>{target_var}</b></p></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------------------
# V. HIGH-CONTRAST DATA INFRASTRUCTURE KPI TILES
# ---------------------------------------------------------------------------------------
total_records = len(df)
total_risk_cases = len(df[df[target_var] == risk_anchor_target])
base_risk_rate = (total_risk_cases / total_records) * 100 if total_records > 0 else 0

kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
with kpi_c1:
    st.markdown(f'<div class="kpi-card"><p style="color:#64748B;font-size:12px;margin:0;font-weight:700;">TOTAL MATRICES RECORDED</p><h2 style="color:#0F172A;margin:5px 0;">{total_records:,}</h2></div>', unsafe_allow_html=True)
with kpi_c2:
    st.markdown(f'<div class="kpi-card" style="border-top-color:#EF4444;"><p style="color:#64748B;font-size:12px;margin:0;font-weight:700;">RISK STATE INSTANCES ({risk_anchor_target})</p><h2 style="color:#EF4444;margin:5px 0;">{total_risk_cases:,}</h2><span style="color:#EF4444;font-size:12px;font-weight:bold;">{base_risk_rate:.2f}% Baseline Rate</span></div>', unsafe_allow_html=True)
with kpi_c3:
    st.markdown(f'<div class="kpi-card" style="border-top-color:#10B981;"><p style="color:#64748B;font-size:12px;margin:0;font-weight:700;">STABLE STATE INSTANCES ({safe_anchor_target})</p><h2 style="color:#10B981;margin:5px 0;">{total_records - total_risk_cases:,}</h2><span style="color:#10B981;font-size:12px;font-weight:bold;">{100-base_risk_rate:.2f}% Concentration</span></div>', unsafe_allow_html=True)
with kpi_c4:
    unique_operational_segments = df[group_var].nunique() if group_var else 1
    st.markdown(f'<div class="kpi-card" style="border-top-color:#F59E0B;"><p style="color:#64748B;font-size:12px;margin:0;font-weight:700;">DISTINCT LOGICAL SEGMENTS</p><h2 style="color:#F59E0B;margin:5px 0;">{unique_operational_segments} Groups</h2><span style="color:#64748B;font-size:12px;">Categorized under {group_var}</span></div>', unsafe_allow_html=True)

# Main Navigation Setup Tabs
tab_descriptive, tab_diagnostic, tab_modeling, tab_findings = st.tabs([
    "📋 Descriptive Slices", "🔍 Disparity Diagnostics", "🤖 Super-Learning Classifiers", "💡 Dynamic Executive Summary"
])

# =======================================================================================
# TAB 1: DATA-AGNOSTIC DESCRIPTIVE SLICES
# =======================================================================================
with tab_descriptive:
    st.header(f"📋 Cross-Tabulation Breakdown vs {target_var}")
    selected_slice = st.selectbox("Select Column to Analyze Distribution Proportions:", schema['categorical_features'] if schema['categorical_features'] else df.columns.tolist())
    
    if selected_slice:
        count_ct = pd.crosstab(df[selected_slice], df[target_var])
        pct_ct = pd.crosstab(df[selected_slice], df[target_var], normalize='index') * 100
        
        split1, split2 = st.columns(2)
        with split1:
            st.markdown("#### Case Allocation Volume Profile")
            st.dataframe(count_ct.style.background_gradient(cmap='Blues', axis=0))
        with split2:
            st.markdown("#### Proportional Density Slices within Sub-Group (%)")
            st.dataframe(pct_ct.style.format("{:.2f}%").background_gradient(cmap='YlOrRd', axis=1))
            
        fig_proportions = px.bar(df, x=selected_slice, color=target_var, barmode='group',
                                 color_discrete_sequence=px.colors.qualitative.Bold,
                                 title=f"Volumetric Profile Visual Distribution: {selected_slice}")
        fig_proportions.update_layout(xaxis={'categoryorder': 'total descending'}, template="plotly_white")
        st.plotly_chart(fig_proportions, use_container_width=True)

# =======================================================================================
# TAB 2: DATA-AGNOSTIC DIAGNOSTIC BIAS PROBING
# =======================================================================================
with tab_diagnostic:
    st.header("🔍 Dynamic Auditing Disparity Probe Engine")
    probe_selection = st.radio("Isolate Auditing Category Variable Vector Layer:", ["Primary Continuous Feature", "Secondary Continuous Feature", "Segmented Grouping Array"], horizontal=True)
    
    if probe_selection == "Primary Continuous Feature" and age_var:
        st.subheader(f"Auditing Disparity Profile: `{age_var}`")
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            fig_box = px.box(df, x=target_var, y=age_var, color=target_var, points="outliers", color_discrete_sequence=['#0284C7', '#EF4444'])
            fig_box.update_layout(template="plotly_white")
            st.plotly_chart(fig_box, use_container_width=True)
        with col_a2:
            df['DYNAMIC_QUANTILE_BINS'] = pd.qcut(df[age_var], q=min(4, df[age_var].nunique()), duplicates='drop').astype(str)
            q_ct = pd.crosstab(df['DYNAMIC_QUANTILE_BINS'], df[target_var], normalize='index') * 100
            fig_q_bar = px.bar(q_ct.reset_index(), x='DYNAMIC_QUANTILE_BINS', y=q_ct.columns.tolist(), barmode='stack', color_discrete_sequence=['#0284C7', '#EF4444'])
            fig_q_bar.update_layout(template="plotly_white")
            st.plotly_chart(fig_q_bar, use_container_width=True)
            
    elif probe_selection == "Secondary Continuous Feature" and income_var:
        st.subheader(f"Auditing Disparity Profile: `{income_var}`")
        col_i1, col_i2 = st.columns(2)
        with col_i1:
            fig_inc_box = px.box(df, x=target_var, y=income_var, color=target_var, points="outliers", color_discrete_sequence=['#10B981', '#EF4444'])
            fig_inc_box.update_layout(template="plotly_white")
            st.plotly_chart(fig_inc_box, use_container_width=True)
        with col_i2:
            mean_financials = df.groupby(target_var)[income_var].mean().reset_index()
            fig_mean_inc = px.bar(mean_financials, x=target_var, y=income_var, color=target_var, color_discrete_sequence=['#10B981', '#EF4444'])
            fig_mean_inc.update_layout(template="plotly_white")
            st.plotly_chart(fig_mean_inc, use_container_width=True)
            
    elif probe_selection == "Segmented Grouping Array" and group_var:
        st.subheader(f"Auditing Disparity Profile: `{group_var}`")
        group_ct_pct = pd.crosstab(df[group_var], df[target_var], normalize='index') * 100
        if risk_anchor_target in group_ct_pct.columns:
            group_ct_pct_sorted = group_ct_pct.sort_values(by=risk_anchor_target, ascending=False)
        else:
            group_ct_pct_sorted = group_ct_pct
        
        fig_grp = px.bar(group_ct_pct_sorted.reset_index(), x=group_var, y=group_ct_pct_sorted.columns.tolist(), barmode='stack', color_discrete_sequence=['#0284C7', '#EF4444'])
        fig_grp.update_layout(template="plotly_white")
        st.plotly_chart(fig_grp, use_container_width=True)

# =======================================================================================
# TAB 3: UNIVERSAL MACHINE LEARNING CLASSIFIERS
# =======================================================================================
if 'dynamic_ml_memory' not in st.session_state:
    st.session_state.dynamic_ml_memory = None
if 'model_pipeline_instance' not in st.session_state:
    st.session_state.model_pipeline_instance = None
if 'trained_champion_model' not in st.session_state:
    st.session_state.trained_champion_model = None

with tab_modeling:
    st.header("🤖 Automated Machine Learning Pipeline Engine")
    holdout_size = st.slider("Validation Train/Test Split Layer Holdout Size Selection (% Split)", 10, 50, 30) / 100.0
    
    if st.button("🚀 Train Model Classifiers Across Active Data Schema", type="primary"):
        with st.spinner("Processing feature extraction pipelines, encoding arrays, and optimizing hyperparameters..."):
            ml_df = df.copy()
            if 'DYNAMIC_QUANTILE_BINS' in ml_df.columns: ml_df = ml_df.drop(columns=['DYNAMIC_QUANTILE_BINS'])
            
            le_tgt = LabelEncoder()
            ml_df[target_var] = le_tgt.fit_transform(ml_df[target_var].astype(str))
            
            X = ml_df.drop(columns=[target_var])
            y = ml_df[target_var]
            
            num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            num_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
            cat_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
            
            transformer_block = ColumnTransformer(transformers=[('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=holdout_size, random_state=42, stratify=y)
            
            X_train_proc = transformer_block.fit_transform(X_train)
            X_test_proc = transformer_block.transform(X_test)
            
            classifiers = {
                'KNN Classifier': KNeighborsClassifier(n_neighbors=5, weights='distance'),
                'Decision Tree Classifier': DecisionTreeClassifier(random_state=42, max_depth=6, class_weight='balanced'),
                'Random Forest Classifier': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=8, class_weight='balanced'),
                'Gradient Boosting Classifier': GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=0.1, max_depth=4)
            }
            
            metrics_store = []
            roc_store = {}
            cm_store = {}
            
            target_risk_index = le_tgt.transform([str(risk_anchor_target).upper()])[0] if str(risk_anchor_target).upper() in le_tgt.classes_ else 1
            
            best_f1 = -1
            champion_name = None
            champion_model_instance = None
            
            for name, clf in classifiers.items():
                clf.fit(X_train_proc, y_train)
                preds = clf.predict(X_test_proc)
                train_preds = clf.predict(X_train_proc)
                
                probs = clf.predict_proba(X_test_proc)[:, 1] if hasattr(clf, "predict_proba") else preds
                
                f1 = f1_score(y_test, preds, pos_label=target_risk_index, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    champion_name = name
                    champion_model_instance = clf
                
                metrics_store.append({
                    'Model Configuration': name,
                    'Train Accuracy': accuracy_score(y_train, train_preds),
                    'Test Accuracy': accuracy_score(y_test, preds),
                    'Precision': precision_score(y_test, preds, pos_label=target_risk_index, zero_division=0),
                    'Recall': recall_score(y_test, preds, pos_label=target_risk_index, zero_division=0),
                    'F1-Score': f1
                })
                
                fpr, tpr, _ = roc_curve(y_test, probs, pos_label=target_risk_index)
                roc_store[name] = (fpr, tpr, auc(fpr, tpr))
                cm_store[name] = confusion_matrix(y_test, preds)
                
            st.session_state.dynamic_ml_memory = {
                'table': pd.DataFrame(metrics_store), 'roc': roc_store, 'cm': cm_store, 'labels': le_tgt.classes_.tolist()
            }
            st.session_state.model_pipeline_instance = transformer_block
            st.session_state.trained_champion_model = champion_model_instance
            st.session_state.ml_features_columns = X.columns.tolist()
            st.session_state.target_label_encoder_instance = le_tgt

    if st.session_state.dynamic_ml_memory is not None:
        cache = st.session_state.dynamic_ml_memory
        st.subheader("📊 Performance Matrix Metrics")
        st.dataframe(cache['table'].style.format({
            'Train Accuracy': "{:.2%}", 'Test Accuracy': "{:.2%}", 'Precision': "{:.2%}", 'Recall': "{:.2%}", 'F1-Score': "{:.2%}"
        }).background_gradient(cmap='Blues'))
        
        fig_metrics = px.bar(cache['table'].melt(id_vars='Model Configuration'), x='Model Configuration', y='value', color='variable', barmode='group', template='plotly_white')
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        cm_col1, cm_col2 = st.columns(2)
        with cm_col1:
            st.markdown("#### 🔄 ROC Curve Stability Trajectories")
            fig_roc = go.Figure()
            for name, (fpr, tpr, area) in cache['roc'].items():
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{name} (AUC = {area:.3f})"))
            fig_roc.update_layout(template='plotly_white', xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(fig_roc, use_container_width=True)
        with cm_col2:
            st.markdown("#### 🧩 Confusion Matrices Slices")
            fig_plt, axes = plt.subplots(2, 2, figsize=(10, 8))
            axes = axes.ravel()
            for idx, (name, matrix) in enumerate(cache['cm'].items()):
                sns.heatmap(matrix, annot=True, fmt='d', ax=axes[idx], cmap='Blues', xticklabels=cache['labels'], yticklabels=cache['labels'], cbar=False)
                axes[idx].set_title(name, fontsize=10, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig_plt)

# =======================================================================================
# TAB 4: 100% COMPLETELY DYNAMIC TEXT BRIEF FINDINGS
# =======================================================================================
with tab_findings:
    st.markdown("### 💡 Automated Strategic Briefing & Action Register")
    
    live_age_tab = pd.crosstab(df['DYNAMIC_QUANTILE_BINS'], df[target_var], normalize='index') * 100 if 'DYNAMIC_QUANTILE_BINS' in df.columns else pd.DataFrame()
    live_group_ct = pd.crosstab(df[group_var], df[target_var], normalize='index') * 100 if group_var else pd.DataFrame()
    
    st.markdown('<div class="anomaly-alert-card">', unsafe_allow_html=True)
    st.markdown('<h4>📌 Identified Operational Risk Divergences</h4>', unsafe_allow_html=True)
    
    if not live_group_ct.empty and risk_anchor_target in live_group_ct.columns:
        worst_grp = live_group_ct.sort_values(by=risk_anchor_target, ascending=False).index[0]
        worst_grp_val = live_group_ct.sort_values(by=risk_anchor_target, ascending=False)[risk_anchor_target].iloc[0]
        st.markdown(f"<li><b>Segment Disparity Filter Risk:</b> Case evaluation analysis indicates non-uniform distribution properties across organizational layers. The tracking node group identified as <b>{worst_grp}</b> holds the highest relative risk signature profile, displaying a <b>{worst_grp_val:.2f}% validation frequency for the {risk_anchor_target} risk track state</b>. This configuration highlights a key area for operational standardization.</li>", unsafe_allow_html=True)
        
    if income_var:
        m_risk = df[df[target_var] == risk_anchor_target][income_var].mean()
        m_stable = df[df[target_var] == safe_anchor_target][income_var].mean() if len(df[df[target_var] == safe_anchor_target]) > 0 else 1
        st.markdown(f"<li><b>Continuous Metric Concentration Filter:</b> Records associated with the <b>{risk_anchor_target}</b> risk path carry an average metrics profile value for `{income_var}` of <b>{m_risk:,.2f}</b>, compared to stable configurations averaging a profile of <b>{m_stable:,.2f}</b>.</li>", unsafe_allow_html=True)
        
    if not live_age_tab.empty and risk_anchor_target in live_age_tab.columns:
        worst_age = live_age_tab.sort_values(by=risk_anchor_target, ascending=False).index[0]
        worst_age_val = live_age_tab.sort_values(by=risk_anchor_target, ascending=False)[risk_anchor_target].iloc[0]
        st.markdown(f"<li><b>Profile Bracket Disparities:</b> Maximum frequency concentrations matching the critical risk outcome state are localized inside the <b>{worst_age}</b> cohort layer, tracking at a <b>{worst_age_val:.2f}% frequency baseline</b>.</li>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<h4>🛠️ Strategic Action Roadmap</h4>')
    st.markdown(f"""
    1. **Enforce Uniform Operational Thresholds:** Review and align evaluation criteria across all segments under `{group_var}` to reduce localized standard deviations.
    2. **Flag Disproportionate Continuous Baselines:** Deploy secondary automated screening workflows on cases falling within high-risk ranges of `{income_var}` to counter structural validation biases.
    3. **Integrate Algorithmic Validation Loops:** Use the top-performing model as an independent secondary audit tool to flag high-probability risk classifications for proactive human review.
    """)

# =======================================================================================
# VII. INTEGRATED REAL-TIME ML CHATBOX PREDICTOR ASSISTANT
# =======================================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 💬 Real-Time Prediction Chatbox")
user_query = st.sidebar.text_input("Ask Assistant (e.g., 'Who will leave?', 'Check high risk'):")

if user_query:
    if st.session_state.trained_champion_model is None:
        st.sidebar.warning("⚠️ Train models on Tab 3 first to enable the chat interface.")
    else:
        clf = st.session_state.trained_champion_model
        proc_pipe = st.session_state.model_pipeline_instance
        feat_cols = st.session_state.ml_features_columns
        le_tgt = st.session_state.target_label_encoder_instance
        
        X_active = df[feat_cols]
        X_active_proc = proc_pipe.transform(X_active)
        
        raw_preds = clf.predict(X_active_proc)
        raw_probs = clf.predict_proba(X_active_proc)[:, 1] if hasattr(clf, "predict_proba") else raw_preds
        
        target_risk_idx_val = le_tgt.transform([str(risk_anchor_target).upper()])[0] if str(risk_anchor_target).upper() in le_tgt.classes_ else 1
        risk_indices = np.where(raw_preds == target_risk_idx_val)[0]
        
        st.sidebar.success(f"🎯 Query Processed via {type(clf).__name__} Engine!")
        st.sidebar.markdown(f"**Identified High-Risk Sub-Entities Count:** `{len(risk_indices)} instances` found inside active data table array.")
        
        if len(risk_indices) > 0:
            high_risk_subset_df = df.iloc[risk_indices].copy()
            high_risk_subset_df['RISK_SCORE_%'] = raw_probs[risk_indices] * 100
            high_risk_subset_df = high_risk_subset_df.sort_values(by='RISK_SCORE_%', ascending=False)
            
            st.sidebar.markdown(f"**Top High-Risk Accounts List (`{risk_anchor_target}` Track):**")
            id_key = [c for c in ['POLICY_NO', 'EmployeeNumber', 'EMPLOYEENUMBER'] if c in df.columns]
            display_cols = id_key + [group_var, age_var] if id_key else [group_var, age_var]
            st.sidebar.dataframe(high_risk_subset_df[display_cols + ['RISK_SCORE_%']].head(10).style.format({'RISK_SCORE_%': "{:.1f}%"}))
            
            st.markdown("---")
            st.markdown(f"### 📊 Chat Assistant Real-Time Analysis Grid: Isolated High-Risk Sub-Segment (`{risk_anchor_target}` Profile)")
            
            c_chat1, c_chat2 = st.columns(2)
            with c_chat1:
                fig_chat_bar = px.histogram(high_risk_subset_df, x=group_var, title=f"High-Risk Structural Density Count Slices by {group_var}", color_discrete_sequence=['#EF4444'], template='plotly_white')
                st.plotly_chart(fig_chat_bar, use_container_width=True)
            with c_chat2:
                fig_chat_scatter = px.scatter(high_risk_subset_df, x=age_var, y=income_var, size='RISK_SCORE_%', hover_data=display_cols, title=f"Risk Matrix Positioning: {age_var} vs {income_var}", color_discrete_sequence=['#EF4444'], template='plotly_white')
                st.plotly_chart(fig_chat_scatter, use_container_width=True)

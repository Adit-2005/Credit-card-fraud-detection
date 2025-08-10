# fraud_detection_visual.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from joblib import load
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

# -----------------------
# 1. Model Definition
# -----------------------
class FraudDetectionModel:
    def __init__(self):
        self.model = None
        self.is_trained = False

    def load_pretrained(self, model_path):
        """Load pre-trained model"""
        if os.path.exists(model_path):
            self.model = load(model_path)
            self.is_trained = True
            return True
        return False

    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model not loaded")
        return self.model.predict_proba(X)[:, 1]

# -----------------------
# 2. Visualization Functions
# -----------------------
def plot_geospatial_fraud(df):
    fig = px.scatter_geo(df[df['is_fraud'] == 1].sample(min(1000, len(df[df['is_fraud']==1])), random_state=42),
                         lat='lat', lon='long',
                         color='amt', size='amt',
                         hover_name='category',
                         projection='natural earth',
                         title='Geospatial Fraud Distribution')
    fig.update_layout(geo=dict(showland=True, landcolor="lightgray"))
    st.plotly_chart(fig, use_container_width=True)

def plot_time_analysis(df):
    if 'trans_date_trans_time' in df.columns:
        df['hour'] = pd.to_datetime(df['trans_date_trans_time']).dt.hour
        df['day'] = pd.to_datetime(df['trans_date_trans_time']).dt.day_name()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        sns.countplot(data=df[df['is_fraud'] == 1], x='hour', ax=ax1, palette='Reds')
        sns.countplot(data=df[df['is_fraud'] == 1], x='day',
                      order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                      ax=ax2, palette='Reds')
        ax1.set_title('Fraud by Hour of Day')
        ax2.set_title('Fraud by Day of Week')
        st.pyplot(fig)

def plot_feature_correlations(df):
    numeric_df = df[['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long', 'is_fraud']]
    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu', title='Feature Correlation Matrix')
    st.plotly_chart(fig, use_container_width=True)

def plot_amount_distribution(df):
    fig = px.violin(df, y='amt', x='is_fraud',
                    color='is_fraud', box=True,
                    title='Transaction Amount Distribution by Fraud Status')
    st.plotly_chart(fig, use_container_width=True)

def plot_category_risk(df):
    fraud_rates = df.groupby('category')['is_fraud'].mean().sort_values(ascending=False)
    fig = px.bar(fraud_rates, orientation='h',
                 title='Fraud Rate by Transaction Category',
                 color=fraud_rates.values,
                 color_continuous_scale='reds')
    st.plotly_chart(fig, use_container_width=True)

def plot_probability_density(y_true, y_prob):
    fig = px.histogram(x=y_prob, color=y_true, nbins=50,
                       marginal='rug', barmode='overlay',
                       title='Predicted Probability Distribution')
    st.plotly_chart(fig, use_container_width=True)

def plot_precision_recall(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines',
                             name=f'PR Curve (AUC={auc(recall, precision):.2f})'))
    fig.update_layout(title='Precision-Recall Curve',
                      xaxis_title='Recall',
                      yaxis_title='Precision')
    st.plotly_chart(fig, use_container_width=True)

def plot_feature_importance(model, feature_names):
    if hasattr(model.named_steps['classifier'], 'feature_importances_'):
        importances = model.named_steps['classifier'].feature_importances_
        fig = px.bar(x=importances, y=feature_names, orientation='h',
                     title='Feature Importance')
        st.plotly_chart(fig, use_container_width=True)

# -------- New Visualizations --------
def plot_fraud_hour_heatmap(df):
    if 'trans_date_trans_time' in df.columns:
        df['hour'] = pd.to_datetime(df['trans_date_trans_time']).dt.hour
        df['day'] = pd.to_datetime(df['trans_date_trans_time']).dt.day_name()
        fraud_only = df[df['is_fraud'] == 1]
        pivot = fraud_only.pivot_table(index='day', columns='hour', values='is_fraud', aggfunc='count').fillna(0)
        days_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        pivot = pivot.reindex(days_order)
        fig = px.imshow(pivot, text_auto=True, aspect="auto",
                        title="Fraud Count Heatmap (Day vs Hour)", color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)

def plot_amount_distribution_log(df):
    df = df.copy()
    df['log_amt'] = np.log1p(df['amt'])
    fig = px.histogram(df, x='log_amt', color='is_fraud',
                       nbins=50, barmode='overlay',
                       title='Transaction Amount Distribution (Log Scale)')
    st.plotly_chart(fig, use_container_width=True)

def plot_fraud_by_city_population(df):
    bins = [0, 10000, 50000, 100000, 500000, 1000000, np.inf]
    labels = ['<10k', '10k-50k', '50k-100k', '100k-500k', '500k-1M', '1M+']
    df['city_pop_group'] = pd.cut(df['city_pop'], bins=bins, labels=labels)
    fraud_rates = df.groupby('city_pop_group')['is_fraud'].mean()
    fig = px.bar(fraud_rates, x=fraud_rates.index, y=fraud_rates.values,
                 title='Fraud Rate by City Population Group', color=fraud_rates.values,
                 color_continuous_scale='reds')
    st.plotly_chart(fig, use_container_width=True)

def plot_category_vs_avg_amount(df):
    fraud_only = df[df['is_fraud'] == 1]
    avg_amounts = fraud_only.groupby('category')['amt'].mean().sort_values(ascending=False)
    fig = px.bar(avg_amounts, x=avg_amounts.index, y=avg_amounts.values,
                 title='Average Fraud Transaction Amount by Category',
                 color=avg_amounts.values, color_continuous_scale='reds')
    st.plotly_chart(fig, use_container_width=True)

def plot_fraud_rate_over_time(df):
    if 'trans_date_trans_time' in df.columns:
        df['date'] = pd.to_datetime(df['trans_date_trans_time']).dt.date
        fraud_rates = df.groupby('date')['is_fraud'].mean()
        fig = px.line(x=fraud_rates.index, y=fraud_rates.values,
                      title='Fraud Rate Over Time', labels={'x': 'Date', 'y': 'Fraud Rate'})
        st.plotly_chart(fig, use_container_width=True)

# -----------------------
# 3. Main Application
# -----------------------
def main():
    st.set_page_config(page_title="Advanced Fraud Detection", page_icon="🔍", layout="wide")
    st.title("🔍 Advanced Credit Card Fraud Detection")

    # Load model
    if 'model' not in st.session_state:
        st.session_state.model = FraudDetectionModel()
        try:
            if st.session_state.model.load_pretrained('pretrained_model.joblib'):
                st.session_state.train_df = pd.read_pickle('training_data.pkl')
                st.success("Pre-trained model loaded successfully!")
            else:
                st.warning("Could not load pre-trained model. Please train the model first.")
        except Exception as e:
            st.warning(f"Could not load model: {str(e)}")

    # Sidebar controls
    threshold = st.sidebar.slider("Fraud Threshold", 0.0, 1.0, 0.3)
    analysis_mode = st.sidebar.radio("Mode", ["Single Check", "Batch Processing", "Data Exploration"])

    # -------------------
    # Data Exploration
    # -------------------
    if analysis_mode == "Data Exploration" and 'train_df' in st.session_state:
        st.header("📊 Data Exploration")
        viz_options = {
            "Geospatial Fraud Map": plot_geospatial_fraud,
            "Temporal Analysis": plot_time_analysis,
            "Feature Correlations": plot_feature_correlations,
            "Amount Distribution": plot_amount_distribution,
            "Amount Distribution (Log Scale)": plot_amount_distribution_log,
            "Category Risk": plot_category_risk,
            "Category vs Avg Fraud Amount": plot_category_vs_avg_amount,
            "Fraud by City Population Group": plot_fraud_by_city_population,
            "Fraud Rate Over Time": plot_fraud_rate_over_time,
            "Fraud Heatmap (Day vs Hour)": plot_fraud_hour_heatmap
        }
        selected_viz = st.selectbox("Choose Visualization", list(viz_options.keys()))
        viz_options[selected_viz](st.session_state.train_df)

    # -------------------
    # Single Transaction
    # -------------------
    elif analysis_mode == "Single Check":
        st.header("🔎 Single Transaction Analysis")
        with st.form("transaction_form"):
            col1, col2 = st.columns(2)
            with col1:
                amount = st.number_input("Amount ($)", min_value=0.0, value=150.0)
                lat = st.number_input("Latitude", value=40.7128)
                long = st.number_input("Longitude", value=-74.0060)
            with col2:
                city_pop = st.number_input("City Population", min_value=0, value=8000000)
                merch_lat = st.number_input("Merchant Latitude", value=40.7135)
                merch_long = st.number_input("Merchant Longitude", value=-74.0065)
            category = st.selectbox("Category", ["entertainment", "food_dining", "gas_transport", "grocery_pos"])

            if st.form_submit_button("Check Fraud Risk"):
                transaction = pd.DataFrame([[amount, lat, long, city_pop, merch_lat, merch_long, category]],
                                           columns=['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long', 'category'])
                proba = st.session_state.model.predict(transaction)[0]
                prediction = "Fraud" if proba > threshold else "Genuine"
                col1.metric("Fraud Probability", f"{proba:.1%}")
                col2.metric("Prediction", prediction,
                            delta="High Risk" if prediction == "Fraud" else "Low Risk",
                            delta_color="inverse")
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=proba,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Fraud Risk Meter"},
                    gauge={
                        'axis': {'range': [0, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, threshold], 'color': "lightgreen"},
                            {'range': [threshold, 1], 'color': "red"}],
                        'threshold': {'value': proba}
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)

    # -------------------
    # Optimized Batch Processing
    # -------------------
    elif analysis_mode == "Batch Processing":
        st.header("📁 Batch Processing")
        uploaded_file = st.file_uploader("Upload transaction data (CSV)", type=["csv"])

        def predict_in_batches(model, df, batch_size=5000):
            results = []
            progress_bar = st.progress(0)
            total_batches = int(np.ceil(len(df) / batch_size))
            for i, start in enumerate(range(0, len(df), batch_size)):
                batch = df.iloc[start:start+batch_size]
                results.extend(model.predict(batch))
                progress_bar.progress((i+1) / total_batches)
            return np.array(results)

        if uploaded_file:
            @st.cache_data
            def batch_predict(file, threshold):
                df = pd.read_csv(file)
                df['Fraud Probability'] = predict_in_batches(st.session_state.model, df)
                df['Prediction'] = np.where(df['Fraud Probability'] > threshold, "Fraud", "Genuine")
                return df

            df = batch_predict(uploaded_file, threshold)

            fraud_count = sum(df['Prediction'] == 'Fraud')
            st.metric("Fraudulent Transactions", f"{fraud_count}/{len(df)} ({fraud_count/len(df):.1%})")

            tab1, tab2 = st.tabs(["📈 Analytics", "📋 Data"])
            with tab1:
                sample_df = df.sample(min(5000, len(df)), random_state=42)
                col1, col2 = st.columns(2)
                with col1:
                    plot_probability_density(sample_df['Prediction'] == 'Fraud', sample_df['Fraud Probability'])
                with col2:
                    plot_precision_recall(sample_df['Prediction'] == 'Fraud', sample_df['Fraud Probability'])
                if 'category' in sample_df.columns:
                    plot_category_risk(sample_df)
                if 'city_pop' in sample_df.columns:
                    plot_fraud_by_city_population(sample_df)
                plot_amount_distribution_log(sample_df)
                try:
                    feature_names = ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long'] + \
                                    list(st.session_state.model.model.named_steps['preprocessor']
                                         .named_transformers_['cat']
                                         .get_feature_names_out(['category']))
                    plot_feature_importance(st.session_state.model.model, feature_names)
                except:
                    pass
            with tab2:
                st.dataframe(df.sort_values('Fraud Probability', ascending=False).head(100))

if __name__ == "__main__":
    main()

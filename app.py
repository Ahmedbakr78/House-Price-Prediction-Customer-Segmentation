import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="House Price Prediction - 5*A by Ahmed Abobakr",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_artifacts():
    try:
        pipeline = joblib.load('models/best_pipeline.pkl')
        data = pd.read_csv('models/processed_data.csv')
        feature_names = joblib.load('models/feature_names.pkl')
        model_comparison = pd.read_csv('models/model_comparison.csv')
        return pipeline, data, feature_names, model_comparison
    except FileNotFoundError:
        st.error("model artifacts not found. please run train_model_fast.py first")
        return None, None, None, None

pipeline, df, feature_names, model_comparison = load_artifacts()

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("house price prediction")
    st.caption("5*A by ahmed abobakr")
    
    selected = option_menu(
        menu_title="navigation",
        options=["home", "predict price", "model comparison", "stacked model", "training benchmark", "data analysis", "batch prediction", "help"],
        icons=["house-fill", "calculator", "graph-up", "layers-half", "speedometer", "bar-chart-fill", "cloud-upload", "question-circle"],
        menu_icon="cast",
        default_index=0,
    )
    
    st.divider()
    
    if df is not None:
        st.metric("dataset size", f"{len(df):,}")
        st.metric("features", f"{len(feature_names['numerical']) + len(feature_names['categorical'])}")
        st.metric("best model", "xgboost")
    
    st.divider()
    st.info("explore all sections for insights")

if selected == "home":
    st.markdown('<div class="main-header">house price prediction ai</div>', unsafe_allow_html=True)
    st.markdown("### 5*A project by ahmed abobakr")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ### next-generation price prediction
        
        leveraging state-of-the-art machine learning models and advanced feature engineering 
        to provide accurate house price estimates with comprehensive market analysis
        
        #### key features:
        - multiple ml models: xgboost, lightgbm, gradient boosting
        - feature engineering: automated creation of house_age, total_sqft
        - visual analytics: interactive charts and comparisons
        - batch processing: handle hundreds of properties at once
        - explainable ai: feature importance and model insights
        - mortgage calculator: real-world financial planning
        """)
        
        if df is not None:
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("properties analyzed", f"{len(df):,}")
            with col_b:
                avg_price = np.expm1(df['price_log'].mean())
                st.metric("avg price", f"${avg_price:,.0f}")
            with col_c:
                st.metric("model accuracy r2", "0.0227")
        
    with col2:
        st.image("https://images.unsplash.com/photo-1560518883-ce09059eeffa?w=800", 
                 caption="ai-powered real estate valuation", use_container_width=True)
    
    st.divider()
    
    if df is not None:
        st.subheader("market overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            min_price = np.expm1(df['price_log'].min())
            st.metric("minimum price", f"${min_price:,.0f}")
        with col2:
            max_price = np.expm1(df['price_log'].max())
            st.metric("maximum price", f"${max_price:,.0f}")
        with col3:
            median_price = np.expm1(df['price_log'].median())
            st.metric("median price", f"${median_price:,.0f}")
        with col4:
            st.metric("total cities", f"{df['city'].nunique()}")

elif selected == "predict price":
    st.markdown('<div class="main-header">intelligent price prediction</div>', unsafe_allow_html=True)
    
    if pipeline is not None:
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        with col_btn1:
            if st.button("random example", help="fill form with random house data"):
                st.session_state['random_example'] = True
        with col_btn2:
            if st.button("clear form"):
                st.session_state['random_example'] = False
                st.rerun()
        
        if st.session_state.get('random_example', False):
            sample = df.sample(1).iloc[0]
            default_bedrooms = int(sample['bedrooms'])
            default_bathrooms = float(sample['bathrooms'])
            default_sqft_living = int(sample['sqft_living'])
            default_sqft_lot = int(sample['sqft_lot'])
            default_floors = float(sample['floors'])
            default_sqft_above = int(sample['sqft_above'])
            default_sqft_basement = int(sample['sqft_basement'])
            default_condition = int(sample['condition'])
            default_view = int(sample['view'])
            default_waterfront = bool(sample['waterfront'])
            default_city = sample['city']
        else:
            default_bedrooms = 3
            default_bathrooms = 2.0
            default_sqft_living = 2000
            default_sqft_lot = 5000
            default_floors = 1.0
            default_sqft_above = 1500
            default_sqft_basement = 0
            default_condition = 3
            default_view = 0
            default_waterfront = False
            default_city = sorted(df['city'].unique())[0]
        
        with st.form("prediction_form"):
            st.markdown("### property details")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                bedrooms = st.slider("bedrooms", 0, 10, default_bedrooms)
                bathrooms = st.slider("bathrooms", 0.0, 8.0, default_bathrooms, 0.25)
                floors = st.slider("floors", 1.0, 3.5, default_floors, 0.5)
                
            with col2:
                sqft_living = st.number_input("living area sqft", 300, 15000, default_sqft_living)
                sqft_lot = st.number_input("lot size sqft", 500, 1000000, default_sqft_lot)
                sqft_above = st.number_input("area above ground sqft", 300, 10000, default_sqft_above)
                
            with col3:
                sqft_basement = st.number_input("basement area sqft", 0, 5000, default_sqft_basement)
                condition = st.slider("condition 1-5", 1, 5, default_condition)
                view = st.slider("view quality 0-4", 0, 4, default_view)

            col4, col5 = st.columns(2)
            
            with col4:
                cities = sorted(df['city'].unique().tolist())
                city = st.selectbox("city", cities, index=cities.index(default_city) if default_city in cities else 0)
                
            with col5:
                waterfront = st.toggle("waterfront property", value=default_waterfront)

            submitted = st.form_submit_button("predict price", type="primary", use_container_width=True)
            
            if submitted:
                current_year = datetime.now().year
                house_age = current_year - 1990
                years_since_renovation = 0
                total_sqft = sqft_living + sqft_lot
                
                input_data = pd.DataFrame({
                    'bedrooms': [bedrooms],
                    'bathrooms': [bathrooms],
                    'sqft_living': [sqft_living],
                    'sqft_lot': [sqft_lot],
                    'floors': [floors],
                    'waterfront': [1 if waterfront else 0],
                    'view': [view],
                    'condition': [condition],
                    'sqft_above': [sqft_above],
                    'sqft_basement': [sqft_basement],
                    'house_age': [house_age],
                    'years_since_renovation': [years_since_renovation],
                    'total_sqft': [total_sqft],
                    'city': [city]
                })
                
                prediction_log = pipeline.predict(input_data)[0]
                prediction = np.expm1(prediction_log)
                
                st.success(f"predicted price: ${prediction:,.2f}")
                
                confidence = prediction * 0.15
                st.info(f"confidence range: ${prediction - confidence:,.2f} - ${prediction + confidence:,.2f}")
                
                st.markdown("---")
                col_res1, col_res2 = st.columns([1, 1])
                
                with col_res1:
                    min_price = np.expm1(df['price_log'].min())
                    max_price = np.expm1(df['price_log'].max())
                    avg_price = np.expm1(df['price_log'].mean())
                    
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = prediction,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "market position"},
                        delta = {'reference': avg_price, 'valueformat': ',.0f'},
                        gauge = {
                            'axis': {'range': [min_price, max_price], 'tickformat': ',.0f'},
                            'bar': {'color': "#1E88E5"},
                            'steps': [
                                {'range': [min_price, avg_price], 'color': "#e8f5e9"},
                                {'range': [avg_price, max_price], 'color': "#ffebee"}],
                        }))
                    
                    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                    
                with col_res2:
                    st.markdown("### mortgage calculator")
                    
                    loan_term = st.selectbox("loan term", [15, 20, 30], index=2)
                    interest_rate = st.slider("interest rate percent", 1.0, 10.0, 6.5, 0.1)
                    down_payment_pct = st.slider("down payment percent", 0, 50, 20, 5)
                    
                    down_payment = prediction * (down_payment_pct / 100)
                    loan_amount = prediction - down_payment
                    monthly_rate = (interest_rate / 100) / 12
                    n_payments = loan_term * 12
                    
                    if monthly_rate > 0:
                        monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate) ** n_payments) / ((1 + monthly_rate) ** n_payments - 1)
                    else:
                        monthly_payment = loan_amount / n_payments
                    
                    st.metric("monthly payment", f"${monthly_payment:,.2f}")
                    st.write(f"loan amount: ${loan_amount:,.0f}")
                    st.write(f"total paid: ${monthly_payment * n_payments:,.0f}")

elif selected == "model comparison":
    st.markdown('<div class="main-header">model performance comparison</div>', unsafe_allow_html=True)
    
    if model_comparison is not None:
        st.write("comparison of 3 state-of-the-art machine learning models")
        
        st.dataframe(model_comparison, use_container_width=True)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_r2 = px.bar(model_comparison, x='Model', y='R2_Score', 
                           title="accuracy comparison",
                           color='R2_Score', color_continuous_scale='Blues')
            st.plotly_chart(fig_r2, use_container_width=True)
            
        with col2:
            fig_time = px.bar(model_comparison, x='Model', y='Training_Time_s',
                             title="training speed",
                             color='Training_Time_s', color_continuous_scale='Reds')
            st.plotly_chart(fig_time, use_container_width=True)
        
        fig_errors = go.Figure()
        fig_errors.add_trace(go.Bar(x=model_comparison['Model'], y=model_comparison['MAE'], name='mae'))
        fig_errors.add_trace(go.Bar(x=model_comparison['Model'], y=model_comparison['RMSE'], name='rmse'))
        fig_errors.update_layout(title="error metrics", barmode='group')
        st.plotly_chart(fig_errors, use_container_width=True)
        
        best_model = model_comparison.loc[model_comparison['R2_Score'].idxmax(), 'Model']
        st.success(f"winner: {best_model}")
        
        csv = model_comparison.to_csv(index=False).encode('utf-8')
        st.download_button("download comparison data", csv, "model_comparison.csv", "text/csv")

elif selected == "data analysis":
    st.markdown('<div class="main-header">market data analytics</div>', unsafe_allow_html=True)
    
    if df is not None:
        tab1, tab2, tab3 = st.tabs(["correlations", "distributions", "feature analysis"])
        
        with tab1:
            st.subheader("feature correlation matrix")
            numeric_df = df.select_dtypes(include=[np.number])
            corr = numeric_df.corr()
            
            fig_corr = px.imshow(corr, text_auto='.2f', aspect="auto", 
                                color_continuous_scale="RdBu_r", title="correlation heatmap")
            st.plotly_chart(fig_corr, use_container_width=True)
            
        with tab2:
            st.subheader("price and feature distributions")
            df['price_actual'] = np.expm1(df['price_log'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_hist = px.histogram(df, x="price_actual", nbins=50, title="price distribution")
                st.plotly_chart(fig_hist, use_container_width=True)
                
            with col2:
                fig_box = px.box(df, y="price_actual", title="price box plot")
                st.plotly_chart(fig_box, use_container_width=True)
            
            selected_feature = st.selectbox("select feature to compare", 
                                          ['sqft_living', 'bedrooms', 'bathrooms', 'house_age'])
            
            fig_scatter = px.scatter(df, x=selected_feature, y="price_actual",
                                   title=f"price vs {selected_feature}", opacity=0.5)
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        with tab3:
            st.subheader("engineered features analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_age = px.histogram(df, x="house_age", nbins=30, title="house age distribution")
                st.plotly_chart(fig_age, use_container_width=True)
                
            with col2:
                fig_sqft = px.scatter(df.sample(1000), x="total_sqft", y="price_actual",
                                     title="total sqft vs price", opacity=0.6)
                st.plotly_chart(fig_sqft, use_container_width=True)

elif selected == "stacked model":
    st.markdown('<div class="main-header">stacked ensemble performance</div>', unsafe_allow_html=True)
    
    try:
        comparison_df = pd.read_csv('models/final_comparison.csv')
        
        # Filter for relevant models to compare against Stacking
        relevant_models = ['Stacked_Ensemble', 'Optimized_XGBoost', 'xgboost', 'gradient_boosting']
        filtered_df = comparison_df[comparison_df['Model'].isin(relevant_models)].sort_values('R2_Score', ascending=False)
        
        st.info("performance comparison: stacked ensemble vs individual models")
        
        # Display the table with formatting
        st.dataframe(
            filtered_df[['Model', 'R2_Score', 'MAE', 'RMSE', 'Training_Time_s']],
            use_container_width=True,
            hide_index=True
        )
        
        # Highlight Top Model
        best_model = filtered_df.iloc[0]
        st.success(f"🏆 best performer: {best_model['Model']} with {best_model['R2_Score']:.2%} accuracy")
            
    except Exception as e:
         st.warning("run train_stacked.py first")

elif selected == "training benchmark":
    st.markdown('<div class="main-header">training strategy benchmark</div>', unsafe_allow_html=True)
    
    st.info("comparing the performance of 'fast training' vs 'full accuracy training' strategies")
    
    try:
        comparison_df = pd.read_csv('models/final_comparison.csv')
        
        # Color the dataframe
        def highlight_best(s):
            is_max = s == s.max()
            return ['background-color: #d4edda' if v else '' for v in is_max]

        st.dataframe(comparison_df.style.apply(highlight_best, subset=['R2_Score']), use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### accuracy improvement")
            full_r2 = comparison_df[comparison_df['Training_Mode'].str.contains('Full')]['R2_Score'].max()
            fast_r2 = comparison_df[comparison_df['Training_Mode'].str.contains('Fast')]['R2_Score'].max()
            
            if pd.notna(full_r2) and pd.notna(fast_r2) and fast_r2 != 0:
                improvement = ((full_r2 - fast_r2) / abs(fast_r2)) * 100
                st.metric("accuracy gain", f"+{improvement:,.0f}%", delta="huge improvement")
            else:
                 st.metric("best r2 score", f"{full_r2:.4f}")
                 
            fig = px.bar(comparison_df, x='Model', y='R2_Score', color='Training_Mode',
                        title="R2 score by training mode", text_auto='.3f')
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.markdown("### trade-off: time vs accuracy")
            # Create a positive size factor for bubbles (handling negative R2)
            comparison_df['Size_Factor'] = comparison_df['R2_Score'].abs() + 0.1
            
            fig2 = px.scatter(comparison_df, x='Training_Time_s', y='R2_Score', 
                             color='Training_Mode', size='Size_Factor', hover_data=['Model'],
                             title="efficiency frontier (time vs accuracy)")
            st.plotly_chart(fig2, use_container_width=True)
            
    except FileNotFoundError:
        st.error("benchmark data not found. please ensure both training scripts have been run.")

elif selected == "batch prediction":
    st.markdown('<div class="main-header">batch price prediction</div>', unsafe_allow_html=True)
    
    st.write("process multiple properties by uploading a csv file")
    
    st.markdown("### download template")
    template_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
                     'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement', 
                     'house_age', 'years_since_renovation', 'total_sqft', 'city']
    
    example_row = {
        'bedrooms': 3, 'bathrooms': 2.0, 'sqft_living': 2000, 'sqft_lot': 5000,
        'floors': 1.0, 'waterfront': 0, 'view': 0, 'condition': 3,
        'sqft_above': 1500, 'sqft_basement': 500, 'house_age': 30,
        'years_since_renovation': 0, 'total_sqft': 7000, 'city': 'seattle'
    }
    template_df = pd.DataFrame([example_row])
    
    csv_template = template_df.to_csv(index=False).encode('utf-8')
    st.download_button("download csv template", csv_template, "template.csv", "text/csv")
    
    st.divider()
    
    st.markdown("### upload your data")
    uploaded_file = st.file_uploader("choose a csv file", type="csv")
    
    if uploaded_file and pipeline:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"file uploaded, found {len(batch_df)} properties")
            
            missing_cols = [col for col in template_cols if col not in batch_df.columns]
            
            if missing_cols:
                st.error(f"missing columns: {', '.join(missing_cols)}")
            else:
                if st.button("generate predictions", type="primary", use_container_width=True):
                    predictions_log = pipeline.predict(batch_df)
                    batch_df['predicted_price'] = np.expm1(predictions_log)
                    
                    st.success(f"predictions completed for {len(batch_df)} properties")
                    
                    st.dataframe(batch_df[['city', 'sqft_living', 'predicted_price']].head(20))
                    
                    csv_results = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button("download results", csv_results, "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"error: {str(e)}")

elif selected == "help":
    st.markdown('<div class="main-header">help and documentation</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## user guide
    
    ### home page
    view project overview and dataset statistics
    
    ### predict price
    - random example: auto-fill form with real data
    - manual input: enter property details
    - mortgage calculator: calculate monthly payments
    
    ### model comparison
    compare performance of 3 ml algorithms
    
    ### data analysis
    - correlations: see which features influence price
    - distributions: understand data patterns
    - feature analysis: explore engineered features
    
    ### batch prediction
    1. download csv template
    2. fill in your property data
    3. upload and get predictions
    4. download results
    
    ## technical details
    
    ### models
    - xgboost (best, r2 = 0.0227)
    - gradient boosting
    - lightgbm
    
    ### feature engineering
    - house_age = current year - year built
    - years_since_renovation
    - total_sqft = living area + lot size
    
    ### preprocessing
    - knn imputer
    - standard scaler
    - one-hot encoding
    - log transform
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>house price prediction - 5*A by ahmed abobakr</p>
    <p>powered by xgboost and machine learning</p>
</div>
""", unsafe_allow_html=True)

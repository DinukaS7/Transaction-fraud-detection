import streamlit as st
import pandas as pd
import joblib

# --- 1. CONFIGURATION (Must be the first command) ---
st.set_page_config(
    page_title="Fraud Detection AI",
    page_icon="Bank.ico", # Ensure Bank.ico is in the same folder as this script
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS FOR MODERN LOOK ---
st.markdown("""
    <style>
    /* Remove default top padding */
    .block-container {
        padding-top: 2rem;
    }
    
    /* Style the predict button */
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        height: 50px;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Input field styling */
    .stTextInput>div>div>input {
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. LOAD MODEL ---
# Using @st.cache_resource to load the model only once prevents lag
@st.cache_resource
def load_model():
    try:
        return joblib.load("fraud_detection.pkl")
    except FileNotFoundError:
        st.error("Model file 'fraud_detection.pkl' not found.")
        return None

model = load_model()

# --- 4. HEADER SECTION ---
col_main, col_img = st.columns([3, 1])

with col_main:
    st.title("🛡️ Fraud Detection")
    st.markdown("Enter transaction details below to assess risk levels using our AI model.")

# --- 5. INPUT FORM (Using Columns for Grid Layout) ---
st.markdown("### 📝 Transaction Details")
with st.container(border=True): # Adds a visible box around inputs
    
    # Row 1
    c1, c2 = st.columns(2)
    with c1:
        transaction_type = st.selectbox("Transaction Type", ["PAYMENT","TRANSFER", "CASH_OUT", "DEPOSIT"])
    with c2:
        amount = st.number_input("Amount ($)", min_value=0.0, value=1000.0, format="%.2f")

    st.divider()

    # Row 2
    c3, c4 = st.columns(2)
    with c3:
        oldbalanceOrg = st.number_input("Sender Old Balance", min_value=0.0, value=10000.0)
    with c4:
        newbalanceOrig = st.number_input("Sender New Balance", min_value=0.0, value=9000.0)

    # Row 3
    c5, c6 = st.columns(2)
    with c5:
        oldbalanceDest = st.number_input("Receiver Old Balance", min_value=0.0, value=0.0)
    with c6:
        newbalanceDest = st.number_input("Receiver New Balance", min_value=0.0, value=0.0)

# --- 6. PREDICTION LOGIC ---
st.write("") # Spacer
if st.button("Analyze Transaction"):
    if model:
        # Prepare input data
        # Note: I corrected the typo 'oldbalance0rg' -> 'oldbalanceOrg'
        input_data = pd.DataFrame([{
            "type": transaction_type,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg, 
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest
        }])

        try:
            # Predict
            prediction = model.predict(input_data)[0]
            
            # Display Results with Animation
            if prediction == 1:
                st.error("🚨 ALERT: This transaction is classified as FRAUD.")
            else:
                st.balloons() # Adds a nice animation for success
                st.success("✅ This transaction appears SAFE.")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
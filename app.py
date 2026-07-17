import streamlit as st
import pickle

# PAGE SETTINGS
st.set_page_config(
    page_title="Spam Mail Detector",
    page_icon="📩",
    layout="centered"
)


# CUSTOM CSS
st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg,#1e3c72,#2a5298);
}

/* Sidebar Background */
section[data-testid="stSidebar"]{
    background: #111827;
}

/* Sidebar Text */
section[data-testid="stSidebar"] *{
    color: white;
}

h1{
    color:white !important;
    text-align:center;
}

p{
    color:white;
    font-size:18px;
}

.stTextArea textarea{
    border-radius:12px;
    font-size:16px;
}

.stButton>button{
    width:100%;
    background:#00c853;
    color:white;
    font-size:18px;
    font-weight:bold;
    border-radius:12px;
    height:50px;
    border:none;
}

.stButton>button:hover{
    background:#00a844;
    color:white;
}

</style>
""", unsafe_allow_html=True)


# LOAD MODEL
model = pickle.load(open("spam_model.pkl", "rb"))

# LOAD TF-IDF
vectorizer = pickle.load(open("tfidf.pkl", "rb"))


st.markdown("<h1>📩 Spam Mail Detector</h1>", unsafe_allow_html=True)

st.markdown(
    "<p style='text-align:center;'>Enter your message below to check whether it is Spam or Not Spam.</p>",
    unsafe_allow_html=True
)


with st.sidebar:
    st.title("ℹ About")
    st.write("""
This application detects whether an email or SMS is:

✅ Not Spam (Normal)

🚨 Spam

Built using Machine Learning.
""")

message = st.text_area("✉ Enter Message")


# PREDICTION

if st.button("🔍 Predict"):

    if message.strip() == "":
        st.warning("⚠ Please enter a message.")

    else:

        text = vectorizer.transform([message])

        prediction = model.predict(text)

        if prediction[0] == "spam":
            st.error("Spam Message Detected")
            st.warning("⚠ Avoid clicking unknown links or sharing personal information.")
        else:
            st.success("Not Spam (Normal) Message")
            st.balloons()

st.markdown("---")
st.caption("© 2026 Spam Mail Detector | Developed by Vishwa Sharma")

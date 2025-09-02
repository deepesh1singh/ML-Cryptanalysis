import streamlit as st, joblib, os
st.title("ML Cryptanalysis Demo (Upgraded)")
st.write("Upload ciphertext file or paste ciphertext and pick a cipher model to run.")
cipher = st.selectbox("Cipher", ["caesar","vigenere","substitution"])
text = st.text_area("Ciphertext", value="")
model_path = st.text_input("Model path", value=f"models/{cipher}_random_forest_model.pkl")
if st.button("Decrypt"):
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        try:
            out = model.predict([text])
            st.write("Model output:", out)
        except Exception as e:
            st.error(f"Model prediction failed: {e}")
    else:
        st.error("Model not found at path: "+model_path)

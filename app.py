
import streamlit as st
from main import run_pipeline

# Show the app title
st.title("📧 Daily Financial Newsletter Generator")

# Text box for the company name
company_name = st.text_input("Enter the company name:", value="TCS")

# Number input for price change percentage
price_change = st.number_input("Enter the price change (%):", value=2.5)

# Button to generate the newsletter
if st.button("Generate Newsletter"):
    newsletter = run_pipeline(company_name, price_change)
    st.success("Newsletter Generated!")
    st.text_area("Your Newsletter:", value=newsletter, height=400)

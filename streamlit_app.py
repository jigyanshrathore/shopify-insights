# streamlit_app.py
# To run this application:
# 1. Make sure your FastAPI app is running in one terminal:
#    uvicorn main:app --reload
#
# 2. Install streamlit:
#    pip install streamlit pandas
#
# 3. Run this Streamlit app in a SECOND terminal:
#    streamlit run streamlit_app.py

import streamlit as st
import requests
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Shopify Store Insights",
    page_icon="🛍️",
    layout="wide"
)

# --- API Endpoint ---
FASTAPI_URL = "http://127.0.0.1:8000/scrape-and-analyze"

# --- UI Components ---
st.title("🛍️ Shopify Store Insights Fetcher")
st.markdown("Enter the URL of a Shopify store to fetch its product catalog, policies, contact details, and competitor information.")

# --- Input Form ---
with st.form("scrape_form"):
    website_url = st.text_input(
        "Shopify Store URL",
        placeholder="e.g., https://memy.co.in or https://hairoriginals.com"
    )
    submitted = st.form_submit_button("Analyze Website")

# --- Logic to call API and display results ---
if submitted and website_url:
    with st.spinner(f"Scraping {website_url} and its competitors... This may take a moment."):
        try:
            # Call the FastAPI backend
            response = requests.post(FASTAPI_URL, json={"website_url": website_url}, timeout=300) # 5 min timeout

            if response.status_code == 200:
                data = response.json()
                st.success("Analysis Complete!")

                # --- Display Primary Brand Data ---
                st.header(f"Analysis for: {data['website_url']}")

                # Use columns for better layout
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Brand Context")
                    st.info(data.get('brand_context', 'Not found.'))

                    st.subheader("Contact Details")
                    st.json(data.get('contact_details', {}))

                with col2:
                    st.subheader("Social Media")
                    st.json(data.get('social_handles', {}))
                    
                    st.subheader("Important Links")
                    st.json(data.get('important_links', {}))

                # Display policies in expanders
                with st.expander("📜 View Policies"):
                    st.subheader("Privacy Policy")
                    st.text(data.get('privacy_policy', 'Not found.'))
                    st.divider()
                    st.subheader("Refund/Return Policy")
                    st.text(data.get('refund_policy', 'Not found.'))

                # Display FAQs
                if data.get('faqs'):
                    with st.expander("❓ View FAQs"):
                        for faq in data['faqs']:
                            st.markdown(f"**Q: {faq['question']}**")
                            st.write(f"A: {faq['answer']}")

                # Display Product Catalog as a DataFrame
                st.subheader("📦 Full Product Catalog")
                if data.get('product_catalog'):
                    df = pd.DataFrame(data['product_catalog'])
                    st.dataframe(df[['title', 'vendor', 'product_type', 'price', 'url']])
                else:
                    st.warning("No products found.")

                # --- Display Competitor Data ---
                st.divider()
                st.header("⚔️ Competitor Analysis")
                if data.get('competitors'):
                    for competitor in data['competitors']:
                        with st.expander(f"Competitor: {competitor['website_url']}"):
                            st.subheader("Brand Context")
                            st.info(competitor.get('brand_context', 'Not found.'))
                            
                            st.subheader("Product Catalog Sample")
                            if competitor.get('product_catalog'):
                                comp_df = pd.DataFrame(competitor['product_catalog'])
                                st.dataframe(comp_df[['title', 'vendor', 'price']].head())
                            else:
                                st.warning("No products found for this competitor.")
                else:
                    st.info("No competitors were found or analyzed.")


            else:
                st.error(f"Error from API: {response.status_code}")
                st.json(response.json())

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to the backend API. Is it running? Error: {e}")

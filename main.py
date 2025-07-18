# main.py
# To run this application:
# 1. Install necessary libraries:
#    pip install fastapi uvicorn requests beautifulsoup4 pydantic google-generativeai python-dotenv sqlalchemy
#
# 2. Create a .env file in the same directory and add your Google API Key:
#    GOOGLE_API_KEY="YOUR_API_KEY_HERE"
#
# 3. Run the server from your terminal:
#    uvicorn main:app --reload
#    This will create a 'shopify_insights.db' file in your directory to store the data.

import os
import re
import json
from typing import List, Optional, Dict, Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv
import google.generativeai as genai

# --- SQLAlchemy Database Setup (using SQLite) ---
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base, Session

# --- Configuration ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("FATAL: GOOGLE_API_KEY not found in .env file. The application cannot run without it.")
    exit(1)

genai.configure(api_key=GOOGLE_API_KEY)

# --- Database Connection (SQLite) ---
DATABASE_URL = "sqlite:///./shopify_insights.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False} # Needed for SQLite with FastAPI
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Pydantic Models (API Schema) ---

class Product(BaseModel):
    id: int
    title: str
    vendor: str
    product_type: str
    price: float
    url: str
    images: List[str]

    class Config:
        orm_mode = True

class FAQItem(BaseModel):
    question: str
    answer: str

    class Config:
        orm_mode = True

# Forward reference for recursive competitor data
class BrandData(BaseModel):
    website_url: HttpUrl
    product_catalog: List[Product]
    hero_products: List[Product]
    social_handles: Dict[str, HttpUrl]
    contact_details: Dict[str, List[str]]
    brand_context: Optional[str] = None
    privacy_policy: Optional[str] = None
    refund_policy: Optional[str] = None
    faqs: Optional[List[FAQItem]] = None
    important_links: Dict[str, HttpUrl]
    competitors: Optional[List['BrandData']] = [] # To hold competitor data

    class Config:
        orm_mode = True

BrandData.update_forward_refs()

class ScrapeRequest(BaseModel):
    website_url: HttpUrl

# --- SQLAlchemy Models (Database Schema) ---

class BrandDB(Base):
    __tablename__ = "brands"
    id = Column(Integer, primary_key=True, index=True)
    website_url = Column(String(255), unique=True, index=True)
    brand_context = Column(Text, nullable=True)
    privacy_policy = Column(Text, nullable=True)
    refund_policy = Column(Text, nullable=True)
    
    products = relationship("ProductDB", back_populates="brand", cascade="all, delete-orphan")
    faqs = relationship("FAQDB", back_populates="brand", cascade="all, delete-orphan")
    
    social_handles_json = Column(Text, name="social_handles")
    contact_details_json = Column(Text, name="contact_details")
    important_links_json = Column(Text, name="important_links")

class ProductDB(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True, index=True)
    shopify_product_id = Column(Integer)
    title = Column(String(255), index=True)
    vendor = Column(String(255))
    product_type = Column(String(255))
    price = Column(Float)
    url = Column(String(512))
    images_json = Column(Text, name="images")
    brand_id = Column(Integer, ForeignKey("brands.id"))
    
    brand = relationship("BrandDB", back_populates="products")

class FAQDB(Base):
    __tablename__ = "faqs"
    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text)
    answer = Column(Text)
    brand_id = Column(Integer, ForeignKey("brands.id"))

    brand = relationship("BrandDB", back_populates="faqs")

# Create database tables
Base.metadata.create_all(bind=engine)

# --- Database Dependency ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- LLM Processor for Unstructured Data & Competitor Analysis ---

class LLMProcessor:
    def __init__(self, api_key: str):
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def get_brand_context(self, text: str) -> Optional[str]:
        prompt = f"Summarize the brand's mission from this 'About Us' text:\n\n{text}"
        try:
            return self.model.generate_content(prompt).text.strip()
        except Exception: return f"Could not summarize. Original: {text[:200]}..."

    def structure_faqs(self, text: str) -> Optional[List[FAQItem]]:
        prompt = f"""
        Extract FAQs into a valid JSON array of objects with "question" and "answer" keys. If none, return [].
        Text: --- {text} ---
        """
        try:
            response = self.model.generate_content(prompt)
            cleaned = response.text.strip().replace('```json', '').replace('```', '').strip()
            faq_data = json.loads(cleaned)
            return [FAQItem(**item) for item in faq_data]
        except Exception: return None

    def find_competitors(self, brand_name: str, brand_context: str) -> List[Dict[str, str]]:
        """Finds competitors using the brand name and context."""
        if not brand_name: return []
        
        prompt = f"""
        Given the brand named "{brand_name}" with this description: "{brand_context[:500]}...",
        list its top 3 direct e-commerce competitors.
        Return a valid JSON array of objects, where each object has "brand_name" and "website_url" keys.
        
        Example: [{{"brand_name": "Competitor A", "website_url": "https://competitor-a.com"}}]
        """
        try:
            response = self.model.generate_content(prompt)
            cleaned_response = response.text.strip().replace('```json', '').replace('```', '').strip()
            return json.loads(cleaned_response)
        except Exception as e:
            print(f"Error finding competitors with LLM: {e}")
            return []

# --- Shopify Scraper ---
class ShopifyScraper:
    def __init__(self, base_url: str, llm_processor: LLMProcessor):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})
        self.llm_processor = llm_processor

    def _get_soup(self, url: str) -> Optional[BeautifulSoup]:
        try:
            r = self.session.get(url, timeout=15)
            r.raise_for_status()
            return BeautifulSoup(r.content, 'html.parser')
        except requests.RequestException: return None

    def get_full_product_catalog(self) -> List[Product]:
        products_url = urljoin(self.base_url, 'products.json')
        catalog = []
        try:
            data = self.session.get(products_url, timeout=15).json()
            for item in data.get('products', []):
                price = float(item['variants'][0].get('price', 0.0)) if item.get('variants') else 0.0
                catalog.append(Product(
                    id=item['id'], title=item.get('title', 'N/A'), vendor=item.get('vendor', 'N/A'),
                    product_type=item.get('product_type', 'N/A'), price=price,
                    url=urljoin(self.base_url, f"products/{item.get('handle', '')}"),
                    images=[img['src'] for img in item.get('images', [])]
                ))
            return catalog
        except Exception: return []

    def _find_links(self, soup: BeautifulSoup, keywords: List[str]) -> Dict[str, HttpUrl]:
        links = {}
        for a in soup.find_all('a', href=True):
            for keyword in keywords:
                if keyword in a.text.lower() or keyword in a['href'].lower():
                    if keyword not in links:
                        full_url = urljoin(self.base_url, a['href'])
                        if urlparse(full_url).scheme in ['http', 'https']:
                            links[keyword] = full_url
        return links

    def extract_page_text(self, url: str) -> str:
        soup = self._get_soup(url)
        if soup:
            for script in soup(['script', 'style']): script.decompose()
            return soup.get_text(separator=' ', strip=True)
        return ""

    def run(self) -> BrandData:
        homepage_soup = self._get_soup(self.base_url)
        if not homepage_soup:
            raise HTTPException(status_code=404, detail=f"Website not found: {self.base_url}")

        product_catalog = self.get_full_product_catalog()
        homepage_text = homepage_soup.get_text().lower()
        hero_products = [p for p in product_catalog if p.title.lower() in homepage_text] if product_catalog else []
        
        link_keywords = ['privacy', 'refund', 'return', 'contact', 'about', 'faq', 'track', 'blog']
        found_links = self._find_links(homepage_soup, link_keywords)

        privacy_policy_text = self.extract_page_text(found_links['privacy']) if found_links.get('privacy') else ""
        refund_url = found_links.get('refund') or found_links.get('return')
        refund_policy_text = self.extract_page_text(refund_url) if refund_url else ""
        faqs_list = self.llm_processor.structure_faqs(self.extract_page_text(found_links['faq'])) if found_links.get('faq') else None
        brand_context = self.llm_processor.get_brand_context(self.extract_page_text(found_links['about'])) if found_links.get('about') else None
        
        social_handles = {}
        social_sites = ['instagram', 'facebook', 'twitter', 'tiktok', 'youtube', 'pinterest', 'linkedin']
        for a in homepage_soup.find_all('a', href=True):
            for site in social_sites:
                if site in a['href']:
                    if site not in social_handles: social_handles[site] = a['href']
        
        contact_page_text = self.extract_page_text(found_links.get('contact')) or homepage_soup.get_text()
        emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', contact_page_text)
        phones = re.findall(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', contact_page_text)
        
        return BrandData(
            website_url=self.base_url, product_catalog=product_catalog, hero_products=hero_products,
            social_handles=social_handles, contact_details={"emails": list(set(emails)), "phones": list(set(phones))},
            brand_context=brand_context, privacy_policy=privacy_policy_text, refund_policy=refund_policy_text,
            faqs=faqs_list, important_links=found_links
        )

# --- Database Persistence Logic ---
def save_brand_data_to_db(db: Session, brand_data: BrandData):
    """Saves a BrandData object and its relationships to the database."""
    db_brand = db.query(BrandDB).filter(BrandDB.website_url == str(brand_data.website_url)).first()
    if db_brand: # If brand exists, delete its old product/faq records to avoid duplicates
        db.query(ProductDB).filter(ProductDB.brand_id == db_brand.id).delete()
        db.query(FAQDB).filter(FAQDB.brand_id == db_brand.id).delete()
    else: # Or create a new one
        db_brand = BrandDB(website_url=str(brand_data.website_url))
        db.add(db_brand)

    # Update brand fields
    db_brand.brand_context = brand_data.brand_context
    db_brand.privacy_policy = brand_data.privacy_policy
    db_brand.refund_policy = brand_data.refund_policy
    
    # --- FIX STARTS HERE ---
    # Convert HttpUrl objects to strings before saving to JSON
    db_brand.social_handles_json = json.dumps({k: str(v) for k, v in brand_data.social_handles.items()})
    db_brand.important_links_json = json.dumps({k: str(v) for k, v in brand_data.important_links.items()})
    # --- FIX ENDS HERE ---
    
    db_brand.contact_details_json = json.dumps(brand_data.contact_details)
    
    # Create new products
    for product in brand_data.product_catalog:
        db.add(ProductDB(
            shopify_product_id=product.id, title=product.title, vendor=product.vendor,
            product_type=product.product_type, price=product.price, url=str(product.url),
            images_json=json.dumps(product.images), brand=db_brand
        ))
    
    # Create new FAQs
    if brand_data.faqs:
        for faq in brand_data.faqs:
            db.add(FAQDB(question=faq.question, answer=faq.answer, brand=db_brand))
            
    db.commit()


# --- FastAPI Application ---

app = FastAPI(
    title="Shopify Insights-Fetcher (Advanced)",
    description="Scrapes Shopify stores, finds competitors, and saves data to a SQLite DB.",
    version="2.0.1" # Incremented version
)

@app.post("/scrape-and-analyze", response_model=BrandData)
def scrape_and_analyze_store(request: ScrapeRequest, db: Session = Depends(get_db)):
    """
    Accepts a URL, scrapes it, finds competitors, scrapes them,
    saves everything to the database, and returns the full analysis.
    """
    try:
        llm_processor = LLMProcessor(api_key=GOOGLE_API_KEY)
        
        # 1. Scrape the primary brand
        primary_scraper = ShopifyScraper(base_url=str(request.website_url), llm_processor=llm_processor)
        primary_brand_data = primary_scraper.run()
        save_brand_data_to_db(db=db, brand_data=primary_brand_data)
        
        # 2. Find and Scrape Competitors
        vendor_name = primary_brand_data.product_catalog[0].vendor if primary_brand_data.product_catalog else urlparse(str(request.website_url)).netloc
        competitors_info = llm_processor.find_competitors(vendor_name, primary_brand_data.brand_context or "")
        
        for comp in competitors_info:
            comp_url = comp.get('website_url')
            if not comp_url: continue
            print(f"Analyzing competitor: {comp.get('brand_name')} at {comp_url}")
            try:
                comp_scraper = ShopifyScraper(base_url=comp_url, llm_processor=llm_processor)
                comp_data = comp_scraper.run()
                primary_brand_data.competitors.append(comp_data) # Add to response
                save_brand_data_to_db(db=db, brand_data=comp_data) # Save to DB
            except Exception as e:
                print(f"Could not scrape competitor {comp_url}: {e}")

        return primary_brand_data
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"An internal server error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@app.get("/", include_in_schema=False)
def root():
    return {"message": "API is running. Use the /docs endpoint for the interactive API documentation."}

from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

sp = create_client(url, key)

# Insert tenant
response = sp.table("Tenants").insert({
    "name": "company_A"
}).execute()
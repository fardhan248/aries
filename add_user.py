from supabase import create_client
import os
from dotenv import load_dotenv
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--tenant", type=str, help="Nama perusahaan")
args = parser.parse_args()

tenant = args.tenant

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

sp = create_client(url, key)

try:
    response = (sp
                .table("Tenants")
                .select("*")
                .eq("name", tenant)
                .execute())
except ValueError as e:
    print(e)
    
id_company = response.data[0]["tenant_id"]

# Insert user
response = sp.table("Users").insert({
    "role": "user",
    "tenant_id": id_company
}).execute()
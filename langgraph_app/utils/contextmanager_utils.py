from supabase import AsyncClient
from chromadb import HttpClient

supabase_client: AsyncClient = None
chroma: HttpClient = None
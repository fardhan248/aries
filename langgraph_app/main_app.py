from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from routes import router
from utils.db_pool import close_db_pool, init_db_pool, get_db_pool
from contextlib import asynccontextmanager
from supabase import acreate_client
from dotenv import load_dotenv
import utils.contextmanager_utils as cm
import os, chromadb

load_dotenv()

supabase_key = os.getenv("SUPABASE_KEY")
supabase_url = os.getenv("SUPABASE_URL")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global supabase_client
    cm.supabase_client = await acreate_client(supabase_url, supabase_key)
    cm.chroma = chromadb.HttpClient(host="chromadb", port=8000)
    
    app.state.pool = await get_db_pool()
    yield
    await close_db_pool()
    
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
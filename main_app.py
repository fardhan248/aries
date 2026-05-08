from db_pool import close_db_pool, init_db_pool, get_db_pool
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from routes import router
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

supabase_key = os.getenv("SUPABASE_KEY")
supabase_url = os.getenv("SUPABASE_URL")

supabase_client: Client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global supabase_client
    supabase_client = create_client(supabase_url, supabase_key)
    # await init_db_pool()
    # yield
    
    # await close_db_pool()
    
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

# Di sini define app fastapi
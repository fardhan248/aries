from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from routes import router
from contextmanager_utils import lifespan
    
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
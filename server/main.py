from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.routes import router
from server.config import get_settings

app = FastAPI(title="Thought Anchors API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("server.main:app", host=settings.HOST, port=settings.PORT, reload=True)

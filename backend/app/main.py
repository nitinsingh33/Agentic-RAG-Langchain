from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.routes import router
from app.utils.logger import logger

def create_app() -> FastAPI:
    
    
    try:
        settings.validate_required_keys()
        logger.info("Configuration validated successfully")
    except ValueError as e:
        logger.critical(f"Configuration validation failed: {e}")
        raise SystemExit(1)
    
    
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description=settings.DESCRIPTION,
        version=settings.VERSION,
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS or ["*"],
        allow_credentials=True,
        allow_methods=settings.ALLOWED_METHODS or ["*"],
        allow_headers=settings.ALLOWED_HEADERS or ["*"],
    )
    
    
    app.include_router(router)
    
    @app.get("/health", tags=["System"])
    async def health_check():
        return {"status": "ok", "project": settings.PROJECT_NAME, "version": settings.VERSION}

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        logger.info(f"➡️ {request.method} {request.url}")
        response = await call_next(request)
        logger.info(f"⬅️ {response.status_code} {request.method} {request.url}")
        return response

    logger.info(f"🚀 FastAPI application created: {settings.PROJECT_NAME} v{settings.VERSION}")
    return app

# Create app instance
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD
    )
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

import structlog
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .rag import KnowledgeBase
from .config import Config

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class QueryRequest(BaseModel):
    """Request model for querying the knowledge base."""

    question: str = Field(
        ...,
        description="The question to ask the knowledge base",
        json_schema_extra={
            "example": "What were my thoughts on microservices architecture?"
        },
    )
    k: Optional[int] = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of similar documents to retrieve. Increase for broader context, decrease for more focused responses.",
    )
    filter_tags: Optional[List[str]] = Field(
        default=None,
        description="List of tags to filter the search results",
        json_schema_extra={"example": ["tech", "architecture"]},
    )


class Source(BaseModel):
    """Model representing a source document."""

    title: str = Field(..., description="Title of the source document")
    source: str = Field(..., description="Path to the source document")
    tags: List[str] = Field(..., description="Tags associated with the document")


class QueryResponse(BaseModel):
    """Response model for knowledge base queries."""

    answer: str = Field(..., description="Generated answer based on the knowledge base")
    sources: List[Source] = Field(
        ..., description="List of source documents used to generate the answer"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = Config.from_toml("config.toml")

    logger.info("initializing_knowledge_base")
    try:
        app.state.rag = KnowledgeBase(config=config, persist_directory="./chroma_db")
        app.state.rag.initialize_vector_store(force_reload=False)
        logger.info("knowledge_base_initialized")
    except Exception as e:
        logger.error("knowledge_base_initialization_failed", error=str(e))
        raise
    yield
    # TODO Cleanup if needed


app = FastAPI(
    title="Knowledge Base API",
    description="""
    API for querying a personal knowledge base built from Obsidian notes.
    Uses RAG (Retrieval Augmented Generation) to provide context-aware responses.
    """,
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.get(
    "/health",
    summary="Check API health",
    response_description="Returns the health status of the API",
)
async def health_check():
    """
    Checks if the API is running and the knowledge base is initialized.
    """
    try:
        if not app.state.rag:
            logger.error("health_check_failed", reason="knowledge_base_not_initialized")
            raise HTTPException(
                status_code=503, detail="Knowledge base not initialized"
            )
        return {"status": "healthy"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("health_check_failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the knowledge base",
    response_description="Returns an answer based on the knowledge base content",
)
async def query(request: QueryRequest):
    """
    Queries the knowledge base with a question and returns an answer based on relevant documents.

    - Use `k` to control how many documents to retrieve (1-10)
    - Optionally filter by tags to narrow the search scope
    - Returns both the answer and the source documents used
    """
    try:
        if not app.state.rag:
            logger.error("query_failed", reason="knowledge_base_not_initialized")
            raise HTTPException(
                status_code=503, detail="Knowledge base not initialized"
            )

        log = logger.bind(
            question=request.question, k=request.k, filter_tags=request.filter_tags
        )
        log.info("processing_query")

        response = await app.state.rag.query(
            question=request.question, k=request.k, filter_tags=request.filter_tags
        )
        log.info("query_processed")
        return response
    except Exception as e:
        logger.error("query_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

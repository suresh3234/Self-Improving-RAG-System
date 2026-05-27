# from src.core.pipeline import run_rag_pipeline
#
# app = FastAPI()
#
# class Query(BaseModel):
#     query: str
#
# @app.get("/")
# def home():
#     return {"message": "RAG API Running"}
#
# @app.post("/query")
# def query_api(q: Query):
#     return run_rag_pipeline(q.query)
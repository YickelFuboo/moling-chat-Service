from .rag.retrieval import search
from .graphrag import search as kg_search
from app.services.common.doc_vector_store_service import DOC_STORE_CONN


# 检索器
RETRIEVALER = search.Dealer(DOC_STORE_CONN)
KG_RETRIEVALER = kg_search.KGSearch(DOC_STORE_CONN)
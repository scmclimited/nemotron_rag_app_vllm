from typing import List, Dict, Any

class HybridVectorStore:
    '''Thin abstraction over deep_rag's vector_db schema.'''

    async def retrieve(
        self,
        query: str,
        cross_doc: bool,
        selected_doc_ids: List[str],
        uploaded_doc_ids: List[str],
    ) -> List[Dict[str, Any]]:
        # TODO: implement pgvector + lexical retrieval using copied vector_db models.
        return []

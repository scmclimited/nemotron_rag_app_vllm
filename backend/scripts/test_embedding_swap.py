#!/usr/bin/env python3
"""
Test script to swap embeddings mid-run and verify pipeline behavior.

This script demonstrates the embedding toggle capability by:
1. Initializing CLIP embeddings
2. Running a query
3. Swapping to Nemotron embeddings mid-execution
4. Comparing results
5. Analyzing performance deltas
"""
import asyncio
import sys
import logging
from pathlib import Path
import json
from typing import Dict, Any

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from app.config import settings
from app.rag.graph import run_rag_query
from app.rag.embedding_providers import (
    get_embedding_provider,
    CLIPEmbeddingProvider,
    NemotronEmbeddingProvider,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


async def test_embedding_swap(
    query: str,
    doc_ids: list[str] | None = None,
    cross_doc: bool = True,
) -> Dict[str, Any]:
    """
    Test embedding swapping during pipeline execution.
    
    Args:
        query: Question to ask
        doc_ids: Optional document IDs to limit search to
        cross_doc: Whether to allow cross-document search
    
    Returns:
        Dictionary with detailed comparison results
    """
    logger.info("=" * 80)
    logger.info("Embedding Swap Test")
    logger.info("=" * 80)
    
    results = {
        "query": query,
        "runs": {},
        "comparison": {},
    }
    
    # Test 1: Run with CLIP
    logger.info("\n[TEST 1] Running with CLIP embeddings...")
    logger.info("-" * 80)
    
    clip_result = await run_rag_query(
        query=query,
        selected_doc_ids=doc_ids or [],
        cross_doc=cross_doc,
        embedding_model="clip",
    )
    
    results["runs"]["clip"] = {
        "answer": clip_result.get("answer", "")[:200],
        "confidence": clip_result.get("confidence", 0.0),
        "citations": len(clip_result.get("citations", [])),
        "metadata": clip_result.get("metadata", {}),
    }
    
    logger.info(f"✓ CLIP run complete")
    logger.info(f"  - Answer: {clip_result.get('answer', '')[:100]}...")
    logger.info(f"  - Confidence: {clip_result.get('confidence', 0.0):.2f}")
    logger.info(f"  - Citations: {len(clip_result.get('citations', []))}")
    
    # Test 2: Run with Nemotron
    logger.info("\n[TEST 2] Running with Nemotron embeddings...")
    logger.info("-" * 80)
    
    nemotron_result = await run_rag_query(
        query=query,
        selected_doc_ids=doc_ids or [],
        cross_doc=cross_doc,
        embedding_model="nemotron",
    )
    
    results["runs"]["nemotron"] = {
        "answer": nemotron_result.get("answer", "")[:200],
        "confidence": nemotron_result.get("confidence", 0.0),
        "citations": len(nemotron_result.get("citations", [])),
        "metadata": nemotron_result.get("metadata", {}),
    }
    
    logger.info(f"✓ Nemotron run complete")
    logger.info(f"  - Answer: {nemotron_result.get('answer', '')[:100]}...")
    logger.info(f"  - Confidence: {nemotron_result.get('confidence', 0.0):.2f}")
    logger.info(f"  - Citations: {len(nemotron_result.get('citations', []))}")
    
    # Test 3: Test mid-run swap (conceptual)
    logger.info("\n[TEST 3] Testing embedding provider switching...")
    logger.info("-" * 80)
    
    try:
        clip_provider = CLIPEmbeddingProvider()
        test_text = "How can machine learning improve efficiency?"
        
        clip_embedding = clip_provider.embed_text(test_text)
        logger.info(f"✓ CLIP embedding created: {len(clip_embedding)} dimensions")
        
        nemotron_provider = NemotronEmbeddingProvider()
        nemotron_embedding = nemotron_provider.embed_text(test_text)
        logger.info(f"✓ Nemotron embedding created: {len(nemotron_embedding)} dimensions")
        
        # Calculate embedding space similarity
        import numpy as np
        cosine_sim = np.dot(clip_embedding, nemotron_embedding)
        results["comparison"]["embedding_space_similarity"] = float(cosine_sim)
        logger.info(f"  - Embedding space cosine similarity: {cosine_sim:.4f}")
        
    except Exception as e:
        logger.error(f"Error testing embedding providers: {e}")
        results["comparison"]["embedding_swap_error"] = str(e)
    
    # Build comparison metrics
    clip_conf = results["runs"]["clip"]["confidence"]
    nemotron_conf = results["runs"]["nemotron"]["confidence"]
    
    results["comparison"]["confidence_delta"] = nemotron_conf - clip_conf
    results["comparison"]["confidence_interpretation"] = (
        "Nemotron more confident" if results["comparison"]["confidence_delta"] > 0
        else "CLIP more confident" if results["comparison"]["confidence_delta"] < 0
        else "Equal confidence"
    )
    
    clip_cites = results["runs"]["clip"]["citations"]
    nemotron_cites = results["runs"]["nemotron"]["citations"]
    
    results["comparison"]["citation_delta"] = nemotron_cites - clip_cites
    results["comparison"]["citation_interpretation"] = (
        "Nemotron retrieved more chunks" if results["comparison"]["citation_delta"] > 0
        else "CLIP retrieved more chunks" if results["comparison"]["citation_delta"] < 0
        else "Equal citations"
    )
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Confidence delta: {results['comparison']['confidence_delta']:+.2f}")
    logger.info(f"  → {results['comparison']['confidence_interpretation']}")
    logger.info(f"Citation delta: {results['comparison']['citation_delta']:+d}")
    logger.info(f"  → {results['comparison']['citation_interpretation']}")
    logger.info(f"Embedding similarity: {results['comparison'].get('embedding_space_similarity', 'N/A')}")
    logger.info("=" * 80)
    
    return results


async def run_batch_tests(queries: list[str]) -> Dict[str, Any]:
    """
    Run multiple test queries comparing both embedding models.
    
    Args:
        queries: List of test questions
    
    Returns:
        Aggregated results across all queries
    """
    logger.info("=" * 80)
    logger.info("BATCH EMBEDDING COMPARISON TEST")
    logger.info("=" * 80)
    
    batch_results = {
        "total_queries": len(queries),
        "queries": {},
        "aggregate": {
            "avg_confidence_delta": 0.0,
            "avg_citation_delta": 0.0,
            "nemotron_better_confidence": 0,
            "clip_better_confidence": 0,
            "nemotron_better_citations": 0,
            "clip_better_citations": 0,
        },
    }
    
    for i, query in enumerate(queries, 1):
        logger.info(f"\n[Query {i}/{len(queries)}]")
        result = await test_embedding_swap(query)
        batch_results["queries"][query] = result
        
        # Aggregate metrics
        conf_delta = result["comparison"].get("confidence_delta", 0)
        cite_delta = result["comparison"].get("citation_delta", 0)
        
        batch_results["aggregate"]["avg_confidence_delta"] += conf_delta / len(queries)
        batch_results["aggregate"]["avg_citation_delta"] += cite_delta / len(queries)
        
        if conf_delta > 0:
            batch_results["aggregate"]["nemotron_better_confidence"] += 1
        elif conf_delta < 0:
            batch_results["aggregate"]["clip_better_confidence"] += 1
        
        if cite_delta > 0:
            batch_results["aggregate"]["nemotron_better_citations"] += 1
        elif cite_delta < 0:
            batch_results["aggregate"]["clip_better_citations"] += 1
    
    logger.info("\n" + "=" * 80)
    logger.info("BATCH RESULTS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Avg confidence delta: {batch_results['aggregate']['avg_confidence_delta']:+.3f}")
    logger.info(f"Nemotron better (confidence): {batch_results['aggregate']['nemotron_better_confidence']}")
    logger.info(f"CLIP better (confidence): {batch_results['aggregate']['clip_better_confidence']}")
    logger.info(f"Nemotron better (citations): {batch_results['aggregate']['nemotron_better_citations']}")
    logger.info(f"CLIP better (citations): {batch_results['aggregate']['clip_better_citations']}")
    logger.info("=" * 80)
    
    return batch_results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test embedding swap functionality"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to test"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run batch test with predefined queries"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (JSON)"
    )
    
    args = parser.parse_args()
    
    if args.batch:
        test_queries = [
            "What are the key features of machine learning?",
            "Explain the difference between supervised and unsupervised learning.",
            "How does neural network training work?",
            "What is the purpose of data normalization?",
            "Describe common classification algorithms.",
        ]
        results = asyncio.run(run_batch_tests(test_queries))
    else:
        query = args.query or "What is machine learning?"
        results = asyncio.run(test_embedding_swap(query))
    
    # Save results if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to {output_path}")
    else:
        logger.info("\nResults:")
        logger.info(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()


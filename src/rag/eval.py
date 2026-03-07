import asyncio 
from llama_index.core.evaluation import (FaithfulnessEvaluator,
                                         RelevancyEvaluator,
                                         BatchEvalRunner)
from src.llm.llm_interface import get_llm 
from src.config.logger import logger 

"""
1. FaithfulnessEvaluator: Checks if the answer is grounded in the retrieved context (no hallucination).
2. RelevancyEvaluator: Is the answer relevant to the question

"""
async def evaluate_router(router,subject_name: str):
    if not subject_name or subject_name == "Unknown":
        logger.warning("[EVAL] Skipping — subject_name is 'Unknown', profile fetch likely failed")
        return {}
    
    llm = get_llm()

    eval_questions = [
        f"What is {subject_name}'s current job title?",
        f"Where did {subject_name} go to school?",
        f"What companies has {subject_name} worked at?",
        f"What is {subject_name}'s location?",
        f"How many years of experience does {subject_name} have?",
    ]

    
    runner = BatchEvalRunner(evaluators={
        "faithfulness":FaithfulnessEvaluator(llm=llm),
        "relevancy":RelevancyEvaluator(llm=llm)},
        workers=2)
    
    results = await runner.aevaluate_queries(query_engine=router,
                                             queries=eval_questions)
    
    for metric,eval_results in results.items():
        scores = [r.score for r in eval_results if r.score is not None]
        avg = sum(scores)/len(scores) if scores else 0 
        logger.info(f"{metric} avg score: {avg:.2f} ({len(scores)}) queries")
    
    return results
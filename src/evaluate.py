"""
Ragas Evaluation for Ribbon Chatbot
RUBRIC: Evaluation Framework (RAGAS) 
- RAGAS evaluation implemented
- Golden dataset created
- All four metrics computed 
- Results saved with pass/fail logic 

TASK: Implement Ragas evaluation with 4 metrics
"""
import os
import json
import logging
import asyncio
import pandas as pd
from pathlib import Path
from typing import List, Dict

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

from src.search_engine import RibbonSearchEngine
from src.config import Config

# HINT: Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluation")


class RibbonChatbotEvaluator:
    """Evaluates Ribbon Chatbot using Ragas metrics"""
    
    def __init__(self):
        # HINT: Initialize search engine and golden dataset path
        self.engine = RibbonSearchEngine() 
        self.golden_dataset_path = Path("data") / "golden_dataset.json"  # HINT: "data", "golden_dataset.json"
    
    def load_golden_dataset(self) -> List[Dict]:
        """
        Load golden dataset for evaluation
        
        HINT: Check if file exists, if not create sample dataset
        """
        if not self.golden_dataset_path.exists(): 
            logger.warning(f"Golden dataset not found at {self.golden_dataset_path}")
            logger.info("Creating sample golden dataset...")
            return self._create_sample_dataset()  
        
        with open(self.golden_dataset_path, 'r') as f: 
            return json.load(f)  
    
    def _create_sample_dataset(self) -> List[Dict]:
        """
        Create sample golden dataset if not exists
        
        HINT: Create list of dicts with 'question' and 'ground_truth' keys
        Save to golden_dataset_path
        """
        sample_data = [
            {
                "question": "What are the baggage allowance rules for international flights?",  # HINT: "What are the baggage allowance rules for international flights?"
                "ground_truth": "The baggage allowance for international flights varies by airline and cabin class. Typically, economy passengers are allowed 20-23 kg of checked baggage, while business and first-class passengers may have higher allowances."  # HINT: Appropriate ground truth answer
            },
            {
                "question": "What is Air India's cancellation policy?",  # HINT: "What is Air India's cancellation policy?"
                "ground_truth": "Air India's cancellation policy depends on the fare type and booking class. Generally, cancellations are subject to fees and restrictions, with some fares being non-refundable."  # HINT: Appropriate ground truth answer
            },
            {
                "question": "Do I need a visa to travel from India to UK?",  # HINT: "Do I need a visa to travel from India to UK?"
                "ground_truth": "Yes, Indian citizens need a visa to travel to the UK."  # HINT: Appropriate ground truth answer
            },
            {
                "question": "What are the refund policies for flight cancellations?",  # HINT: "What are the refund policies for flight cancellations?"
                "ground_truth": "Refund policies vary by airline. Generally, refunds are processed within 7-14 business days."  # HINT: Appropriate ground truth answer
            },
            {
                "question": "What documents do I need for international travel?",  # HINT: "What documents do I need for international travel?"
                "ground_truth": "You will need a valid passport and possibly a visa depending on your destination."  # HINT: Appropriate ground truth answer
            }
        ]
        
        # HINT: Save sample dataset
        self.golden_dataset_path.parent.mkdir(exist_ok=True)
        with open(self.golden_dataset_path, 'w') as f:  
            json.dump(sample_data, f, indent=2)
        
        logger.info(f"Sample golden dataset saved to {self.golden_dataset_path}")
        return sample_data
    
    def generate_responses(self, questions: List[str]) -> tuple:
        """
        Generate responses for questions
        
        HINT: For each question:
        1. Search for documents
        2. Synthesize response
        3. Collect contexts
        Return (answers, contexts)
        """
        answers = []
        contexts = []
        
        for question in questions:
            logger.info(f"Generating answer for: {question}")
            
            try:
                # HINT: Search for relevant documents
                docs, _ = self.engine.search_by_text(question, k=5) 
                
                # HINT: Generate answer
                answer = self.engine.synthesize_response(docs, question)
                
                # HINT: Collect contexts (retrieved documents)
                context_texts = [doc.page_content  for doc in docs]
                
                answers.append(answer)
                contexts.append(context_texts)
                
            except Exception as e:
                logger.error(f"Error generating answer for '{question}': {e}")
                answers.append("Error generating answer.") 
                contexts.append([])
        
        return answers, contexts
    
    async def run_ragas_evaluation(self):
        """
        Run Ragas evaluation
        
        HINT: This method should:
        1. Load golden dataset
        2. Generate responses
        3. Prepare dataset dict
        4. Run Ragas evaluation with 4 metrics
        5. Save results
        """
        logger.info("=" * 70)
        logger.info("Starting Ragas Evaluation...")
        logger.info("=" * 70)
        
        # HINT: Load golden dataset
        golden_data = self.load_golden_dataset()  
        
        if not golden_data:
            logger.error("No evaluation data available")
            return None
        
        logger.info(f"Loaded {len(golden_data)} test cases")
        
        # HINT: Extract questions and ground truths
        questions = [item["question"] for item in golden_data] 
        ground_truths = [item["ground_truth"] if isinstance(item["ground_truth"], str) else item["ground_truth"][0] for item in golden_data]
        
        # HINT: Generate answers and contexts
        logger.info("\nGenerating responses...")
        answers, contexts = self.generate_responses(questions) 
        
        # HINT: Prepare dataset for Ragas
        dataset_dict = {
            "question": questions,  
            "answer": answers,   
            "contexts": contexts,  
            "ground_truth": ground_truths  
        }
        
        # HINT: Create HuggingFace Dataset
        hf_dataset = Dataset.from_dict(dataset_dict) 
        
        logger.info("\nRunning Ragas metrics...")
        logger.info("Metrics: faithfulness, answer_relevancy, context_precision, context_recall")
        
        # HINT: Run evaluation
        try:
            results = evaluate(
                hf_dataset,
                metrics=[
                    faithfulness,  # HINT: faithfulness
                    answer_relevancy,  # HINT: answer_relevancy
                    context_precision,  # HINT: context_precision
                    context_recall   # HINT: context_recall
                ],
            )
            
            logger.info("\n" + "=" * 70)
            logger.info("EVALUATION RESULTS")
            logger.info("=" * 70)
            logger.info(f"\nRagas Scores:")
            scores = results.to_pandas()
            faith = scores['faithfulness'].mean()
            relevancy = scores['answer_relevancy'].mean()
            precision = scores['context_precision'].mean()
            recall = scores['context_recall'].mean()

            logger.info(f"  Faithfulness:       {faith:.4f}")  
            logger.info(f"  Answer Relevancy:   {relevancy:.4f}")   
            logger.info(f"  Context Precision:  {precision:.4f}") 
            logger.info(f"  Context Recall:     {recall:.4f}")
            logger.info("=" * 70)
            
            # HINT: Save detailed results
            self._save_results(results, dataset_dict)  # HINT: _save_results
            
            return results
            
        except Exception as e:
            logger.error(f"Ragas evaluation failed: {e}")
            logger.error("Make sure you have OPENAI_API_KEY set for Ragas to work")
            return None
    
    def _save_results(self, results: dict, dataset_dict: dict):
        """
        Save evaluation results to file
        
        HINT: Save summary JSON and detailed CSV
        """
        output_dir = Path("reports")  # HINT: "reports"
        output_dir.mkdir(exist_ok=True)
        scores = results.to_pandas()
        # HINT: Save summary
        summary = {
        "faithfulness": float(scores['faithfulness'].mean()),
        "answer_relevancy": float(scores['answer_relevancy'].mean()),
        "context_precision": float(scores['context_precision'].mean()),
        "context_recall": float(scores['context_recall'].mean()),
        "total_test_cases": len(dataset_dict["question"])
    }
        
        summary_path = output_dir / "evaluation_summary.json"  # HINT: "evaluation_summary.json"
        with open(summary_path, 'w') as f:  # HINT: 'w'
            json.dump(summary, f, indent=2)  # HINT: dump
        
        logger.info(f"\n✅ Evaluation summary saved to {summary_path}")
        
        # HINT: Save detailed results
        detailed_df = pd.DataFrame(dataset_dict)  # HINT: dataset_dict
        detailed_path = output_dir / "evaluation_detailed.csv"  # HINT: "evaluation_detailed.csv"
        detailed_df.to_csv(detailed_path, index=False)
        
        logger.info(f"✅ Detailed results saved to {detailed_path}")
    
    def run(self):
        """Run evaluation (sync wrapper)"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.run_ragas_evaluation())


def run_evaluation():
    """
    Main evaluation function
    
    HINT: Run evaluator and check if results pass thresholds
    """
    evaluator = RibbonChatbotEvaluator()  
    results = evaluator.run()
    if asyncio.iscoroutine(results):
      results = asyncio.run(results)
    
    
    if results:
        # HINT: Check if evaluation passes minimum thresholds
        min_faithfulness =0.7 
        min_relevancy = 0.7 
        
        scores = results.to_pandas()
        passed = (
            float(scores['faithfulness'].mean()) >= min_faithfulness and
            float(scores['answer_relevancy'].mean()) >= min_relevancy
        )
        
        if passed:
            logger.info("\n✅ EVALUATION PASSED")
            return 0
        else:
            logger.warning("\n⚠️  EVALUATION BELOW THRESHOLDS")
            return 1
    else:
        logger.error("\n❌ EVALUATION FAILED")
        return 1


if __name__ == "__main__":
    import sys
    exit_code = run_evaluation()
    sys.exit(exit_code)
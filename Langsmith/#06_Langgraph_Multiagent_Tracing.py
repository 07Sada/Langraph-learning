from click import prompt
from ast import operator
from sqlalchemy import desc
from dotenv import load_dotenv
from typing import TypedDict, List, Annotated
from pydantic import BaseModel, Field
import operator
import os 

from langsmith import traceable
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()
llm_model = ChatOpenAI(model='gpt-4o-mini', temperature=0)

class EvaluationSchema(BaseModel):
    feedback:str = Field(description="Detailed Feedback for the essay")
    score: int = Field(description="Score out of 10", ge=1, le=10)

structured_model = llm_model.with_structured_output(EvaluationSchema)

# Loading the essay 
ESSAY_FILE = 'test_essay.txt'
with open(ESSAY_FILE, 'r', encoding='utf-8') as file:
    essay = file.read()

class UPSCState(TypedDict, total=False):
    essay: str
    language_feedback: str
    analysis_feedback: str 
    clearity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[List[int], operator.add]
    avg_score: float

@traceable(name="evaluate_language_fn", tags=["dimension:language"], metadata={"dimension": "language"})
def evaluate_language(state: UPSCState):
    prompt = (
        "Evaluate the language quality of the following essay and provide feedback "
        "and assign a score out of 10. \n\n"+ state['essay']
    )
    model_output = structured_model.invoke(prompt)
    return {'language_feedback': model_output.feedback, 'individual_scores': [model_output.score]}

@traceable(name="evaluate_analysis_fn", tags=["dimension:analysis"], metadata={"dimension": "analysis"})
def evaluate_analysis(state: UPSCState):
    prompt = (
        "Evaluate the depth of analysis of the following essay and provide feedback "
        "and assign a score out of 10. \n\n"+ state['essay']
    )
    model_output = structured_model.invoke(prompt)
    return {'analysis_feedback': model_output.feedback, 'individual_scores': [model_output.score]}

@traceable(name="evaluate_thought_fn", tags=["dimension:clarity"], metadata={"dimension": "clarity_of_thought"})
def evaluate_thought(state: UPSCState):
    prompt = (
        "Evaluate the clarity of thought of the following essay and provide feedback "
        "and assign a score out of 10. \n\n"+ state['essay']
    )
    model_output = structured_model.invoke(prompt)
    return {'clearity_feedback': model_output.feedback, 'individual_scores': [model_output.score]}

@traceable(name="final_evaluation_fn", tags=["aggregate"])
def final_evaluation(state:UPSCState):
    prompt = (
        "Based on the following feedback, create a summarized overall feedback. \n\n"
        f"Language feedback: {state.get('language_feedback', '')}\n"
        f"Deptgh of analysis: {state.get('analysis_feedback', '')}\n"
        f"Clearity of thoughts: {state.get('clearity_feedback', '')}\n"
    )
    overall = llm_model.invoke(prompt).content
    scores = state.get("individual_scores", []) or []
    avg = (sum(scores) / len(scores)) if scores else 0.0
    return {"overall_feedback": overall, "avg_score": avg}

graph = StateGraph(UPSCState)
graph.add_node('evaluate_language', evaluate_language)
graph.add_node('evaluate_analysis', evaluate_analysis)
graph.add_node('evaluate_thought', evaluate_thought)
graph.add_node('final_evaluation', final_evaluation)


graph.add_edge(START, 'evaluate_language')
graph.add_edge(START, 'evaluate_analysis')
graph.add_edge(START, 'evaluate_thought')
graph.add_edge('evaluate_language', 'final_evaluation')
graph.add_edge('evaluate_analysis', 'final_evaluation')
graph.add_edge('evaluate_thought', 'final_evaluation')
graph.add_edge('final_evaluation', END)

workflow = graph.compile()

if __name__ == "__main__":
    result = workflow.invoke(
        {'essay':essay},
        config={
            'run_name':'evaluate_upsc_essay',
            'tags': ['essay', 'langgraph', 'evaluation'],
            'metadata':{
                'essay_length': len(essay),
                'model': 'gpt-4o-mini',
                'dimensions': ['language', 'analysis', 'clearity']
            }
        }
    )

    print("\n=== Evaluation Results ===")
    print("Language feedback:\n", result.get("language_feedback", ""), "\n")
    print("Analysis feedback:\n", result.get("analysis_feedback", ""), "\n")
    print("Clarity feedback:\n", result.get("clearity_feedback", ""), "\n")
    print("Overall feedback:\n", result.get("overall_feedback", ""), "\n")
    print("Individual scores:", result.get("individual_scores", []))
    print("Average score:", result.get("avg_score", 0.0))
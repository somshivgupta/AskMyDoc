from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(q: Query):
    context = retriever.retrieve(q.question)
    answer = generate_answer(q.question, context)
    return {"answer": answer}
from .rag_vectorstore import load_documents, split_into_chunks, create_faiss_index, load_faiss_index, retrieve_context
from openai import OpenAI

def handle_knowledge_base(files, kb_folder, index_path):
    import os
    if files:
        os.makedirs(kb_folder, exist_ok=True)
        for file in files:
            with open(os.path.join(kb_folder, file.name), "wb") as f:
                f.write(file.getbuffer())
        raw_docs = load_documents(kb_folder)
        chunks = split_into_chunks(raw_docs)
        return create_faiss_index(chunks, index_path)
    else:
        try:
            return load_faiss_index(index_path)
        except:
            return None

def handle_user_question(db, question, df, api_key):
    if db is None:
        return "RAG context unavailable. Please upload KB documents."

    context = retrieve_context(db, question)
    example_data = df.sample(1).to_dict(orient="records")[0]
    enhanced_prompt = f"""Use the following example performance data:\n{example_data}\n\nUse the retrieved KB context:\n{context}\n\nThen answer this:\n{question}\n\nBe concise and technical."""
    
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": enhanced_prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content

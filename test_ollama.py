import modal
app = modal.App("test-ollama")
image = (
    modal.Image.debian_slim()
    .pip_install(
        "langchain<0.3.0",
        "langchain-community<0.3.0",
        "llama-cpp-python",
        "langchain-core<0.3.0",
    )
)
@app.function(image=image)
def test():
    from langchain_core.prompts import PromptTemplate
    from langchain_community.llms import LlamaCpp
    
    llm = LlamaCpp(
        model_path="/model/empty.gguf", # this will fail to load but we can check imports
        temperature=0.1,
    )
    prompt = PromptTemplate.from_template("Question: {input}\nAnswer:")
    chain = prompt | llm
    
    print(chain)

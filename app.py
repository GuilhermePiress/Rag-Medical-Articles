import streamlit as st
from langchain_openai import ChatOpenAI
import io
from utils import (
    generate_slide_markdown_from_summary,
    export_markdown_to_pptx,
    build_compressed_retriever,
    build_qa_chain,
    translate_to_english,
    load_and_chunk_pdf,
    generate_summary_from_documents
)
#---------------------------------------------------------------------------------------------
st.title("Bem-vindo ao Doutora Sarah 👩")

st.sidebar.markdown("""
#### 📄 O que este app faz:
- 🧠 **Cria um resumo clínico do artigo**
- 📊 **Gera uma apresentação básica usando o resumo**
- ❓ **Responde dúvidas sobre o artigo caso perguntado**

#### 🔎 Leia antes de usar o app
⚠️ **Atenção:**
- O artigo **deve estar em inglês**
- **Figuras e imagens não são processadas**
- O foco é em **conteúdo clínico textual**
""")
## Model Option
model_name = st.sidebar.selectbox(
    "🧠 Escolha o modelo de linguagem:",
    options=["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
    index=2  # default = gpt-4o
)

# Temperature slider
temperature = st.sidebar.slider(
    "🎯 Temperatura da resposta:",
    min_value=0.0,
    max_value=1.0,
    value=0.0, # Default
    step=0.1,
    help="Valores maiores tornam a resposta mais criativa, valores menores mais objetivas"
)

# Cria a primeira área de criação de resumo do artigo
st.markdown("## 📄 Resumo do artigo")
st.write("Insira o artigo científico em inglês e em formato PDF")

# Streamlit session keys we need to not run the entire code again
for k in ["current_pdf", "llm", "qa_chain", "summary_text", "pptx_bytes", "summary_documents", "rag_documents", "num_pages"]:
    if k not in st.session_state:
        st.session_state[k] = None

# Cria o botão para receber o pdf
pdf_file = st.file_uploader("Escolha um arquivo PDF", type=["pdf"])

## Abre o pdf e lê os dados
if pdf_file is not None:
    # If the filename changed, reset the pipeline so we rebuild for this PDF
    if st.session_state.current_pdf != pdf_file.name:
        st.session_state.current_pdf = pdf_file.name
        st.session_state.qa_chain = None
        st.session_state.llm = None
        st.session_state.summary_text = None      # ← add this
        st.session_state.pptx_bytes = None        # ← add this

        st.info("⏳ Lendo o PDF...")
        # Summary chunks
        summary_chunks, num_pages = load_and_chunk_pdf(pdf_file, chunk_size=1500, chunk_overlap=200)
        # RAG chunks
        rag_chunks, _ = load_and_chunk_pdf(pdf_file, chunk_size=800, chunk_overlap=100)

        # Store both
        st.session_state.summary_documents = summary_chunks
        st.session_state.rag_documents = rag_chunks
        st.session_state.num_pages = num_pages

    # ✅ Always show success message if document is loaded
    st.success(f"✅ Arquivo carregado com {st.session_state.num_pages} página(s)")
    # Ask for API key
    api_key = st.text_input("🔑 Insira sua OpenAI API Key para criação do resumo:", 
                            type="password",
                            key="api_key_input")

    # Only continue if API key is provided
    if api_key:
        # Create (or reuse) the LLM once
        if st.session_state.llm is None:
            st.session_state.llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                openai_api_key=api_key
            )
        llm = st.session_state.llm
        # ===== IMPORTANT GUARD =====
        # Build summarization/PPT/retriever/QA ONLY if we don't already have a qa_chain
        if st.session_state.qa_chain is None:
            st.info("✍️ Criando o resumo do artigo e o arquivo em power point, isso pode levar alguns minutos...")
            summary_text = generate_summary_from_documents(st.session_state.summary_documents, llm)
            st.session_state.summary_text = summary_text
    #--------------------------------------------------------------------------------------
            # Create powerpoint file
            # Step 1 – Generate slide markdown from the summary
            slide_markdown = generate_slide_markdown_from_summary(st.session_state.summary_text, llm)
            
            # Step 2 – Create presentation and keep it in memory
            pptx_io = io.BytesIO()
            export_markdown_to_pptx(slide_markdown, output_path=pptx_io)
            pptx_io.seek(0)
            st.session_state.pptx_bytes = pptx_io.getvalue() 
        
    # ------------------------------------------------------------------------------------------------
            # Create QA file
            # Cria o retriever usando os documentos processados e o QA
            compression_retriever = build_compressed_retriever(st.session_state.rag_documents, llm)
            qa_chain = build_qa_chain(llm, compression_retriever)
            st.session_state.qa_chain = qa_chain  # Armazena para uso depois
                 
if st.session_state.qa_chain is not None and st.session_state.llm is not None:

    # This fix the result on the screen and not vanish after the questions
    if st.session_state.summary_text:
        st.markdown("#### 📝 Resumo do Artigo")
        st.write(st.session_state.summary_text)
        st.success(f"✅ Resumo criado com sucesso")
    if st.session_state.pptx_bytes:
        st.markdown("## 📊 Apresentação em PowerPoint")
        st.success("✅ Apresentação criada com sucesso, fazer o download clicando no botão abaixo")
        st.download_button(
            label="📥 Baixar apresentação (.pptx)",
            data=st.session_state.pptx_bytes,  # <-- use cached bytes
            file_name="resumo_artigo.pptx",
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
        )   

    st.markdown("## ❓ Pergunte sobre o artigo")

    st.info(
        "Você pode fazer perguntas sobre o conteúdo clínico do artigo.\n\n"
        "- A resposta será baseada **apenas no conteúdo do artigo**.\n"
        "- Escreva sua pergunta em **português**.\n"
        "- Você pode fazer **quantas perguntas quiser!**"
    )

    question_pt = st.text_input("Digite sua pergunta:")

    if question_pt:
        # Traduz a pergunta para o inglês antes de enviar para o modelo
        question_en = translate_to_english(question_pt, st.session_state.llm)

        # Executa a QA chain
        result = st.session_state.qa_chain.invoke({"question": question_en})
        # Exibe a resposta
        st.markdown("### 🧠 Resposta baseada no artigo:")
        st.write(result["answer"])
        st.success("✅ Pergunta respondida. Para fazer uma nova pergunta só digitar uma nova pergunta e apertar o enter!")

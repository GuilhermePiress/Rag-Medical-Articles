# 🧠 Doutora Sarah

**Doutora Sarah** é um aplicativo clínico interativo construído com [Streamlit](https://streamlit.io/) que utiliza uma arquitetura RAG (Retrieval-Augmented Generation) com modelos da OpenAI para auxiliar profissionais de saúde e estudantes na análise de artigos médicos.

---

## ✅ Funcionalidades

- 📄 **Upload de PDF clínico em inglês**
- 🧠 **Geração de resumo clínico estruturado**
- 📊 **Criação automática de apresentações PowerPoint a partir do resumo**
- ❓ **Perguntas e respostas sobre o conteúdo do artigo (QA)**
- 🌐 **Interface simples e interativa via navegador (Streamlit)**

---

## ⚙️ Como usar

1. Faça upload de um artigo **em inglês** no formato PDF.
2. Insira sua **chave da API da OpenAI**.
3. O app irá:
   - Ler e processar o conteúdo do PDF
   - Gerar um resumo clínico técnico e didático
   - Criar uma apresentação com slides (.pptx) para download
   - Permitir perguntas em português sobre o conteúdo do artigo

---

## 🧩 Tecnologias Utilizadas

- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [OpenAI GPT (gpt-3.5, gpt-4, gpt-4o)](https://platform.openai.com/docs)
- [FAISS](https://github.com/facebookresearch/faiss) para busca vetorial
- [PyPDFLoader](https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf) para leitura de PDFs
- [python-pptx](https://python-pptx.readthedocs.io/) para geração de slides

---

## 🚨 Requisitos e Observações

- O arquivo **deve estar em inglês** (resumos e QA são traduzidos para português automaticamente)
- **Figuras e imagens não são processadas**
- A aplicação foca **exclusivamente no conteúdo textual clínico**

---

## 🧪 Execução Local

Clone o repositório e instale os requisitos:

```bash
git clone https://github.com/seu-usuario/doutora-sarah.git
cd doutora-sarah
pip install -r requirements.txt
streamlit run app.py
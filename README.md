# RAG-Chatbot-LLM
This project enables users to interact with multiple PDFs, extracting information from the documents, and querying the content using a conversational AI interface. The application utilizes Groq's advanced AI model for generating responses to user queries, providing real-time answers based on the uploaded PDF documents.

**Features**:

PDF Upload: Users can upload multiple PDF files and the system processes them to extract text.

Text Chunking: The extracted text is split into smaller chunks for efficient searching and retrieval.

Vector Store: A vector store powered by Chroma is used to index the document's content, allowing for fast and accurate querying.

Conversational Interface: The app provides an interactive Q&A interface where users can ask questions related to the document. The system retrieves relevant information and answers based on the content of the uploaded PDFs.

Groq API Integration: Instead of traditional OpenAI models, the application is integrated with Groq's LLM for natural language processing, offering powerful and efficient question answering.

Custom Prompts: A custom template is used to improve the rephrasing of follow-up questions into standalone queries for better accuracy.


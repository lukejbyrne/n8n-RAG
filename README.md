# Local-RAG: Local Retrieval Augmented Generation

Local-RAG is a Python application that combines local document retrieval with a pre-trained language model (LLM) to provide fast and accurate responses. It uses ChromaDB for efficient document indexing and retrieval, and SentenceTransformers for document embeddings.

## Key Features

- Fast document retrieval using ChromaDB.
- Pre-trained LLM (e.g., OpenAI's GPT-3) for generating responses.
- Integration with Google PaLM for advanced language modeling.

## Installation Steps

1. Clone the repository: `git clone https://github.com/your-username/local-rag.git`
2. Navigate to the project directory: `cd local-rag`
3. Create a virtual environment (optional):
   - Run `./reset_env.sh` (Linux/macOS) or `reset_env.bat` (Windows) to create and activate a virtual environment.
4. Install dependencies: `pip install -r requirements.txt`

## Usage

1. Set up ChromaDB and Google PaLM by following the instructions in `app.py`.
2. Run the chat interface: `python chat.py`
3. Type your queries and enjoy the fast and accurate responses!

## Contributing

Contributions are welcome! Please follow these guidelines:

- Fork the repository and create a new branch for your changes.
- Make your changes and ensure they follow the project's coding style.
- Write clear and concise commit messages.
- Submit a pull request and describe your changes.

## License

This project is licensed under the MIT License.

## Acknowledgments

- ChromaDB: https://github.com/chroma-core/chroma
- SentenceTransformers: https://github.com/UKPLab/sentence-transformers
- OpenAI's GPT-3: https://openai.com/blog/gpt-3-apps/
- Google PaLM: https://ai.googleblog.com/2022/10/palm-scaling-language-model-training.html

## Badges

[![GitHub stars](https://img.shields.io/github/stars/your-username/local-rag)](https://github.com/your-username/local-rag/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/your-username/local-rag)](https://github.com/your-username/local-rag/network)
[![GitHub issues](https://img.shields.io/github/issues/your-username/local-rag)](https://github.com/your-username/local-rag/issues)
[![GitHub license](https://img.shields.io/github/license/your-username/local-rag)](https://github.com/your-username/local-rag/blob/main/LICENSE)

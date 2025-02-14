# Company Documents Chat - AI-Powered HR Assistant

## Overview
This project is an AI-powered HR assistant that answers employee questions based on company policies. It retrieves relevant information from internal company documents using Google Drive, embeds the text into a Pinecone vector database, and queries the Groq chat model to provide accurate responses.

## Features
- **Google Drive Integration**: Fetches and processes company documents automatically.
- **Pinecone Vector Database**: Stores and retrieves document embeddings efficiently.
- **Google Gemini API**: Generates embeddings for text chunks.
- **Groq Chat Model**: Provides AI-powered responses to user queries.
- **Interactive Chat Interface**: Users can ask HR-related questions via a console-based chat system.
- **Automated Document Processing**: New and updated documents are automatically processed and indexed.

---

## Prerequisites
Before running the project, ensure you have the following:
- Python 3.8+
- A Google Cloud service account with Drive API access
- A Pinecone API key
- A Google Gemini API key
- A Groq API key

### Required Python Packages
Install the dependencies using pip:
```bash
pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client pinecone-client google-generativeai rich dotenv groq
```

---

## Setup
### 1. Environment Variables
Create a `.env` file in the project root and add the following environment variables:
```env
GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE=path/to/service-account.json
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENV=your-pinecone-environment
PINECONE_INDEX_NAME=your-pinecone-index-name
GOOGLE_GEMINI_API_KEY=your-google-gemini-api-key
GROQ_API_KEY=your-groq-api-key
GOOGLE_DRIVE_FOLDER_ID=your-google-drive-folder-id
```

### 2. Configure Google Drive API
- Enable the Google Drive API in the Google Cloud Console.
- Generate a service account JSON key and specify its path in the `.env` file.

### 3. Initialize Pinecone Index
Ensure your Pinecone index exists, or the script will create one automatically.

---

## Running the Application

### Start the HR Chat Assistant
```bash
python chat.py
```

### Manually Update Indexed Documents
```bash
python app.py
```

---

## How It Works
### Document Processing (`app.py`)
1. **Fetch Files**: Retrieves documents from Google Drive.
2. **Process Text**: Downloads, splits, and embeds document content.
3. **Store in Pinecone**: Saves text chunks and embeddings in a vector database.
4. **Track Changes**: Detects new, modified, or deleted files.

### Chat Assistant (`chat.py`)
1. **User Query**: Accepts a question from the user.
2. **Retrieve Relevant Data**: Queries Pinecone for similar document sections.
3. **Generate Response**: Uses Groq's chat model to generate an HR response.

---

## Example Usage
```
Your Question> What is the company's leave policy?

Answer: Employees are entitled to 28 days of annual leave, including public holidays. Refer to the HR document for more details.
```

---

## Future Improvements
- Implement a web-based UI.
- Support multiple file formats (e.g., PDFs, Word docs).
- Enhance document processing speed.
- Add authentication for user queries.

---

## License
This project is licensed under the MIT License.

## Contributions
Feel free to submit pull requests or open issues for suggestions and improvements.


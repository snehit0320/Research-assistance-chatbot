# Research Assistance Chatbot

🧠 **ResearchMate AI** — Advanced Research Assistant with Quality Control

A comprehensive research assistance tool that helps researchers generate high-quality academic papers with automatic quality checking, AI-powered content generation, and intelligent document analysis.

## Features

### 💬 Chat Assistant
- Upload PDFs or set a research topic for AI-powered Q&A
- Intelligent RAG (Retrieval-Augmented Generation) system
- Context-aware responses based on uploaded documents or research topics
- Session management and chat history

### 📝 Paper Generator with Quality Control
- Generate complete research papers with all standard sections:
  - Abstract
  - Introduction
  - Literature Review
  - Methodology
  - Results and Discussion
  - Conclusion
  - References
- **Automatic Quality Checks:**
  - ✅ AI Content Detection
  - ✅ Plagiarism Checking
  - ✅ Text Humanization
  - ✅ Grammar Correction
- Export to DOCX and PDF formats

### 📊 Analytics
- Session statistics and insights
- Keyword extraction and visualization
- Document analysis metrics

## Installation

1. Clone this repository:
```bash
git clone https://github.com/snehit0320/Research-assistance-chatbot.git
cd Research-assistance-chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Setup

Before running the application, you need to configure API keys:

1. **OpenRouter API Key**: Get your free API key from [OpenRouter](https://openrouter.ai/)
2. **RapidAPI Key**: Get your API key from [RapidAPI](https://rapidapi.com/)

Edit the API keys in the Python files:
```python
OPENROUTER_API_KEY = "your-openrouter-api-key"
RAPIDAPI_KEY = "your-rapidapi-key"
```

## Usage

Run the application:
```bash
python researchmate_ai_fixed.py
```

Or use the complete version:
```bash
python researchmate_ai_complete.py
```

The application will launch a Gradio interface that you can access in your browser or via a shareable link.

## Files

- `researchmate_ai_complete.py` - Complete version with all features
- `researchmate_ai_fixed.py` - Fixed/optimized version with enhanced section generation

## Technologies Used

- **Gradio** - Web interface
- **ChromaDB** - Vector database for RAG
- **Sentence Transformers** - Embeddings
- **PyMuPDF** - PDF processing
- **arXiv API** - Research paper fetching
- **OpenRouter** - AI model access
- **RapidAPI** - Quality check services

## API Services

This application uses several external APIs for quality checking:
- AI Detection API
- Plagiarism Checker API
- Text Humanization API
- Grammar Check API

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available for educational purposes.

## Disclaimer

This tool is designed to assist researchers. Always review and verify generated content. Ensure compliance with your institution's academic integrity policies.


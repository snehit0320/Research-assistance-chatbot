# ResearchMate AI - Advanced Research Assistant

A comprehensive research assistance chatbot with quality control features including AI content detection, plagiarism checking, text humanization, and grammar correction.

## Features

- 📚 **Document Upload & Chat**: Upload PDFs or set research topics to interact with your documents
- 🤖 **AI-Powered Responses**: Uses OpenRouter API with multiple free models for reliable responses
- ✅ **Quality Control**:
  - AI Content Detection
  - Plagiarism Checking
  - Text Humanization
  - Grammar Correction
- 📝 **Paper Generation**: Generate complete research papers with quality checks
- 📊 **Analytics**: Session insights and keyword analysis

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/snehit0320/Research-assistance-chatbot.git
cd Research-assistance-chatbot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or if using the Jupyter notebook, dependencies will be installed automatically when you run the first cell.

### 3. Configure API Keys

1. Copy the example environment file:
   ```bash
   cp env.example .env
   ```

2. Edit the `.env` file and add your API keys:
   ```
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   RAPIDAPI_KEY=your_rapidapi_key_here
   ```

3. **Get API Keys**:
   - **OpenRouter API Key**: Sign up at [https://openrouter.ai/keys](https://openrouter.ai/keys)
   - **RapidAPI Key**: Sign up at [https://rapidapi.com/](https://rapidapi.com/) and subscribe to:
     - AI Detection API
     - Plagiarism Checker API
     - Humanize AI Content API
     - TextGears API

### 4. Run the Application

#### Option A: Using Jupyter Notebook

1. Open `chatbot.ipynb` in Jupyter Notebook or Google Colab
2. Run all cells
3. The Gradio interface will launch automatically

#### Option B: Using Python Scripts

Run either of the Python scripts:
```bash
python researchmate_ai_complete.py
```
or
```bash
python researchmate_ai_fixed.py
```

## File Descriptions

- **`chatbot.ipynb`**: Jupyter notebook version of the application (recommended for interactive use)
- **`researchmate_ai_complete.py`**: Complete standalone Python script version
- **`researchmate_ai_fixed.py`**: Fixed/improved version of the Python script with enhanced section generation
- **`env.example`**: Template for environment variables (copy to `.env` and fill in your keys)
- **`.gitignore`**: Git ignore file that excludes sensitive files like `.env`

## Security Notes

⚠️ **Important**: 
- Never commit your `.env` file to git
- The `.gitignore` file is configured to exclude `.env` files
- API keys are now loaded from environment variables, not hardcoded
- If you find any API keys in the repository history, please regenerate them immediately

## Usage

1. **Chat Assistant**: 
   - Set a research topic OR upload PDF files
   - Ask questions about your research topic or uploaded documents
   
2. **Paper Generator**:
   - Enter a research topic
   - Optionally enable quality processing
   - Generate and download papers in DOCX and PDF formats

3. **Analytics**:
   - View session statistics
   - See keyword analysis and visualizations

## Technologies Used

- **Gradio**: Web interface
- **OpenRouter**: AI model access
- **ChromaDB**: Vector database for RAG (Retrieval Augmented Generation)
- **Sentence Transformers**: Text embeddings
- **arXiv API**: Research paper fetching
- **RapidAPI**: Quality checking services

## Contributing

Contributions are welcome! Please ensure:
- API keys are never committed
- Use environment variables for all sensitive data
- Follow security best practices

## License

This project is open source and available for educational purposes.

## Support

For issues or questions, please open an issue on GitHub.


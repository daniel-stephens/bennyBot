# BennyBot Readme

BennyBot is an intelligent conversational assistant designed to provide responses using a Retrieval-Augmented Generation (RAG) model. It employs audio input and output for seamless interaction. This document outlines the structure and functionality of BennyBot, along with setup and usage instructions.

## Features
1. **Wake Word Detection:** Activates the system upon hearing the wake word "Hey Benny."
2. **Speech-to-Text Conversion:** Converts audio input into text.
3. **Answer Generation:** Uses a Retrieval-Augmented Generation (RAG) model powered by a large language model (LLM) to provide contextually relevant answers.
4. **Text-to-Speech Conversion:** Converts the generated response back into audio for a complete conversational experience.

## System Workflow
1. **Activation:** The wake word "Hey Benny" activates the system.
2. **Audio Input:** The user speaks their query into the system.
3. **Speech-to-Text:** Audio input is processed and converted into text using a speech recognition module.
4. **Query Processing:** The text input is passed to the RAG model to retrieve context and generate an appropriate response.
5. **Text-to-Speech:** The response text is converted to audio and played back to the user.

## Components
### 1. Wake Word Detector
- **Library/Tool:** Porcupine (or any similar wake word detection library).
- **Wake Word:** "Hey Benny."

### 2. Speech-to-Text
- **Library/Tool:** OpenAI Whisper, Google Speech-to-Text API, or any preferred ASR system.
- **Functionality:** Transcribes the user's audio input into text.

### 3. RAG Model
- **Libraries/Tools:**
  - LangChain for integrating the RAG model.
  - Chroma for document retrieval.
  - OpenAI or other LLM for text generation.
- **Functionality:** Retrieves context from a database and generates a response based on the user's query.

### 4. Text-to-Speech
- **Library/Tool:** Google Text-to-Speech (gTTS), AWS Polly, or pyttsx3.
- **Functionality:** Converts the response text into audio output.

## Setup Instructions
### Prerequisites
- Python 3.8 or later
- Libraries:
  - `pvporcupine` for wake word detection
  - `speechrecognition` for speech-to-text conversion
  - `langchain` and `chromadb` for RAG model integration
  - `pyttsx3` or equivalent for text-to-speech conversion

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/bennybot.git
   cd bennybot
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up API keys for LLM and speech recognition services in an `.env` file:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   SPEECH_API_KEY=your_speech_api_key
   ```

### Configuration
- Modify the wake word detection configuration to use "Hey Benny."
- Set up the RAG model with relevant documents or a database for retrieval.

## Usage
1. Run the bot:
   ```bash
   python bennybot.py
   ```
2. Say "Hey Benny" to activate the system.
3. Ask your query after activation.
4. Listen to BennyBot's audio response.

## Future Enhancements
- Multilingual support for input and output.
- Enhanced context retrieval for domain-specific queries.
- Integration with external APIs for additional functionalities.

## Contributing
Contributions are welcome! Please submit a pull request or raise an issue for any bugs or feature requests.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.


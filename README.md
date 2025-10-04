# 👁️ U-Tube AI: Beyond the Transcript

> While most AI agents only listen to YouTube videos, mine watches, capturing the complete picture to deliver unparalleled understanding and truly comprehensive notes.

![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)

---

## ✨ About The Project

U-Tube AI is a multi-modal AI agent that transforms YouTube videos into rich, actionable knowledge assets. Unlike traditional tools that rely solely on a video's transcript, U-Tube AI analyzes **both the audio and the visual stream**. It reads slides, understands diagrams, and captures code on screen, fusing what's said with what's shown to create notes that are incredibly detailed, accurate, and truly comprehensive.

This system is perfect for students, researchers, and professionals who need to efficiently digest complex tutorials, technical presentations, or academic lectures without losing critical visual context.

---

## 🧠 Key Features

### 1. True Visual Comprehension: Seeing What Others Miss
My core innovation is a pipeline that treats the video's visual stream as a first-class citizen, not an afterthought. It uses **adaptive frame sampling** to intelligently pinpoint key visual moments—like slides, diagrams, or code on screen—and employs **OCR** to extract this critical information. This captures the essential context that purely transcript-based systems are blind to.

### 2. Intelligent Dual-Model Synthesis
This isn't a brute-force approach. The system uses a nimble, cost-effective model (**Gemini-Flash**) to rapidly map the video's entire topic structure and create an outline. Then, a high-power, creative model (**Gemini-Pro**) performs a "deep dive" on each section, expertly synthesizing the spoken transcript and the extracted visual data into a single, coherent narrative.

### 3. Purpose-Built RAG for Deep Note Generation
I intentionally chose **ChromaDB** over speed-focused alternatives like FAISS. Why? Because generating high-quality notes requires more than just fast vector search—it demands a seamless RAG pipeline that preserves rich metadata and context. This choice is fundamental to the system's ability to assemble complex, structured documents rather than just answering simple questions.

### 4. From Raw Video to Actionable Knowledge
The final output isn't just a summary; it's a **synthesized knowledge asset**. By fusing what's said with what's shown, the system produces incredibly rich and reliable notes. This multi-modal understanding is then delivered in user-friendly formats like **Markdown**, **Word**, and **PDF**.

---

### Workflow at Glance:

[START: User provides a YouTube Video URL]
 |
 |
 +--> [1. Data Ingestion & Parallel Processing]
 |    |
 |    |--> [A. VISUAL ANALYSIS PATH]
 |    |    |
 |    |    +--> [Step A1: Video Stream Access] -> (Raw Video Frames)
 |    |    |
 |    |    +--> [Step A2: Adaptive Frame Sampling] -> (Selects Keyframes with high information density like slides, code, diagrams)
 |    |    |
 |    |    L--> [Step A3: Optical Character Recognition (OCR)] -> (Output: Extracted text from visuals, e.g., "def function():", "Q3 Financials")
 |    |
 |    |
 |    L--> [B. AUDIO ANALYSIS PATH]
 |         |
 |         L--> [Step B1: Transcript Extraction] -> (Output: Full spoken transcript with timestamps)
 |
 |
 +--> [2. Initial Structuring with Gemini-Flash]
 |    (Input: Both visual text and spoken transcript are fed to the nimble model)
 |    |
 |    L--> [Step 2A: Topic & Structure Mapping] -> (Output: A high-level outline of the video, e.g., "1. Intro, 2. Core Concept, 3. Demo, 4. Q&A")
 |
 |
 +--> [3. Deep Synthesis with Gemini-Pro]
 |    (For EACH section in the outline, the powerful model performs a deep dive)
 |    |
 |    L--> [Step 3A: Multi-Modal Fusion]
 |         (Input: Outline section + relevant transcript part + relevant visual text)
 |         |
 |         L--> (Output: A rich, synthesized narrative chunk for that specific section, combining what was said and shown)
 |
 |
 +--> [4. RAG-Powered Document Assembly with ChromaDB]
 |    |
 |    +--> [Step 4A: Storing Context] -> (Each synthesized chunk is stored in ChromaDB, preserving its metadata and relationship to the outline)
 |    |
 |    L--> [Step 4B: Coherent Retrieval] -> (The system retrieves the chunks in order, ensuring context flows correctly)
 |    |
 |    L--> (Output: Final, fully assembled document content in a raw, structured format)
 |
 |
 L--> [5. Final Output Generation]
      (User's choice determines the final format)
      |
      |--> [Choice A: Markdown] --> (Generates a clean, well-structured `notes.md` file)
      |
      |--> [Choice B: Word Document] --> (Generates a professional `notes.docx` file with headings)
      |
      L--> [Choice C: PDF] --> (Generates a portable, easy-to-share `notes.pdf` document)

## 🛠️ Tech Stack

### Core Framework
* **Language:** Python

### AI & Machine Learning
* **LLM Models:** Google Gemini Pro & Gemini Flash
* **LLM SDK:** `google-generativeai` (for interacting with the Gemini API)
* **Vector Database (RAG):** ChromaDB
* **Video/Image Processing:** OpenCV
* **Optical Character Recognition (OCR):** Pytesseract
* **Image Manipulation:** Pillow (often used with OpenCV/Pytesseract for preprocessing frames)

### Data Handling & APIs
* **Video Downloader:** `pytube` (or a similar library for accessing the video stream)
* **Transcript Fetching:** `youtube-transcript-api`

### Document & Environment
* **Document Generation:**
    * `python-docx` (for creating .docx Word files)
    * `fpdf2` or `ReportLab` (for generating PDF documents)
* **Environment Management:** `python-dotenv` (for loading API keys from the `.env` file)

---

## ⚙️ Getting Started

Follow these steps to get a local copy up and running.

### Prerequisites

You will need a Google AI Studio API key to use the Gemini models.
* Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### Installation for Windows
    git clone https://github.com/Rishab27279/U-Tube-AI.git
    cd U-Tube-AI
    python -m venv venv
    .\venv\Scripts\activate
    pip install -r requirements.txt
    copy example.env .env
 **Add your API Key:** Open the newly created `.env` file and add your Google AI Studio API key.
    ```env
    GOOGLE_API_KEY=YOUR_API_KEY_HERE
    ```

### Installation for MacOS & Linux

    git clone https://github.com/Rishab27279/U-Tube-AI.git
    cd U-Tube-AI
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    cp example.env .env
**Add your API Key:** Open the newly created `.env` file and add your Google AI Studio API key.
    ```env
    GOOGLE_API_KEY=YOUR_API_KEY_HERE
    ```

---

## 📖 Usage

To start generating notes, run the below code in terminal where you followed the above mentioned process

```sh
streamlit run app.py
```

## 🤝 Contributing

I warmly welcome contributions! Whether it's bug fixes, new features, or documentation improvements, please feel free to open issues and submit pull requests.

- 🐛 Bug fixes
- ✨ New features or feature ideas
- 📖 Documentation improvements
- 🎨 UI/UX enhancements

## ⭐ Support

If you find this project helpful, your support would mean the world:

- **Give it a star ⭐**
- **Share your feedback**
- **Try it out and report any issues**

<br>
<p align="center">
  ---
  <br>
  <strong>Made by Rishab with ❤️ & 🔥 for the community</strong>
  <br>
</p>

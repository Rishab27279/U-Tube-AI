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

                                              [START: User provides YouTube URL, Task, & Language]
                                                                |
                                                                V
                                               [1. Transcript Generation (via Flash Model)]
                                                                |
                                                                V
                                                          (Raw Transcript)
                                                                |
                                                                V
                                               [2. Language Processing & Translation]
                                                                |
                                                                |
                     +------------------------------------------+------------------------------------------+
                     |                                          |                                          |
                     V                                          V                                          V
    [BRANCH 1: INTERACTIVE TASKS]                   (Processed Transcript)                  [BRANCH 2: GENERATIVE TASKS]
                     |                                                                                         |
                     |                                                                                         |
    +----------------+-----------------+                                                   +-------------------+-----------------+
    |                                  |                                                   |                                     |
    V                                  V                                                   V                                     V
    [Task A: Translate Video]          [Task B: Chat with Video]                           [Task C: Important Points]            [Task D: Note Generation]
    |                                  |                                                   |                                     |
    L--> (Output: Final                +-->[B1. Vectorize & Create ChromaDB]                L-->[C1. Topic Extraction (Flash)]    (Uses "Imp. Points" as base)
         Processed Transcript)         |                                                        |                                |
                                       +-->[B3. Find Chunks & Time Frames (Flash)]             L-->(Output: Detailed              +-->[D1. Initial Note Gen (Pro)]
                                       |                                                            Topics & Subtopics)           |   (Transcript-only notes)
                                       +-->[B5. Initial Answer Gen (Pro)]                                                         |
                                       |     |                                                                                    +-->[D2. Visuals Decision]
                                       |     L-->(Output: Answer, is_ocr_needed, confidence)                                      |    |
                                       |                                                                                         |    +-->(If Visuals = NO)-->[D6]
                                       +-->[B6. OCR Decision]                                                                     |
                                       |    |                                                                                     L-->(If Visuals = YES)
                                       |    +-->(If YES)-->[B7. Refine with OCR (Pro)]                                             |
                                       |                                                                                         |    +-->[D3. Visual Extraction]
                                       +-->[B8. Confidence Check & Loop]                                                         |    |    (Adaptive Sampling/OCR)
                                       |    (If low, repeat B5-B8 up to 3x)                                                        |    |
                                       |                                                                                         |    +-->[D4. Iterative Loop (Flash)]
                                       L-->(Output: Final, high-confidence answer)                                                |    |    (Enhances 2 sub-topics
                                                                                                                                 |    |     at a time with OCR)
                                                                                                                                 |    |
                                                                                                                                 |    L-->[D5. Final Assembly]
                                                                                                                                 |         |
                                                                                                                                 |         V
                                                                                                                                 L-->[D6. Download Options]
                                                                                                                                      |
                                                                                                                                      +-->(.md File)
                                                                                                                                      +-->(.pdf Document)
                                                                                                                                      L-->(.docx Document)

## 🛠️ Tech Stack

### Core Framework
* **Language:** Python
* **UI Framework:** Streamlit

### AI & Machine Learning
* **LLM Models:** Google Gemini Pro & Gemini Flash
* **LLM SDK:** `google-generativeai` (for interacting with the Gemini API)
* **Vector Database (RAG):** ChromaDB
* **Video/Image Processing:** OpenCV

### Data Handling & APIs
* **Transcript Fetching:** `youtube-transcript-api`

### Gen AI
* **Langchain**
* **Role based Prompting**
* **Few Shot Prompting**
* **Case based Prompting**

### Document & Environment
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

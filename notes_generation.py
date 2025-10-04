from collections import defaultdict
import os
import json
import time
import re
from typing import List, Dict, Tuple, Optional
import google.generativeai as genai
from dotenv import load_dotenv
import shutil
from typing import Dict, List

from main import (
    download_full_video_safe,
    extract_video_frames_optimized,
    get_transcript,
    translate_transcript_batch,
)

NOTES_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "subtopic_id": {"type": "string"},
            "title": {"type": "string"},
            "time_range": {"type": "string"},
            "overview": {"type": "string"},
            "detailed_notes": {"type": "string"},
            "key_takeaways": {
                "type": "array",
                "items": {"type": "string"}
            },
            "visual_elements": {
                "type": "array",
                "items": {"type": "string"}
            },
            "tables": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "headers": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "rows": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        }
                    },
                    "required": ["title", "headers", "rows"]
                }
            }
        },
        "required": ["subtopic_id", "title", "time_range", "overview", "detailed_notes", "key_takeaways"]
    }
}

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)


def detect_language(text: str) -> str:
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"""Identify the language of this text. Return ONLY the language name in lowercase (e.g., 'english', 'hindi', 'spanish').

Text: {text[:500]}

Language:"""
        
        response = model.generate_content(prompt)
        language = response.text.strip().lower()
        return language
    except Exception as e:
        print(f"Language detection error: {e}")
        return "english" 


def perform_ocr_on_image(image_path: str) -> str:
    try:
        uploaded_file = genai.upload_file(image_path)
        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = """
You are analyzing a single frame from an educational video. Extract maximum learning value.

Return STRICT JSON with keys:
{
  "text_verbatim": "Exact text/code/commands/formulas as seen (verbatim, preserve line breaks).",
  "visual_concepts": "Describe diagrams/drawings/whiteboard content and what they illustrate.",
  "math_notation": "List important equations/symbols with short explanations.",
  "teaching_aids": "Arrows, highlights, underlines, boxes, annotations used to explain.",
  "key_insights": "One or two sentences on the concept being taught in this frame."
}

Rules:
- If a field is not applicable, use an empty string "" (not null).
- Do NOT add any commentary outside the JSON.
- Do NOT wrap JSON in code fences.
- Preserve formatting for code/terminal blocks inside text_verbatim.
"""

        response = model.generate_content([prompt, uploaded_file])
        response_text = (response.text or "").strip()


        if response_text.startswith("```"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        try:
            parsed = json.loads(response_text)
            for k in ["text_verbatim", "visual_concepts", "math_notation", "teaching_aids", "key_insights"]:
                if k not in parsed:
                    parsed[k] = ""
            clean_json = json.dumps(parsed, ensure_ascii=False)
        except Exception:
            clean_json = json.dumps({
                "text_verbatim": response_text,
                "visual_concepts": "",
                "math_notation": "",
                "teaching_aids": "",
                "key_insights": ""
            }, ensure_ascii=False)

        try:
            genai.delete_file(uploaded_file.name)
        except Exception:
            pass

        return clean_json

    except Exception as e:
        print(f"OCR error: {e}")
        return json.dumps({
            "text_verbatim": "",
            "visual_concepts": "",
            "math_notation": "",
            "teaching_aids": "",
            "key_insights": ""
        }, ensure_ascii=False)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def parse_timestamp(timestamp_str: str) -> int: # To convert time stamp to seconds
    try:
        parts = timestamp_str.split(':')
        if len(parts) == 2: 
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3: 
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except:
        return 0
    return 0


def format_seconds_to_timestamp(seconds: int) -> str:
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


# ============================================================================
#  Topic Structure Extraction with Flash
# ============================================================================


def extract_topic_structure_with_flash(transcript_chunks: List[Dict], 
                                       model_name: str = "gemini-2.5-flash") -> Dict:

    print("\nStage 1: Extracting topic structure with Gemini Flash...")
    
    if not transcript_chunks or len(transcript_chunks) == 0:
        print("No transcript chunks provided")
        return create_fallback_structure([])
    
    video_duration_seconds = 0
    if transcript_chunks and 'metadata' in transcript_chunks[-1]:     # Metadata to Cal Duration
        last_chunk_meta = transcript_chunks[-1]['metadata']
        video_duration_seconds = last_chunk_meta.get('end_time', 0)
    
    if video_duration_seconds == 0:     # Parse Timestrap string to Cal Duration
        try:
            last_timestamp = transcript_chunks[-1].get('timestamp', '0:00')
            if isinstance(last_timestamp, str) and ':' in last_timestamp:
                parts = last_timestamp.split(':')
                if len(parts) == 2:  # MM:SS
                    video_duration_seconds = int(parts[0]) * 60 + int(parts[1])
                elif len(parts) == 3:  # HH:MM:SS
                    video_duration_seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        except Exception as e:
            print(f"Error parsing timestamp: {e}")
    
    if video_duration_seconds == 0: # Estimate if all fails
        print("Could not determine real duration, using estimate")
        video_duration_seconds = len(transcript_chunks) * 25  
    video_duration_minutes = video_duration_seconds // 60
    video_duration_formatted = f"{video_duration_minutes}:{video_duration_seconds % 60:02d}"
    
    print(f"Video duration: {video_duration_minutes}m {video_duration_seconds % 60}s "
          f"({video_duration_seconds} seconds total)")
    print(f"Number of chunks: {len(transcript_chunks)}")
    

    full_transcript = "\n".join([     # Build transcript
        f"[{chunk.get('timestamp', 'N/A')}] {chunk.get('text', '')}"
        for chunk in transcript_chunks[:100]
    ])
    
    if len(full_transcript.strip()) < 50:
        print("Transcript too short, using fallback")
        return create_fallback_structure(transcript_chunks)
    
    print(f"Transcript length: {len(full_transcript)} characters")
    
    # FEW-SHOT LEARNING PROMPT (Examples)
    prompt = f"""
    You are an expert at creating optimal video note structures. Your goal is to break content into subtopics that are MEANINGFUL, SUBSTANTIAL, and DISTINCT.

    TRANSCRIPT TO ANALYZE ({video_duration_minutes} minutes):
    {full_transcript[:25000]}

    CORE PRINCIPLES:
    1. Each subtopic should represent a COMPLETE CONCEPT (not a sentence)
    2. Subtopics should be 30-90 seconds of content each (substantial, not trivial)
    3. Create subtopics based on CONCEPTUAL SHIFTS, not arbitrary time intervals
    4. Avoid over-segmentation - don't split what should be together
    5. Balance: Too few subtopics = information overload; Too many = fragmentation

    ═══════════════════════════════════════════════════════════════════
    EXAMPLE 1: 5-MINUTE PYTHON TUTORIAL VIDEO (BAD vs GOOD)
    ═══════════════════════════════════════════════════════════════════
    BAD STRUCTURE (Over-Segmented - 15 subtopics):
    Topic 1: Introduction
    - 1.1 Greeting and Welcome (7 seconds) Too trivial
    - 1.2 Topic Overview (8 seconds) Too trivial
    - 1.3 Why This Matters (10 seconds) Should merge with 1.2

    Topic 2: The Problem
    - 2.1 Problem Description (25 seconds) Could be one subtopic
    - 2.2 Example Scenario Part 1 (20 seconds) Artificial split
    - 2.3 Example Scenario Part 2 (18 seconds) Belongs with 2.2
    - 2.4 Why It's Important (12 seconds) Too short

    Why This Is Bad:
    - 15 subtopics for 5 minutes = one every 20 seconds (micro-fragmentation)
    - Breaks unified concepts into artificial pieces
    - Creates 18+ pages of repetitive notes
    - Harder to navigate than just watching the video

    GOOD STRUCTURE (Optimal - 5 subtopics):
    Topic 1: Introduction & Problem Statement (0:00 - 1:30)
    - 1.1 Why Virtual Environments Matter (0:00 - 0:45)
        - Covers: intro, motivation, real-world problem setup
    - 1.2 The Dependency Conflict Problem (0:45 - 1:30)
        - Covers: complete explanation with examples

    Topic 2: Understanding VENV Solution (1:30 - 3:30)
    - 2.1 How VENV Creates Isolation (1:30 - 2:30)
        - Covers: mechanism, components, why it works
    - 2.2 Creating and Activating Environments (2:30 - 3:30)
        - Covers: complete workflow, commands, verification

    Topic 3: Best Practices (3:30 - 5:00)
    - 3.1 Workflow Integration & IDE Tools (3:30 - 5:00)
        - Covers: practical usage, automation, recommendations

    Why This Is Good:
    - 5 subtopics for 5 minutes = one per minute (substantial)
    - Each subtopic is a complete, meaningful unit (60-90 seconds)
    - Grouped related concepts together logically
    - Creates 5-6 pages of useful, scannable notes

    ═══════════════════════════════════════════════════════════════════
    EXAMPLE 2: 30-MINUTE MACHINE LEARNING LECTURE (BAD vs GOOD)
    ═══════════════════════════════════════════════════════════════════
    BAD STRUCTURE (Under-Segmented - 3 subtopics):
    Topic 1: Neural Networks Overview (0:00 - 10:00) Too broad
    - 1.1 Everything About Neural Networks (0:00 - 10:00)
        - 10 minutes crammed into one subtopic - information overload

    Topic 2: Training Concepts (10:00 - 22:00) Too broad
    - 2.1 All About Training (10:00 - 22:00)
        - 12 minutes without meaningful breaks

    Topic 3: Applications (22:00 - 30:00) Too broad
    - 3.1 Use Cases (22:00 - 30:00)
        - 8 minutes undifferentiated

    Why This Is Bad:
    - Only 3 mega-subtopics for 30 minutes
    - Each subtopic tries to cover 8-12 minutes (overwhelming)
    - No way to navigate to specific concepts
    - Forces readers to scan huge blocks of text

    GOOD STRUCTURE (Optimal - 15 subtopics):
    Topic 1: Neural Network Fundamentals (0:00 - 8:00)
    - 1.1 Perceptron Architecture (0:00 - 2:00)
    - 1.2 Activation Functions (2:00 - 4:30)
    - 1.3 Forward Propagation Mechanics (4:30 - 8:00)

    Topic 2: Training Process (8:00 - 18:00)
    - 2.1 Loss Functions Explained (8:00 - 10:30)
    - 2.2 Gradient Descent Fundamentals (10:30 - 13:00)
    - 2.3 Backpropagation Algorithm (13:00 - 16:00)
    - 2.4 Optimization Techniques (16:00 - 18:00)

    Topic 3: Practical Implementation (18:00 - 25:00)
    - 3.1 Dataset Preparation (18:00 - 20:30)
    - 3.2 Network Architecture Design (20:30 - 23:00)
    - 3.3 Hyperparameter Tuning (23:00 - 25:00)

    Topic 4: Real-World Applications (25:00 - 30:00)
    - 4.1 Computer Vision Use Cases (25:00 - 27:00)
    - 4.2 NLP Applications (27:00 - 29:00)
    - 4.3 Best Practices & Pitfalls (29:00 - 30:00)

    Why This Is Good:
    - 15 subtopics for 30 minutes = one every 2 minutes (digestible)
    - Each concept gets focused treatment (2 minutes each)
    - Easy to find specific topics (e.g., "just review backpropagation")
    - Creates 18-22 pages of well-organized reference material

    ═══════════════════════════════════════════════════════════════════

    YOUR TASK:
    Analyze the provided {video_duration_minutes}-minute transcript and create an OPTIMAL structure.

    Ask yourself:
    1. What are the MAIN CONCEPTUAL SECTIONS? (These become topics)
    2. Within each section, what are DISTINCT SUB-CONCEPTS worth 30-90 seconds each? (These become subtopics)
    3. Can I merge similar/short segments into one cohesive subtopic?
    4. Does each subtopic represent something I'd want to "jump to" or "review specifically"?

    NATURAL HEURISTICS (Don't force these, but use as judgment guides):
    - Short videos (5-10 min): Typically 4-8 subtopics TOTAL
    - Medium videos (10-30 min): Typically 12-25 subtopics TOTAL
    - Long videos (30-60 min): Typically 25-50 subtopics TOTAL
    - Each subtopic should cover 30-90 seconds minimum (not 5-10 seconds)

    CRITICAL TIMESTAMP CONSTRAINT 

    VIDEO DURATION: {video_duration_formatted} ({video_duration_seconds} seconds)

    MANDATORY RULES:
    • ALL topic and subtopic end_time values MUST be <= {video_duration_formatted}
    • NEVER create end_time values beyond the video length
    • If content extends beyond {video_duration_formatted}, TRUNCATE it to fit
    • MAXIMUM valid timestamp = {video_duration_formatted}
    • If unsure, use earlier timestamps—NEVER exceed the video length

    Example: If video is 5:30 and you have content to 8:00, you MUST:
    Compress it: Topic 4 becomes "05:00 - 05:30" (NOT "06:00 - 08:00")
    NEVER write timestamps like "06:25", "07:00", "08:00" for a 5:30 video

    ═══════════════════════════════════════════════════════════════════

    OUTPUT FORMAT (JSON):
    {{
    "video_title": "Inferred title based on content",
    "total_duration": "{video_duration_formatted}",
    "content_density": "light|medium|dense",
    "topics": [
        {{
        "topic_id": 1,
        "title": "Main Topic Title",
        "start_time": "00:00",
        "end_time": "MM:SS",
        "subtopics": [
            {{
            "subtopic_id": "1.1",
            "title": "Meaningful Sub-Concept Title (not micro-fragment)",
            "start_time": "00:00",
            "end_time": "MM:SS",
            "description": "Brief description of complete concept",
            "content_type": "explanation|demonstration|example|code",
            "complexity": "basic|intermediate|advanced"
            }}
        ]
        }}
    ]
    }}

    REQUIREMENTS:
    - Return ONLY valid JSON, no markdown or extra text
    - Each subtopic must be SUBSTANTIAL (30+ seconds minimum)
    - Merge related short segments into cohesive subtopics
    - Think like a student: "Would I want to review THIS specific thing?"
    - Avoid: micro-fragmentation, artificial splits, trivial segments
    - Timestamps must be accurate and match transcript
    """
    
    try:
        print("Calling Gemini API...")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)

        print(f"DEBUG: Response type: {type(response)}")
        
        if not response:
            print("API returned None/null response")
            return create_fallback_structure(transcript_chunks)
        
        if not hasattr(response, 'text'):
            print("Response has no 'text' attribute")
            return create_fallback_structure(transcript_chunks)
        
        response_text = response.text
        
        if not response_text:
            print("Response.text is None or empty")
            return create_fallback_structure(transcript_chunks)
        
        response_text = response_text.strip()
        
        print(f"Got API response: {len(response_text)} characters")
        print(f"First 200 chars: {response_text[:200]}")
        
        if len(response_text) < 20:
            print(f"Response too short: '{response_text}'")
            return create_fallback_structure(transcript_chunks)
        
        cleaned_json = extract_json_from_response(response_text)
        
        print(f"Cleaned JSON length: {len(cleaned_json) if cleaned_json else 0}")
        
        if not cleaned_json or len(cleaned_json.strip()) < 10:
            print("Could not extract valid JSON from response")
            return create_fallback_structure(transcript_chunks)
        
        try:
            topic_structure = json.loads(cleaned_json)
        except json.JSONDecodeError as json_err:
            print(f"JSON parse error: {json_err}")
            print(f"   Attempted to parse: {cleaned_json[:500]}")
            return create_fallback_structure(transcript_chunks)
        
        if not isinstance(topic_structure, dict):
            print(f"Parsed JSON is not a dict: {type(topic_structure)}")
            return create_fallback_structure(transcript_chunks)
        
        if 'topics' not in topic_structure:
            print(f"JSON missing 'topics' key. Keys: {list(topic_structure.keys())}")
            return create_fallback_structure(transcript_chunks)
        
        topics = topic_structure.get('topics', [])
        
        if not isinstance(topics, list) or len(topics) == 0:
            print("No valid topics found")
            return create_fallback_structure(transcript_chunks)
        
        valid_topics = []
        for topic in topics:
            if not isinstance(topic, dict):
                continue
            
            if 'subtopics' not in topic or not isinstance(topic['subtopics'], list):
                continue
            
            if len(topic['subtopics']) == 0:
                continue
            
            valid_topics.append(topic)
        
        if not valid_topics:
            print("No valid topics with subtopics found")
            return create_fallback_structure(transcript_chunks)
        
        topic_structure['topics'] = valid_topics
        
        total_subtopics = sum(len(t.get('subtopics', [])) for t in valid_topics)
        avg_subtopics_per_min = total_subtopics / video_duration_minutes if video_duration_minutes > 0 else 0
        
        print(f"Extracted {len(valid_topics)} topics with {total_subtopics} subtopics")
        print(f"Density: {avg_subtopics_per_min:.1f} subtopics/minute")
        
        if avg_subtopics_per_min > 2.5:
            print(f"High subtopic density ({avg_subtopics_per_min:.1f}/min) - may be over-segmented")
        elif avg_subtopics_per_min < 0.5:
            print(f"Low subtopic density ({avg_subtopics_per_min:.1f}/min) - may be under-segmented")
        else:
            print(f"Good subtopic density ({avg_subtopics_per_min:.1f}/min)")
        
        return topic_structure
        
    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return create_fallback_structure(transcript_chunks)


def create_fallback_structure(transcript_chunks: List[Dict]) -> Dict:
    """
    Create basic structure as fallback
    ALWAYS returns a valid Dict, never None
    FIXED: Uses real duration from timestamps
    """
    print("Using fallback structure (topic extraction failed)")
    
    if transcript_chunks and len(transcript_chunks) > 0: # Calculate real end time
        try:
            # From metadata
            if 'metadata' in transcript_chunks[-1]:
                end_seconds = transcript_chunks[-1]['metadata'].get('end_time', 300)
                end_time = f"{end_seconds // 60}:{end_seconds % 60:02d}"
                total_duration = end_time
            # From timestamp string
            else:
                end_time = str(transcript_chunks[-1].get('timestamp', '5:00'))
                total_duration = end_time
        except (IndexError, AttributeError, KeyError) as e:
            print(f"⚠️ Error calculating duration: {e}")
            end_time = "5:00"
            total_duration = "5:00"
    else:
        end_time = "5:00"
        total_duration = "5:00"
    
    return {
        "video_title": "Video Content (Fallback Structure)",
        "total_duration": total_duration,
        "content_density": "unknown",
        "topics": [{
            "topic_id": 1,
            "title": "Complete Video Content",
            "start_time": "0:00",
            "end_time": end_time,
            "subtopics": [{
                "subtopic_id": "1.1",
                "title": "Video Content Summary",
                "start_time": "0:00",
                "end_time": end_time,
                "description": "Complete video transcript content",
                "content_type": "explanation",
                "complexity": "intermediate"
            }]
        }]
    }

# ============================================================================
#  Deep Content Generation with Pro
# ============================================================================

def generate_deep_content_with_pro(topic_structure: Dict, 
                                   transcript_chunks: List[Dict], 
                                   model_name: str = "gemini-2.5-pro") -> Dict:
    # Full content generation for each topic for notes
    print("\nStage 2: Generating deep content with Gemini Pro...")
    print("This may take 30-90 seconds due to Pro's deep reasoning...")
    
    full_transcript = "\n".join([
        f"[{chunk.get('timestamp', 'N/A')}] {chunk.get('text', '')}" 
        for chunk in transcript_chunks[:150]  # Till 150 its good, more than that will be too much for Model.
    ])
    
    topics_summary = [] # Simplified structure for prompt
    for topic in topic_structure.get('topics', [])[:5]:  # Limit to first 5 topics as more will cross input token limit
        topic_info = {
            'topic_id': topic.get('topic_id'),
            'title': topic.get('title'),
            'subtopics': [
                {
                    'subtopic_id': st.get('subtopic_id'),
                    'title': st.get('title'),
                    'time_range': f"{st.get('start_time')} - {st.get('end_time')}"
                }
                for st in topic.get('subtopics', [])
            ]
        }
        topics_summary.append(topic_info)
    
    prompt = f"""You are a world-class educational content synthesizer. Your mission is to transform a raw video transcript into a set of clear, concise, and deeply informative study notes for a human learner. Your primary goal is to maximize understanding and retention while eliminating all redundancy.

**ANTI-REDUNDANCY DIRECTIVE:** Each piece of information should only appear in the most appropriate section. Do not repeat facts or concepts across the 'explanation', 'key_points', and 'examples' sections. Each section must serve a unique purpose.

**TOPIC STRUCTURE (Your guide):**
{json.dumps(topics_summary, indent=2)}

**FULL TRANSCRIPT (Your data source):**
{full_transcript[:40000]}

--------------------------------------------------
**TASK: For EACH sub-topic, generate the following four distinct components:**

1.  **Detailed Explanation (The "Why" and "How"):**
    - **Purpose:** To provide a conceptual, narrative understanding. Explain the core concepts, their context, and why they matter.
    - **Instructions:** Write 3-5 fluid paragraphs. Focus on the relationships between ideas. **Do not** include verbatim code or step-by-step lists here; instead, explain the *purpose* of those examples.

2.  **Key Points (The "What"):**
    - **Purpose:** To isolate the most critical, dense facts for quick review and memorization.
    - **Instructions:** Create a bulleted list of 5-10 points. Focus on specific, verifiable facts, technical terms, names, and data points. This section should be factual and concise, not conversational.

3.  **Examples (The "Show Me"):**
    - **Purpose:** To provide only concrete, practical illustrations.
    - **Instructions:** If the transcript contains them, extract verbatim code snippets, step-by-step procedures, terminal commands, or real-world applications. **Do not** add any explanatory prose here; this section is for concrete examples only. If none exist, return an empty list.

4.  **Learning Objectives (The "So What?"):**
    - **Purpose:** To frame the knowledge in terms of skills gained by the learner.
    - **Instructions:** Write 2-4 points. Each point must start from the learner's perspective, such as "Understand the difference between..." or "Be able to implement...".

--------------------------------------------------
**OUTPUT FORMAT (JSON) - Adhere Strictly to this Schema:**
{{
  "topics": [
    {{
      "topic_id": 1,
      "subtopics": [
        {{
          "subtopic_id": "1.1",
          "detailed_content": {{
            "explanation": "The conceptual 'Why & How', written in fluid paragraphs.",
            "key_points": ["A concise list of the factual 'What'."],
            "examples": ["Verbatim code or step-by-step 'Show Me' examples."],
            "learning_objectives": ["The skill-based 'So What?' for the learner."]
          }}
        }}
      ]
    }}
  ]
}}

--------------------------------------------------
**CRITICAL REQUIREMENTS:**
- **Adhere to the Anti-Redundancy Directive:** This is the most important rule.
- **Be Comprehensive:** Capture all critical details from the transcript.
- **Maintain a Professional Tone:** Write as an expert educator.
- **Return ONLY valid JSON.**
"""
    
    try:
        model = genai.GenerativeModel(model_name)
        
        response = model.generate_content(prompt)
        
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        enhanced_content = json.loads(response_text.strip())
        
        merged_structure = topic_structure.copy()  # Merge with original structure
        for i, topic in enumerate(merged_structure.get('topics', [])):
            if i < len(enhanced_content.get('topics', [])):
                enhanced_topic = enhanced_content['topics'][i]
                for j, subtopic in enumerate(topic.get('subtopics', [])):
                    if j < len(enhanced_topic.get('subtopics', [])):
                        subtopic['detailed_content'] = enhanced_topic['subtopics'][j].get('detailed_content', {})
        
        print("Generated comprehensive content for all topics")
        return merged_structure
        
    except json.JSONDecodeError as e:
        print(f"JSON parse error in Pro generation: {str(e)}")
        print("Falling back to original structure...")
        return topic_structure
    except Exception as e:
        print(f"Error in Pro content generation: {str(e)}")
        print("Falling back to original structure...")
        return topic_structure


# ============================================================================
#  Adaptive Frame Sampling for different types of Videos
# ============================================================================

def calculate_adaptive_frame_count(subtopic: Dict) -> int:
    base_frames = 8
    
    content_type = subtopic.get('content_type', 'explanation').lower()
    if content_type in ['code', 'demonstration', 'tutorial']:
        base_frames += 4  # More frames
    elif content_type in ['theory', 'explanation']:
        base_frames -= 2  # Less frames
    
    complexity = subtopic.get('complexity', 'intermediate').lower()
    if complexity == 'advanced':
        base_frames += 2
    elif complexity == 'basic':
        base_frames -= 1
    
    start_time = parse_timestamp(subtopic.get('start_time', '0:00')) # Adjust based on duration
    end_time = parse_timestamp(subtopic.get('end_time', '0:00'))
    duration = end_time - start_time
    
    if duration > 600: 
        base_frames += 3
    elif duration < 120: 
        base_frames -= 2
    
    return max(6, min(15, base_frames))


def adaptive_frame_sampling(video_path: str, subtopic: Dict, output_dir: str) -> List[str]:
    # Extract frames based on the above adaptive frame count.

    frame_count = calculate_adaptive_frame_count(subtopic)
    
    start_time = parse_timestamp(subtopic.get('start_time', '0:00'))
    end_time = parse_timestamp(subtopic.get('end_time', '0:00'))
    duration = end_time - start_time
    
    if duration <= 0:
        print(f"  ⚠️ Invalid duration for subtopic {subtopic.get('subtopic_id')}")
        return []
    
    timestamps = [] # Timestamps for frame extraction
    interval = duration / (frame_count + 1)
    
    for i in range(1, frame_count + 1):
        timestamp = start_time + (i * interval)
        timestamps.append(timestamp)
    
    print(f"  Extracting {frame_count} frames from {subtopic.get('start_time')} to {subtopic.get('end_time')}")
    
    try:
        frame_paths = extract_video_frames_optimized(video_path, timestamps, output_dir)
        print(f"  Extracted {len(frame_paths)} frames successfully")
        return frame_paths
    except Exception as e:
        print(f"  Frame extraction failed: {str(e)}")
        return []


# ============================================================================
#  Batch OCR Processing
# ============================================================================

def batch_ocr_processing(frame_paths: List[str]) -> List[Dict[str, str]]:
    # Process Multiple frames in a single API call with Batch using Flash Model
    if not frame_paths:
        return []
    
    ocr_results = []
    MAX_FRAMES_PER_BATCH = 16  # Safe limit for batch processing for performance
    
    try:
        print(f"  Uploading {len(frame_paths)} frames for batch OCR...")
        
        uploaded_files = []
        for frame_path in frame_paths[:MAX_FRAMES_PER_BATCH]:
            if os.path.exists(frame_path):
                try:
                    uploaded = genai.upload_file(frame_path)
                    uploaded_files.append({
                        'file': uploaded,
                        'path': frame_path
                    })
                except Exception as e:
                    print(f"Failed to upload {frame_path}: {e}")
        
        if not uploaded_files:
            return []
        
        print(f"  Uploaded {len(uploaded_files)} frames")
        print(f"  Running batch OCR on all frames...")
        
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = """Extract ALL readable text from these video frames.

For EACH frame, provide:
1. Frame number (1, 2, 3...)
2. All text visible (code, commands, slides, diagrams, terminal output)
3. Whether frame contains meaningful text (true/false)

Return as JSON array:
[
  {
    "frame_number": 1,
    "extracted_text": "python3 -m venv myenv",
    "has_text": true
  },
  {
    "frame_number": 2,
    "extracted_text": "",
    "has_text": false
  }
]

Return ONLY the JSON array, no markdown.
"""
        
        content = [prompt] + [uf['file'] for uf in uploaded_files] # All frames in 1 API request
        
        response = model.generate_content(
            content,
            request_options={'timeout': 120}
        )
        
        response_text = response.text.strip()
        
        if response_text.startswith('```'):
            response_text = response_text[7:]
        if response_text.startswith('```'):
            response_text = response_text[3:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        ocr_data = json.loads(response_text)
        
        for i, uf in enumerate(uploaded_files):
            if i < len(ocr_data):
                result = ocr_data[i]
                ocr_results.append({
                    'frame_path': uf['path'],
                    'frame_number': result.get('frame_number', i + 1),
                    'extracted_text': result.get('extracted_text', ''),
                    'has_text': result.get('has_text', False)
                })
            else:
                ocr_results.append({
                    'frame_path': uf['path'],
                    'frame_number': i + 1,
                    'extracted_text': '',
                    'has_text': False
                })
        
        for uf in uploaded_files: # Delete frames after Refresh Button
            try:
                genai.delete_file(uf['file'].name)
            except:
                pass
        
        text_frames = sum(1 for r in ocr_results if r['has_text'])
        print(f"  OCR complete: {text_frames}/{len(ocr_results)} frames contain text")
        
        return ocr_results
        
    except json.JSONDecodeError as e:
        print(f"  JSON parse error in batch OCR: {e}")
        return []
    except Exception as e:
        print(f"  Batch OCR error: {e}")
        return []

import logging

# For Logging to occur only once.
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

logger = logging.getLogger(__name__)

synthesis_metrics = defaultdict(int)


def extract_json_from_response(response_text: str) -> Optional[str]:
    # Code to extract JSON from various response formats
    if not response_text:
        return None
    
    text = response_text.strip()
    
    if text.startswith("```"):
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        text = text.strip()
    
    if text.endswith("```"):
        text = text[:-3].strip()
    
    json_pattern = r'\{[\s\S]*\}'
    match = re.search(json_pattern, text)
    
    if match:
        text = match.group(0)
    
    prefixes = ["Here's the JSON:", "Here is the JSON:", "json", "JSON:", "Response:", "Output:"]
    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    
    return text if text else None


def validate_notes_structure(notes_list: List[Dict]) -> Tuple[bool, List[Dict]]:
    # Checks/Validates notes structures.

    if not notes_list or not isinstance(notes_list, list):
        return False, []
    
    cleaned_notes = []
    for note in notes_list:
        if not isinstance(note, dict):
            continue
        
        if not note.get('subtopic_id') or not note.get('title'):
            continue
        
        visual_elem = note.get('visual_elements', [])
        if visual_elem is None:
            note['visual_elements'] = []
        elif isinstance(visual_elem, str):
            note['visual_elements'] = [visual_elem] if visual_elem else []
        elif not isinstance(visual_elem, list):
            note['visual_elements'] = []
        
        takeaways = note.get('key_takeaways', [])
        if takeaways is None:
            note['key_takeaways'] = []
        elif isinstance(takeaways, str):
            note['key_takeaways'] = [takeaways] if takeaways else []
        elif not isinstance(takeaways, list):
            note['key_takeaways'] = []
        
        note['overview'] = str(note.get('overview', ''))
        note['detailed_notes'] = str(note.get('detailed_notes', ''))
        note['title'] = str(note.get('title', ''))
        note['time_range'] = str(note.get('time_range', '00:00 - 00:00'))
        
        if len(note['detailed_notes']) < 50: # Check for minimum content quality
            logger.warning(f"⚠️ Note {note['subtopic_id']} has suspiciously short content ({len(note['detailed_notes'])} chars)")
        
        cleaned_notes.append(note)
    
    return len(cleaned_notes) > 0, cleaned_notes


def retry_with_exponential_backoff( # Google cloud uses this, like retries func with expontial backoff
    func,                           # Tries to call following func max times if error occurs due to trafficing or any otherreason.
    max_retries: int = 3,           # Helps to retry some transient errors.
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_errors: tuple = (429, 500, 503, 504)
):
    """
    Retry function with exponential backoff for transient errors
    Based on Google Cloud best practices
    """
    import random
    
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            error_str = str(e).lower()
            is_retryable = (
                any(str(code) in error_str for code in retryable_errors) or
                'resource_exhausted' in error_str or
                'unavailable' in error_str or
                'deadline' in error_str or
                'overloaded' in error_str
            )
            
            if not is_retryable or attempt >= max_retries - 1:
                raise  # Re-raise if not retryable or final attempt
            
            # Calculate delay with exponential backoff + jitter
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, delay * 0.1)  # Add 10% jitter
            total_delay = delay + jitter
            
            logger.warning(
                f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                f"Retrying in {total_delay:.2f}s..."
            )
            synthesis_metrics['retries'] += 1
            time.sleep(total_delay)
    
    raise Exception(f"Failed after {max_retries} retries")


def create_fallback_notes(subtopics_batch: List[Dict]) -> List[Dict]:
    # Create some notes if synthesis fails. Always returns a valid list to let the pipeline running.
    logger.warning(f"Creating fallback notes for {len(subtopics_batch)} subtopics")
    synthesis_metrics['fallback_used'] += 1
    
    fallback_notes = []
    
    try:
        for subtopic in subtopics_batch:
            content = subtopic.get('detailed_content', {})
            
            if isinstance(content, dict):
                detailed_text = content.get('explanation', 'Content for this segment')
            elif isinstance(content, str):
                detailed_text = content
            else:
                detailed_text = "Content for this segment"
            
            fallback_notes.append({
                "subtopic_id": subtopic.get('subtopic_id', '1.1'),
                "title": subtopic.get('title', 'Subtopic'),
                "time_range": f"{subtopic.get('start_time', '00:00')} - {subtopic.get('end_time', '00:00')}",
                "overview": "Content from this video segment (fallback mode)",
                "detailed_notes": detailed_text[:500] if detailed_text else "Content unavailable",
                "key_takeaways": ["Content covered in this segment"],
                "visual_elements": []
            })
    except Exception as e:
        logger.error(f"Error creating fallback notes: {e}")
        fallback_notes = [{
            "subtopic_id": "1.1",
            "title": "Video Content",
            "time_range": "00:00 - 00:00",
            "overview": "Video content (minimal fallback)",
            "detailed_notes": "Content synthesis failed",
            "key_takeaways": [],
            "visual_elements": []
        }]
    
    return fallback_notes

def parse_time_to_seconds(time_str: str) -> int:
    try:
        parts = str(time_str).split(':')
        if len(parts) == 2:  # MM:SS
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:  # HH:MM:SS
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        else:
            return 60  
    except Exception as e:
        print(f"Error parsing time '{time_str}': {e}")
        return 60  

def convert_table_to_markdown(table: Dict) -> str:
    # For Matrix, tables, etc.. (dont remember)
    title = table.get('title', 'Table')
    headers = table.get('headers', [])
    rows = table.get('rows', [])
    
    if not headers or not rows:
        return f"**{title}:** (empty table)"
    
    md_lines = [f"\n**{title}:**\n"]
    
    md_lines.append("| " + " | ".join(headers) + " |")
    
    md_lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    
    for row in rows:
        padded_row = row + [''] * (len(headers) - len(row))
        md_lines.append("| " + " | ".join(padded_row[:len(headers)]) + " |")
    
    return "\n".join(md_lines) + "\n"


def validate_and_normalize_notes(notes_list: List[Dict]) -> tuple:
    normalized_notes = []
    
    for note in notes_list:
        if not isinstance(note, dict):
            continue
        
        if not note.get('subtopic_id') or not note.get('title'):
            continue
        
        note['overview'] = str(note.get('overview', ''))
        note['detailed_notes'] = str(note.get('detailed_notes', ''))
        
        if not isinstance(note.get('key_takeaways'), list):
            note['key_takeaways'] = []
        if not isinstance(note.get('visual_elements'), list):
            note['visual_elements'] = []
        
        if 'tables' in note and note['tables']: #  Convert tables to Markdown by calling above func.
            markdown_tables = []
            for table in note['tables']:
                md_table = convert_table_to_markdown(table)
                markdown_tables.append(md_table)
            
            note['detailed_notes'] += "\n\n" + "\n\n".join(markdown_tables)
        
        normalized_notes.append(note)
    
    return len(normalized_notes) > 0, normalized_notes

# ---------------------------------------------------------------------------
#  Synthesize Comprehensive Notes with Flash Model
# ---------------------------------------------------------------------------

def synthesize_notes_with_flash(
    subtopics_batch: List[Dict], 
    ocr_data: Dict[str, List[Dict]], 
    model_name: str = "gemini-2.5-flash",
    timeout: int = 60,
    max_retries: int = 3
) -> Dict[str, any]:
    
    logger.info(f"  Synthesizing comprehensive notes for {len(subtopics_batch)} subtopics...")
    synthesis_metrics['total_calls'] += 1
    
    if not subtopics_batch or len(subtopics_batch) == 0:
        logger.warning("Empty subtopics batch")
        return {'notes': [], 'quality': 'fallback', 'fallback_used': True, 'error': 'Empty input'}
    
    batch_content = [] # Visual Data
    for subtopic in subtopics_batch:
        subtopic_id = subtopic.get('subtopic_id')
        detailed_content = subtopic.get('detailed_content', {})
        ocr_results = ocr_data.get(subtopic_id, [])
        
        start_time = subtopic.get('start_time', '0:00')
        end_time = subtopic.get('end_time', '1:00')
        start_seconds = parse_time_to_seconds(start_time)
        end_seconds = parse_time_to_seconds(end_time)
        duration_seconds = max(10, end_seconds - start_seconds)
        
        if duration_seconds < 40:
            sentence_count = "4-5 sentences"
            word_count = "70-90 words"
            takeaway_count = "4-5"
        elif duration_seconds < 70:
            sentence_count = "6-8 sentences"
            word_count = "100-130 words"
            takeaway_count = "5-6"
        else:
            sentence_count = "8-10 sentences"
            word_count = "140-180 words"
            takeaway_count = "6-7"
        
        visual_text = "\n".join([  # Combine OCR text from visual frames
            f"[Frame {ocr.get('frame_number', i+1)}]: {ocr['extracted_text']}" 
            for i, ocr in enumerate(ocr_results) 
            if ocr.get('has_text', False)
        ])
        
        if not visual_text:
            visual_text = "No text detected in video frames"
        
        batch_content.append({
            'subtopic_id': subtopic_id,
            'title': subtopic.get('title', ''),
            'time_range': f"{start_time} - {end_time}",
            'duration_seconds': duration_seconds,
            'content': detailed_content,
            'visual_information': visual_text,
            'sentence_constraint': sentence_count,
            'word_constraint': word_count,
            'takeaway_constraint': f"EXACTLY {takeaway_count} bullets"
        })

    
    prompt = f"""You are a professional educational content synthesizer creating COMPREHENSIVE study notes from multi-modal inputs.

MODE: COMPREHENSIVE
Goal: Capture 95% of information with rich detail, technical accuracy, and practical examples.

UNIQUE DATA SOURCES YOU HAVE:
1. Deep Content Analysis (from Gemini Pro's reasoning)
2. Visual Frame Text (OCR from video frames)
3. Timestamp Context (precise timing for review)

SUBTOPICS TO PROCESS:
{json.dumps(batch_content, indent=2, ensure_ascii=False)}

═══════════════════════════════════════════════════════════════════
COMPREHENSIVE NOTE-TAKING PRINCIPLES:
═══════════════════════════════════════════════════════════════════

1. **Detail Capture Priority**
   Include ALL commands, code snippets, version numbers
   Preserve exact syntax (python3 -m venv, not "use python")
   Capture warnings, gotchas, edge cases
   Include specific examples with parameters

2. **Visual Content Integration**
   Extract code from OCR frame text
   Note diagram elements, flowchart steps
   Identify slide text, terminal commands, IDE screenshots
   If OCR shows code/commands, include them EXACTLY

3. **Tables & Matrices (CRITICAL)**
    For ALL matrices, payoff tables, game diagrams:
      - Use the "tables" field in JSON
      - Each table MUST have: "title", "headers", "rows"
   
   Example:
   {{
     "tables": [
       {{
         "title": "Prisoner's Dilemma Payoff Matrix",
         "headers": ["", "Cooperate", "Defect"],
         "rows": [
           ["Cooperate", "(3,3)", "(0,5)"],
           ["Defect", "(5,0)", "(1,1)"]
         ]
       }}
     ]
   }}

4. **Contextual Completeness**
   Explain WHY, not just WHAT
   Include practical implications
   Capture numeric details (file sizes, percentages)

5. **Technical Accuracy**
   Use exact terminology from video
   Include file paths, flags, arguments
   Preserve error messages or output examples

═══════════════════════════════════════════════════════════════════
OUTPUT STRUCTURE (Per Subtopic):
═══════════════════════════════════════════════════════════════════

For each subtopic:

1. **Overview** (1-2 sentences, 25-35 words)
   - Concise summary of core concept
   - Include key technical term or command

2. **Detailed Notes** ({batch_content[0]['sentence_constraint']}, {batch_content[0]['word_constraint']})
   - Start with concept explanation
   - Include specific examples with concrete details
   - Integrate OCR text (commands, code, slide content)
   - Explain mechanism/workflow
   - Note practical implications
   - Add warnings or best practices
   
   FORMATTING:
   • Use clear paragraphs (2-3 sentences each)
   • Bold important commands: **python3 -m venv myenv**
   • Note warnings: Don't commit venv/ folder

3. **Key Takeaways** ({batch_content[0]['takeaway_constraint']} bullets)
   - Format: "• Specific, actionable point with details"
   - Include commands: "• Create with: python3 -m venv <name>"
   - Technical specifics: "• Each venv averages 20-100MB"

4. **Visual Elements** (if applicable)
   - Extract from OCR frame text
   - Format: ["Command: python3 -m venv myenv", "Terminal output: (venv) user@host"]
   - Empty list [] if no visual text

5. **Tables** (if applicable)
   - For matrices, payoff structures, comparison tables
   - Use structured format with title, headers, rows

REQUIRED JSON FIELDS:
- subtopic_id (string)
- title (string)
- time_range (string: MM:SS - MM:SS)
- overview (string)
- detailed_notes (string)
- key_takeaways (array of strings)
- visual_elements (array of strings)
- tables (array of table objects - USE THIS FOR MATRICES!)

CRITICAL SUCCESS CRITERIA:
Include EVERY command, code snippet from content and OCR
Explain WHY and HOW, not just WHAT
Extract and format ALL visual text from OCR
Use "tables" field for ALL matrices/game theory tables
Follow word counts: {batch_content[0]['word_constraint']}
Use {batch_content[0]['takeaway_constraint']} takeaway bullets
Bold commands using **text**
Use for warnings

Return ONLY valid JSON array matching the schema.
"""
    
    def call_api():
        try:
            model = genai.GenerativeModel(
                model_name,
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": NOTES_SCHEMA  # Defined at start of this file for structured output
                }
            )
            
            response = model.generate_content(
                prompt,
                request_options={'timeout': timeout}
            )
            
            if not response or not response.text:
                raise ValueError("Empty response from API (likely safety block)")
            
            return response
            
        except Exception as e:
            error_str = str(e).lower()
            if 'quota' in error_str or '429' in error_str:
                synthesis_metrics['quota_errors'] += 1
                logger.error(f"QUOTA EXCEEDED: {e}")
            elif 'safety' in error_str or 'blocked' in error_str:
                synthesis_metrics['safety_blocks'] += 1
                logger.warning(f"Safety block: {e}")
            elif '503' in error_str or 'overloaded' in error_str:
                synthesis_metrics['overload_errors'] += 1
                logger.warning(f"Service overloaded: {e}")
            elif '504' in error_str or 'timeout' in error_str:
                synthesis_metrics['timeout_errors'] += 1
                logger.warning(f"Timeout: {e}")
            
            raise
    
    try:
        response = retry_with_exponential_backoff( # retry logic of API 
            call_api,
            max_retries=max_retries,
            retryable_errors=(429, 500, 503, 504)
        )
        
        response_text = response.text.strip()
        logger.info(f"  Response length: {len(response_text)} characters")
        
        notes_list = json.loads(response_text)
        
        is_valid, normalized_notes = validate_and_normalize_notes(notes_list)
        
        if not is_valid or len(normalized_notes) == 0:
            raise ValueError(f"Invalid notes structure: got {len(notes_list)} notes, {len(normalized_notes)} valid")
        
        expected_ids = {s.get('subtopic_id') for s in subtopics_batch}
        generated_ids = {n.get('subtopic_id') for n in normalized_notes}
        missing_ids = expected_ids - generated_ids
        
        if missing_ids:
            logger.warning(f"Missing notes for subtopics: {missing_ids}")
            synthesis_metrics['incomplete_responses'] += 1
        
        synthesis_metrics['success'] += 1
        logger.info(f"  Synthesized {len(normalized_notes)} note sections")
        
        return {
            'notes': normalized_notes,
            'quality': 'high',
            'fallback_used': False,
            'error': None
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"  JSON parse error: {str(e)}")
        logger.error(f"  Response preview: {response_text[:500] if 'response_text' in locals() else 'N/A'}")
        synthesis_metrics['json_errors'] += 1
        
        return {
            'notes': create_fallback_notes(subtopics_batch),
            'quality': 'fallback',
            'fallback_used': True,
            'error': f'JSON parse error: {str(e)}'
        }
        
    except ValueError as e:
        logger.error(f"  Validation error: {str(e)}")
        synthesis_metrics['validation_errors'] += 1
        
        return {
            'notes': create_fallback_notes(subtopics_batch),
            'quality': 'fallback',
            'fallback_used': True,
            'error': f'Validation error: {str(e)}'
        }
        
    except Exception as e:
        error_str = str(e).lower()
        
        if 'quota' in error_str or 'authentication' in error_str or 'permission' in error_str:
            logger.critical(f"CRITICAL ERROR: {e}")
            logger.critical("  This requires immediate attention - stopping gracefully")
            raise
        
        logger.error(f"  Unexpected error: {type(e).__name__}: {str(e)}")
        synthesis_metrics['other_errors'] += 1
        
        return {
            'notes': create_fallback_notes(subtopics_batch),
            'quality': 'fallback',
            'fallback_used': True,
            'error': f'{type(e).__name__}: {str(e)}'
        }


def print_synthesis_metrics():
    # To see how the synthesis went.
    print("\n" + "="*70)
    print("NOTE SYNTHESIS HEALTH REPORT")
    print("="*70)
    
    if synthesis_metrics['total_calls'] == 0:
        print("No synthesis calls made")
        return
    
    total = synthesis_metrics['total_calls']
    success = synthesis_metrics['success']
    fallback = synthesis_metrics['fallback_used']
    
    print(f"Total synthesis calls: {total}")
    print(f"Successful: {success} ({success/total*100:.1f}%)")
    print(f"Fallback used: {fallback} ({fallback/total*100:.1f}%)")
    
    if synthesis_metrics['retries'] > 0:
        print(f"\nRetries: {synthesis_metrics['retries']}")
    
    errors = {
        'JSON errors': synthesis_metrics['json_errors'],
        'Validation errors': synthesis_metrics['validation_errors'],
        'Quota errors': synthesis_metrics['quota_errors'],
        'Safety blocks': synthesis_metrics['safety_blocks'],
        'Timeout errors': synthesis_metrics['timeout_errors'],
        'Overload errors': synthesis_metrics['overload_errors'],
        'Incomplete responses': synthesis_metrics['incomplete_responses'],
        'Other errors': synthesis_metrics['other_errors'],
    }
    
    error_total = sum(errors.values())
    if error_total > 0:
        print(f"\nError Breakdown:")
        for error_type, count in errors.items():
            if count > 0:
                print(f"  - {error_type}: {count}")
    
    # Health assessment
    print(f"\n{'='*70}")
    if fallback / total > 0.5:
        print("UNHEALTHY: >50% fallback rate - investigate API issues!")
    elif fallback / total > 0.2:
        print("DEGRADED: >20% fallback rate - monitor closely")
    else:
        print("HEALTHY: Low fallback rate")
    print("="*70)


def create_fallback_notes(subtopics_batch: List[Dict]) -> List[Dict]:
    # Notes if our note gen failed.
    notes = []
    for subtopic in subtopics_batch:
        detailed = subtopic.get('detailed_content', {})
        notes.append({
            'subtopic_id': subtopic.get('subtopic_id'),
            'title': subtopic.get('title', 'Section'),
            'time_range': f"{subtopic.get('start_time')} - {subtopic.get('end_time')}",
            'overview': subtopic.get('description', 'Content from this section'),
            'detailed_notes': detailed.get('explanation', 'Detailed content not available'),
            'key_takeaways': detailed.get('key_points', ['See video for details']),
            'visual_elements': []
        })
    return notes


def validate_notes_structure_for_display(notes_structure: Dict) -> tuple:
    # Check if the note structure has correct format.

    if not isinstance(notes_structure, dict):
        return False, f"notes_structure is not a dict: {type(notes_structure)}"
    
    required_keys = ['video_title', 'topics']
    for key in required_keys:
        if key not in notes_structure:
            return False, f"Missing required key: '{key}'"
    
    topics = notes_structure.get('topics', [])
    if not isinstance(topics, list):
        return False, f"'topics' is not a list: {type(topics)}"
    
    if len(topics) == 0:
        return False, "No topics found in structure"
    
    for topic_idx, topic in enumerate(topics):
        if not isinstance(topic, dict):
            return False, f"Topic {topic_idx} is not a dict: {type(topic)}"
        
        if 'title' not in topic:
            return False, f"Topic {topic_idx} missing 'title'"
        
        if 'subtopics' not in topic:
            return False, f"Topic {topic_idx} missing 'subtopics'"
        
        subtopics = topic.get('subtopics', [])
        if not isinstance(subtopics, list):
            return False, f"Topic {topic_idx} 'subtopics' not a list: {type(subtopics)}"
        
        for sub_idx, subtopic in enumerate(subtopics):
            if not isinstance(subtopic, dict):
                return False, (f"Topic {topic_idx}, Subtopic {sub_idx} is not a dict: "
                             f"{type(subtopic)}, value: {str(subtopic)[:100]}")
            
            required_fields = ['subtopic_id', 'title']
            for field in required_fields:
                if field not in subtopic:
                    return False, (f"Topic {topic_idx}, Subtopic {sub_idx} "
                                 f"missing required field '{field}'")
    
    return True, "Structure valid"


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def generate_comprehensive_notes(video_id: str, 
                                 video_url: str, 
                                 working_dir: str = "./working",
                                 include_visuals: bool = True) -> Dict:

    print("\n" + "="*80)
    if include_visuals:
        print("COMPREHENSIVE NOTES GENERATION PIPELINE WITH VISION (ENHANCED MODE)")
    else:
        print("COMPREHENSIVE NOTES GENERATION PIPELINE (FAST TEXT-ONLY MODE)")
    print("="*80)
    
    os.makedirs(working_dir, exist_ok=True)
    frames_dir = os.path.join(working_dir, f"frames_{video_id}")
    os.makedirs(frames_dir, exist_ok=True)
    
    video_path = None  # Track video file for cleanup
    
    try:
        print("\n📝 Getting transcript...")
        transcript_chunks = get_transcript(video_id)
        
        if not transcript_chunks or len(transcript_chunks) == 0:
            return {"error": "Failed to get transcript or transcript is empty"}
        
        print(f"Got {len(transcript_chunks)} transcript chunks")
        
        try:
            first_text = transcript_chunks[0].get('text', '') if transcript_chunks else '' # Translation if needed
            if first_text:
                language = detect_language(first_text)
                print(f"Detected language: {language}")
                
                if language.lower() != 'english':
                    print("Translating to English...")
                    translated = translate_transcript_batch(transcript_chunks, target_language='english')
                    if translated:
                        transcript_chunks = translated
                    time.sleep(2) 
        except Exception as e:
            print(f"Language detection/translation failed: {e}")
            print("   Continuing with original transcript...")
        
        topic_structure = extract_topic_structure_with_flash(transcript_chunks)
        
        if not topic_structure or not isinstance(topic_structure, dict):
            return {"error": "Failed to extract topic structure (returned None or invalid)"}
        
        if not topic_structure.get('topics'):
            return {"error": "Topic structure has no topics"}
        
        print(f"Stage 1 complete: {len(topic_structure['topics'])} topics extracted")
        time.sleep(3)  
        
        enhanced_structure = generate_deep_content_with_pro(topic_structure, transcript_chunks)
        
        if not enhanced_structure or not isinstance(enhanced_structure, dict):
            print("Stage 2 failed, using Stage 1 structure")
            enhanced_structure = topic_structure
        
        print(f"Stage 2 complete")
        time.sleep(35) 
        
        if include_visuals:
            print("\nStage 3: Downloading video for frame extraction...")
            video_path = download_full_video_safe(video_id, max_size_gb=5.2)
            
            if not video_path or not os.path.exists(video_path):
                print("Video download failed or file too large (>5GB)")
                print("   Continuing without visual frames (transcript-only notes)")
                video_path = None
            else:
                print(f"Video downloaded: {video_path}")
        else:
            print("\nStage 3: SKIPPED - Fast mode enabled (no visual processing)")
            video_path = None
        
        final_notes_structure = {
            'video_id': video_id,
            'video_title': enhanced_structure.get('video_title', 'Video Notes'),
            'total_duration': enhanced_structure.get('total_duration', 'Unknown'),
            'topics': []
        }
        
        topics_list = enhanced_structure.get('topics', [])
        
        for topic_idx, topic in enumerate(topics_list):
            if not isinstance(topic, dict):
                print(f"Skipping invalid topic at index {topic_idx}: {type(topic)}")
                continue
            
            print(f"\n{'='*80}")
            print(f"Processing Topic {topic.get('topic_id', topic_idx+1)}: {topic.get('title', 'Untitled')}")
            print(f"{'='*80}")
            
            topic_notes = {
                'topic_id': topic.get('topic_id', topic_idx + 1),
                'title': topic.get('title', 'Untitled Topic'),
                'subtopics': []
            }
            
            subtopics = topic.get('subtopics', [])
            
            if not isinstance(subtopics, list):
                print(f"Topic subtopics not a list: {type(subtopics)}")
                continue
            
            if len(subtopics) == 0:
                print(f"Topic has no subtopics, skipping")
                continue
            
            ocr_data = {}
            
            if video_path and include_visuals:
                print(f"\nStage 4: Extracting frames and running OCR for {len(subtopics)} subtopics...")
                
                for subtopic in subtopics:
                    subtopic_id = subtopic.get('subtopic_id')
                    print(f"\n    Subtopic {subtopic_id}: {subtopic.get('title')}")
                    
                    try:
                        frame_results = adaptive_frame_sampling(video_path, subtopic, frames_dir)

                        if frame_results and len(frame_results) > 0:
                            frame_paths = [f['path'] for f in frame_results if isinstance(f, dict) and 'path' in f]
                            
                            print(f"      Extracted {len(frame_paths)} frames")
                            
                            if frame_paths:
                                ocr_results = []
                                MAX_FRAMES_PER_BATCH = 16
                                for i in range(0, len(frame_paths), MAX_FRAMES_PER_BATCH):
                                    batch = frame_paths[i:i+MAX_FRAMES_PER_BATCH]
                                    ocr_results.extend(batch_ocr_processing(batch))
                                
                                text_frames = sum(1 for r in ocr_results if r.get('has_text', False))
                                print(f"      OCR complete: {text_frames}/{len(ocr_results)} frames contain text")
                                
                                ocr_data[subtopic_id] = ocr_results
                            else:
                                ocr_data[subtopic_id] = []
                        else:
                            ocr_data[subtopic_id] = []
                            
                    except Exception as e:
                        print(f"      Frame/OCR error: {e}")
                        ocr_data[subtopic_id] = []
                
                total_ocr_frames = sum(len(v) for v in ocr_data.values())
                print(f"\n  Stage 4 complete: Processed {total_ocr_frames} frames across {len(ocr_data)} subtopics")
            else:
                print(f"\n  Stage 4 skipped: {'Fast mode enabled' if not include_visuals else 'No video file available'} (transcript-only mode)")
                ocr_data = {}
            
            processed_subtopics = []
            batch_size = 2
            
            for batch_idx in range(0, len(subtopics), batch_size):
                batch = subtopics[batch_idx:batch_idx + batch_size]
                
                print(f"\n  Stage 5: Processing batch {batch_idx // batch_size + 1} ({len(batch)} subtopics)...")
                
                try:
                    result = synthesize_notes_with_flash(batch, ocr_data)
                    
                    if not result or not isinstance(result, dict):
                        print(f"  Synthesis returned invalid result: {type(result)}")
                        continue
                    
                    if result.get('fallback_used'):
                        print(f"  Fallback notes used (reason: {result.get('error')})")
                    
                    notes_list = result.get('notes', [])
                    
                    if not isinstance(notes_list, list):
                        print(f"  Result 'notes' is not a list: {type(notes_list)}")
                        continue
                    
                    for note in notes_list:
                        if not isinstance(note, dict):
                            print(f"  Note is not a dict: {type(note)}")
                            print(f"     Value: {str(note)[:200]}")
                            continue
                        
                        if not note.get('subtopic_id') or not note.get('title'):
                            print(f"  Note missing required fields")
                            print(f"     Keys: {list(note.keys())}")
                            continue
                        
                        visual_count = len(note.get('visual_elements', []))
                        if visual_count > 0:
                            print(f"{visual_count} visual elements captured")
                        
                        processed_subtopics.append(note)
                    
                    print(f"  Batch complete: {len(notes_list)} notes processed")
                    
                except Exception as e:
                    print(f"  Error processing batch: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                if batch_idx + batch_size < len(subtopics):
                    print(f"  Rate limit buffer (3 seconds)...")
                    time.sleep(3)
            
            topic_notes['subtopics'] = processed_subtopics
            
            print(f"\n Topic complete: {len(processed_subtopics)}/{len(subtopics)} subtopics processed")
            
            if len(processed_subtopics) > 0:
                final_notes_structure['topics'].append(topic_notes)
            else:
                print(f"  Topic has no valid subtopics, skipping")
            
            if topic_idx < len(topics_list) - 1:
                print("\nBuffer between topics (5 seconds)...")
                time.sleep(5)
        
        print("\nCleaning up temporary files...")
        
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
                print(f"Removed video file: {video_path}")
            except Exception as e:
                print(f"Failed to remove video file: {e}")
        
        if os.path.exists(frames_dir):
            try:
                shutil.rmtree(frames_dir)
                print(f"Removed frames directory: {frames_dir}")
            except Exception as e:
                print(f"Failed to remove frames directory: {e}")
        
        notes_file = os.path.join(working_dir, f"notes_{video_id}.json")
        
        try:
            with open(notes_file, 'w', encoding='utf-8') as f:
                json.dump(final_notes_structure, f, indent=2, ensure_ascii=False)
            print(f"\nNotes saved to: {notes_file}")
        except Exception as e:
            print(f"Failed to save notes file: {e}")
        
        print("\n" + "="*80)
        print("NOTES GENERATION COMPLETE!")
        print(f"Notes saved to: {notes_file}")
        print(f"Statistics:")
        print(f"   - Total Topics: {len(final_notes_structure['topics'])}")
        
        total_subtopics = 0
        total_visual_elements = 0
        for topic in final_notes_structure['topics']:
            total_subtopics += len(topic.get('subtopics', []))
            for subtopic in topic.get('subtopics', []):
                total_visual_elements += len(subtopic.get('visual_elements', []))
        
        print(f"   - Total Subtopics: {total_subtopics}")
        print(f"   - Visual Elements Captured: {total_visual_elements}")
        print(f"   - OCR Enabled: {'Yes' if (video_path and include_visuals) else 'No (transcript-only)'}")
        print(f"   - Processing Mode: {'Enhanced (with visuals)' if include_visuals else 'Fast (text-only)'}")
        print("="*80)
        
        print("\nValidating final structure...")
        is_valid, error_msg = validate_notes_structure_for_display(final_notes_structure)
        
        if not is_valid:
            print(f"Structure validation warning: {error_msg}")
            print("   Notes may have formatting issues")
        else:
            print("Structure validation passed")
        
        return final_notes_structure
        
    except Exception as e:
        print(f"\nPipeline error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
                print(f"Cleaned up video file after error")
            except:
                pass
        
        if os.path.exists(frames_dir):
            try:
                shutil.rmtree(frames_dir)
                print(f"Cleaned up frames directory after error")
            except:
                pass
        
        return {"error": f"Pipeline failed: {str(e)}"}

    
# ============================================================================
# Format Notes for Display (in Markdown format 'md')
# ============================================================================

def format_notes_for_display(notes_structure: Dict) -> str:
    # Format notes structure into readable Markdown text
    
    if not isinstance(notes_structure, dict):
        return f"# Error\n\nInvalid notes structure: expected dict, got {type(notes_structure)}"
    
    video_title = notes_structure.get('video_title', 'Video Notes')
    total_duration = notes_structure.get('total_duration', 'Unknown')
    
    markdown = f"# {video_title}\n\n"
    markdown += f"**Duration:** {total_duration}\n"
    markdown += f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    markdown += "---\n\n"
    
    topics = notes_structure.get('topics', [])
    
    if not isinstance(topics, list):
        markdown += f"\nError: Topics is not a list (type: {type(topics)})\n"
        return markdown
    
    if len(topics) == 0:
        markdown += "\nNo topics found in notes.\n"
        return markdown
    
    for topic_idx, topic in enumerate(topics):
        if not isinstance(topic, dict):
            markdown += f"\nSkipping invalid topic {topic_idx + 1} (not a dict)\n\n"
            continue
        
        topic_id = topic.get('topic_id', topic_idx + 1)
        topic_title = topic.get('title', 'Untitled Topic')
        
        markdown += f"\n## {topic_id}. {topic_title}\n\n"
        
        subtopics = topic.get('subtopics', [])
        
        if not isinstance(subtopics, list):
            markdown += f"Subtopics not a list for this topic (type: {type(subtopics)})\n\n"
            continue
        
        if len(subtopics) == 0:
            markdown += "No subtopics found for this topic.\n\n"
            continue
        
        for sub_idx, subtopic in enumerate(subtopics):
            if not isinstance(subtopic, dict):
                markdown += f"Skipping invalid subtopic {sub_idx + 1} (type: {type(subtopic)})\n\n"
                print(f"WARNING: Subtopic {sub_idx + 1} in topic {topic_id} is not a dict: {type(subtopic)}")
                print(f"Value: {str(subtopic)[:200]}")
                continue
            
            subtopic_id = subtopic.get('subtopic_id', f"{topic_id}.{sub_idx + 1}")
            subtopic_title = subtopic.get('title', 'Untitled Subtopic')
            time_range = subtopic.get('time_range', 'N/A')
            overview = subtopic.get('overview', '')
            detailed_notes = subtopic.get('detailed_notes', '')
            
            markdown += f"### {subtopic_id} {subtopic_title}\n"
            markdown += f"*Time: {time_range}*\n\n"
            
            if overview and isinstance(overview, str):
                markdown += f"**Overview:** {overview}\n\n"
            
            if detailed_notes and isinstance(detailed_notes, str):
                markdown += f"{detailed_notes}\n\n"
            
            takeaways = subtopic.get('key_takeaways', [])
            
            if not isinstance(takeaways, list):
                if isinstance(takeaways, str):
                    takeaways = [takeaways] if takeaways else []
                else:
                    takeaways = []
            
            valid_takeaways = [
                t for t in takeaways 
                if t and isinstance(t, str) and len(t.strip()) > 1
            ]
            
            if valid_takeaways:
                markdown += "**Key Takeaways:**\n"
                for point in valid_takeaways:
                    markdown += f"- {point.strip()}\n"
                markdown += "\n"
            
            visuals = subtopic.get('visual_elements', [])
            
            if not isinstance(visuals, list):
                if isinstance(visuals, str):
                    visuals = [visuals] if visuals else []
                else:
                    visuals = []
            
            valid_visuals = [
                v for v in visuals 
                if v and isinstance(v, str) and len(v.strip()) > 1
            ]
            
            if valid_visuals:
                markdown += "**Visual Elements:**\n"
                for element in valid_visuals:
                    markdown += f"- {element.strip()}\n"
                markdown += "\n"
            
            markdown += "---\n\n"
    
    return markdown



# ============================================================================
# EXPORT FUNCTION for app.py
# ============================================================================

def process_video_for_notes(video_url: str, 
                           include_visuals: bool = True,
                           working_dir: str = "./working") -> Tuple[bool, str, Optional[Dict]]:
    try:
        video_id_match = re.search(
            r'(?:v=|/videos/|embed/|youtu\.be/|/v/|watch\?v=|&v=)([^#&?\n]*)', 
            video_url
        )
        if not video_id_match:
            return False, "Invalid YouTube URL", None
        
        video_id = video_id_match.group(1)
        
        print(f"\n{'='*80}")
        print(f"NOTES GENERATION MODE: {'ENHANCED (with visuals)' if include_visuals else 'FAST (text-only)'}")
        print(f"{'='*80}\n")
        
        notes_structure = generate_comprehensive_notes(
            video_id, 
            video_url, 
            working_dir,
            include_visuals=include_visuals
        )
        
        if 'error' in notes_structure:
            return False, f"Error: {notes_structure['error']}", None
        
        formatted_notes = format_notes_for_display(notes_structure)
        
        return True, formatted_notes, notes_structure
        
    except Exception as e:
        import traceback
        error_msg = f"Error processing video: {str(e)}\n{traceback.format_exc()}"
        return False, error_msg, None



# ============================================================================
# TEST CODE - Hardcoded Video Link
# ============================================================================

# if __name__ == "__main__":
#     # HARDCODED TEST VIDEO URL
#     test_video_url = "https://www.youtube.com/watch?v=aFczzFYjUec"  
    
#     print("="*80)
#     print("TESTING NOTES GENERATION")
#     print("="*80)
#     print(f"\nVideo URL: {test_video_url}")
#     print("\nStarting pipeline... This will take 2-6 minutes\n")
    
#     # Call the main function directly
#     success, result, structure = process_video_for_notes(test_video_url)
    
#     if success:
#         print("\n" + "="*80)
#         print("SUCCESS! NOTES GENERATED")
#         print("="*80)
        
#         # Print the formatted notes
#         print("\n" + result)
        
#         # Print summary statistics
#         if structure:
#             print("\n" + "="*80)
#             print("SUMMARY STATISTICS")
#             print("="*80)
#             print(f"Video Title: {structure.get('video_title')}")
#             print(f"Duration: {structure.get('total_duration')}")
#             print(f"Total Topics: {len(structure.get('topics', []))}")
            
#             total_subtopics = sum(len(t.get('subtopics', [])) for t in structure.get('topics', []))
#             print(f"Total Subtopics: {total_subtopics}")
            
#             # Show topic breakdown
#             print("\nTopic Breakdown:")
#             for topic in structure.get('topics', []):
#                 print(f"\n  {topic.get('topic_id')}. {topic.get('title')}")
#                 for subtopic in topic.get('subtopics', []):
#                     print(f"     - {subtopic.get('subtopic_id')}: {subtopic.get('title')} ({subtopic.get('time_range')})")
        
#         # Save to files
#         video_id = structure.get('video_id', 'test_video')
        
#         # Save Markdown
#         md_filename = f"notes_{video_id}.md"
#         with open(md_filename, 'w', encoding='utf-8') as f:
#             f.write(result)
#         print(f"\nSaved Markdown: {md_filename}")
        
#         # Save JSON
#         json_filename = f"notes_{video_id}.json"
#         with open(json_filename, 'w', encoding='utf-8') as f:
#             json.dump(structure, f, indent=2, ensure_ascii=False)
#         print(f"Saved JSON: {json_filename}")
        
#     else:
#         print("\n" + "="*80)
#         print("FAILED")
#         print("="*80)
#         print(f"\nError Message:\n{result}")
    
#     print("\n" + "="*80)
#     print("TEST COMPLETE")
#     print("="*80)

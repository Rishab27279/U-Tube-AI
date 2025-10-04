import os
import re
import time
import json
import glob
import shutil
import subprocess
import yt_dlp
import streamlit as st
import google.generativeai as genai

from typing import List, Optional
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer

load_dotenv()

## "all-mpnet-base-v2" Model for local embeddings as Gemini has monetization issues.

class LocalSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vecs = self.model.encode(texts, batch_size=32, show_progress_bar=False)
        return [v.tolist() for v in (vecs if hasattr(vecs, "__len__") else [vecs])]

    def embed_query(self, query: str) -> List[float]:
        return self.model.encode(query).tolist()
    
# ----------------------------------------------------------------------------------------------------------------------
#  Func to extract YouTube video ID from URL
#-----------------------------------------------------------------------------------------------------------------------

def extract_video_id(youtube_url):
    match = re.search(r'(?:https?://)?(?:www\.)?(?:m\.)?(?:youtube\.com/(?:watch\?v=|embed/|v/|.+\?v=)|youtu\.be/)([a-zA-Z0-9_-]{11})', youtube_url)
    if match:
        return match.group(1)
    else:
        st.error("Invalid YouTube URL. Please enter a valid video link.")
        return None

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ----------------------------------------------------------------------------------------------------------------------
#  Download YouTube Audio and Video
#-----------------------------------------------------------------------------------------------------------------------

def download_youtube_audio(video_id):
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f'{video_id}_audio.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return f"{video_id}_audio.mp3"
    except Exception as e:
        print(f"Error downloading audio: {e}")
        return None

def check_video_duration(youtube_url, max_hours=10):
    try:
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            duration = info.get('duration', 0)
            duration_hours = duration / 3600
            
            print(f"[video] Duration: {duration_hours:.1f} hours ({duration/60:.1f} minutes)")
            
            if duration_hours > max_hours:
                print(f"[video] Video too long ({duration_hours:.1f}h > {max_hours}h), will use fallback")
                return False, duration
            
            return True, duration
    except Exception as e:
        print(f"[video] Error checking duration: {e}")
        return False, 0

def download_full_video_safe(video_id, max_size_gb=5.2):
    try:
        youtube_url = f"https://www.youtube.com/watch?v={video_id}"
        
        duration_ok, duration = check_video_duration(youtube_url, max_hours=10)
        if not duration_ok:
            return None
        
        print(f"[video] Starting download of {duration/60:.1f} minute video...")
        
        ydl_opts = {
            "format": "135+140/best[height<=480]",
            "outtmpl": f"{video_id}_full.%(ext)s",
            "writeinfojson": False,
            "writesubtitles": False,
            "sleep_interval": 2,
            "retries": 3,
            "fragment_retries": 5,
            "timeout": 300,
            "quiet": False,
            "no_warnings": False,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        
        video_files = [f for f in os.listdir('.') if f.startswith(f'{video_id}_full') and f.endswith('.mp4')]
        if not video_files:
            print("[video] No video file was created")
            return None
            
        video_file = video_files[0]
        size_gb = os.path.getsize(video_file) / (1024**3)
        print(f"[video] Downloaded: {size_gb:.2f} GB")
        
        if size_gb > max_size_gb:
            print(f"[video] File too large ({size_gb:.2f} GB > {max_size_gb} GB), removing")
            try:
                os.remove(video_file)
            except:
                pass
            return None
        
        return video_file
        
    except Exception as e:
        print(f"[video] Error downloading: {e}")
        return None
    
# ----------------------------------------------------------------------------------------------------------------------
#  Load Downladed Video
#-----------------------------------------------------------------------------------------------------------------------

def extract_frames_from_local_video(video_file, timestamps, video_id, frames_per_segment=3):
    import cv2
    
    frames = []
    
    try:
        print(f"[frames] Extracting frames from {video_file}")
        cap = cv2.VideoCapture(video_file)
        
        if not cap.isOpened():
            print("[frames] Error: Could not open video file")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"[frames] Video info: {fps:.1f} FPS, {total_frames} frames, {duration/60:.1f} minutes")
        
        for i, ts in enumerate(timestamps[:4]):
            if ts >= duration:
                print(f"[frames] Timestamp {ts}s beyond video duration {duration:.1f}s, skipping")
                continue
                
            print(f"[frames] Processing timestamp {i+1}/4: {ts}s")
            
            frame_positions = []
            if frames_per_segment >= 3:
                for offset in [-2, 0, 2]:
                    pos = max(0, min(duration-1, ts + offset))
                    frame_positions.append(pos)
            else:
                frame_positions = [ts]
            
            for j, frame_time in enumerate(frame_positions):
                frame_number = int(frame_time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                
                ret, frame = cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                    if width > 640:
                        scale = 640.0 / width
                        new_width = 640
                        new_height = int(height * scale)
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    frame_filename = f"{video_id}_frame_{ts}_{j}.jpg"
                    cv2.imwrite(frame_filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    
                    frames.append({
                        'timestamp': int(frame_time),
                        'path': frame_filename
                    })
                    
                    print(f"[frames] Extracted frame at {frame_time:.1f}s -> {frame_filename}")
        
        cap.release()
        print(f"[frames] Successfully extracted {len(frames)} frames")
        return frames
        
    except Exception as e:
        print(f"[frames] Error extracting frames: {e}")
        try:
            if 'cap' in locals():
                cap.release()
        except:
            pass
        return []

def extract_youtube_static_thumbnails(video_id, timestamps):  # If vid not available, then Thumbnails
    import requests
    
    frames = []
    print("[thumbnails] Using YouTube static thumbnails as fallback")
    
    thumb_qualities = [
        "maxresdefault",
        "hqdefault",
        "mqdefault",
    ]
    
    for i, ts in enumerate(timestamps[:2]):
        for quality in thumb_qualities:
            thumb_url = f"https://img.youtube.com/vi/{video_id}/{quality}.jpg"
            try:
                response = requests.get(thumb_url, timeout=10)
                if response.status_code == 200 and len(response.content) > 1000:
                    frame_path = f"{video_id}_thumb_{ts}_{i}.jpg"
                    with open(frame_path, 'wb') as f:
                        f.write(response.content)
                    frames.append({'timestamp': ts, 'path': frame_path})
                    print(f"[thumbnails] Downloaded {quality} thumbnail -> {frame_path}")
                    break
            except Exception as e:
                print(f"[thumbnails] Failed to get {quality} thumbnail: {e}")
                continue
    
    return frames

# ----------------------------------------------------------------------------------------------------------------------
# Code to understand the type of Video Content (Implementation vs Conceptual)
# True if Implementation-focused, False if Conceptual 
#-----------------------------------------------------------------------------------------------------------------------

def detect_query_type(query):
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
        Analyze this user query and classify it as either 'implementation' or 'conceptual'.

        Query: "{query}"

        Classification Rules:
        - IMPLEMENTATION: Hands-on activities, step-by-step processes, coding, building, creating, demonstrating, showing how to do something, visual tutorials, practical examples
        - CONCEPTUAL: Theoretical explanations, definitions, understanding principles, discussing ideas, analyzing concepts, explaining why/what

        Examples:
        - "Show me how to implement this algorithm" → implementation
        - "What is machine learning?" → conceptual  
        - "Walk through the code step by step" → implementation
        - "Explain the concept of recursion" → conceptual
        - "Create a function that does X" → implementation
        - "Why does this approach work?" → conceptual

        Respond with ONLY one word: "implementation" or "conceptual"
        """
        
        response = model.generate_content(prompt)
        result = response.text.strip().lower()
        
        # Return True for implementation, False for conceptual
        return result == "implementation"
        
    except Exception as e:
        print(f"Error in query classification: {e}")
        return detect_query_type_fallback(query)

def detect_query_type_fallback(query):
    """
    Fallback keyword-based detection if LLM fails
    """
    implementation_keywords = [
        'code', 'extract', 'program', 'function', 'write', 'syntax', 'implement', 
        'create', 'build', 'develop', 'script', 'coding', 'variable', 'loop', 
        'class', 'method', 'algorithm', 'solution', 'example', 'demonstration',
        'screen', 'display', 'show', 'editor', 'terminal', 'console', 'ide',
        'notebook', 'jupyter', 'vs code', 'vscode', 'typing', 'keyboard',
        'how to', 'step by step', 'hands on', 'practical', 'tutorial', 
        'walkthrough', 'demo', 'live coding', 'actual', 'real'
    ]
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in implementation_keywords)

# ----------------------------------------------------------------------------------------------------------------------
#  Extraction of Video Frames
#-----------------------------------------------------------------------------------------------------------------------

def extract_video_frames_optimized(video_id, timestamps, buffer_seconds=5, frames_per_segment=3):
    video_file_path = None
    try:
        print(f"[extract] Starting frame extraction for video {video_id}")
        print(f"[extract] Target timestamps: {timestamps}")
        
        download_result = download_full_video_safe(video_id, max_size_gb=5.2)
        
        if download_result:
            video_file_path = download_result
            status = "success" if video_file_path else "failed"
            
            if video_file_path and os.path.exists(video_file_path) and status == "success":
                print(f"[extract] Using full video download approach")
                frames = extract_frames_from_local_video(video_file_path, timestamps, video_id, frames_per_segment)
                
                if frames:
                    print(f"[extract] Successfully extracted {len(frames)} frames from video")
                    for frame in frames:
                        frame['video_file'] = video_file_path
                    return frames
                else:
                    print("[extract] Frame extraction from video failed, trying fallback")
            else:
                print(f"[extract] Video download failed: {status}")
        else:
            print("[extract] Video download failed or file too large")
        
        print("[extract] Using YouTube thumbnails as fallback")
        frames = extract_youtube_static_thumbnails(video_id, timestamps)
        
        if frames:
            print(f"[extract] Got {len(frames)} thumbnail frames")
            return frames
        else:
            print("[extract] All frame extraction methods failed")
            return []
            
    except Exception as e:
        print(f"[extract] Error in frame extraction: {e}")
        return []
        
    finally:
        if video_file_path and os.path.exists(video_file_path):
            try:
                file_size = os.path.getsize(video_file_path) / (1024**3)
                print(f"[info] Keeping video file: {video_file_path} ({file_size:.2f} GB)")
            except Exception as e:
                print(f"[info] Could not get video file info: {e}")


def cleanup_leftover_videos(video_id): # Clean small leftover files (frames, audio, segments) - but not full videos
    try:
        import glob
        
        patterns = [
            f"{video_id}_audio.*",      
            f"{video_id}_segment_*",      
            f"{video_id}_frame_*",      
            f"{video_id}_thumb_*"       
        ]
        
        cleaned_files = []
        for pattern in patterns:
            for file_path in glob.glob(pattern):
                try:
                    file_size = os.path.getsize(file_path) / (1024*1024)  # MB
                    os.remove(file_path)
                    cleaned_files.append((file_path, file_size))
                    print(f"[cleanup] Removed: {file_path} ({file_size:.1f} MB)")
                except Exception as e:
                    print(f"[cleanup] Failed to remove {file_path}: {e}")
        
        if cleaned_files:
            total_size = sum(size for _, size in cleaned_files)
            print(f"[cleanup] Cleaned {len(cleaned_files)} files, freed {total_size:.1f} MB")
        
        return len(cleaned_files)
        
    except Exception as e:
        print(f"[cleanup] Error in leftover cleanup: {e}")
        return 0

def cleanup_video_files_for_streamlit(video_id): # Only after user's session is done.

    try:
        import glob
        
        patterns = [
            f"{video_id}_full.*",       
            f"{video_id}_segment_*",  
            f"{video_id}_audio.*", 
            f"{video_id}_frame_*",      
            f"{video_id}_thumb_*"      
        ]
        
        cleaned_files = []
        total_size_gb = 0
        
        for pattern in patterns:
            for file_path in glob.glob(pattern):
                try:
                    file_size_gb = os.path.getsize(file_path) / (1024**3)  # GB
                    
                    if os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                        except PermissionError:
                            import stat
                            os.chmod(file_path, stat.S_IWRITE)
                            os.remove(file_path)
                    
                    cleaned_files.append((file_path, file_size_gb))
                    total_size_gb += file_size_gb
                    
                    if file_size_gb > 0.01: 
                        print(f"[cleanup] Removed: {file_path} ({file_size_gb:.2f} GB)")
                    else:
                        print(f"[cleanup] Removed: {file_path}")
                        
                except Exception as e:
                    print(f"[cleanup] Failed to remove {file_path}: {e}")
        
        if cleaned_files:
            print(f"[cleanup] ✅ Cleaned {len(cleaned_files)} files, freed {total_size_gb:.2f} GB total")
        else:
            print(f"[cleanup] No files found for video {video_id}")
        
        return cleaned_files
        
    except Exception as e:
        print(f"[cleanup] Error in complete cleanup: {e}")
        return []

def cleanup_specific_file_types(video_id, file_types=['frames', 'audio', 'segments']):
    try:
        import glob
        
        type_patterns = {
            'frames': f"{video_id}_frame_*",
            'audio': f"{video_id}_audio.*", 
            'segments': f"{video_id}_segment_*",
            'thumbs': f"{video_id}_thumb_*",
            'full_video': f"{video_id}_full.*"
        }
        
        patterns = [type_patterns[ft] for ft in file_types if ft in type_patterns]
        
        cleaned_files = []
        for pattern in patterns:
            for file_path in glob.glob(pattern):
                try:
                    file_size = os.path.getsize(file_path) / (1024*1024)  
                    os.remove(file_path)
                    cleaned_files.append((file_path, file_size))
                    print(f"[cleanup] Removed {file_path} ({file_size:.1f} MB)")
                except Exception as e:
                    print(f"[cleanup] Failed to remove {file_path}: {e}")
        
        return cleaned_files
        
    except Exception as e:
        print(f"[cleanup] Error in selective cleanup: {e}")
        return []

# ----------------------------------------------------------------------------------------------------------------------
# Extract Transcript using Gemini-2.5-Flash
#-----------------------------------------------------------------------------------------------------------------------

def get_transcript(video_id, lang=None):
    try:
        print(f"DEBUG: Starting transcript extraction for video {video_id}")
        audio_path = download_youtube_audio(video_id)
        if not audio_path or not os.path.exists(audio_path):
            print("ERROR: Audio download failed")
            return None

        print(f"DEBUG: Audio downloaded to {audio_path}")
        audio_file = genai.upload_file(audio_path)
        print(f"DEBUG: Audio uploaded to Gemini: {audio_file.name}")
        time.sleep(5)

        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content([
            """Transcribe this audio with the following requirements:
            1. Identify different speakers as Speaker_A, Speaker_B, etc.
            2. Include precise timestamps in format [MM:SS] for each speaker segment
            3. Provide clean, accurate transcription
            4. Format each segment as: [MM:SS] Speaker_X: transcript text
            5. Ensure timestamps are continuous and accurate
            6. Create segments every 10-15 seconds for detailed breakdown

            Example format:
            [00:00] Speaker_A: Hello everyone, welcome to today's video
            [00:15] Speaker_B: Thank you for having me on the show
            [00:30] Speaker_A: Let's talk about our main topic today
            """,
            audio_file
        ])

        try:
            os.remove(audio_path)
        except:
            pass
        try:
            genai.delete_file(audio_file.name)
        except:
            pass
        print("DEBUG: Cleaned up audio files")

        transcript_text = response.text
        print(f"DEBUG: Raw transcript length: {len(transcript_text)} characters")

        lines = transcript_text.strip().split('\n')
        all_segments = []
        print(f"DEBUG: Total lines in transcript: {len(lines)}")

        timestamp_patterns = [
            r'\[(\d{1,2}):(\d{2})\]\s+(Speaker_\w+):\s*(.*)',
            r'\[(\d{1,2}):(\d{2})\]\s*(\w+):\s*(.*)',
            r'(\d{1,2}):(\d{2})\s+(Speaker_\w+):\s*(.*)',
            r'\[(\d{1,2}):(\d{2})\]\s*(.*)',
        ]

        for i, line in enumerate(lines):
            if not line.strip():
                continue
            matched = False
            for pattern in timestamp_patterns:
                m = re.match(pattern, line.strip())
                if m:
                    groups = m.groups()
                    minutes, seconds = groups[0], groups[1]
                    if len(groups) >= 4:
                        speaker = groups[2] if groups[2].startswith("Speaker_") else "Speaker_A"
                        text = groups[3]
                    else:
                        speaker = "Speaker_A"
                        text = groups[-1]

                    start_time = int(minutes) * 60 + int(seconds)
                    clean_text = ' '.join(text.replace('\xa0', ' ').split())
                    if clean_text:
                        all_segments.append({
                            'text': clean_text,
                            'start': start_time,
                            'speaker': speaker,
                            'duration': 10
                        })
                        print(f"DEBUG: Segment {len(all_segments)}: [{start_time}s] {speaker}: {clean_text[:60]}...")
                    matched = True
                    break
            if not matched:
                print(f"DEBUG: Unmatched line {i}: {line.strip()[:100]}...")

        if len(all_segments) == 0:
            print("DEBUG: No timestamped segments found, creating artificial segments")
            sentences = re.split(r'[.!?]+', transcript_text)
            segment_duration = 15
            for i, sentence in enumerate(sentences):
                clean_sentence = sentence.strip()
                if len(clean_sentence) > 20:
                    all_segments.append({
                        'text': clean_sentence,
                        'start': i * segment_duration,
                        'speaker': 'Speaker_A',
                        'duration': segment_duration
                    })

        if len(all_segments) == 0:
            print("ERROR: No segments could be created from transcript")
            return None

        video_duration = max(seg['start'] + seg['duration'] for seg in all_segments)
        print(f"DEBUG: Estimated video duration: {video_duration} seconds ({video_duration/60:.1f} minutes)")

        checkpoint_interval = 30
        overlap_seconds = 5
        step_size = checkpoint_interval - overlap_seconds

        structured_transcript = []
        chunk_start = 0
        chunk_id = 0

        while chunk_start < video_duration:
            chunk_end = chunk_start + checkpoint_interval
            chunk_segments, chunk_text, chunk_speakers = [], "", []
            for segment in all_segments:
                segment_end = segment['start'] + segment['duration']
                if (segment['start'] < chunk_end and segment_end > chunk_start):
                    chunk_segments.append(segment)
                    chunk_text += segment['text'] + ' '
                    chunk_speakers.append(segment['speaker'])

            if chunk_segments and chunk_text.strip():
                dominant_speaker = max(set(chunk_speakers), key=chunk_speakers.count) if chunk_speakers else 'Speaker_A'
                structured_transcript.append({
                    'start_time': chunk_start,
                    'end_time': min(chunk_end, video_duration),
                    'text': chunk_text.strip(),
                    'segments': chunk_segments,
                    'speakers': chunk_speakers,
                    'dominant_speaker': dominant_speaker
                })
                chunk_id += 1
                print(f"DEBUG: Chunk {chunk_id}: {chunk_start}s-{min(chunk_end, video_duration)}s, "
                      f"{len(chunk_segments)} segments, {len(chunk_text.split())} words")

            chunk_start += step_size

        print(f"DEBUG: Created {len(structured_transcript)} final chunks")

        structured_chunks = []
        for i, checkpoint in enumerate(structured_transcript):
            all_speakers_csv = ', '.join(list(set(checkpoint.get('speakers', []))))
            structured_chunks.append({
                'text': checkpoint['text'],
                'metadata': {
                    'chunk_id': i,
                    'start_time': checkpoint['start_time'],
                    'end_time': checkpoint['end_time'],
                    'video_id': video_id,
                    'duration': checkpoint['end_time'] - checkpoint['start_time'],
                    'dominant_speaker': checkpoint['dominant_speaker'],
                    'all_speakers': all_speakers_csv,
                    'word_count': len(checkpoint['text'].split()),
                    'segment_count': len(checkpoint['segments'])
                }
            })

        print(f"DEBUG: Final output: {len(structured_chunks)} chunks ready for embedding")
        time.sleep(1)
        return structured_chunks

    except Exception as e:
        print(f"ERROR: Transcript extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
# -----------------------------------------------------------------------------------------------------------------------
# Translate if not English
#-----------------------------------------------------------------------------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.2,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

def translate_transcript_batch(transcript_chunks, batch_size=3):
    try:
        translated_chunks = []
        prompt = ChatPromptTemplate.from_template("""
        You are an expert multilingual translator specializing in mixed-language content.

        CRITICAL TRANSLATION RULES:
        1. Convert ALL non-English words to natural English
        2. Handle transliterated content (Hindi in Roman script)
        3. Convert Devanagari script to English
        4. Preserve technical terms and proper nouns
        5. Maintain natural flow and meaning
        6. Convert code-switched sentences to pure English

        Return ONLY clean English translations, numbered as provided.

        Text segments to translate to PURE ENGLISH:
        {batch_text}
        """)
        chain = prompt | llm

        for i in range(0, len(transcript_chunks), batch_size):
            batch = transcript_chunks[i:i + batch_size]
            batch_text = "\n\n".join([f"{j+1}. {chunk['text']}" for j, chunk in enumerate(batch)])
            response = chain.invoke({"batch_text": batch_text})

            parts = re.split(r'\n\s*\d+\.\s*', response.content.strip())
            if parts and not parts[0].strip():
                parts = parts[1:]

            for j, chunk in enumerate(batch):
                txt = parts[j].strip() if j < len(parts) else chunk['text']
                translated_chunks.append({'text': txt, 'metadata': chunk['metadata']})
            time.sleep(0.5)

        return translated_chunks

    except Exception as e:
        st.error(f"Error translating transcript: {e}")
        return None

def _sanitize_metadata(md: dict) -> dict: 
    safe = {} # Metadata contains complex information (nested dicts, lists, numbers, etc.) that can create issues when serializing (saving to CSV/JSON).
    for k, v in md.items(): # Sanitize metadata to ensure all values are strings or simple types.
        if isinstance(v, (str, int, float, bool)) or v is None:
            safe[k] = v
        elif isinstance(v, list):
            safe[k] = ', '.join(list({str(x) for x in v}))
        elif isinstance(v, dict):
            continue
        else:
            safe[k] = str(v)
    return safe
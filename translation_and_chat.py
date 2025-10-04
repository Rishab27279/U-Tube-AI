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

from main import (
    LocalSentenceTransformerEmbeddings,
    download_full_video_safe,
    extract_video_frames_optimized,
    get_transcript,
    translate_transcript_batch,
    _sanitize_metadata,
    cleanup_leftover_videos,
    cleanup_video_files_for_streamlit,
    cleanup_specific_file_types,
    detect_query_type,
    extract_video_id
)

load_dotenv()

# ----------------------------------------------------------------------------------------------------------------------
#  Chroma DB
#-----------------------------------------------------------------------------------------------------------------------

def create_vector_store(translated_chunks, video_id):
    try:
        embeddings = LocalSentenceTransformerEmbeddings('all-mpnet-base-v2')

        texts, metadatas = [], []
        for chunk in translated_chunks:
            texts.append(chunk['text'])
            md = dict(chunk['metadata'])
            md['video_id'] = video_id
            metadatas.append(_sanitize_metadata(md))

        collection_name = f"video_{video_id}"
        vectorstore = Chroma.from_texts(
            texts=texts,
            metadatas=metadatas,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=f"./chroma_db_{video_id}"
        )
        print(f"Created embeddings and stored {len(texts)} chunks for video {video_id}")
        return vectorstore
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

def load_existing_vectorstore(video_id): # if chroma db exists, then load it.
    try:
        embeddings = LocalSentenceTransformerEmbeddings('all-mpnet-base-v2')
        collection_name = f"video_{video_id}"
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=f"./chroma_db_{video_id}"
        )
        return vectorstore
    except Exception as e:
        print(f"Error loading existing vector store: {e}")
        return None

# ----------------------------------------------------------------------------------------------------------------------
#  Semantic Search
#-----------------------------------------------------------------------------------------------------------------------

def semantic_search_with_phase_filtering(vectorstore, query, top_k=3, intro_skip_percent=0.2):
    try:
        candidate_k = min(top_k * 3, 12)
        all_docs = vectorstore.similarity_search(query=query, k=candidate_k)
        
        if not all_docs:
            return []
        
        all_timestamps = [doc.metadata.get('start_time', 0) for doc in all_docs]
        vid_dur = max(all_timestamps) if all_timestamps else 3600
        
        is_implementation_query = detect_query_type(query) # Conceptual or Implementation
        
        if is_implementation_query and vid_dur > 300:
            intro_cutoff = vid_dur * intro_skip_percent
            implementation_start = vid_dur * 0.25
            implementation_end = vid_dur * 0.85
            
            implementation_docs = [
                doc for doc in all_docs 
                if implementation_start <= doc.metadata.get('start_time', 0) <= implementation_end
            ]
            
            post_intro_docs = [
                doc for doc in all_docs 
                if doc.metadata.get('start_time', 0) >= intro_cutoff
            ]
            
            if len(implementation_docs) >= top_k:
                filtered_docs = implementation_docs[:top_k]
                phase_info = f"implementation phase ({implementation_start/60:.1f}-{implementation_end/60:.1f} min)"
            elif len(post_intro_docs) >= top_k:
                filtered_docs = post_intro_docs[:top_k] 
                phase_info = f"post-intro phase ({intro_cutoff/60:.1f}+ min)"
            else:
                filtered_docs = all_docs[:top_k]
                phase_info = "all phases (insufficient filtered results)"
            
            print(f"[search] Implementation query detected, using {phase_info}")
        else:
            filtered_docs = all_docs[:top_k]
            print(f"[search] Concept query or short video, using regular search")
        
        return [{'text': d.page_content, 'metadata': d.metadata} for d in filtered_docs]
        
    except Exception as e:
        print(f"Error during semantic search: {e}")
        return None

def semantic_search(vectorstore, query, top_k=3):
    return semantic_search_with_phase_filtering(vectorstore, query, top_k)

# ----------------------------------------------------------------------------------------------------------------------
# Navigate the Video using Gemini-2.5-Flash
#-----------------------------------------------------------------------------------------------------------------------

flash_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

def navigate_video_segments(query, relevant_chunks):
    try:
        if not relevant_chunks:
            return []
        
        timestamps = [ch['metadata'].get('start_time', 0) for ch in relevant_chunks]
        vid_dur = max(timestamps) if timestamps else 3600
        
        is_implementation_query = detect_query_type(query)
        
        if is_implementation_query and vid_dur > 600:
            hands_on_start = int(vid_dur * 0.3)
            hands_on_end = int(vid_dur * 0.8)
            
            hands_on_timestamps = [
                ts for ts in timestamps 
                if hands_on_start <= ts <= hands_on_end
            ]
            
            if len(hands_on_timestamps) >= 2:
                unique_timestamps = sorted(list(set(hands_on_timestamps)))
                selected = unique_timestamps[:4]
                print(f"[navigation] Implementation query: targeting hands-on section {hands_on_start/60:.1f}-{hands_on_end/60:.1f} min")
                print(f"[navigation] Selected timestamps: {selected}")
                return selected
            else:
                print(f"[navigation] Implementation query: insufficient hands-on timestamps, using later section")
                later_cutoff = int(vid_dur * 0.4)
                later_timestamps = [ts for ts in timestamps if ts >= later_cutoff]
                if later_timestamps:
                    return sorted(list(set(later_timestamps)))[:4]
        
        context_lines = []
        for ch in relevant_chunks:
            stime = ch['metadata'].get('start_time', 0)
            context_lines.append(f"[{stime//60:02}:{stime%60:02}] {ch['text'][:100]}...")
        context_text = "\n".join(context_lines)

        navigation_prompt = f"""
        You are a video navigation expert. Analyze these transcript segments and return timestamps
        for the most relevant moments.

        Query: {query}

        Segments:
        {context_text}

        Return ONLY a JSON array of the most relevant timestamps (in seconds).
        Example: [120, 240, 360]
        """

        msg = flash_model.invoke(navigation_prompt)
        raw = msg.content if isinstance(msg.content, str) else str(msg.content)

        m = re.search(r'\[\s*(?:\d+\s*(?:,\s*\d+\s*)*)\]', raw, re.S)
        payload = m.group(0) if m else "[]"
        llm_timestamps = json.loads(payload)

        clean = []
        for t in llm_timestamps:
            try:
                val = int(t)
                if val >= 0 and (not clean or clean[-1] != val):
                    clean.append(val)
            except:
                continue
        
        result = clean if clean else timestamps[:3]
        print(f"[navigation] LLM-based navigation returned: {result}")
        return result

    except Exception as e:
        print(f"Error in video navigation: {e}")
        return [ch['metadata']['start_time'] for ch in relevant_chunks[:3]]
    
# ----------------------------------------------------------------------------------------------------------------------
#  Generate Initial Answer using Gemini-2.5-Pro
#-----------------------------------------------------------------------------------------------------------------------
    
pro_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.1,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

def generate_initial_answer(query, relevant_chunks):
    try:
        timestamps = [ch['metadata'].get('start_time', 0) for ch in relevant_chunks]
        avg_timestamp = sum(timestamps) / len(timestamps) if timestamps else 0
        vid_dur = max(timestamps) if timestamps else 3600
        
        context_parts = []
        for ch in relevant_chunks:
            spk = ch['metadata'].get('dominant_speaker', 'Speaker')
            stime = ch['metadata'].get('start_time', 0)
            context_parts.append(f"{spk} at {stime//60:02}:{stime%60:02} - {ch['text']}")
        context = "\n\n".join(context_parts)

        is_implementation_query = detect_query_type(query)
        
        context_hint = ""
        if is_implementation_query:
            phase_percent = (avg_timestamp / vid_dur) * 100
            if phase_percent > 25:
                context_hint = f"""
                IMPORTANT: This appears to be an implementation/hands-on query. The context is from the practical section of the video (around {avg_timestamp/60:.1f} minutes, {phase_percent:.0f}% through).
                
                Focus on:
                - Actual code being written or demonstrated on screen
                - Live coding sessions and practical examples  
                - Visual programming elements (IDE, terminal, editor)
                - Step-by-step implementation details
                - Screen content rather than just verbal explanation
                
                Set needs_ocr to TRUE for implementation queries as the actual code/content is likely shown visually.
                """


        answer_prompt = ChatPromptTemplate.from_template(f"""
        You are a video analysis expert. Answer the user's question based on the transcript context.
        
        {context_hint}

        Provide your response in this EXACT JSON format (no additional text):
        {{{{
            "answer": "Your detailed answer here",
            "needs_ocr": true,
            "confidence": 0.95,
            "reasoning": "Why you made these decisions"
        }}}}

        Set needs_ocr to true if visual elements (slides, charts, text, diagrams, code, equations) would help answer better.

        Context: {{context}}
        Question: {{question}}

        JSON Response:
        """)

        chain = answer_prompt | pro_model
        resp = chain.invoke({"context": context, "question": query})

        raw = resp.content.strip()
        raw = re.sub(r'``````', '', raw, flags=re.IGNORECASE)
        
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            raw = json_match.group(0)

        data = json.loads(raw)
        
        if 'answer' not in data:
            data['answer'] = "Unable to generate proper response format"
        if 'confidence' not in data:
            data['confidence'] = 0.5
        if 'needs_ocr' not in data:
            data['needs_ocr'] = False
        if 'reasoning' not in data:
            data['reasoning'] = "Standard analysis approach used"
            
        data['timestamps'] = timestamps
        return data

    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return {
            "answer": f"Error parsing model response: {str(e)}",
            "needs_ocr": False,
            "confidence": 0.0,
            "reasoning": "JSON format error",
            "timestamps": timestamps if 'timestamps' in locals() else []
        }
    except Exception as e:
        print(f"Error generating initial answer: {e}")
        return {
            "answer": f"Error processing question: {str(e)}",
            "needs_ocr": False,
            "confidence": 0.0,
            "reasoning": "System error occurred",
            "timestamps": timestamps if 'timestamps' in locals() else []
        }

# ----------------------------------------------------------------------------------------------------------------------
#  OCR code if OCR = True (For image/text/notes/slides/codes/etc)
#-----------------------------------------------------------------------------------------------------------------------

def process_with_ocr(query, frames, initial_answer):
    try:
        if not frames:
            return initial_answer

        uploads = []
        for fr in frames:
            if os.path.exists(fr['path']):
                uploads.append(genai.upload_file(fr['path']))
        if not uploads:
            return initial_answer
        time.sleep(1)

        ocr_prompt = f"""
        You are an expert video content analyzer with advanced OCR and visual reasoning capabilities.

        Original Question: {query}
        Initial Answer from Audio: {initial_answer['answer']}

        Analyze these {len(uploads)} video frames with deep visual reasoning and enhance the answer.

        Provide response in this EXACT JSON format:
        {{
            "enhanced_answer": "Comprehensive answer combining deep visual analysis with audio transcript",
            "visual_elements_found": ["list", "elements"],
            "confidence": 0.0-1.0,
            "ocr_text_extracted": "All readable text found across frames",
            "visual_insights": "Insights from cross-frame analysis",
            "frame_by_frame_analysis": "Key findings per frame"
        }}
        """

        model = genai.GenerativeModel('gemini-2.5-pro')
        resp = model.generate_content([ocr_prompt] + uploads)

        for up in uploads:
            try:
                genai.delete_file(up.name)
            except:
                pass
        for fr in frames:
            try:
                if os.path.exists(fr['path']):
                    os.remove(fr['path'])
            except:
                pass

        text = resp.text.strip()
        m = re.search(r'\{.*\}', text, re.S)
        if m:
            text = m.group(0)

        ocr_data = json.loads(text)

        enhanced = dict(initial_answer)
        enhanced['answer'] = ocr_data.get('enhanced_answer', initial_answer['answer'])
        enhanced['confidence'] = float(ocr_data.get('confidence', initial_answer.get('confidence', 0.6)))
        enhanced['visual_elements'] = ocr_data.get('visual_elements_found', [])
        enhanced['ocr_text'] = ocr_data.get('ocr_text_extracted', "")
        enhanced['visual_insights'] = ocr_data.get('visual_insights', "")
        enhanced['reasoning'] = f"Enhanced with visual analysis: {ocr_data.get('visual_insights', 'Visual elements processed')}"
        return enhanced

    except json.JSONDecodeError as e:
        print(f"OCR JSON parsing error: {e}")
        return initial_answer
    except Exception as e:
        print(f"Error in OCR processing: {e}")
        return initial_answer
    
# ----------------------------------------------------------------------------------------------------------------------
#  Complete Q&A Pipeline with 3 times Recursion Logic
#-----------------------------------------------------------------------------------------------------------------------

def video_qa_pipeline(query, vectorstore, video_id, attempt=1, max_attempts=3):
    try:
        print(f"Q&A Pipeline - Attempt {attempt}")
        top_k = min(3 + attempt, 8)
        relevant = semantic_search(vectorstore, query, top_k=top_k)
        if not relevant:
            return {
                "answer": "No relevant information found in the video for this question.",
                "needs_ocr": False,
                "confidence": 0.0,
                "reasoning": "No matching content found in transcript",
                "timestamps": [],
                "visual_elements": [],
                "attempt": attempt
            }

        print(f"Found {len(relevant)} relevant chunks")
        timestamps = navigate_video_segments(query, relevant)
        print(f"Navigation identified {len(timestamps)} key timestamps: {timestamps}")

        initial_result = generate_initial_answer(query, relevant)
        print(f"Initial confidence: {initial_result['confidence']:.2f}, OCR needed: {initial_result['needs_ocr']}")

        if initial_result['needs_ocr']:
            print("Processing visual elements...")
            frames = extract_video_frames_optimized(video_id, timestamps[:4])
            if frames:
                final_result = process_with_ocr(query, frames, initial_result)
                print(f"Enhanced with {len(frames)} frames")
            else:
                print("No frames extracted, using transcript-only answer")
                final_result = initial_result
        else:
            print("Using transcript-only answer")
            final_result = initial_result

        confidence_threshold = max(0.3, 0.6 - (0.15 * (attempt - 1)))
        print(f"Final confidence: {final_result['confidence']:.2f} (threshold: {confidence_threshold:.2f})")

        if final_result['confidence'] < confidence_threshold and attempt < max_attempts:
            print(f"Low confidence, retrying with expanded search...")
            return video_qa_pipeline(query, vectorstore, video_id, attempt + 1, max_attempts)

        final_result['attempt'] = attempt
        return final_result

    except Exception as e:
        print(f"Error in Q&A pipeline: {e}")
        return {
            "answer": f"Error processing your question: {str(e)}",
            "needs_ocr": False,
            "confidence": 0.0,
            "reasoning": "System error occurred during processing",
            "timestamps": [],
            "visual_elements": [],
            "attempt": attempt
        }

def process_video_and_qa(youtube_url, target_lang="en"):
    try:
        print("Starting complete video processing pipeline...")
        
        video_id = extract_video_id(youtube_url)
        if not video_id:
            return None, "Invalid YouTube URL provided"
        
        print(f"Video ID extracted: {video_id}")
        
        db_path = f"./chroma_db_{video_id}"
        if os.path.exists(db_path):
            print("Found existing database, loading...")
            vectorstore = load_existing_vectorstore(video_id)
            if vectorstore:
                print("Database loaded successfully")
                return vectorstore, "Database loaded successfully"
            else:
                print("Failed to load existing database, recreating...")
                shutil.rmtree(db_path, ignore_errors=True)
        
        print("Extracting transcript...")
        transcript_chunks = get_transcript(video_id)
        if not transcript_chunks:
            return None, "Failed to extract transcript from video"
        
        print(f"Extracted {len(transcript_chunks)} transcript chunks")
        
        print("Translating content to English...")
        translated_chunks = translate_transcript_batch(transcript_chunks)
        if not translated_chunks:
            return None, "Failed to translate transcript"
        
        print(f"Translated {len(translated_chunks)} chunks")
        
        print("Creating vector store...")
        vectorstore = create_vector_store(translated_chunks, video_id)
        if not vectorstore:
            return None, "Failed to create vector store"
        
        print("Video processing complete!")
        return vectorstore, "Video processed successfully"
        
    except Exception as e:
        print(f"Error in video processing pipeline: {e}")
        return None, f"Error processing video: {str(e)}"

def answer_question(vectorstore, video_id, question):
    try:
        result = video_qa_pipeline(question, vectorstore, video_id)
        return result
    except Exception as e:
        return {
            "answer": f"Error answering question: {str(e)}",
            "confidence": 0.0,
            "reasoning": "System error",
            "timestamps": [],
            "attempt": 1
        }

# ------------------------------------ XXXXX END XXXXX --------------------------------------------------------------------------------

#### TEST CODE - To run standalone test ####

# if __name__ == "__main__":
#     print("esting Python Code Extraction from Programming with Mosh...")
    
#     # Test with Mosh's Python tutorial
#     print("Processing video...")
#     vectorstore, result = process_video_and_qa("https://www.youtube.com/watch?v=kqtD5dpn9C8")
    
#     if vectorstore and "successfully" in str(result):
#         video_id = extract_video_id("https://www.youtube.com/watch?v=kqtD5dpn9C8")
#         print(f"Video {video_id} ready for testing!")
        
#         # Test 1: Weight converter program extraction
#         print("\n" + "="*80)
#         print("Test 1: Weight Converter Program")
#         print("="*80)
        
#         result1 = answer_question(
#             vectorstore, 
#             video_id, 
#             "Extract and explain the complete Python code for the weight converter program shown in the video. Include all the variable assignments, input statements, if/else logic, and print statements exactly as written."
#         )
        
#         print(f"Answer: {result1['answer']}")
#         print(f"Confidence: {result1['confidence']:.2f}")
#         print(f"Visual Analysis: {'Yes' if result1.get('visual_elements') else 'No'}")
#         print(f"Key Timestamps: {result1['timestamps']}")
#         if result1.get('ocr_text'):
#             print(f"OCR Text: {result1['ocr_text'][:200]}...")
        
#         # Test 2: Input and type conversion code
#         print("\n" + "="*80) 
#         print("Test 2: User Input and Type Conversion")
#         print("="*80)
        
#         result2 = answer_question(
#             vectorstore,
#             video_id, 
#             "Show me the exact Python code for handling user input and type conversion that's displayed on screen. Include the birth year calculator example with proper error handling."
#         )
        
#         print(f"Answer: {result2['answer']}")
#         print(f"Confidence: {result2['confidence']:.2f}")
#         print(f"Visual Analysis: {'Yes' if result2.get('visual_elements') else 'No'}")
        
#         # Test 3: Loop examples
#         print("\n" + "="*80)
#         print("Test 3: Loop Code Examples") 
#         print("="*80)
        
#         result3 = answer_question(
#             vectorstore,
#             video_id,
#             "Extract the for loop and while loop code examples shown in the video, including the exact syntax, indentation, and the triangle pattern example with asterisks."
#         )
        
#         print(f"Answer: {result3['answer']}")
#         print(f"Confidence: {result3['confidence']:.2f}")
#         print(f"Visual Analysis: {'Yes' if result3.get('visual_elements') else 'No'}")
        
#         # Test 4: List methods demonstration
#         print("\n" + "="*80)
#         print("Test 4: List Methods Code")
#         print("="*80)
        
#         result4 = answer_question(
#             vectorstore,
#             video_id,
#             "Show me the Python list methods code examples from the video - append, insert, remove, clear methods with the exact syntax and variable names used."
#         )
        
#         print(f"Answer: {result4['answer']}")
#         print(f"Confidence: {result4['confidence']:.2f}")
#         print(f"Visual Analysis: {'Yes' if result4.get('visual_elements') else 'No'}")
        
#         # Test 5: String methods examples  
#         print("\n" + "="*80)
#         print("Test 5: String Methods Implementation")
#         print("="*80)
        
#         result5 = answer_question(
#             vectorstore,
#             video_id,
#             "Extract the string methods code examples - upper(), lower(), find(), replace() methods with the 'Python for Beginners' course variable example exactly as shown."
#         )
        
#         print(f"Answer: {result5['answer']}")
#         print(f"Confidence: {result5['confidence']:.2f}")
#         print(f"Visual Analysis: {'Yes' if result5.get('visual_elements') else 'No'}")
        
#         # Summary
#         print("TESTING SUMMARY")
#         print("="*80)
        
#         results = [result1, result2, result3, result4, result5]
#         avg_confidence = sum(r['confidence'] for r in results) / len(results)
#         visual_tests = sum(1 for r in results if r.get('visual_elements'))
        
#         print(f"Average Confidence: {avg_confidence:.2f}")
#         print(f"Tests Using Visual Analysis: {visual_tests}/{len(results)}")
#         print(f"All Tests Completed Successfully!")
        
#         # Cleanup
#         print("\nCleaning up temporary files...")
#         cleanup_leftover_videos(video_id)
        
#     else:
#         print("Failed to process video")
#         if result:
#             print(f"Error: {result}")

# if __name__ == "__main__":
#     print("DEEP PIPELINE DIAGNOSTIC")
    
#     video_id = "klkOdh4l0Eo"
#     query = "What is Latent Space Computing?"
    
#     # Load database
#     vectorstore = load_existing_vectorstore(video_id)
#     print(f"Database loaded: {vectorstore is not None}")
    
#     if vectorstore:
#         print("\nSTEP 1: Raw Database Search")
#         raw_docs = vectorstore.similarity_search(query, k=10)
#         print(f"Raw search results: {len(raw_docs)} documents")
        
#         if raw_docs:
#             for i, doc in enumerate(raw_docs[:2]):
#                 print(f"Doc {i+1}: {doc.page_content[:100]}...")
#                 print(f"Metadata: {doc.metadata}")
#         else:
#             print("PROBLEM: Database returns ZERO documents!")
            
#             # Check if database has ANY content
#             print("\nChecking if database is completely empty...")
#             try:
#                 all_docs = vectorstore.similarity_search("the", k=50)  # Search for common word
#                 print(f"Total documents in database: {len(all_docs)}")
#                 if len(all_docs) == 0:
#                     print("DATABASE IS COMPLETELY EMPTY!")
#                 else:
#                     print("Database has content but search is failing")
#                     # Show some sample docs
#                     for i, doc in enumerate(all_docs[:3]):
#                         print(f"Sample {i+1}: {doc.page_content[:80]}...")
#             except Exception as e:
#                 print(f"Database access error: {e}")

    
        
#         print("\nSTEP 2: Filtered Search")
#         filtered_results = semantic_search_with_phase_filtering(vectorstore, query, top_k=8)
#         print(f"Filtered search results: {len(filtered_results)} chunks")
        
#         if not filtered_results:
#             print("PROBLEM: Filtered search returns EMPTY!")
            
        
#         for i, result in enumerate(filtered_results[:2]):
#             print(f"Chunk {i+1}: {result['text'][:100]}...")
            
#         print("\nSTEP 3: Answer Generation")
#         try:
#             answer = generate_initial_answer(query, filtered_results)
#             print(f"Generated answer: {answer}")
            
#             if answer.get('confidence', 0) == 0.0:
#                 print("PROBLEM: Answer generation returns 0.0 confidence!")
#                 print(f"Answer: {answer.get('answer', 'No answer')}")
#                 print(f"Reasoning: {answer.get('reasoning', 'No reasoning')}")
#         except Exception as e:
#             print(f"Answer generation failed: {e}")


# if __name__ == "__main__":
#     print("Testing")
    
#     # Test with Mosh's Python tutorial
#     print("Processing video...")
#     vectorstore, result = process_video_and_qa("https://www.youtube.com/watch?v=klkOdh4l0Eo")
    
#     if vectorstore and "successfully" in str(result):
#         video_id = extract_video_id("https://www.youtube.com/watch?v=klkOdh4l0Eo")
#         print(f"Video {video_id} ready for testing!")
        
        
#         print("\n" + "="*80)
#         print("Test 1")
#         print("="*80)
        
#         result1 = answer_question(
#             vectorstore, 
#             video_id, 
#             "Explain the working of HRM's. Also write its full-form and explain the reason."
#         )
        
#         print(f"Answer: {result1['answer']}")
#         print(f"Confidence: {result1['confidence']:.2f}")
#         print(f"Visual Analysis: {'Yes' if result1.get('visual_elements') else 'No'}")
#         print(f"Key Timestamps: {result1['timestamps']}")
#         if result1.get('ocr_text'):
#             print(f"OCR Text: {result1['ocr_text'][:200]}...")
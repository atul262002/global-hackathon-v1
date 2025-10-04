#!/usr/bin/env python3
"""
voice_driven_video_extractor_multi_request.py

Enhanced pipeline that handles multiple distinct requests in voiceover
and combines them sequentially in the output video.

Key Features:
1. Parses multiple requests from voiceover (e.g., "show A, then B, then C")
2. Processes each request independently with semantic matching
3. Selects best clip for each request based on confidence
4. Combines clips in the order requested

"""

import os
import sys
import json
import warnings
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import tempfile
import re

warnings.filterwarnings('ignore')

import torch
import whisper
import cv2
import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip, concatenate_videoclips
from transformers import (
    Blip2Processor, Blip2ForConditionalGeneration,
    AutoTokenizer, AutoModelForCausalLM,
)
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer, util

# =====================================================================
# DATA STRUCTURES
# =====================================================================

@dataclass
class SearchIntent:
    """Structured search intent from LLM"""
    original_command: str
    interpretation: str
    search_queries: List[str]
    avoid_queries: List[str]
    clip_duration: float
    confidence: float
    reasoning: str

@dataclass
class RequestSegment:
    """Individual request from multi-part voiceover"""
    index: int
    text: str
    intent: Optional[SearchIntent] = None

@dataclass
class VideoMoment:
    """A matched moment in the video"""
    frame_number: int
    timestamp: float
    score: float
    matched_query: str
    description: str
    frame_path: str
    semantic_similarity: float = 0.0
    request_index: int = 0  # Which request this belongs to

# =====================================================================
# SEMANTIC MATCHER
# =====================================================================

class SemanticMatcher:
    """Uses sentence transformers for semantic similarity"""
    
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Semantic] Loading sentence transformer...")
        
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.model.to(self.device)
            self.enabled = True
            print(f"[Semantic] Ready!\n")
        except Exception as e:
            print(f"[Semantic] Failed: {e}")
            print(f"[Semantic] Using fallback mode\n")
            self.enabled = False
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts"""
        if not self.enabled:
            return self._fallback_similarity(text1, text2)
        
        try:
            embeddings = self.model.encode([text1, text2], convert_to_tensor=True)
            similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
            return max(0.0, similarity)
        except Exception as e:
            return self._fallback_similarity(text1, text2)
    
    def _fallback_similarity(self, text1: str, text2: str) -> float:
        """Simple keyword-based similarity as fallback"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)

# =====================================================================
# LLM QUERY GENERATOR (ENHANCED FOR MULTI-REQUEST)
# =====================================================================

class LLMQueryGenerator:
    """Converts natural language to structured search queries"""
    
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        print(f"[LLM] Loading {model_name} on {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            ).to(self.device)
            
            self.model.eval()
            self.enabled = True
            print(f"[LLM] Ready!\n")
            
        except Exception as e:
            print(f"[LLM] Failed: {e}")
            print(f"[LLM] Using fallback mode\n")
            self.model = None
            self.enabled = False
    
    def parse_multiple_requests(self, command: str) -> List[RequestSegment]:
        """Parse command into multiple distinct requests"""
        print(f"[LLM] Parsing multi-request command...")
        
        # Split by common separators
        separators = [',', ' and ', ' then ', ';', ' also ', ' plus ']
        
        segments = [command]
        for sep in separators:
            new_segments = []
            for seg in segments:
                parts = seg.split(sep)
                new_segments.extend([p.strip() for p in parts if p.strip()])
            segments = new_segments
        
        # Clean up segments
        request_segments = []
        for i, text in enumerate(segments):
            # Remove common prefixes
            text = re.sub(r'^(show|extract|find|get|give me|show me)\s+', '', text, flags=re.IGNORECASE)
            text = text.strip()
            
            if text and len(text) > 5:  # Minimum length
                request_segments.append(RequestSegment(index=i, text=text))
        
        if not request_segments:
            request_segments = [RequestSegment(index=0, text=command)]
        
        print(f"[LLM] Found {len(request_segments)} distinct requests:")
        for seg in request_segments:
            print(f"  {seg.index + 1}. {seg.text}")
        print()
        
        return request_segments
    
    def generate_search_queries(self, command: str) -> SearchIntent:
        """Generate structured search intent from natural language"""
        if not self.enabled:
            return self._fallback_queries(command)
        
        print(f"[LLM] Analyzing: '{command}'")
        
        prompt = self._build_prompt(command)
        response = self._generate(prompt)
        intent = self._parse_response(command, response)
        
        print(f"[LLM] Generated {len(intent.search_queries)} queries\n")
        
        return intent
    
    def _build_prompt(self, command: str) -> str:
        """Build LLM prompt for query generation"""
        prompt = f"""You are a video search assistant. Analyze this specific request:

"{command}"

Generate visual descriptions to find matching video moments for THIS SPECIFIC request.

Respond with JSON:
{{
  "interpretation": "what user wants in one sentence",
  "search_queries": [
    "specific visual description 1",
    "specific visual description 2",
    "specific visual description 3",
    "specific visual description 4"
  ],
  "avoid": ["what to avoid"],
  "clip_duration": 8.0,
  "confidence": 0.9,
  "reasoning": "brief explanation"
}}

Guidelines:
- Focus ONLY on the visible elements mentioned in THIS request
- Be specific about what's happening in the scene
- 3-5 diverse descriptions
- clip_duration: 6-12 seconds

Request: "{command}"

JSON:
{{"""
        return prompt
    
    def _generate(self, prompt: str) -> str:
        """Generate LLM response"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if prompt in response:
                response = response.split(prompt)[-1]
            return response.strip()
        except Exception as e:
            print(f"[LLM] Generation error: {e}")
            return "{}"
    
    def _parse_response(self, command: str, response: str) -> SearchIntent:
        """Parse JSON response"""
        try:
            response = response.strip()
            if '```' in response:
                parts = response.split('```')
                for part in parts:
                    if '{' in part:
                        response = part.replace('json', '').strip()
                        break
            
            start = response.find('{')
            end = response.rfind('}')
            if start == -1 or end == -1:
                raise ValueError("No JSON found")
            
            json_str = response[start:end+1]
            data = json.loads(json_str)
            
            return SearchIntent(
                original_command=command,
                interpretation=data.get("interpretation", command),
                search_queries=data.get("search_queries", [command]),
                avoid_queries=data.get("avoid", []),
                clip_duration=float(data.get("clip_duration", 8.0)),
                confidence=float(data.get("confidence", 0.7)),
                reasoning=data.get("reasoning", "Generated from command")
            )
        except Exception as e:
            print(f"[LLM] Parse error: {e}, using fallback")
            return self._fallback_queries(command)
    
    def _fallback_queries(self, command: str) -> SearchIntent:
        """Fallback when LLM unavailable"""
        words = command.lower().split()
        key_terms = [w for w in words if len(w) > 3 and w not in ['when', 'where', 'what', 'show', 'find', 'clip', 'part']]
        
        queries = [
            command,
            " ".join(key_terms),
            f"scene showing {' '.join(key_terms)}",
            f"moment with {' '.join(key_terms)}"
        ]
        
        return SearchIntent(
            original_command=command,
            interpretation=f"Find: {command}",
            search_queries=queries,
            avoid_queries=[],
            clip_duration=8.0,
            confidence=0.5,
            reasoning="Fallback mode"
        )

# =====================================================================
# VIDEO MATCHER
# =====================================================================

class VideoFrameMatcher:
    """Multi-model video-text matching with semantic filtering"""
    
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"[Matcher] Loading BLIP-2...")
        self._load_blip2()
        
        print(f"[Matcher] Loading semantic matcher...")
        self.semantic_matcher = SemanticMatcher(device=self.device)
        
        print(f"[Matcher] Ready!\n")
    
    def _load_blip2(self):
        """Load BLIP-2 for vision-language tasks"""
        model_name = "Salesforce/blip2-opt-2.7b"
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.model.eval()
    
    def match_frames(
        self,
        frames: List[Tuple[int, float, np.ndarray]],
        intent: SearchIntent,
        request_index: int,
        min_score: float = 0.30,
        min_semantic_similarity: float = 0.35
    ) -> List[VideoMoment]:
        """Match video frames to search queries with semantic filtering"""
        print(f"[Matcher] Searching {len(frames)} frames for request {request_index + 1}...")
        
        matches = []
        intent_text = intent.original_command
        
        for i, (frame_num, timestamp, frame_rgb) in enumerate(frames):
            if i % 50 == 0:
                print(f"  Processing frame {i+1}/{len(frames)}...", end='\r')
            
            score, matched_query, description = self._score_frame(
                frame_rgb, intent.search_queries, intent.avoid_queries
            )
            
            semantic_sim = self.semantic_matcher.compute_similarity(
                description, intent_text
            )
            
            if score >= min_score and semantic_sim >= min_semantic_similarity:
                match = VideoMoment(
                    frame_number=frame_num,
                    timestamp=timestamp,
                    score=score,
                    matched_query=matched_query,
                    description=description,
                    frame_path="",
                    semantic_similarity=semantic_sim,
                    request_index=request_index
                )
                matches.append(match)
        
        print(f"\n[Matcher] Found {len(matches)} matches for request {request_index + 1}\n")
        return matches
    
    def _score_frame(
        self,
        frame_rgb: np.ndarray,
        search_queries: List[str],
        avoid_queries: List[str]
    ) -> Tuple[float, str, str]:
        """Score frame with BLIP-2"""
        if not search_queries:
            return 0.0, "", ""
        
        image = Image.fromarray(frame_rgb.astype("uint8"))
        
        try:
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_length=80)
            caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            best_score = 0.0
            best_query = search_queries[0]
            
            for query in search_queries:
                query_words = set(query.lower().split())
                caption_words = set(caption.lower().split())
                overlap = len(query_words & caption_words)
                score = overlap / max(len(query_words), 1)
                
                prompt = f"Question: Does this image show {query}? Answer:"
                inputs = self.processor(image, text=prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    generated_ids = self.model.generate(**inputs, max_length=80)
                answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                if any(word in answer.lower() for word in ["yes", "shows", "visible", "contains"]):
                    score += 0.4
                
                if score > best_score:
                    best_score = score
                    best_query = query
            
            if avoid_queries:
                for avoid_q in avoid_queries:
                    if any(word in caption.lower() for word in avoid_q.lower().split()):
                        best_score *= 0.5
            
            best_score = min(1.0, best_score)
            
            return best_score, best_query, caption
            
        except Exception as e:
            return 0.0, search_queries[0] if search_queries else "", ""

# =====================================================================
# MAIN PIPELINE (MULTI-REQUEST)
# =====================================================================

class VoiceDrivenVideoExtractor:
    """Main pipeline with multi-request support"""
    
    def __init__(
        self,
        llm_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device: Optional[str] = None,
        sample_fps: float = 1.0,
        min_score: float = 0.3,
        min_semantic_similarity: float = 0.35,
        clips_per_request: int = 1,
        tmp_dir: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_fps = sample_fps
        self.min_score = min_score
        self.min_semantic_similarity = min_semantic_similarity
        self.clips_per_request = clips_per_request
        self.tmp_dir = tmp_dir or tempfile.mkdtemp(prefix="video_extract_")
        
        print("="*70)
        print("MULTI-REQUEST SEMANTIC VIDEO EXTRACTOR")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Min Score: {self.min_score}")
        print(f"Min Semantic Similarity: {self.min_semantic_similarity}")
        print(f"Clips per Request: {self.clips_per_request}")
        print(f"Temp Dir: {self.tmp_dir}\n")
        
        print("[1/3] Loading Whisper...")
        self.whisper = whisper.load_model("base", device=self.device)
        
        print("[2/3] Loading LLM...")
        self.llm = LLMQueryGenerator(model_name=llm_model, device=self.device)
        
        print("[3/3] Loading Vision Models...")
        self.matcher = VideoFrameMatcher(device=self.device)
        
        print("="*70)
        print("READY\n")

    def _transcribe(self, audio_path: str) -> str:
        """Transcribe audio"""
        result = self.whisper.transcribe(audio_path)
        return result.get("text", "").strip()
    
    def extract_video_clips(
        self,
        video_path: str,
        voiceover_path: str,
        output_path: str = "extracted_video.mp4",
        output_dir: str = "extracted_frames"
    ) -> Dict[str, Any]:
        """Main extraction pipeline for multi-request"""
        print("\n" + "="*70)
        print("EXTRACTION START")
        print("="*70)
        
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Transcribe
        print("\n[STEP 1/6] Transcribing voiceover...")
        # transcription = self._transcribe(voiceover_path)
        # transcription = "Extract frames cooking vegetables in a pan, showing vegetable cutting, stirring vegetables on the stove, and making tomato mixture in a grinder"
        transcription="extarct the video which shows player playing hit the ball with bat  and ball goes out of ground  "
        print(f"Command: '{transcription}'")
        
        if not transcription.strip():
            raise ValueError("Empty transcription")
        
        # Step 2: Parse multiple requests
        print("\n[STEP 2/6] Parsing multiple requests...")
        request_segments = self.llm.parse_multiple_requests(transcription)
        
        # Step 3: Generate queries for each request
        print("\n[STEP 3/6] Generating search queries for each request...")
        for segment in request_segments:
            segment.intent = self.llm.generate_search_queries(segment.text)
            print(f"  Request {segment.index + 1}: {segment.intent.interpretation}")
        
        # Step 4: Extract frames
        print("\n[STEP 4/6] Extracting frames...")
        frames = self._extract_frames(video_path)
        print(f"Extracted {len(frames)} frames")
        
        # Step 5: Match frames for each request
        print("\n[STEP 5/6] Matching frames for each request...")
        all_matches = []
        for segment in request_segments:
            matches = self.matcher.match_frames(
                frames, segment.intent, segment.index,
                self.min_score, self.min_semantic_similarity
            )
            all_matches.extend(matches)
        
        if not all_matches:
            print("\n[RESULT] No matches found for any request")
            return self._empty_result(transcription, request_segments, output_dir_path)
        
        # Step 6: Select best clips for each request
        print("\n[STEP 6/6] Selecting best clips for each request...")
        selected_moments = self._select_best_per_request(all_matches, len(request_segments))
        
        print(f"\n✓ Selected {len(selected_moments)} clips total")
        for i in range(len(request_segments)):
            count = sum(1 for m in selected_moments if m.request_index == i)
            print(f"  Request {i + 1}: {count} clip(s)")
        print()
        
        # Save frames
        saved_moments = self._save_frames(selected_moments, frames, output_dir_path)
        
        # Create video montage in request order
        video_output = self._create_video_montage(
            video_path, saved_moments, output_path, request_segments
        )
        
        # Generate result
        result = {
            "transcription": transcription,
            "requests": [
                {
                    "index": seg.index,
                    "text": seg.text,
                    "intent": asdict(seg.intent)
                } for seg in request_segments
            ],
            "moments": [asdict(m) for m in saved_moments],
            "num_requests": len(request_segments),
            "num_clips": len(saved_moments),
            "output_video": video_output,
            "output_dir": str(output_dir_path)
        }
        
        report_path = output_dir_path / "extraction_report.json"
        with open(report_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print("\n" + "="*70)
        print("EXTRACTION COMPLETE")
        print("="*70)
        print(f"Total requests: {len(request_segments)}")
        print(f"Total clips: {len(saved_moments)}")
        print(f"Output video: {video_output}")
        print(f"Report: {report_path}")
        print("="*70 + "\n")
        
        return result
    
    def _extract_frames(self, video_path: str) -> List[Tuple[int, float, np.ndarray]]:
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(fps / self.sample_fps))
        
        frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / fps
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append((frame_idx, timestamp, rgb))
            
            frame_idx += 1
        
        cap.release()
        return frames
    
    def _select_best_per_request(
        self,
        matches: List[VideoMoment],
        num_requests: int
    ) -> List[VideoMoment]:
        """Select best clips for each request separately"""
        selected = []
        
        for req_idx in range(num_requests):
            # Get matches for this request
            req_matches = [m for m in matches if m.request_index == req_idx]
            
            if not req_matches:
                continue
            
            # Compute combined score
            for m in req_matches:
                m.score = 0.7 * m.score + 0.3 * m.semantic_similarity
            
            # Sort by score
            req_matches.sort(key=lambda m: m.score, reverse=True)
            
            # Cluster to avoid duplicates
            times = np.array([m.timestamp for m in req_matches]).reshape(-1, 1)
            clustering = DBSCAN(eps=30.0, min_samples=1).fit(times)
            
            clusters = {}
            for moment, label in zip(req_matches, clustering.labels_):
                clusters.setdefault(label, []).append(moment)
            
            # Select best from each cluster
            representatives = []
            for cluster in clusters.values():
                best = max(cluster, key=lambda m: m.score)
                representatives.append(best)
            
            # Take top N
            representatives.sort(key=lambda m: m.score, reverse=True)
            selected.extend(representatives[:self.clips_per_request])
        
        # Sort by request index, then by timestamp
        selected.sort(key=lambda m: (m.request_index, m.timestamp))
        
        return selected
    
    def _save_frames(
        self,
        moments: List[VideoMoment],
        frames: List[Tuple[int, float, np.ndarray]],
        output_dir: Path
    ) -> List[VideoMoment]:
        """Save matched frames"""
        frame_dict = {num: rgb for num, _, rgb in frames}
        
        saved = []
        for i, moment in enumerate(moments):
            if moment.frame_number in frame_dict:
                filename = f"req{moment.request_index + 1}_clip{i+1:02d}_t{moment.timestamp:.1f}s_score{moment.score:.2f}.jpg"
                filepath = output_dir / filename
                
                rgb = frame_dict[moment.frame_number]
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(filepath), bgr)
                
                moment.frame_path = str(filepath)
                saved.append(moment)
                print(f"  ✓ {filename}")
                print(f"    Description: {moment.description}")
        
        return saved
    
    def _create_video_montage(
        self,
        video_path: str,
        moments: List[VideoMoment],
        output_path: str,
        request_segments: List[RequestSegment]
    ) -> str:
        """Create video montage combining all requests"""
        print(f"\n  Creating multi-request video montage...")
        
        clips = []
        for i, moment in enumerate(moments):
            req_idx = moment.request_index
            clip_duration = request_segments[req_idx].intent.clip_duration
            
            print(f"  Extracting clip {i+1}/{len(moments)} (Request {req_idx + 1}) at {moment.timestamp:.1f}s")
            
            clip_path = os.path.join(self.tmp_dir, f"clip_{i:03d}.mp4")
            self._extract_clip(video_path, moment.timestamp, clip_duration, clip_path)
            clips.append(clip_path)
        
        print(f"  Combining {len(clips)} clips in request order...")
        video_clips = [VideoFileClip(path) for path in clips]
        final = concatenate_videoclips(video_clips, method="compose")
        final.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=30,
            verbose=False,
            logger=None
        )
        
        final.close()
        for clip in video_clips:
            clip.close()
        
        for clip_path in clips:
            try:
                os.remove(clip_path)
            except:
                pass
        
        return output_path
    
    def _extract_clip(
        self,
        video_path: str,
        center_time: float,
        duration: float,
        output_path: str
    ):
        """Extract single clip"""
        video = VideoFileClip(video_path)
        
        start = max(0.0, center_time - duration / 2.0)
        end = min(video.duration, center_time + duration / 2.0)
        
        if end - start < duration:
            if start <= 1e-6:
                end = min(duration, video.duration)
            else:
                start = max(0.0, end - duration)
        
        clip = video.subclip(start, end)
        clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=30,
            verbose=False,
            logger=None
        )
        
        clip.close()
        video.close()
    
    def _empty_result(self, transcription, request_segments, output_dir):
        """Return empty result"""
        return {
            "transcription": transcription,
            "requests": [
                {
                    "index": seg.index,
                    "text": seg.text,
                    "intent": asdict(seg.intent) if seg.intent else None
                } for seg in request_segments
            ],
            "moments": [],
            "num_requests": len(request_segments),
            "num_clips": 0,
            "output_video": None,
            "output_dir": str(output_dir)
        }

# =====================================================================
# CLI
# =====================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Multi-Request Video Extractor - Handle multiple requests in one command",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract clips for multiple requests
  python %(prog)s --video input.mp4 --voice "show A, then B, then C" --out output.mp4
  
  # Get 2 clips per request
  python %(prog)s -v video.mp4 -a voice.wav --clips-per-request 2
  
  # Adjust thresholds
  python %(prog)s -v video.mp4 -a voice.wav --min-score 0.25 --min-similarity 0.3

Key Features:
  - Parses multiple distinct requests from voiceover
  - Processes each request independently
  - Combines clips in the order requested
  - Semantic matching for each request
        """
    )
    
    parser.add_argument("--video", "-v", required=True, help="Input video file")
    parser.add_argument("--voice", "-a", required=True, help="Voiceover audio file")
    parser.add_argument("--out", "-o", default="extracted_video.mp4", help="Output video")
    parser.add_argument("--frames-dir", default="extracted_frames", help="Frames directory")
    parser.add_argument("--llm", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="LLM model")
    parser.add_argument("--fps", type=float, default=1.0, help="Sampling FPS")
    parser.add_argument("--min-score", type=float, default=0.25, help="Min matching score")
    parser.add_argument("--min-similarity", type=float, default=0.30, help="Min semantic similarity")
    parser.add_argument("--clips-per-request", type=int, default=1, help="Clips to extract per request")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Device")
    parser.add_argument("--tmp", help="Temp directory")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)
    
    if not os.path.exists(args.voice):
        print(f"Error: Audio not found: {args.voice}")
        sys.exit(1)
    
    print(f"\nVideo: {args.video}")
    print(f"Voice: {args.voice}")
    print(f"Output: {args.out}\n")
    
    extractor = VoiceDrivenVideoExtractor(
        llm_model=args.llm,
        device=args.device,
        sample_fps=args.fps,
        min_score=args.min_score,
        min_semantic_similarity=args.min_similarity,
        clips_per_request=args.clips_per_request,
        tmp_dir=args.tmp
    )
    
    try:
        result = extractor.extract_video_clips(
            video_path=args.video,
            voiceover_path=args.voice,
            output_path=args.out,
            output_dir=args.frames_dir
        )

        if result["output_video"]:
            print("\n✓ SUCCESS!")
            print(f"\nCommand: \"{result['transcription']}\"")
            print(f"\nRequests processed:")
            for req in result['requests']:
                print(f"  {req['index'] + 1}. {req['text']}")
                print(f"     Interpretation: {req['intent']['interpretation']}")
            print(f"\nTotal clips: {result['num_clips']}")
            print(f"\nOutput video: {result['output_video']}")
            print(f"Frames directory: {result['output_dir']}")
        else:
            print("\n✗ No matching moments found")
            print("\nTry:")
            print("  - Lower thresholds: --min-score 0.2 --min-similarity 0.25")
            print("  - More specific commands")
            print("  - Higher sampling: --fps 2")
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
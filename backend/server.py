#!/usr/bin/env python3
"""
flask_video_api.py

Flask API wrapper for the multi-request video extraction pipeline.
Handles video and voiceover uploads, processes them, and returns results.

Endpoints:
- POST /api/extract - Upload video and voiceover for extraction
- GET /api/status/<task_id> - Check processing status
- GET /api/download/<filename> - Download extracted video
- GET /api/health - Health check
"""

import os
import sys
import uuid
import json
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from werkzeug.utils import secure_filename

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS



from main import VoiceDrivenVideoExtractor


# =====================================================================
# CONFIGURATION
# =====================================================================

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
FRAMES_FOLDER = 'frames'
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a', 'flac'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['FRAMES_FOLDER'] = FRAMES_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create necessary directories
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, FRAMES_FOLDER]:
    Path(folder).mkdir(parents=True, exist_ok=True)

# Task storage (in production, use Redis or a database)
tasks: Dict[str, Dict[str, Any]] = {}
task_lock = threading.Lock()

# Global extractor instance (initialized on first use)
extractor: Optional[VoiceDrivenVideoExtractor] = None
extractor_lock = threading.Lock()

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def allowed_file(filename: str, allowed_extensions: set) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def get_extractor() -> VoiceDrivenVideoExtractor:
    """Get or initialize the video extractor (singleton pattern)"""
    global extractor
    
    if extractor is None:
        with extractor_lock:
            if extractor is None:  # Double-check locking
                print("Initializing VoiceDrivenVideoExtractor...")
                extractor = VoiceDrivenVideoExtractor(
                    llm_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    device=None,  # Auto-detect
                    sample_fps=1.0,
                    min_score=0.25,
                    min_semantic_similarity=0.30,
                    clips_per_request=1,
                    tmp_dir=None
                )
                print("VoiceDrivenVideoExtractor initialized!")
    
    return extractor

def create_task(task_id: str) -> Dict[str, Any]:
    """Create a new task entry"""
    task = {
        'id': task_id,
        'status': 'pending',  # pending, processing, completed, failed
        'created_at': datetime.now().isoformat(),
        'progress': 0,
        'message': 'Task created',
        'result': None,
        'error': None
    }
    
    with task_lock:
        tasks[task_id] = task
    
    return task

def update_task(task_id: str, **kwargs):
    """Update task status"""
    with task_lock:
        if task_id in tasks:
            tasks[task_id].update(kwargs)
            tasks[task_id]['updated_at'] = datetime.now().isoformat()

def process_extraction(task_id: str, video_path: str, voiceover_path: str, 
                      output_path: str, frames_dir: str, params: Dict[str, Any]):
    """Background task for video extraction"""
    try:
        update_task(task_id, status='processing', progress=10, 
                   message='Initializing extraction...')
        
        # Get extractor
        ext = get_extractor()
        
        update_task(task_id, progress=20, message='Extracting video clips...')
        
        # Run extraction
        result = ext.extract_video_clips(
            video_path=video_path,
            voiceover_path=voiceover_path,
            output_path=output_path,
            output_dir=frames_dir
        )
        
        update_task(task_id, progress=90, message='Finalizing...')
        
        # Prepare result
        response_result = {
            'transcription': result['transcription'],
            'num_requests': result['num_requests'],
            'num_clips': result['num_clips'],
            'requests': result['requests'],
            'moments': result['moments'],
            'output_video': os.path.basename(result['output_video']) if result['output_video'] else None,
            'output_dir': os.path.basename(result['output_dir']),
            'download_url': f"/api/download/{os.path.basename(output_path)}" if result['output_video'] else None
        }
        
        update_task(
            task_id,
            status='completed',
            progress=100,
            message='Extraction completed successfully',
            result=response_result
        )
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error in task {task_id}: {error_msg}")
        import traceback
        traceback.print_exc()
        
        update_task(
            task_id,
            status='failed',
            progress=0,
            message='Extraction failed',
            error=error_msg
        )

# =====================================================================
# API ENDPOINTS
# =====================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'video-extraction-api',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/extract', methods=['POST'])
def extract_video():
    """
    Main extraction endpoint
    
    Accepts:
    - multipart/form-data with 'video' and 'voiceover' files
    - OR JSON with 'video_path' and 'voiceover_path' for existing files
    
    Optional parameters:
    - min_score (float): Minimum matching score (default: 0.25)
    - min_similarity (float): Minimum semantic similarity (default: 0.30)
    - clips_per_request (int): Number of clips per request (default: 1)
    - sample_fps (float): Sampling FPS (default: 1.0)
    """
    try:
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Parse parameters
        params = {
            'min_score': float(request.form.get('min_score', 0.25)),
            'min_similarity': float(request.form.get('min_similarity', 0.30)),
            'clips_per_request': int(request.form.get('clips_per_request', 1)),
            'sample_fps': float(request.form.get('sample_fps', 1.0))
        }
        
        # Handle file uploads or paths
        if 'video' in request.files and 'voiceover' in request.files:
            # File upload mode
            video_file = request.files['video']
            voiceover_file = request.files['voiceover']
            
            # Validate files
            if video_file.filename == '' or voiceover_file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not allowed_file(video_file.filename, ALLOWED_VIDEO_EXTENSIONS):
                return jsonify({'error': f'Invalid video format. Allowed: {ALLOWED_VIDEO_EXTENSIONS}'}), 400
            
            if not allowed_file(voiceover_file.filename, ALLOWED_AUDIO_EXTENSIONS):
                return jsonify({'error': f'Invalid audio format. Allowed: {ALLOWED_AUDIO_EXTENSIONS}'}), 400
            
            # Save uploaded files
            video_filename = secure_filename(f"{task_id}_{video_file.filename}")
            voiceover_filename = secure_filename(f"{task_id}_{voiceover_file.filename}")
            
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
            voiceover_path = os.path.join(app.config['UPLOAD_FOLDER'], voiceover_filename)
            
            video_file.save(video_path)
            voiceover_file.save(voiceover_path)
            
        elif request.is_json:
            # Path mode (for existing files)
            data = request.get_json()
            video_path = data.get('video_path')
            voiceover_path = data.get('voiceover_path')
            
            if not video_path or not voiceover_path:
                return jsonify({'error': 'video_path and voiceover_path required'}), 400
            
            if not os.path.exists(video_path):
                return jsonify({'error': f'Video file not found: {video_path}'}), 404
            
            if not os.path.exists(voiceover_path):
                return jsonify({'error': f'Voiceover file not found: {voiceover_path}'}), 404
        
        else:
            return jsonify({'error': 'Invalid request. Provide files or JSON with paths'}), 400
        
        # Prepare output paths
        output_filename = f"{task_id}_extracted.mp4"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        frames_dir = os.path.join(app.config['FRAMES_FOLDER'], task_id)
        
        # Create task
        task = create_task(task_id)
        
        # Start background processing
        thread = threading.Thread(
            target=process_extraction,
            args=(task_id, video_path, voiceover_path, output_path, frames_dir, params),
            daemon=True
        )
        thread.start()
        
        # Return task info
        return jsonify({
            'task_id': task_id,
            'status': 'pending',
            'message': 'Extraction started',
            'status_url': f"/api/status/{task_id}"
        }), 202
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status/<task_id>', methods=['GET'])
def get_status(task_id: str):
    """Get task status"""
    with task_lock:
        task = tasks.get(task_id)
    
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    
    return jsonify(task)

@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename: str):
    """Download extracted video"""
    try:
        # Security: only allow downloading from output folder
        safe_filename = secure_filename(filename)
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], safe_filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='video/mp4'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/frames/<task_id>/<filename>', methods=['GET'])
def download_frame(task_id: str, filename: str):
    """Download individual frame"""
    try:
        safe_task_id = secure_filename(task_id)
        safe_filename = secure_filename(filename)
        frames_dir = os.path.join(app.config['FRAMES_FOLDER'], safe_task_id)
        
        return send_from_directory(frames_dir, safe_filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/api/tasks', methods=['GET'])
def list_tasks():
    """List all tasks (for debugging)"""
    with task_lock:
        task_list = list(tasks.values())
    
    return jsonify({
        'tasks': task_list,
        'count': len(task_list)
    })

@app.route('/api/cleanup/<task_id>', methods=['DELETE'])
def cleanup_task(task_id: str):
    """Clean up task files"""
    try:
        # Remove from tasks dict
        with task_lock:
            if task_id in tasks:
                del tasks[task_id]
        
        # Remove files
        safe_task_id = secure_filename(task_id)
        
        # Remove uploaded files
        for f in os.listdir(app.config['UPLOAD_FOLDER']):
            if f.startswith(task_id):
                try:
                    os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f))
                except:
                    pass
        
        # Remove output files
        for f in os.listdir(app.config['OUTPUT_FOLDER']):
            if f.startswith(task_id):
                try:
                    os.remove(os.path.join(app.config['OUTPUT_FOLDER'], f))
                except:
                    pass
        
        # Remove frames directory
        frames_dir = os.path.join(app.config['FRAMES_FOLDER'], safe_task_id)
        if os.path.exists(frames_dir):
            import shutil
            shutil.rmtree(frames_dir, ignore_errors=True)
        
        return jsonify({'message': 'Task cleaned up successfully'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =====================================================================
# ERROR HANDLERS
# =====================================================================

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        'error': 'File too large',
        'max_size': f'{MAX_FILE_SIZE / (1024*1024):.0f}MB'
    }), 413

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# =====================================================================
# MAIN
# =====================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Flask API for Video Extraction')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print("="*70)
    print("VIDEO EXTRACTION API")
    print("="*70)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Frames folder: {FRAMES_FOLDER}")
    print(f"Max file size: {MAX_FILE_SIZE / (1024*1024):.0f}MB")
    print("="*70)
    print("\nEndpoints:")
    print("  POST   /api/extract          - Upload and extract")
    print("  GET    /api/status/<task_id> - Check status")
    print("  GET    /api/download/<file>  - Download result")
    print("  GET    /api/health           - Health check")
    print("="*70)
    print(f"\nStarting server on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop\n")
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )

#!/usr/bin/env python3
"""
Production-Ready Flask API for Video Text Detection with Subtitle Placement
Supports both file uploads and video URL processing with subtitle positioning.

Author: Videograph AI 
Version: 2.1.0
License: MIT
"""

import os
import cv2
import numpy as np
import json
import logging
import time
import requests
import tempfile
import shutil
import uuid
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict
from urllib.parse import urlparse
import mimetypes

# Flask imports
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

# OCR Engine Imports
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Advanced image processing
from skimage import restoration, filters, morphology
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    # Flask config
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size for Render
    
    # Upload config
    UPLOAD_FOLDER = '/tmp/uploads'
    TEMP_FOLDER = '/tmp/temp'
    RESULTS_FOLDER = '/tmp/results'
    
    # Video processing config
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
    MAX_VIDEO_DURATION = int(os.environ.get('MAX_VIDEO_DURATION', '300'))  # 5 minutes for free tier
    
    # Download config
    DOWNLOAD_TIMEOUT = int(os.environ.get('DOWNLOAD_TIMEOUT', '180'))  # 3 minutes
    MAX_DOWNLOAD_SIZE = 100 * 1024 * 1024  # 100MB
    
    # Subtitle config
    SUBTITLE_MARGIN = 20  # pixels from edges
    SUBTITLE_HEIGHT = 60  # estimated subtitle height
    SUBTITLE_WIDTH_RATIO = 0.8  # 80% of video width

@dataclass
class BoundingBox:
    """Represents a bounding box with coordinates"""
    left: int
    top: int
    right: int
    bottom: int
    
    @property
    def width(self) -> int:
        return self.right - self.left
    
    @property
    def height(self) -> int:
        return self.bottom - self.top
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.left + self.width // 2, self.top + self.height // 2)
    
    def intersection_over_union(self, other: 'BoundingBox') -> float:
        """Calculate IoU with another bounding box"""
        x1 = max(self.left, other.left)
        y1 = max(self.top, other.top)
        x2 = min(self.right, other.right)
        y2 = min(self.bottom, other.bottom)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def expand(self, padding: int) -> 'BoundingBox':
        """Expand bounding box by padding"""
        return BoundingBox(
            left=max(0, self.left - padding),
            top=max(0, self.top - padding),
            right=self.right + padding,
            bottom=self.bottom + padding
        )
    
    def overlaps_with(self, other: 'BoundingBox', threshold: float = 0.1) -> bool:
        """Check if this box overlaps significantly with another"""
        return self.intersection_over_union(other) > threshold

@dataclass
class TextDetection:
    """Represents a text detection with temporal information"""
    start_time: float
    end_time: float
    box: BoundingBox
    confidence: float = 0.0
    text_content: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'start_time': self._format_timestamp(self.start_time),
            'end_time': self._format_timestamp(self.end_time),
            'box': {
                'top': self.box.top,
                'left': self.box.left,
                'right': self.box.right,
                'bottom': self.box.bottom
            },
            'confidence': round(self.confidence, 3),
            'text_content': self.text_content
        }
    
    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Format timestamp as HH:MM:SS.mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

@dataclass
class SubtitlePlacement:
    """Represents a suggested subtitle placement"""
    start_time: float
    end_time: float
    position: str  # 'middle_center' or 'top_center'
    box: BoundingBox
    priority: int  # 1 = highest priority
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'start_time': TextDetection._format_timestamp(self.start_time),
            'end_time': TextDetection._format_timestamp(self.end_time),
            'position': self.position,
            'box': {
                'top': self.box.top,
                'left': self.box.left,
                'right': self.box.right,
                'bottom': self.box.bottom
            },
            'priority': self.priority
        }

class AdvancedPreprocessor:
    """Advanced image preprocessing pipeline for optimal OCR performance"""
    
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu and TORCH_AVAILABLE
        
    def enhance_frame(self, frame: np.ndarray) -> List[np.ndarray]:
        """Apply multiple enhancement techniques and return variants"""
        variants = []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variants.append(gray)
        
        # CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        variants.append(enhanced)
        
        # Denoising
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        variants.append(denoised)
        
        # Edge preserving filter
        edge_preserved = cv2.edgePreservingFilter(
            cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 
            flags=1, sigma_s=50, sigma_r=0.4
        )
        edge_gray = cv2.cvtColor(edge_preserved, cv2.COLOR_BGR2GRAY)
        variants.append(edge_gray)
        
        # Morphological enhancement
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph_enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        variants.append(morph_enhanced)
        
        return variants

class OCREngine:
    """Abstract base class for OCR engines"""
    
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
    
    def detect_text(self, image: np.ndarray) -> List[Tuple[BoundingBox, float, str]]:
        """Detect text in image and return bounding boxes with confidence"""
        raise NotImplementedError

class EasyOCREngine(OCREngine):
    """EasyOCR implementation with GPU support"""
    
    def __init__(self, confidence_threshold: float = 0.7, use_gpu: bool = False):
        super().__init__(confidence_threshold)
        if not EASYOCR_AVAILABLE:
            raise ImportError("EasyOCR not available. Install with: pip install easyocr")
        
        self.reader = easyocr.Reader(['en'], gpu=use_gpu and torch.cuda.is_available())
        logger.info(f"EasyOCR initialized with GPU: {use_gpu and torch.cuda.is_available()}")
    
    def detect_text(self, image: np.ndarray) -> List[Tuple[BoundingBox, float, str]]:
        """Detect text using EasyOCR"""
        try:
            results = self.reader.readtext(image)
            detections = []
            
            for (bbox_coords, text, confidence) in results:
                if confidence < self.confidence_threshold:
                    continue
                
                coords = np.array(bbox_coords).astype(int)
                left = int(np.min(coords[:, 0]))
                top = int(np.min(coords[:, 1]))
                right = int(np.max(coords[:, 0]))
                bottom = int(np.max(coords[:, 1]))
                
                bbox = BoundingBox(left, top, right, bottom)
                detections.append((bbox, confidence, text.strip()))
            
            return detections
        except Exception as e:
            logger.error(f"EasyOCR detection failed: {e}")
            return []

class VideoTitleExtractor:
    """Extracts video title from URL or file metadata"""
    
    @staticmethod
    def extract_from_url(url: str) -> str:
        """Extract title from URL"""
        try:
            # Try to get title from URL structure
            parsed = urlparse(url)
            path = parsed.path
            
            # Extract filename without extension
            filename = os.path.basename(path)
            if filename:
                name, _ = os.path.splitext(filename)
                # Clean up the name
                title = name.replace('_', ' ').replace('-', ' ')
                # Capitalize words
                title = ' '.join(word.capitalize() for word in title.split())
                return title if title else "Video"
            
            return "Video"
        except Exception:
            return "Video"
    
    @staticmethod
    def extract_from_file(filepath: str) -> str:
        """Extract title from file path"""
        try:
            filename = os.path.basename(filepath)
            name, _ = os.path.splitext(filename)
            # Clean up the name
            title = name.replace('_', ' ').replace('-', ' ')
            # Remove UUIDs and timestamps
            title = re.sub(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', '', title)
            title = re.sub(r'\d{4}-\d{2}-\d{2}', '', title)
            title = ' '.join(word.capitalize() for word in title.split() if word)
            return title if title else "Video"
        except Exception:
            return "Video"

class SubtitlePlacer:
    """Handles subtitle placement logic"""
    
    def __init__(self, video_width: int, video_height: int):
        self.video_width = video_width
        self.video_height = video_height
        self.margin = Config.SUBTITLE_MARGIN
        self.subtitle_height = Config.SUBTITLE_HEIGHT
        self.subtitle_width = int(video_width * Config.SUBTITLE_WIDTH_RATIO)
    
    def get_middle_center_box(self) -> BoundingBox:
        """Get bounding box for middle center position"""
        left = (self.video_width - self.subtitle_width) // 2
        right = left + self.subtitle_width
        top = (self.video_height - self.subtitle_height) // 2
        bottom = top + self.subtitle_height
        
        return BoundingBox(left, top, right, bottom)
    
    def get_top_center_box(self) -> BoundingBox:
        """Get bounding box for top center position"""
        left = (self.video_width - self.subtitle_width) // 2
        right = left + self.subtitle_width
        top = self.margin
        bottom = top + self.subtitle_height
        
        return BoundingBox(left, top, right, bottom)
    
    def suggest_placement(self, detections: List[TextDetection], 
                         start_time: float, end_time: float) -> SubtitlePlacement:
        """Suggest best subtitle placement for given time range"""
        middle_box = self.get_middle_center_box()
        top_box = self.get_top_center_box()
        
        # Check if middle center conflicts with any detections in this time range
        conflicting_detections = [
            d for d in detections 
            if (d.start_time <= end_time and d.end_time >= start_time and
                d.box.overlaps_with(middle_box))
        ]
        
        if not conflicting_detections:
            # Middle center is free
            return SubtitlePlacement(
                start_time=start_time,
                end_time=end_time,
                position='middle_center',
                box=middle_box,
                priority=1
            )
        else:
            # Fall back to top center
            return SubtitlePlacement(
                start_time=start_time,
                end_time=end_time,
                position='top_center',
                box=top_box,
                priority=2
            )

class VideoDownloader:
    """Handles video downloading from URLs with safety checks"""
    
    def __init__(self, timeout: int = 180, max_size: int = 100 * 1024 * 1024):
        self.timeout = timeout
        self.max_size = max_size
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'VideoTextDetector/2.1 (Video Processing Service)'
        })
    
    def is_valid_video_url(self, url: str) -> bool:
        """Check if URL appears to be a valid video URL"""
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False
            
            # Check file extension
            path = parsed.path.lower()
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v']
            return any(path.endswith(ext) for ext in video_extensions)
        except Exception:
            return False
    
    def get_video_info(self, url: str) -> Dict:
        """Get video information without downloading"""
        try:
            response = self.session.head(url, timeout=30)
            response.raise_for_status()
            
            content_length = response.headers.get('content-length')
            content_type = response.headers.get('content-type', '')
            
            return {
                'content_length': int(content_length) if content_length else None,
                'content_type': content_type,
                'url': url
            }
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            raise Exception(f"Cannot access video URL: {str(e)}")
    
    def download_video(self, url: str, output_path: str) -> str:
        """Download video from URL with progress tracking"""
        logger.info(f"Starting download from: {url}")
        
        # Validate URL
        if not self.is_valid_video_url(url):
            raise ValueError("Invalid video URL format")
        
        # Get video info
        video_info = self.get_video_info(url)
        
        # Check file size
        if video_info['content_length'] and video_info['content_length'] > self.max_size:
            raise ValueError(f"Video file too large: {video_info['content_length']} bytes")
        
        try:
            # Download with streaming
            response = self.session.get(url, stream=True, timeout=self.timeout)
            response.raise_for_status()
            
            downloaded_size = 0
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        # Check size limit
                        if downloaded_size > self.max_size:
                            raise ValueError("Download size exceeded limit")
            
            logger.info(f"Download completed: {downloaded_size} bytes")
            return output_path
            
        except Exception as e:
            # Clean up partial download
            if os.path.exists(output_path):
                os.remove(output_path)
            logger.error(f"Download failed: {e}")
            raise Exception(f"Download failed: {str(e)}")

class ProductionVideoTextDetector:
    """Production-grade video text detection system"""
    
    def __init__(self, 
                 ocr_engine: str = 'easyocr',
                 confidence_threshold: float = 0.7,
                 focus_region_ratio: float = 0.3,
                 use_gpu: bool = False,
                 frame_skip: int = 10):  # Increased for faster processing
        
        self.focus_region_ratio = focus_region_ratio
        self.frame_skip = frame_skip
        self.preprocessor = AdvancedPreprocessor(use_gpu)
        
        # Initialize OCR engine
        if ocr_engine == 'easyocr' and EASYOCR_AVAILABLE:
            self.ocr_engine = EasyOCREngine(confidence_threshold, use_gpu)
        else:
            raise ValueError(f"OCR engine '{ocr_engine}' not available")
        
        logger.info(f"ProductionVideoTextDetector initialized with {ocr_engine} engine")
    
    def _extract_focus_region(self, frame: np.ndarray) -> np.ndarray:
        """Extract bottom region for text detection"""
        height = frame.shape[0]
        focus_start = int(height * (1 - self.focus_region_ratio))
        return frame[focus_start:, :]
    
    def _process_frame(self, frame: np.ndarray, timestamp: float) -> List[Tuple[BoundingBox, float, str]]:
        """Process a single frame and return detections"""
        focus_region = self._extract_focus_region(frame)
        enhanced_variants = self.preprocessor.enhance_frame(focus_region)
        
        all_detections = []
        # Process only first 2 variants for speed
        for variant in enhanced_variants[:2]:
            detections = self.ocr_engine.detect_text(variant)
            
            height_offset = int(frame.shape[0] * (1 - self.focus_region_ratio))
            adjusted_detections = []
            
            for bbox, confidence, text in detections:
                adjusted_bbox = BoundingBox(
                    left=bbox.left,
                    top=bbox.top + height_offset,
                    right=bbox.right,
                    bottom=bbox.bottom + height_offset
                )
                adjusted_detections.append((adjusted_bbox, confidence, text))
            
            all_detections.extend(adjusted_detections)
        
        return self._non_max_suppression(all_detections)
    
    def _non_max_suppression(self, detections: List[Tuple[BoundingBox, float, str]], 
                            iou_threshold: float = 0.5) -> List[Tuple[BoundingBox, float, str]]:
        """Apply non-maximum suppression"""
        if not detections:
            return []
        
        detections.sort(key=lambda x: x[1], reverse=True)
        
        filtered = []
        for current in detections:
            bbox, confidence, text = current
            
            should_keep = True
            for filtered_bbox, _, _ in filtered:
                if bbox.intersection_over_union(filtered_bbox) > iou_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                filtered.append(current)
        
        return filtered
    
    def process_video(self, video_path: str) -> Dict:
        """Process video and return comprehensive text detection results"""
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        logger.info(f"Processing video: {video_path}")
        start_time = time.time()
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Video properties: {width}x{height}, {fps:.2f} FPS, {duration:.2f}s")
        
        # Check duration limit
        if duration > Config.MAX_VIDEO_DURATION:
            cap.release()
            raise ValueError(f"Video too long: {duration:.2f}s (max: {Config.MAX_VIDEO_DURATION}s)")
        
        all_detections = []
        frame_count = 0
        processed_frames = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % self.frame_skip != 0:
                    frame_count += 1
                    continue
                
                timestamp = frame_count / fps if fps > 0 else 0
                detections = self._process_frame(frame, timestamp)
                
                for bbox, confidence, text in detections:
                    detection = TextDetection(
                        start_time=timestamp,
                        end_time=timestamp + (self.frame_skip / fps if fps > 0 else 1),
                        box=bbox,
                        confidence=confidence,
                        text_content=text
                    )
                    all_detections.append(detection)
                
                processed_frames += 1
                frame_count += 1
                
                if processed_frames % 10 == 0:
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    logger.info(f"Progress: {progress:.1f}%")
        
        finally:
            cap.release()
        
        # Calculate merged region
        merged_region = None
        if all_detections:
            min_left = min(d.box.left for d in all_detections)
            min_top = min(d.box.top for d in all_detections)
            max_right = max(d.box.right for d in all_detections)
            max_bottom = max(d.box.bottom for d in all_detections)
            merged_region = BoundingBox(min_left, min_top, max_right, max_bottom)
        
        results = {
            "video": str(video_path.name),
            "video_properties": {
                "width": width,
                "height": height,
                "fps": fps,
                "duration": duration,
                "focus_region": f"bottom {int(self.focus_region_ratio * 100)}%"
            },
            "processing_stats": {
                "total_frames": total_frames,
                "processed_frames": processed_frames,
                "frame_skip": self.frame_skip,
                "processing_time": time.time() - start_time,
                "detections_found": len(all_detections)
            },
            "detections": [detection.to_dict() for detection in all_detections],
            "merged_blocked_region": {
                'top': merged_region.top,
                'left': merged_region.left,
                'right': merged_region.right,
                'bottom': merged_region.bottom
            } if merged_region else None
        }
        
        processing_time = time.time() - start_time
        logger.info(f"Processing completed in {processing_time:.2f}s")
        logger.info(f"Found {len(all_detections)} text regions")
        
        return results
    
    def process_video_with_subtitles(self, video_path: str, video_url: str = None) -> Dict:
        """Process video with subtitle placement suggestions"""
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        logger.info(f"Processing video with subtitles: {video_path}")
        start_time = time.time()
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Video properties: {width}x{height}, {fps:.2f} FPS, {duration:.2f}s")
        
        # Check duration limit
        if duration > Config.MAX_VIDEO_DURATION:
            cap.release()
            raise ValueError(f"Video too long: {duration:.2f}s (max: {Config.MAX_VIDEO_DURATION}s)")
        
        # Extract title
        if video_url:
            title = VideoTitleExtractor.extract_from_url(video_url)
        else:
            title = VideoTitleExtractor.extract_from_file(str(video_path))
        
        all_detections = []
        frame_count = 0
        processed_frames = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % self.frame_skip != 0:
                    frame_count += 1
                    continue
                
                timestamp = frame_count / fps if fps > 0 else 0
                detections = self._process_frame(frame, timestamp)
                
                for bbox, confidence, text in detections:
                    detection = TextDetection(
                        start_time=timestamp,
                        end_time=timestamp + (self.frame_skip / fps if fps > 0 else 1),
                        box=bbox,
                        confidence=confidence,
                        text_content=text
                    )
                    all_detections.append(detection)
                
                processed_frames += 1
                frame_count += 1
                
                if processed_frames % 10 == 0:
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    logger.info(f"Progress: {progress:.1f}%")
        
        finally:
            cap.release()
        
        # Generate subtitle placements
        subtitle_placer = SubtitlePlacer(width, height)
        subtitle_placements = []
        
        # Create subtitle placement suggestions every 5 seconds
        time_interval = 5.0  # seconds
        current_time = 0.0
        
        while current_time < duration:
            end_time = min(current_time + time_interval, duration)
            placement = subtitle_placer.suggest_placement(all_detections, current_time, end_time)
            subtitle_placements.append(placement)
            current_time = end_time
        
        results = {
            "title": title,
            "video_dimensions": {
                "width": width,
                "height": height
            },
            "video_properties": {
                "fps": fps,
                "duration": duration,
                "total_frames": total_frames
            },
            "processing_stats": {
                "processed_frames": processed_frames,
                "frame_skip": self.frame_skip,
                "processing_time": time.time() - start_time,
                "detections_found": len(all_detections)
            },
            "text_detections": [detection.to_dict() for detection in all_detections],
            "subtitle_placements": [placement.to_dict() for placement in subtitle_placements]
        }
        
        processing_time = time.time() - start_time
        logger.info(f"Processing completed in {processing_time:.2f}s")
        logger.info(f"Found {len(all_detections)} text regions and {len(subtitle_placements)} subtitle placements")
        
        return results

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Ensure directories exist
for folder in [Config.UPLOAD_FOLDER, Config.TEMP_FOLDER, Config.RESULTS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Initialize components
downloader = VideoDownloader(Config.DOWNLOAD_TIMEOUT, Config.MAX_DOWNLOAD_SIZE)

# Initialize detector only when needed to avoid startup issues
detector = None

def get_detector():
    global detector
    if detector is None:
        detector = ProductionVideoTextDetector(frame_skip=15)  # Faster processing
    return detector

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API information"""
    return jsonify({
        'name': 'Video Text Detection API with Subtitle Placement',
        'version': '2.1.0',
        'status': 'running',
        'endpoints': {
            '/': 'API information',
            '/health': 'Health check',
            '/process-url': 'Process video from URL (POST)',
            '/process-file': 'Process uploaded video (POST)',
            '/process-subtitles': 'Process video with subtitle placement (POST)',
            '/api-info': 'Detailed API information'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'version': '2.1.0',
        'easyocr_available': EASYOCR_AVAILABLE,
        'torch_available': TORCH_AVAILABLE
    })

@app.route('/process-subtitles', methods=['POST'])
def process_video_with_subtitles():
    """Process video from URL with subtitle placement suggestions"""
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({
                'error': 'Video URL is required',
                'message': 'Please provide a video URL in the request body'
            }), 400
        
        video_url = data['url'].strip()
        
        # Validate URL format
        if not video_url.startswith(('http://', 'https://')):
            return jsonify({
                'error': 'Invalid URL format',
                'message': 'URL must start with http:// or https://'
            }), 400
        
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Validate video URL
        if not downloader.is_valid_video_url(video_url):
            return jsonify({
                'error': 'Invalid video URL',
                'message': 'URL does not appear to be a valid video file. Supported formats: mp4, avi, mov, mkv, webm'
            }), 400
        
        # Download video
        temp_video_path = os.path.join(Config.TEMP_FOLDER, f"{task_id}.mp4")
        
        try:
            logger.info(f"Starting video download for task {task_id}")
            downloader.download_video(video_url, temp_video_path)
            logger.info(f"Video download completed for task {task_id}")
        except ValueError as e:
            return jsonify({
                'error': 'Video validation failed',
                'message': str(e)
            }), 400
        except Exception as e:
            return jsonify({
                'error': 'Download failed',
                'message': str(e)
            }), 400
        
        # Process video with subtitle placement
        try:
            logger.info(f"Starting video processing for task {task_id}")
            detector_instance = get_detector()
            results = detector_instance.process_video_with_subtitles(temp_video_path, video_url)
            
            # Clean up temp file
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            
            logger.info(f"Video processing completed for task {task_id}")
            
            # Add task metadata
            response_data = {
                'task_id': task_id,
                'status': 'completed',
                'timestamp': time.time(),
                'source_url': video_url,
                **results
            }
            
            return jsonify(response_data)
            
        except ValueError as e:
            # Clean up on validation error
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            return jsonify({
                'error': 'Video processing validation failed',
                'message': str(e)
            }), 400
        except Exception as e:
            # Clean up on processing error
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            logger.error(f"Processing failed for task {task_id}: {e}")
            return jsonify({
                'error': 'Video processing failed',
                'message': str(e)
            }), 500
            
    except Exception as e:
        logger.error(f"Process subtitles error: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'An unexpected error occurred during processing'
        }), 500

@app.route('/process-url', methods=['POST'])
def process_video_url():
    """Process video from URL"""
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({'error': 'Video URL is required'}), 400
        
        video_url = data['url']
        
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Validate URL
        if not downloader.is_valid_video_url(video_url):
            return jsonify({'error': 'Invalid video URL format'}), 400
        
        # Download video
        temp_video_path = os.path.join(Config.TEMP_FOLDER, f"{task_id}.mp4")
        
        try:
            downloader.download_video(video_url, temp_video_path)
        except Exception as e:
            return jsonify({'error': f'Download failed: {str(e)}'}), 400
        
        # Process video
        try:
            detector_instance = get_detector()
            results = detector_instance.process_video(temp_video_path)
            
            # Clean up temp file
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            
            return jsonify({
                'task_id': task_id,
                'status': 'completed',
                'results': results
            })
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Process URL error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/process-file', methods=['POST'])
def process_video_file():
    """Process uploaded video file"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Supported: mp4, avi, mov, mkv, webm'}), 400
        
        # Save uploaded file
        task_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        filepath = os.path.join(Config.UPLOAD_FOLDER, f"{task_id}_{filename}")
        file.save(filepath)
        
        try:
            # Process video
            detector_instance = get_detector()
            results = detector_instance.process_video(filepath)
            
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
            
            return jsonify({
                'task_id': task_id,
                'status': 'completed',
                'results': results
            })
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
            
    except RequestEntityTooLarge:
        return jsonify({'error': 'File too large (max 100MB)'}), 413
    except Exception as e:
        logger.error(f"Process file error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api-info', methods=['GET'])
def api_info():
    """Get detailed API information and usage"""
    return jsonify({
        'name': 'Video Text Detection API with Subtitle Placement',
        'version': '2.1.0',
        'description': 'API for detecting text in videos and suggesting subtitle placements using OCR technology',
        'endpoints': {
            '/': 'API information',
            '/health': 'Health check',
            '/process-url': {
                'method': 'POST',
                'description': 'Process video from URL (legacy endpoint)',
                'body': {'url': 'string - Video URL'},
                'response': 'JSON with detection results'
            },
            '/process-file': {
                'method': 'POST',
                'description': 'Process uploaded video file (legacy endpoint)',
                'body': 'Form data with video file',
                'response': 'JSON with detection results'
            },
            '/process-subtitles': {
                'method': 'POST',
                'description': 'Process video from URL with subtitle placement suggestions',
                'body': {'url': 'string - Video URL'},
                'response': 'JSON with text detections and subtitle placement suggestions'
            },
            '/api-info': 'Detailed API information'
        },
        'supported_formats': list(Config.ALLOWED_EXTENSIONS),
        'limits': {
            'max_file_size': f"{Config.MAX_CONTENT_LENGTH // (1024*1024)}MB",
            'max_duration': f"{Config.MAX_VIDEO_DURATION}s",
            'max_download_size': f"{Config.MAX_DOWNLOAD_SIZE // (1024*1024)}MB"
        },
        'features': [
            'Text detection in video frames',
            'OCR with confidence scores',
            'Bounding box coordinates (top, left, right, bottom format)',
            'Timestamp information',
            'Focus region processing (bottom 30%)',
            'Video title extraction',
            'Subtitle placement suggestions',
            'Priority-based subtitle positioning',
            'Professional subtitle workflow integration'
        ],
        'subtitle_placement': {
            'positions': ['middle_center', 'top_center'],
            'priority': 'middle_center preferred, top_center as fallback',
            'detection_avoidance': 'Automatically avoids detected text regions',
            'time_segments': 'Suggestions provided in 5-second intervals'
        }
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist. Check /api-info for available endpoints.'
    }), 404

@app.errorhandler(413)
def payload_too_large(error):
    return jsonify({
        'error': 'File too large',
        'message': f'Maximum file size is {Config.MAX_CONTENT_LENGTH // (1024*1024)}MB'
    }), 413

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred. Please try again later.'
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
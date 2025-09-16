# Progress tracking system for audio processing
import json
import os
import time
from django.core.cache import cache
from django.conf import settings

class ProcessingProgress:
    """Track processing progress for audio denoising tasks"""
    
    def __init__(self, task_id):
        self.task_id = task_id
        self.cache_key = f"processing_progress_{task_id}"
        
    def update_progress(self, stage, percentage, message, details=None):
        """Update processing progress"""
        progress_data = {
            'task_id': self.task_id,
            'stage': stage,
            'percentage': percentage,
            'message': message,
            'details': details or {},
            'timestamp': time.time(),
            'status': 'processing' if percentage < 100 else 'completed'
        }
        
        # Store in cache for 1 hour
        cache.set(self.cache_key, progress_data, 3600)
        
        # Also store in file for persistence
        progress_file = os.path.join(settings.MEDIA_ROOT, 'progress', f'{self.task_id}.json')
        os.makedirs(os.path.dirname(progress_file), exist_ok=True)
        
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
            
        return progress_data
    
    def get_progress(self):
        """Get current progress"""
        # Try cache first
        progress = cache.get(self.cache_key)
        if progress:
            return progress
            
        # Fall back to file
        progress_file = os.path.join(settings.MEDIA_ROOT, 'progress', f'{self.task_id}.json')
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    return json.load(f)
            except:
                pass
                
        return None
    
    def set_error(self, error_message, details=None):
        """Set error status"""
        return self.update_progress(
            stage='error',
            percentage=0,
            message=f"Error: {error_message}",
            details={'error': error_message, 'error_details': details}
        )
    
    def complete(self, message, output_data=None):
        """Mark processing as complete"""
        # If output_data contains folder_name, use it directly
        if output_data and isinstance(output_data, dict) and 'folder_name' in output_data:
            details = output_data
        else:
            # Legacy support - wrap in output_files
            details = {'output_files': output_data or []}
            
        return self.update_progress(
            stage='completed',
            percentage=100,
            message=message,
            details=details
        )

# Processing stages with estimated percentages
PROCESSING_STAGES = {
    'upload': {'start': 0, 'end': 10, 'message': 'Uploading file...'},
    'validation': {'start': 10, 'end': 15, 'message': 'Validating audio file...'},
    'loading_model': {'start': 15, 'end': 25, 'message': 'Loading AI model...'},
    'preprocessing': {'start': 25, 'end': 30, 'message': 'Preprocessing audio...'},
    'denoising': {'start': 30, 'end': 80, 'message': 'Applying AI denoising...'},
    'postprocessing': {'start': 80, 'end': 90, 'message': 'Post-processing audio...'},
    'generating_visualizations': {'start': 90, 'end': 95, 'message': 'Generating visualizations...'},
    'finalizing': {'start': 95, 'end': 100, 'message': 'Finalizing results...'}
}

def get_stage_progress(stage, sub_percentage=0):
    """Get overall progress percentage for a stage"""
    if stage not in PROCESSING_STAGES:
        return 0
        
    stage_info = PROCESSING_STAGES[stage]
    stage_range = stage_info['end'] - stage_info['start']
    return stage_info['start'] + (stage_range * sub_percentage / 100)

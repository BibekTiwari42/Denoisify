import os
import numpy as np
import wave
import csv
import torch
from model_training.model import WaveUNet
from denoiser.mmse_stsa import mmse_stsa
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent tkinter errors
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
from typing import List, Dict, Tuple, Optional

# Try to import PESQ and STOI
try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    print("Warning: PESQ not available. Install with: pip install pesq")
    PESQ_AVAILABLE = False

try:
    from pystoi.stoi import stoi
    STOI_AVAILABLE = True
except ImportError:
    print("Warning: STOI not available. Install with: pip install pystoi")
    STOI_AVAILABLE = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = os.path.join("model_training", "checkpoints", "unet_best.pth")

class ModelEvaluator:
    """Comprehensive evaluation class for Wave-U-Net + MMSE-STSA system"""
    
    def __init__(self, checkpoint_path: str = CHECKPOINT_PATH):
        self.device = DEVICE
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.results = []
        
    def load_model(self) -> bool:
        """Load the Wave-U-Net model from checkpoint"""
        try:
            if not os.path.exists(self.checkpoint_path):
                print(f"Error: Checkpoint not found at {self.checkpoint_path}")
                return False
                
            self.model = WaveUNet(in_ch=1, out_ch=1, depth=5, base_ch=24).to(self.device)
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
                
            self.model.eval()
            print(f"Model loaded successfully from {self.checkpoint_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


def read_wav(filename):
    """Read WAV file and return samples and sample rate"""
    try:
        with wave.open(filename, 'rb') as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            frames = wf.readframes(n_frames)
            
            dtype = np.int16 if sampwidth == 2 else np.uint8
            samples = np.frombuffer(frames, dtype=dtype)
            
            # Convert to mono if stereo
            if n_channels > 1:
                samples = samples[::n_channels]
                
            samples = samples.astype(np.float32)
            
            # Normalize to [-1, 1]
            if sampwidth == 2:
                samples = samples / 32768.0
            elif sampwidth == 1:
                samples = samples / 128.0 - 1.0
            elif sampwidth == 3:
                samples = samples / 8388608.0
            elif sampwidth == 4:
                samples = samples / 2147483648.0
                
            return samples, framerate
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None, None

def compute_metrics(clean: np.ndarray, processed: np.ndarray, sr: int) -> Dict[str, float]:
    """Compute all available audio quality metrics"""
    metrics = {}
    
    # Ensure same length
    min_len = min(len(clean), len(processed))
    clean = clean[:min_len]
    processed = processed[:min_len]
    
    # SNR (Signal-to-Noise Ratio)
    noise = clean - processed
    signal_power = np.sum(clean ** 2)
    noise_power = np.sum(noise ** 2)
    metrics['snr'] = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    # SDR (Signal-to-Distortion Ratio)
    metrics['sdr'] = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    # PESQ (Perceptual Evaluation of Speech Quality)
    if PESQ_AVAILABLE:
        try:
            if sr == 8000:
                pesq_score = pesq(sr, clean, processed, 'nb')
            elif sr == 16000:
                pesq_score = pesq(sr, clean, processed, 'wb')
            else:
                # Resample to 16kHz for wideband PESQ
                from scipy import signal
                clean_16k = signal.resample(clean, int(len(clean) * 16000 / sr))
                processed_16k = signal.resample(processed, int(len(processed) * 16000 / sr))
                pesq_score = pesq(16000, clean_16k, processed_16k, 'wb')
            metrics['pesq'] = pesq_score
        except Exception as e:
            print(f"PESQ computation failed: {e}")
            metrics['pesq'] = -1
    else:
        metrics['pesq'] = -1
    
    # STOI (Short-Time Objective Intelligibility)
    if STOI_AVAILABLE:
        try:
            stoi_score = stoi(clean, processed, sr, extended=False)
            metrics['stoi'] = stoi_score
        except Exception as e:
            print(f"STOI computation failed: {e}")
            metrics['stoi'] = -1
    else:
        metrics['stoi'] = -1
    
    # ESTOI (Extended STOI)
    if STOI_AVAILABLE:
        try:
            estoi_score = stoi(clean, processed, sr, extended=True)
            metrics['estoi'] = estoi_score
        except Exception as e:
            print(f"ESTOI computation failed: {e}")
            metrics['estoi'] = -1
    else:
        metrics['estoi'] = -1
    
    return metrics

    def process_waveunet(self, noisy: np.ndarray, sr: int) -> np.ndarray:
        """Process audio through Wave-U-Net model"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        waveform = torch.from_numpy(noisy.astype(np.float32)).unsqueeze(0)
        
        # Segmented processing for long audio
        SEGMENT_LENGTH = 16384
        OVERLAP = 4096
        STEP = SEGMENT_LENGTH - OVERLAP
        
        original_len = waveform.shape[1]
        num_chunks = (original_len - OVERLAP + STEP - 1) // STEP
        pad_len = max(0, num_chunks * STEP + OVERLAP - original_len)
        waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        
        window = torch.hann_window(SEGMENT_LENGTH).to(self.device)
        denoised_audio = torch.zeros_like(waveform)
        normalization = torch.zeros_like(waveform)
        
        with torch.no_grad():
            for i in range(num_chunks):
                start = i * STEP
                end = start + SEGMENT_LENGTH
                chunk = waveform[:, start:end].to(self.device)
                input_tensor = chunk.unsqueeze(0)
                output = self.model(input_tensor).squeeze(0).squeeze(0)
                denoised_audio[:, start:end] += (output * window).unsqueeze(0).cpu()
                normalization[:, start:end] += window.unsqueeze(0).cpu()
        
        denoised_audio /= normalization.clamp(min=1e-8)
        denoised_audio = denoised_audio[:, :original_len]
        
        return denoised_audio.squeeze(0).cpu().numpy()
    
    def process_single_file(self, noisy_path: str, clean_path: str) -> Optional[Dict]:
        """Process a single audio file and compute metrics"""
        # Read audio files
        noisy, sr1 = read_wav(noisy_path)
        clean, sr2 = read_wav(clean_path)
        
        if noisy is None or clean is None:
            print(f"Error reading files: {noisy_path}, {clean_path}")
            return None
            
        if sr1 != sr2:
            print(f"Sample rate mismatch: {sr1} vs {sr2}")
            return None
        
        sr = sr1
        
        # Align lengths
        min_len = min(len(noisy), len(clean))
        noisy = noisy[:min_len]
        clean = clean[:min_len]
        
        # Process through Wave-U-Net
        print(f"Processing {os.path.basename(noisy_path)} through Wave-U-Net...")
        waveunet_output = self.process_waveunet(noisy, sr)
        waveunet_output = waveunet_output[:min_len]
        
        # Apply MMSE-STSA postprocessing
        print(f"Applying MMSE-STSA postprocessing...")
        try:
            postprocessed = mmse_stsa(waveunet_output, sr)
            postprocessed = postprocessed[:min_len]
        except Exception as e:
            print(f"MMSE-STSA failed: {e}")
            postprocessed = waveunet_output
        
        # Compute metrics for all stages
        noisy_metrics = compute_metrics(clean, noisy, sr)
        waveunet_metrics = compute_metrics(clean, waveunet_output, sr)
        final_metrics = compute_metrics(clean, postprocessed, sr)
        
        result = {
            'file': os.path.basename(noisy_path),
            'sample_rate': sr,
            'duration': len(clean) / sr,
            # Original noisy metrics
            'noisy_snr': noisy_metrics['snr'],
            'noisy_sdr': noisy_metrics['sdr'],
            'noisy_pesq': noisy_metrics['pesq'],
            'noisy_stoi': noisy_metrics['stoi'],
            'noisy_estoi': noisy_metrics['estoi'],
            # Wave-U-Net only metrics
            'waveunet_snr': waveunet_metrics['snr'],
            'waveunet_sdr': waveunet_metrics['sdr'],
            'waveunet_pesq': waveunet_metrics['pesq'],
            'waveunet_stoi': waveunet_metrics['stoi'],
            'waveunet_estoi': waveunet_metrics['estoi'],
            # Final (Wave-U-Net + MMSE-STSA) metrics
            'final_snr': final_metrics['snr'],
            'final_sdr': final_metrics['sdr'],
            'final_pesq': final_metrics['pesq'],
            'final_stoi': final_metrics['stoi'],
            'final_estoi': final_metrics['estoi'],
            # Improvements
            'snr_improvement_waveunet': waveunet_metrics['snr'] - noisy_metrics['snr'],
            'snr_improvement_final': final_metrics['snr'] - noisy_metrics['snr'],
            'pesq_improvement_waveunet': waveunet_metrics['pesq'] - noisy_metrics['pesq'] if noisy_metrics['pesq'] > 0 else -1,
            'pesq_improvement_final': final_metrics['pesq'] - noisy_metrics['pesq'] if noisy_metrics['pesq'] > 0 else -1,
        }
        
        return result
        
    def evaluate_dataset(self, noisy_dir: str, clean_dir: str, output_dir: str = "evaluation_results") -> List[Dict]:
        """Evaluate the model on a complete dataset"""
        if not self.load_model():
            return []
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all WAV files
        noisy_files = [f for f in os.listdir(noisy_dir) if f.lower().endswith('.wav')]
        
        print(f"Found {len(noisy_files)} files to evaluate")
        results = []
        
        start_time = time.time()
        
        for i, filename in enumerate(noisy_files):
            print(f"\n[{i+1}/{len(noisy_files)}] Processing {filename}")
            
            noisy_path = os.path.join(noisy_dir, filename)
            clean_path = os.path.join(clean_dir, filename)
            
            if not os.path.exists(clean_path):
                print(f"Warning: No clean reference found for {filename}")
                continue
                
            result = self.process_single_file(noisy_path, clean_path)
            if result:
                results.append(result)
                
        total_time = time.time() - start_time
        print(f"\nEvaluation completed in {total_time:.2f} seconds")
        
        # Save results
        self.results = results
        csv_path = os.path.join(output_dir, "evaluation_results.csv")
        self.save_results_csv(csv_path)
        
        # Generate analysis and plots
        self.generate_analysis_report(output_dir)
        
        return results

    def save_results_csv(self, csv_path: str):
        """Save evaluation results to CSV file"""
        if not self.results:
            print("No results to save")
            return
            
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = self.results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.results:
                writer.writerow(row)
        
        print(f"Results saved to {csv_path}")
    
    def generate_summary_statistics(self) -> Dict:
        """Generate summary statistics for all metrics"""
        if not self.results:
            return {}
            
        df = pd.DataFrame(self.results)
        
        # Metrics to analyze
        metrics = ['snr', 'sdr', 'pesq', 'stoi', 'estoi']
        stages = ['noisy', 'waveunet', 'final']
        
        summary = {}
        
        for metric in metrics:
            summary[metric] = {}
            for stage in stages:
                col_name = f'{stage}_{metric}'
                if col_name in df.columns:
                    values = df[col_name]
                    # Filter out invalid values (-1)
                    valid_values = values[values > -1] if metric in ['pesq', 'stoi', 'estoi'] else values
                    
                    if len(valid_values) > 0:
                        summary[metric][stage] = {
                            'mean': float(valid_values.mean()),
                            'std': float(valid_values.std()),
                            'min': float(valid_values.min()),
                            'max': float(valid_values.max()),
                            'median': float(valid_values.median()),
                            'count': len(valid_values)
                        }
        
        # Improvement statistics
        summary['improvements'] = {}
        improvement_cols = [col for col in df.columns if 'improvement' in col]
        for col in improvement_cols:
            values = df[col]
            valid_values = values[values > -999]  # Filter out invalid improvements
            if len(valid_values) > 0:
                summary['improvements'][col] = {
                    'mean': float(valid_values.mean()),
                    'std': float(valid_values.std()),
                    'positive_count': int((valid_values > 0).sum()),
                    'total_count': len(valid_values)
                }
        
        return summary
    
    def create_comparison_plots(self, output_dir: str):
        """Create comprehensive comparison plots"""
        if not self.results:
            print("No results available for plotting")
            return
            
        df = pd.DataFrame(self.results)
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Metrics comparison across stages
        metrics = ['snr', 'sdr']
        if PESQ_AVAILABLE:
            metrics.append('pesq')
        if STOI_AVAILABLE:
            metrics.extend(['stoi', 'estoi'])
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Prepare data for plotting
            data_to_plot = []
            labels = []
            
            for stage in ['noisy', 'waveunet', 'final']:
                col_name = f'{stage}_{metric}'
                if col_name in df.columns:
                    values = df[col_name]
                    if metric in ['pesq', 'stoi', 'estoi']:
                        values = values[values > -1]  # Filter invalid values
                    if len(values) > 0:
                        data_to_plot.append(values)
                        labels.append(stage.capitalize())
            
            if data_to_plot:
                ax.boxplot(data_to_plot, labels=labels)
                ax.set_title(f'{metric.upper()} Comparison')
                ax.set_ylabel(f'{metric.upper()}')
                ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(metrics), len(axes)):
            fig.delaxes(axes[i])
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Improvement analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # SNR improvements
        if 'snr_improvement_waveunet' in df.columns and 'snr_improvement_final' in df.columns:
            ax = axes[0, 0]
            x = df['snr_improvement_waveunet']
            y = df['snr_improvement_final']
            ax.scatter(x, y, alpha=0.6)
            ax.plot([-10, 20], [-10, 20], 'r--', alpha=0.5)  # Reference line
            ax.set_xlabel('Wave-U-Net SNR Improvement (dB)')
            ax.set_ylabel('Final SNR Improvement (dB)')
            ax.set_title('SNR Improvement Comparison')
            ax.grid(True, alpha=0.3)
        
        # PESQ improvements (if available)
        if PESQ_AVAILABLE and 'pesq_improvement_final' in df.columns:
            ax = axes[0, 1]
            improvements = df['pesq_improvement_final']
            valid_improvements = improvements[improvements > -1]
            if len(valid_improvements) > 0:
                ax.hist(valid_improvements, bins=20, alpha=0.7, edgecolor='black')
                ax.set_xlabel('PESQ Improvement')
                ax.set_ylabel('Count')
                ax.set_title('PESQ Improvement Distribution')
                ax.grid(True, alpha=0.3)
        
        # Processing time vs file duration
        if 'duration' in df.columns:
            ax = axes[1, 0]
            ax.scatter(df['duration'], df.index, alpha=0.6)  # Using index as proxy for processing order
            ax.set_xlabel('File Duration (seconds)')
            ax.set_ylabel('Processing Order')
            ax.set_title('File Duration Distribution')
            ax.grid(True, alpha=0.3)
        
        # Overall improvement summary
        ax = axes[1, 1]
        metrics_to_show = ['snr_improvement_final']
        if PESQ_AVAILABLE:
            metrics_to_show.append('pesq_improvement_final')
        
        improvement_data = []
        improvement_labels = []
        for metric in metrics_to_show:
            if metric in df.columns:
                values = df[metric]
                if 'pesq' in metric:
                    values = values[values > -1]
                if len(values) > 0:
                    improvement_data.append(values)
                    improvement_labels.append(metric.replace('_improvement_final', '').upper())
        
        if improvement_data:
            ax.boxplot(improvement_data, labels=improvement_labels)
            ax.set_title('Overall Improvement Summary')
            ax.set_ylabel('Improvement')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'improvement_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to {output_dir}")
    
    def generate_analysis_report(self, output_dir: str):
        """Generate comprehensive analysis report"""
        # Generate summary statistics
        summary = self.generate_summary_statistics()
        
        # Save summary to JSON
        import json
        with open(os.path.join(output_dir, 'summary_statistics.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create plots
        self.create_comparison_plots(output_dir)
        
        # Generate text report
        report_path = os.path.join(output_dir, 'evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("WAVE-U-NET + MMSE-STSA EVALUATION REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Total files evaluated: {len(self.results)}\n")
            f.write(f"Evaluation date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model checkpoint: {self.checkpoint_path}\n")
            f.write(f"Device used: {self.device}\n\n")
            
            if summary:
                f.write("SUMMARY STATISTICS:\n")
                f.write("-" * 30 + "\n")
                
                for metric, stages in summary.items():
                    if metric == 'improvements':
                        continue
                    f.write(f"\n{metric.upper()}:\n")
                    for stage, stats in stages.items():
                        if isinstance(stats, dict):
                            f.write(f"  {stage.capitalize():12} - Mean: {stats['mean']:6.2f}, "
                                  f"Std: {stats['std']:5.2f}, Range: [{stats['min']:5.2f}, {stats['max']:5.2f}]\n")
                
                if 'improvements' in summary:
                    f.write(f"\nIMPROVEMENTS:\n")
                    for imp_type, stats in summary['improvements'].items():
                        positive_pct = (stats['positive_count'] / stats['total_count']) * 100
                        f.write(f"  {imp_type:25} - Mean: {stats['mean']:6.2f}, "
                               f"Positive: {stats['positive_count']}/{stats['total_count']} ({positive_pct:.1f}%)\n")
        
        print(f"Comprehensive evaluation report saved to {output_dir}")
        print(f"Summary statistics saved to summary_statistics.json")
        print(f"Detailed report saved to evaluation_report.txt")

# Convenience functions for backward compatibility
def evaluate_folder(noisy_dir: str, clean_dir: str, csv_path: str = "evaluation_results.csv", 
                   output_dir: str = "evaluation_results"):
    """Legacy function for backward compatibility"""
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_dataset(noisy_dir, clean_dir, output_dir)
    
    # Save simple CSV for backward compatibility
    if results:
        simple_results = []
        for r in results:
            simple_results.append({
                'file': r['file'],
                'snr_noisy': r['noisy_snr'],
                'snr_waveunet': r['waveunet_snr'], 
                'snr_post': r['final_snr'],
                'sdr_noisy': r['noisy_sdr'],
                'sdr_waveunet': r['waveunet_sdr'],
                'sdr_post': r['final_sdr']
            })
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['file', 'snr_noisy', 'snr_waveunet', 'snr_post', 'sdr_noisy', 'sdr_waveunet', 'sdr_post']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in simple_results:
                writer.writerow(row)
    
    return results

def quick_evaluate(noisy_dir: str, clean_dir: str, output_dir: str = None):
    """Quick evaluation with automatic output directory"""
    if output_dir is None:
        output_dir = f"evaluation_{int(time.time())}"
    
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_dataset(noisy_dir, clean_dir, output_dir)
    
    if results:
        print(f"\n{'='*60}")
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"Files evaluated: {len(results)}")
        print(f"Results saved to: {output_dir}")
        print("Generated files:")
        print(f"  - evaluation_results.csv (detailed metrics)")
        print(f"  - summary_statistics.json (summary stats)")
        print(f"  - evaluation_report.txt (human-readable report)")
        print(f"  - metrics_comparison.png (metrics plots)")
        print(f"  - improvement_analysis.png (improvement plots)")
        
        # Print quick summary
        df = pd.DataFrame(results)
        print(f"\nQUICK SUMMARY:")
        print(f"  Average SNR improvement (Wave-U-Net): {df['snr_improvement_waveunet'].mean():.2f} dB")
        print(f"  Average SNR improvement (Final): {df['snr_improvement_final'].mean():.2f} dB")
        if PESQ_AVAILABLE and 'pesq_improvement_final' in df.columns:
            valid_pesq = df['pesq_improvement_final'][df['pesq_improvement_final'] > -1]
            if len(valid_pesq) > 0:
                print(f"  Average PESQ improvement: {valid_pesq.mean():.2f}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Wave-U-Net + MMSE-STSA Evaluation")
    parser.add_argument("--noisy_dir", type=str, required=True, 
                       help="Directory with noisy WAV files")
    parser.add_argument("--clean_dir", type=str, required=True,
                       help="Directory with clean reference WAV files")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for results (default: auto-generated)")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH,
                       help="Path to model checkpoint")
    parser.add_argument("--quick", action="store_true",
                       help="Use quick evaluation mode")
    parser.add_argument("--legacy", action="store_true",
                       help="Use legacy evaluation mode")
    
    args = parser.parse_args()
    
    if args.legacy:
        # Legacy mode - backward compatibility
        csv_path = "evaluation_results.csv"
        results = evaluate_folder(args.noisy_dir, args.clean_dir, csv_path)
    else:
        # New comprehensive mode
        if args.output_dir is None:
            args.output_dir = f"evaluation_{int(time.time())}"
        
        evaluator = ModelEvaluator(args.checkpoint)
        
        if args.quick:
            results = quick_evaluate(args.noisy_dir, args.clean_dir, args.output_dir)
        else:
            results = evaluator.evaluate_dataset(args.noisy_dir, args.clean_dir, args.output_dir)
    
    print(f"\nEvaluation completed with {len(results) if results else 0} files processed.")
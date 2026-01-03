"""
System Information Utilities
Detect GPU VRAM and system resources across different platforms
"""

import subprocess
import platform
import logging
from typing import Optional, Dict, Any
import json
import re

logger = logging.getLogger(__name__)

class SystemInfo:
    """System information detection"""
    
    @staticmethod
    def get_gpu_memory_gb() -> float:
        """
        Get available GPU memory in GB
        Supports macOS (Metal/MPS) and other platforms
        """
        system = platform.system().lower()
        
        if system == "darwin":  # macOS
            return SystemInfo._get_macos_gpu_memory()
        elif system == "linux":
            return SystemInfo._get_linux_gpu_memory()
        elif system == "windows":
            return SystemInfo._get_windows_gpu_memory()
        else:
            logger.warning(f"Unknown system: {system}, defaulting to 8GB GPU memory")
            return 8.0
    
    @staticmethod
    def _get_macos_gpu_memory() -> float:
        """Get GPU memory on macOS using system_profiler and sysctl"""
        try:
            # Method 1: Try system_profiler for discrete GPUs
            result = subprocess.run([
                "system_profiler", "SPDisplaysDataType", "-json"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                displays = data.get("SPDisplaysDataType", [])
                
                for display in displays:
                    # Look for VRAM info
                    vram_info = display.get("sppci_model", "")
                    if "Metal" in vram_info or "GPU" in vram_info:
                        # Try to extract VRAM from description
                        vram_match = re.search(r'(\d+)\s*GB', vram_info)
                        if vram_match:
                            return float(vram_match.group(1))
            
            # Method 2: For Apple Silicon, check unified memory
            result = subprocess.run([
                "sysctl", "-n", "hw.memsize"
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                total_memory_bytes = int(result.stdout.strip())
                total_memory_gb = total_memory_bytes / (1024**3)
                
                # Apple Silicon uses unified memory
                # Check if this is Apple Silicon
                result = subprocess.run([
                    "sysctl", "-n", "machdep.cpu.brand_string"
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    cpu_info = result.stdout.strip().lower()
                    if "apple" in cpu_info or "m1" in cpu_info or "m2" in cpu_info or "m3" in cpu_info:
                        # Apple Silicon - unified memory
                        # Reserve some memory for system (typically 25-30%)
                        available_for_gpu = total_memory_gb * 0.7
                        logger.info(f"Apple Silicon detected: {total_memory_gb:.1f}GB total, ~{available_for_gpu:.1f}GB available for GPU")
                        return available_for_gpu
            
            # Method 3: Check for specific Apple Silicon models
            result = subprocess.run([
                "system_profiler", "SPHardwareDataType", "-json"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                hardware = data.get("SPHardwareDataType", [])
                
                for hw in hardware:
                    chip_type = hw.get("chip_type", "").lower()
                    memory = hw.get("physical_memory", "")
                    
                    if any(chip in chip_type for chip in ["m1", "m2", "m3", "apple"]):
                        # Extract memory amount
                        memory_match = re.search(r'(\d+)\s*GB', memory)
                        if memory_match:
                            total_gb = float(memory_match.group(1))
                            available_gb = total_gb * 0.7  # 70% available for GPU
                            logger.info(f"Apple {chip_type.upper()} with {total_gb}GB unified memory, ~{available_gb:.1f}GB for GPU")
                            return available_gb
            
        except Exception as e:
            logger.warning(f"Failed to detect macOS GPU memory: {e}")
        
        # Fallback for macOS
        logger.warning("Could not detect macOS GPU memory, defaulting to 16GB")
        return 16.0
    
    @staticmethod
    def _get_linux_gpu_memory() -> float:
        """Get GPU memory on Linux"""
        try:
            # Try nvidia-smi for NVIDIA GPUs
            result = subprocess.run([
                "nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                memory_mb = float(result.stdout.strip())
                memory_gb = memory_mb / 1024
                logger.info(f"NVIDIA GPU detected: {memory_gb:.1f}GB VRAM")
                return memory_gb
            
            # Try AMD ROCm
            result = subprocess.run([
                "rocm-smi", "--showmeminfo", "vram"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                # Parse AMD GPU memory info
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if "Total" in line:
                        match = re.search(r'(\d+)\s*MB', line)
                        if match:
                            memory_mb = float(match.group(1))
                            memory_gb = memory_mb / 1024
                            logger.info(f"AMD GPU detected: {memory_gb:.1f}GB VRAM")
                            return memory_gb
            
            # Try lspci for integrated GPUs
            result = subprocess.run([
                "lspci", "-v"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                # Look for GPU info in lspci output
                if "VGA" in result.stdout or "Display" in result.stdout:
                    logger.info("Integrated GPU detected, estimating 8GB shared memory")
                    return 8.0
                    
        except Exception as e:
            logger.warning(f"Failed to detect Linux GPU memory: {e}")
        
        # Fallback for Linux
        logger.warning("Could not detect Linux GPU memory, defaulting to 8GB")
        return 8.0
    
    @staticmethod
    def _get_windows_gpu_memory() -> float:
        """Get GPU memory on Windows"""
        try:
            # Try wmic for Windows GPU info
            result = subprocess.run([
                "wmic", "path", "win32_VideoController", "get", "AdapterRAM"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip() and line.strip() != "AdapterRAM":
                        try:
                            memory_bytes = int(line.strip())
                            if memory_bytes > 0:
                                memory_gb = memory_bytes / (1024**3)
                                logger.info(f"Windows GPU detected: {memory_gb:.1f}GB VRAM")
                                return memory_gb
                        except ValueError:
                            continue
                            
        except Exception as e:
            logger.warning(f"Failed to detect Windows GPU memory: {e}")
        
        # Fallback for Windows
        logger.warning("Could not detect Windows GPU memory, defaulting to 8GB")
        return 8.0
    
    @staticmethod
    def get_system_summary() -> Dict[str, Any]:
        """Get comprehensive system information"""
        system = platform.system()
        gpu_memory = SystemInfo.get_gpu_memory_gb()
        
        summary = {
            "platform": system,
            "gpu_memory_gb": gpu_memory,
            "reserved_memory_gb": max(2.0, gpu_memory * 0.15),  # Reserve 15% or 2GB minimum
            "available_memory_gb": gpu_memory * 0.85,
            "python_version": platform.python_version(),
            "architecture": platform.machine()
        }
        
        try:
            # Get CPU info
            if system == "Darwin":
                result = subprocess.run([
                    "sysctl", "-n", "machdep.cpu.brand_string"
                ], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    summary["cpu"] = result.stdout.strip()
            
            # Get total system memory
            if system == "Darwin":
                result = subprocess.run([
                    "sysctl", "-n", "hw.memsize"
                ], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    memory_bytes = int(result.stdout.strip())
                    summary["system_memory_gb"] = memory_bytes / (1024**3)
        except:
            pass
        
        return summary
    
    @staticmethod
    def log_system_info():
        """Log system information for debugging"""
        info = SystemInfo.get_system_summary()
        
        logger.info("=== SYSTEM INFORMATION ===")
        logger.info(f"Platform: {info['platform']}")
        logger.info(f"Architecture: {info['architecture']}")
        logger.info(f"GPU Memory: {info['gpu_memory_gb']:.1f}GB")
        logger.info(f"Available for LLM: {info['available_memory_gb']:.1f}GB")
        logger.info(f"Reserved: {info['reserved_memory_gb']:.1f}GB")
        
        if "cpu" in info:
            logger.info(f"CPU: {info['cpu']}")
        if "system_memory_gb" in info:
            logger.info(f"System Memory: {info['system_memory_gb']:.1f}GB")
        
        logger.info("==========================")
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Callable, Any, Dict, Union
import logging
from tqdm import tqdm

class BatchProcessor:
    """Process multiple images in batch with progress tracking and error handling."""
    
    def __init__(
        self,
        max_workers: int = None,
        show_progress: bool = True
    ):
        """
        Initialize the batch processor.
        
        Args:
            max_workers: Maximum number of worker threads
            show_progress: Whether to show progress bar
        """
        self.max_workers = max_workers or os.cpu_count()
        self.show_progress = show_progress
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up logging configuration."""
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def process_batch(
        self,
        items: List[Any],
        process_fn: Callable,
        save_results: bool = True,
        output_dir: Union[str, Path] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a batch of items using the provided function.
        
        Args:
            items: List of items to process (file paths or images)
            process_fn: Function to process each item
            save_results: Whether to save results to disk
            output_dir: Directory to save results if save_results is True
            **kwargs: Additional arguments to pass to process_fn

        Returns:
            Dictionary with results and statistics
        """
        if save_results and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        errors = {}
        processed_count = 0
        error_count = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(self._process_item, item, process_fn, 
                              save_results, output_dir, **kwargs): item 
                for item in items
            }
            
            # Process results as they complete
            iterator = as_completed(future_to_item)
            if self.show_progress:
                iterator = tqdm(iterator, total=len(items), 
                              desc="Processing items")
            
            for future in iterator:
                item = future_to_item[future]
                try:
                    result = future.result()
                    results[str(item)] = result
                    processed_count += 1
                except Exception as e:
                    self.logger.error(f"Error processing {item}: {str(e)}")
                    errors[str(item)] = str(e)
                    error_count += 1
        
        # Compile statistics
        stats = {
            "total_items": len(items),
            "processed": processed_count,
            "errors": error_count,
            "success_rate": processed_count / len(items) if items else 0
        }
        
        return {
            "results": results,
            "errors": errors,
            "stats": stats
        }
    
    def _process_item(
        self,
        item: Any,
        process_fn: Callable,
        save_results: bool,
        output_dir: Path,
        **kwargs
    ) -> Any:
        """
        Process a single item and optionally save results.
        
        Args:
            item: Item to process
            process_fn: Processing function
            save_results: Whether to save results
            output_dir: Output directory for saved results
            **kwargs: Additional arguments for process_fn

        Returns:
            Processing results
        """
        # Process the item
        result = process_fn(item, **kwargs)
        
        # Save results if requested
        if save_results and output_dir:
            if isinstance(item, (str, Path)):
                # For file paths, use original filename with suffix
                output_path = output_dir / f"{Path(item).stem}_processed{Path(item).suffix}"
            else:
                # For other items, generate unique name
                output_path = output_dir / f"result_{id(item)}.jpg"
            
            # Handle different result types
            if hasattr(result, 'save'):  # PIL Image
                result.save(output_path)
            elif isinstance(result, dict):  # Dictionary results
                import json
                with open(output_path.with_suffix('.json'), 'w') as f:
                    json.dump(result, f, indent=2)
            
            self.logger.info(f"Saved result to {output_path}")
        
        return result

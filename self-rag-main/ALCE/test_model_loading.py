from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import logging
import os
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('t5_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MODEL_PATH = "/gpfsnyu/scratch/yx2432/models/models--google--t5_xxl_true_nli_mixture/snapshots/aa6cfe1dd4257853bfdd772992045f41bfc14988"

# Handle SIGFPE (floating point exception)
def handle_sigfpe(signum, frame):
    logger.error("Floating point exception encountered")
    sys.exit(1)

signal.signal(signal.SIGFPE, handle_sigfpe)

def load_model():
    """Load model with explicit error handling"""
    try:
        # Print environment info for debugging
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Set environment variables to control numerical stability
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        
        logger.info("Loading model...")
        # Try with bfloat16 precision instead of float16
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,  # bfloat16 is often more stable than float16
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        logger.info(f"Model loaded successfully! Device: {next(model.parameters()).device}")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        return None, None

def test_nli(model, tokenizer, passage, claim):
    """Run NLI test with memory safeguards"""
    if not model or not tokenizer:
        logger.error("Cannot run test - model not loaded")
        return
    
    try:
        logger.info(f"\nTesting: Passage='{passage[:50]}...', Claim='{claim[:50]}...'")
        
        # Clear GPU cache and validate memory
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        
        # Prepare input with error checking
        input_text = f"premise: {passage} hypothesis: {claim}"
        logger.info(f"Tokenizing input...")
        inputs = tokenizer(input_text, return_tensors="pt")
        logger.info(f"Input shape: {inputs.input_ids.shape}")
        
        # Move to GPU with validation - get device from model instead of hardcoding
        device = next(model.parameters()).device
        logger.info(f"Moving inputs to device: {device}...")
        inputs = inputs.to(device)
        
        # Generation with stability settings
        logger.info("Running generation...")
        try:
            # Try with torch.no_grad() instead of inference_mode
            with torch.no_grad():
                # Even more simplified generation parameters
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    max_length=20,        # Use a smaller max_length
                    num_beams=1,
                    do_sample=False,
                    use_cache=True        # Enable KV-cache for more efficient generation
                )
            
            # Decode with validation
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"✅ NLI Result: {result}")
            print(f"Result: {result}")
        except RuntimeError as e:
            # If it fails, try a single token generation
            logger.warning(f"First generation attempt failed: {str(e)}")
            logger.info("Trying with simpler generation settings...")
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    max_new_tokens=1,      # Generate just 1 token
                    num_beams=1,
                    do_sample=False
                )
            
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"✅ Simple generation result: {result}")
            print(f"Result: {result}")
            
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA out of memory! Try reducing max_length")
    except RuntimeError as e:
        logger.error(f"Runtime error: {str(e)}")
        # Print more detailed GPU info when runtime error occurs
        if torch.cuda.is_available():
            logger.error(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
            logger.error(f"GPU memory reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")
    except FloatingPointError as e:
        logger.error(f"Floating point error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
    finally:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    logger.info("===== Starting T5 NLI Test =====")
    
    try:
        model, tokenizer = load_model()
        
        if model and tokenizer:
            test_cases = [
                ("Pelé, a Brazilian footballer, has the highest goals in world football", 
                 "Pelé has the highest goals in world football"),
                ("The capital of France is Paris", 
                 "Paris is the capital of France")
            ]
            
            for passage, claim in test_cases:
                test_nli(model, tokenizer, passage, claim)
    except Exception as e:
        logger.error(f"Critical error: {str(e)}", exc_info=True)
    
    logger.info("===== Test Completed =====")
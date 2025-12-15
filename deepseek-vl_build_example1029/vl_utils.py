
#!/usr/bin/env python
# coding: utf-8

import torch
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images


class DeepseekVLV2Chat:
    def __init__(self, model_path="./deepseek-vl2-tiny", dtype=torch.bfloat16):
        """
        Initialize the DeepseekVLV2Chat class.
        
        Args:
            model_path (str): Path to the model directory
            dtype (torch.dtype): Data type for model computation
        """
        self.model_path = model_path
        self.dtype = dtype
        
        # Load processor and tokenizer
        self.vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer
        
        # Load model
        self.vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        self.vl_gpt = self.vl_gpt.to(dtype).cuda().eval()
    
    def chat(self, conversation, max_new_tokens=512, do_sample=False, use_cache=True):
        """
        Process a conversation with the model.
        
        Args:
            conversation (list): List of conversation turns with role, content and images
            max_new_tokens (int): Maximum number of new tokens to generate
            do_sample (bool): Whether to use sampling
            use_cache (bool): Whether to use cache for generation
            
        Returns:
            str: Model's response
        """
        with torch.no_grad():
            # Load images
            pil_images = load_pil_images(conversation)
            
            # Prepare inputs
            prepare_inputs = self.vl_chat_processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True,
                system_prompt=""
            ).to(self.vl_gpt.device)
            
            # Prepare input embeddings
            inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
            
            # Generate output
            outputs = self.vl_gpt.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                use_cache=use_cache
            )
            
            # Decode the response
            answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
            
            # Clean up GPU memory
            del inputs_embeds, outputs, prepare_inputs
            if 'pil_images' in locals():
                del pil_images
            
            # Run garbage collection
            torch.cuda.empty_cache()
            
            return answer
    
    def __del__(self):
        """Clean up when the object is deleted."""
        if hasattr(self, 'vl_gpt'):
            del self.vl_gpt
        if hasattr(self, 'vl_chat_processor'):
            del self.vl_chat_processor
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
            
        torch.cuda.empty_cache()


# Example usage:
if __name__ == "__main__":
    # Initialize the chat model
    chat_model = DeepseekVLV2Chat("./deepseek-vl2-tiny")
    
    # First conversation
    conversation1 = [
        {
            "role": "<|User|>",
            "content": "<image>\n<|ref|>describe this<|/ref|>.",
            "images": ["./images/image3.png"],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    
    response1 = chat_model.chat(conversation1, max_new_tokens=512)
    print("Response 1:", response1)
    
    # Second conversation
    conversation2 = [
        {
            "role": "<|User|>",
            "content": "<image>\n<|ref|>summarize this picture to see what conclusion you get through this<|/ref|>.",
            "images": ["./images/image3.png"],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    
    response2 = chat_model.chat(conversation2, max_new_tokens=1024)
    print("Response 2:", response2)
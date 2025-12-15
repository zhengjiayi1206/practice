#!/usr/bin/env python
# coding: utf-8

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
import os
import uuid
from pathlib import Path

from vl_utils import DeepseekVLV2Chat

# Initialize FastAPI app
app = FastAPI(title="DeepSeek-VL2 API", description="API for DeepSeek-VL2 vision language model")

# Global model instance
model: Optional[DeepseekVLV2Chat] = None

class Message(BaseModel):
    role: str
    content: str
    images: Optional[List[str]] = None  # Local image paths

class ConversationRequest(BaseModel):
    messages: List[Message]
    max_new_tokens: int = 512
    do_sample: bool = False
    use_cache: bool = True

class ConversationResponse(BaseModel):
    response: str

@app.on_event("startup")
async def load_model():
    """
    Load the model when the application starts.
    """
    global model
    try:
        model = DeepseekVLV2Chat("./deepseek-vl2-tiny")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

@app.on_event("shutdown")
async def cleanup():
    """
    Clean up resources when the application shuts down.
    """
    global model
    if model:
        del model
        torch.cuda.empty_cache()

def validate_image_paths(image_paths: List[str]) -> List[str]:
    """
    Validate that image paths exist.
    
    Args:
        image_paths: List of image file paths
        
    Returns:
        Validated image paths
        
    Raises:
        ValueError: If any path doesn't exist or is not a file
    """
    validated_paths = []
    for path in image_paths:
        # Convert to Path object for easier handling
        p = Path(path)
        
        # Check if path exists and is a file
        if not p.exists():
            raise ValueError(f"Image path does not exist: {path}")
        
        if not p.is_file():
            raise ValueError(f"Path is not a file: {path}")
            
        validated_paths.append(str(p.absolute()))
        
    return validated_paths

def process_conversation_messages(messages: List[Message]):
    """
    Process messages to format conversation for the model.
    
    Args:
        messages: List of message objects
        
    Returns:
        Formatted conversation
    """
    conversation = []
    
    for msg in messages:
        # Format message for model
        formatted_msg = {
            "role": msg.role,
            "content": msg.content,
        }
        
        # Add images if they exist
        if msg.images:
            try:
                validated_paths = validate_image_paths(msg.images)
                formatted_msg["images"] = validated_paths
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        conversation.append(formatted_msg)
    
    # Ensure the last message is from assistant (empty content for generation)
    if conversation[-1]["role"] != "<|Assistant|>":
        conversation.append({"role": "<|Assistant|>", "content": ""})
    
    return conversation

@app.post("/chat", response_model=ConversationResponse)
async def chat(request: ConversationRequest):
    """
    Chat with the DeepSeek-VL2 model.
    
    Args:
        request: Conversation request containing messages and generation parameters
        
    Returns:
        Model's response
    """
    global model
    
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Process messages
        conversation = process_conversation_messages(request.messages)
        
        # Get model response
        response = model.chat(
            conversation=conversation,
            max_new_tokens=request.max_new_tokens,
            do_sample=request.do_sample,
            use_cache=request.use_cache
        )
        
        return ConversationResponse(response=response)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Status information
    """
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
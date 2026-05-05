from fastapi import FastAPI, UploadFile, File, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
import json
import asyncio
from pdf_extractor import extract_images_from_pdf
from model_runner import QwenModelRunner

app = FastAPI(title="Scientific Figure Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

runner = QwenModelRunner()

UPLOAD_DIR = "/home/drive4/FigCapsHF/backend/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount the uploads directory so the frontend can display the images
app.mount("/images", StaticFiles(directory=UPLOAD_DIR), name="images")

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    print(f"Received file: {file.filename}")
    pdf_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    output_dir = os.path.join(UPLOAD_DIR, file.filename.replace(".pdf", "_images"))
    print(f"Extracting images to: {output_dir}")
    
    # Extract images from PDF
    image_paths = extract_images_from_pdf(pdf_path, output_dir)
    
    # Return a job ID
    job_id = file.filename
    print(f"Extracted {len(image_paths)} images.")
    
    return {
        "job_id": job_id,
        "image_count": len(image_paths)
    }

@app.get("/stream/{job_id}")
async def stream_results(request: Request, job_id: str):
    """
    Streams the generated paragraphs back to the client one by one as Server-Sent Events.
    """
    output_dir = os.path.join(UPLOAD_DIR, job_id.replace(".pdf", "_images"))
    
    if not os.path.exists(output_dir):
        return {"error": "Images not found. Please upload again."}
        
    image_paths = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    async def event_generator():
        # Pre-load model to avoid long delay on first image
        runner.load_model()
        
        for i, img_path in enumerate(image_paths):
            if await request.is_disconnected():
                print("Client disconnected, stopping generation.")
                break
                
            print(f"Generating paragraph for {img_path}...")
            # Run inference
            # Note: This is synchronous/blocking, which is fine for a single-user prototype.
            paragraph = runner.generate_paragraph(img_path)
            
            data = {
                "image_name": os.path.basename(img_path),
                # Provide a relative URL for the frontend to fetch the image
                "image_url": f"http://127.0.0.1:8000/images/{job_id.replace('.pdf', '_images')}/{os.path.basename(img_path)}",
                "paragraph": paragraph,
                "progress": f"{i+1}/{len(image_paths)}"
            }
            
            yield {
                "event": "message",
                "id": str(i),
                "data": json.dumps(data)
            }
            
            # Tiny sleep to yield to event loop
            await asyncio.sleep(0.1)
            
        # Send a final 'done' event
        yield {
            "event": "done",
            "data": "generation complete"
        }
            
    return EventSourceResponse(event_generator())

class ChatRequest(BaseModel):
    image_name: str
    job_id: str
    paragraph: str
    question: str

@app.post("/chat")
async def chat_with_figure(request: ChatRequest):
    """
    Handles a per-image chat question from the frontend.
    """
    try:
        # Determine the image path based on the job ID and image name
        output_dir = os.path.join(UPLOAD_DIR, request.job_id.replace(".pdf", "_images"))
        img_path = os.path.join(output_dir, request.image_name)
        
        if not os.path.exists(img_path):
            return {"error": "Image not found on server."}
            
        print(f"Chat request for {request.image_name}: {request.question}")
        
        # Run inference through the new chat method
        answer = runner.chat_about_figure(img_path, request.paragraph, request.question)
        
        return {"answer": answer}
    except Exception as e:
        print(f"Chat error: {str(e)}")
        return {"error": "An error occurred while generating the response."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

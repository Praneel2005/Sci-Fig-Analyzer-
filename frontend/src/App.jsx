import React, { useState, useRef } from 'react';
import { UploadCloud, FileText, RefreshCw, CheckCircle2 } from 'lucide-react';
import axios from 'axios';
import './index.css';

const API_URL = 'http://localhost:8000';

function App() {
  const [file, setFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState('');
  const [results, setResults] = useState([]);
  const [isDragActive, setIsDragActive] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  
  const fileInputRef = useRef(null);

  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.type === 'application/pdf') {
        handleFileUpload(droppedFile);
      } else {
        alert("Please upload a PDF file.");
      }
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileUpload(e.target.files[0]);
    }
  };

  const handleFileUpload = async (selectedFile) => {
    setFile(selectedFile);
    setIsUploading(true);
    setResults([]);
    setIsComplete(false);
    
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(`${API_URL}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      const { job_id, image_count } = response.data;
      
      if (image_count === 0) {
        alert("No figures were found in this PDF.");
        setIsUploading(false);
        setFile(null);
        return;
      }

      setIsUploading(false);
      setIsProcessing(true);
      setProgress(`0/${image_count}`);
      
      startEventStream(job_id);
      
    } catch (error) {
      console.error("Upload error:", error);
      alert("Error uploading file. Make sure the backend is running.");
      setIsUploading(false);
      setFile(null);
    }
  };

  const startEventStream = (jobId) => {
    const eventSource = new EventSource(`${API_URL}/stream/${jobId}`);
    
    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setResults(prev => {
        const exists = prev.find(r => r.image_name === data.image_name);
        if (exists) return prev;
        return [...prev, data];
      });
      setProgress(data.progress);
    };
    
    eventSource.addEventListener('done', () => {
      setIsProcessing(false);
      setIsComplete(true);
      eventSource.close();
    });
    
    eventSource.onerror = (error) => {
      console.error("SSE Error:", error);
      eventSource.close();
      setIsProcessing(false);
    };
  };

  const resetApp = () => {
    setFile(null);
    setResults([]);
    setIsProcessing(false);
    setIsComplete(false);
    setProgress('');
  };

  return (
    <div className="app-container">
      
      {!file && !isUploading && (
        <div className="hero-section">
          {/* Canva-style Floating Images */}
          <img src="/img1.png" alt="Researcher" className="floating-image img-left" />
          <img src="/img2.png" alt="Abstract Data" className="floating-image img-right" />
          
          <div className="main-card">
            <h1>Use AI to extract and analyze your scientific figures</h1>
            <p className="subtitle">
              Upload any research PDF. We'll extract the figures and generate publication-ready paragraph descriptions in seconds.
            </p>
            
            <button className="upload-btn" onClick={() => fileInputRef.current.click()}>
              <UploadCloud size={20} />
              Upload your research PDF
            </button>
            <p className="upload-hint">or drop it here</p>

            <div 
              className={`upload-zone ${isDragActive ? 'drag-active' : ''}`}
              onDragEnter={handleDragEnter}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current.click()}
              style={{marginTop: '2rem'}}
            >
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileInput} 
                accept="application/pdf" 
                style={{ display: 'none' }} 
              />
              <div style={{color: '#6b7280', fontWeight: 500}}>Drag & drop area</div>
            </div>
          </div>
        </div>
      )}

      {isUploading && (
        <div className="loading-container" style={{background: 'rgba(255,255,255,0.9)', padding: '4rem', borderRadius: '24px', width: '100%', maxWidth: '600px', boxShadow: '0 20px 40px rgba(0,0,0,0.1)'}}>
          <div className="spinner"></div>
          <h2 style={{fontSize: '1.5rem', marginBottom: '0.5rem'}}>Parsing PDF...</h2>
          <p style={{color: '#6b7280'}}>Extracting all embedded figures and images using PyMuPDF</p>
        </div>
      )}

      {(isProcessing || results.length > 0) && (
        <div className="results-container">
          <div className="results-header">
            <div>
              <h2 style={{fontSize: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem', margin: 0}}>
                <FileText size={20} color="var(--primary)" />
                {file?.name}
              </h2>
              {isProcessing ? (
                <div style={{color: 'var(--primary)', fontWeight: 600, fontSize: '0.9rem', marginTop: '0.5rem'}}>
                  Generating AI Analysis: {progress} <span className="typing-indicator"></span>
                </div>
              ) : (
                <div style={{color: '#10b981', fontWeight: 600, fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '0.25rem', marginTop: '0.5rem'}}>
                  <CheckCircle2 size={16} /> Analysis Complete
                </div>
              )}
            </div>
            
            <button className="reset-btn" onClick={resetApp}>
              <RefreshCw size={16} /> Start Over
            </button>
          </div>

          <div className="figures-grid">
            {results.map((result, index) => (
              <div key={index} className="figure-card">
                <div className="figure-image-container">
                  <img 
                    src={result.image_url} 
                    alt={`Figure ${index + 1}`} 
                    className="figure-image"
                  />
                </div>
                <div className="figure-content">
                  <div className="figure-title">Figure {index + 1}</div>
                  <div className="figure-paragraph">
                    {result.paragraph}
                  </div>
                </div>
              </div>
            ))}
          </div>
          
          {isProcessing && results.length > 0 && (
             <div style={{textAlign: 'center', padding: '3rem', color: '#6b7280', fontWeight: 500}}>
                Qwen2.5-VL is analyzing the next figure...
             </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;

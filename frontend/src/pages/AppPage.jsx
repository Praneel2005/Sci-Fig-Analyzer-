import React, { useState, useRef, useEffect } from 'react';
import { UploadCloud, FileText, RefreshCw, CheckCircle2, MessageSquare, Send } from 'lucide-react';
import axios from 'axios';

const API_URL = 'http://localhost:8000';

const FigureCard = ({ result, index, jobId }) => {
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);
  const chatEndRef = useRef(null);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    if (isChatOpen) scrollToBottom();
  }, [messages, isChatOpen]);

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;
    
    const userMsg = inputValue.trim();
    setMessages(prev => [...prev, { role: 'user', text: userMsg }]);
    setInputValue('');
    setIsChatLoading(true);

    try {
      const response = await axios.post(`${API_URL}/chat`, {
        image_name: result.image_name,
        job_id: jobId,
        paragraph: result.paragraph,
        question: userMsg
      });

      setMessages(prev => [...prev, { role: 'ai', text: response.data.answer || response.data.error }]);
    } catch (error) {
      setMessages(prev => [...prev, { role: 'ai', text: "Sorry, an error occurred while asking the question." }]);
    } finally {
      setIsChatLoading(false);
    }
  };

  return (
    <div className="figure-card">
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
        
        <div className="chat-container">
          {!isChatOpen ? (
            <button className="chat-toggle-btn" onClick={() => setIsChatOpen(true)}>
              <MessageSquare size={16} /> Ask a question about this figure
            </button>
          ) : (
            <div className="chat-box">
              <div className="chat-messages">
                <div className="chat-message ai">
                  Hi! I'm the AI that analyzed this figure. What would you like to know?
                </div>
                {messages.map((msg, i) => (
                  <div key={i} className={`chat-message ${msg.role}`}>
                    {msg.text}
                  </div>
                ))}
                {isChatLoading && (
                  <div className="chat-loading">
                    <span className="typing-indicator"></span> Thinking...
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>
              <div className="chat-input-area">
                <input 
                  type="text" 
                  className="chat-input" 
                  placeholder="Ask about a specific label, trend, or detail..." 
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                />
                <button 
                  className="chat-send-btn" 
                  onClick={handleSendMessage}
                  disabled={isChatLoading || !inputValue.trim()}
                >
                  <Send size={16} />
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const AppPage = () => {
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
        <div className="workspace-intro">
          <div className="main-card workspace-card">
            <h2>Start your analysis</h2>
            <p className="subtitle">
              Upload your research PDF below. We will extract the figures and generate analytical paragraphs.
            </p>
            
            <button className="upload-btn" onClick={() => fileInputRef.current.click()}>
              <UploadCloud size={20} />
              Upload PDF
            </button>

            <div 
              className={`upload-zone ${isDragActive ? 'drag-active' : ''}`}
              onDragEnter={handleDragEnter}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current.click()}
            >
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileInput} 
                accept="application/pdf" 
                style={{ display: 'none' }} 
              />
              <div className="upload-hint-text">Drag & drop area</div>
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
              <FigureCard key={index} result={result} index={index} jobId={file?.name} />
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

export default AppPage;

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import PyPDF2
from docx import Document
from pptx import Presentation
import pandas as pd
import markdown
from bs4 import BeautifulSoup
import io

class DocumentProcessor:
    """Processes different document formats and extracts text content"""
    
    @staticmethod
    def process_file(file_path: str) -> Dict[str, Any]:
        """
        Process a file and return its content based on file extension
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Dict containing the extracted content and metadata
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.pdf':
                return DocumentProcessor._process_pdf(file_path)
            elif file_extension == '.docx':
                return DocumentProcessor._process_docx(file_path)
            elif file_extension == '.pptx':
                return DocumentProcessor._process_pptx(file_path)
            elif file_extension == '.csv':
                return DocumentProcessor._process_csv(file_path)
            elif file_extension in ['.txt', '.md']:
                return DocumentProcessor._process_text(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }
    
    @staticmethod
    def _process_pdf(file_path: str) -> Dict[str, Any]:
        """Extract text from PDF file"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n\n"
        
        return {
            "success": True,
            "content": text.strip(),
            "file_type": "pdf",
            "page_count": len(pdf_reader.pages)
        }
    
    @staticmethod
    def _process_docx(file_path: str) -> Dict[str, Any]:
        """Extract text from DOCX file"""
        doc = Document(file_path)
        full_text = []
        
        for para in doc.paragraphs:
            full_text.append(para.text)
        
        return {
            "success": True,
            "content": "\n\n".join(full_text).strip(),
            "file_type": "docx"
        }
    
    @staticmethod
    def _process_pptx(file_path: str) -> Dict[str, Any]:
        """Extract text from PPTX file"""
        prs = Presentation(file_path)
        text = []
        
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text.append(shape.text)
        
        return {
            "success": True,
            "content": "\n\n".join(text).strip(),
            "file_type": "pptx",
            "slide_count": len(prs.slides)
        }
    
    @staticmethod
    def _process_csv(file_path: str) -> Dict[str, Any]:
        """Extract content from CSV file"""
        df = pd.read_csv(file_path)
        
        # Convert dataframe to a readable string format
        csv_content = df.to_string(index=False)
        
        return {
            "success": True,
            "content": csv_content,
            "file_type": "csv",
            "rows": len(df),
            "columns": list(df.columns)
        }
    
    @staticmethod
    def _process_text(file_path: str) -> Dict[str, Any]:
        """Extract text from TXT or MD file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # If it's a markdown file, convert to plain text
        if file_path.lower().endswith('.md'):
            html = markdown.markdown(content)
            soup = BeautifulSoup(html, 'html.parser')
            content = soup.get_text()
        
        return {
            "success": True,
            "content": content.strip(),
            "file_type": "markdown" if file_path.lower().endswith('.md') else "text"
        }
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks
        
        Args:
            text: The text to chunk
            chunk_size: Maximum size of each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of chunks with metadata
        """
        if not text:
            return []
            
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk = text[start:end]
            
            chunks.append({
                "text": chunk,
                "start": start,
                "end": end,
                "length": len(chunk)
            })
            
            if end == text_length:
                break
                
            # Move the start position, but include overlap
            start = end - overlap
        
        return chunks

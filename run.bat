@echo off
title Personal Agentic RAG Chatbot
cd /d "%~dp0"
echo ========================================================
echo   Starting Personal Agentic RAG Chatbot...
echo   URL: http://localhost:8501
echo ========================================================
python -m streamlit run app.py --server.port 8501
pause

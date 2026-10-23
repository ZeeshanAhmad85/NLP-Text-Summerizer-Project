import os
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from src.TextSummarizer.pipeline.prediction import PredictionPipeline
from src.TextSummarizer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.TextSummarizer.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from src.TextSummarizer.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from src.TextSummarizer.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from src.TextSummarizer.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline
from src.TextSummarizer.logging import logger

app = FastAPI(
    title="NLP Text Summarization API",
    description="Abstractive Dialogue & Text Summarization System using Transformer Seq2Seq (PEGASUS)",
    version="1.0.0"
)

# Global lazy pipeline instance
pipeline_instance = None


def get_prediction_pipeline() -> PredictionPipeline:
    global pipeline_instance
    if pipeline_instance is None:
        pipeline_instance = PredictionPipeline()
    return pipeline_instance


class SummarizationRequest(BaseModel):
    text: str = Field(
        ...,
        description="The dialogue or document text to be summarized.",
        min_length=5,
        example="Hannah: Hey, do you have Betty's number? Amanda: Lemme check. Yes, it's 555-1234. Hannah: Thanks!"
    )
    max_length: Optional[int] = Field(128, ge=10, le=512, description="Maximum token length of the summary.")
    min_length: Optional[int] = Field(15, ge=1, le=256, description="Minimum token length of the summary.")
    num_beams: Optional[int] = Field(4, ge=1, le=10, description="Beam search width.")


class SummarizationResponse(BaseModel):
    summary: str
    input_tokens: int
    output_tokens: int
    device: str


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NLP Text Summarizer | Academic Demonstration</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0e17;
            --bg-secondary: #121826;
            --bg-card: rgba(22, 30, 46, 0.7);
            --border: rgba(255, 255, 255, 0.08);
            --border-hover: rgba(59, 130, 246, 0.4);
            --accent: #3b82f6;
            --accent-glow: rgba(59, 130, 246, 0.25);
            --accent-gradient: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --success: #10b981;
            --danger: #ef4444;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            background-image: 
                radial-gradient(circle at 15% 20%, rgba(59, 130, 246, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 85% 80%, rgba(139, 92, 246, 0.08) 0%, transparent 40%);
        }
        header {
            padding: 1.5rem 2rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
            backdrop-filter: blur(12px);
            background: rgba(10, 14, 23, 0.75);
            position: sticky;
            top: 0;
            z-index: 50;
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        .logo-badge {
            width: 36px;
            height: 36px;
            border-radius: 10px;
            background: var(--accent-gradient);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 1.1rem;
            color: white;
            box-shadow: 0 4px 12px var(--accent-glow);
        }
        .brand h1 {
            font-size: 1.15rem;
            font-weight: 600;
            letter-spacing: -0.02em;
        }
        .badge {
            font-size: 0.72rem;
            padding: 0.25rem 0.6rem;
            border-radius: 9999px;
            background: rgba(59, 130, 246, 0.12);
            color: #60a5fa;
            border: 1px solid rgba(59, 130, 246, 0.3);
            font-weight: 500;
        }
        main {
            flex: 1;
            max-width: 1200px;
            width: 100%;
            margin: 0 auto;
            padding: 2.5rem 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 2rem;
        }
        .hero {
            text-align: center;
            max-width: 720px;
            margin: 0 auto;
        }
        .hero h2 {
            font-size: 2.4rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            line-height: 1.2;
            margin-bottom: 0.75rem;
            background: linear-gradient(180deg, #ffffff 0%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hero p {
            color: var(--text-secondary);
            font-size: 1.05rem;
            line-height: 1.6;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }
        @media (max-width: 860px) {
            .grid { grid-template-columns: 1fr; }
        }
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.5rem;
            backdrop-filter: blur(16px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
            display: flex;
            flex-direction: column;
            gap: 1rem;
            transition: border-color 0.2s ease;
        }
        .card:hover {
            border-color: var(--border-hover);
        }
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .card-title {
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        textarea {
            width: 100%;
            min-height: 240px;
            background: rgba(10, 14, 23, 0.6);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1rem;
            color: var(--text-primary);
            font-family: inherit;
            font-size: 0.95rem;
            line-height: 1.6;
            resize: vertical;
            outline: none;
            transition: all 0.2s ease;
        }
        textarea:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }
        .controls {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            align-items: center;
            justify-content: space-between;
        }
        .control-group {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        label {
            font-size: 0.85rem;
            color: var(--text-muted);
            font-weight: 500;
        }
        select, input[type="number"] {
            background: rgba(10, 14, 23, 0.8);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text-primary);
            padding: 0.4rem 0.75rem;
            font-size: 0.85rem;
            outline: none;
        }
        .button-row {
            display: flex;
            gap: 0.75rem;
            margin-top: 0.5rem;
        }
        button.btn-primary {
            background: var(--accent-gradient);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 10px;
            font-weight: 600;
            font-size: 0.95rem;
            cursor: pointer;
            box-shadow: 0 4px 16px var(--accent-glow);
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }
        button.btn-primary:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
        }
        button.btn-primary:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        button.btn-secondary {
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-secondary);
            border: 1px solid var(--border);
            padding: 0.75rem 1rem;
            border-radius: 10px;
            font-weight: 500;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        button.btn-secondary:hover {
            background: rgba(255, 255, 255, 0.1);
            color: var(--text-primary);
        }
        .output-box {
            min-height: 240px;
            background: rgba(10, 14, 23, 0.6);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.25rem;
            line-height: 1.7;
            font-size: 0.95rem;
            color: #e2e8f0;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .output-text {
            white-space: pre-wrap;
        }
        .meta-tags {
            display: flex;
            gap: 0.75rem;
            margin-top: 1rem;
            padding-top: 0.75rem;
            border-top: 1px solid var(--border);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: var(--text-muted);
        }
        .meta-tag {
            background: rgba(255, 255, 255, 0.03);
            padding: 0.2rem 0.5rem;
            border-radius: 6px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        .meta-tag span { color: var(--text-primary); }
        .spinner {
            display: inline-block;
            width: 14px;
            height: 14px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 0.8s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        footer {
            padding: 1.5rem 2rem;
            border-top: 1px solid var(--border);
            text-align: center;
            font-size: 0.85rem;
            color: var(--text-muted);
        }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <div class="logo-badge">TS</div>
            <div>
                <h1>PEGASUS Text Summarizer</h1>
            </div>
            <span class="badge">Seq2Seq Transformer</span>
        </div>
        <div>
            <a href="/docs" target="_blank" style="color: var(--text-secondary); text-decoration: none; font-size: 0.85rem; font-weight: 500;">API Docs &rarr;</a>
        </div>
    </header>

    <main>
        <section class="hero">
            <h2>Abstractive Summarization Pipeline</h2>
            <p>Fine-tuned sequence-to-sequence model designed for multi-turn dialogue comprehension and concise narrative condensation.</p>
        </section>

        <div class="grid">
            <div class="card">
                <div class="card-header">
                    <span class="card-title">Source Input</span>
                    <button class="btn-secondary" id="btnSample" onclick="loadSample()">Load Sample Dialogue</button>
                </div>
                <textarea id="inputText" placeholder="Paste dialogue or narrative text here (minimum 3 words)..."></textarea>
                <div class="controls">
                    <div class="control-group">
                        <label for="maxLength">Max Tokens:</label>
                        <select id="maxLength">
                            <option value="64">Concise (64)</option>
                            <option value="128" selected>Standard (128)</option>
                            <option value="256">Detailed (256)</option>
                        </select>
                    </div>
                    <div class="control-group">
                        <label for="numBeams">Beam Search:</label>
                        <select id="numBeams">
                            <option value="2">2 (Fast)</option>
                            <option value="4" selected>4 (Balanced)</option>
                            <option value="8">8 (High Quality)</option>
                        </select>
                    </div>
                </div>
                <div class="button-row">
                    <button class="btn-primary" id="btnSummarize" onclick="summarizeText()">
                        <span id="btnText">Generate Summary</span>
                    </button>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <span class="card-title">Generated Summary</span>
                    <span id="statusBadge" class="badge" style="display:none;">Ready</span>
                </div>
                <div class="output-box" id="outputBox">
                    <div class="output-text" id="outputText" style="color: var(--text-muted); font-style: italic;">
                        The generated summary will appear here once inference completes...
                    </div>
                    <div class="meta-tags" id="metaSection" style="display:none;">
                        <div class="meta-tag">Input: <span id="metaInTokens">-</span> tok</div>
                        <div class="meta-tag">Summary: <span id="metaOutTokens">-</span> tok</div>
                        <div class="meta-tag">Device: <span id="metaDevice">-</span></div>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <footer>
        <p>NLP Text Summarization System &bull; Designed & Maintained by Zeeshan Ahmad &bull; Evaluated for M.Sc. Graduate Admissions</p>
    </footer>

    <script>
        const sampleText = "Amanda: I baked some chocolate chip cookies today! Anyone wants some?\\nJerry: Oh yes please! Can you save three for me?\\nAmanda: Sure thing! I'll drop them off at your desk after lunch.\\nJerry: You're the best, Amanda! Thanks a lot!";

        function loadSample() {
            document.getElementById('inputText').value = sampleText;
        }

        async function summarizeText() {
            const text = document.getElementById('inputText').value;
            const maxLength = parseInt(document.getElementById('maxLength').value);
            const numBeams = parseInt(document.getElementById('numBeams').value);
            const btn = document.getElementById('btnSummarize');
            const btnText = document.getElementById('btnText');
            const outputText = document.getElementById('outputText');
            const metaSection = document.getElementById('metaSection');

            if (!text || text.trim().split(/\\s+/).length < 3) {
                alert('Please enter at least 3 words to summarize.');
                return;
            }

            btn.disabled = true;
            btnText.innerHTML = '<span class="spinner"></span> Summarizing...';
            outputText.style.color = 'var(--text-muted)';
            outputText.innerText = 'Running sequence-to-sequence beam search inference...';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        text: text,
                        max_length: maxLength,
                        min_length: 10,
                        num_beams: numBeams
                    })
                });

                const data = await response.json();
                if (!response.ok) {
                    throw new Error(data.detail || 'Inference failed');
                }

                outputText.style.color = '#e2e8f0';
                outputText.innerText = data.summary;

                document.getElementById('metaInTokens').innerText = data.input_tokens;
                document.getElementById('metaOutTokens').innerText = data.output_tokens;
                document.getElementById('metaDevice').innerText = data.device.toUpperCase();
                metaSection.style.display = 'flex';

            } catch (err) {
                outputText.style.color = 'var(--danger)';
                outputText.innerText = 'Error: ' + err.message;
            } finally {
                btn.disabled = false;
                btnText.innerText = 'Generate Summary';
            }
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serves the interactive web interface demo."""
    return HTMLResponse(content=HTML_TEMPLATE, status_code=200)


@app.get("/health")
async def health():
    """Health check endpoint providing status and runtime info."""
    import torch
    return {
        "status": "healthy",
        "service": "NLP-Text-Summarizer",
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


@app.post("/predict", response_model=SummarizationResponse)
async def predict_route(request: SummarizationRequest):
    """Executes abstractive text summarization on the provided text."""
    try:
        pipeline = get_prediction_pipeline()
        result = pipeline.predict(
            text=request.text,
            max_length=request.max_length,
            min_length=request.min_length,
            num_beams=request.num_beams
        )
        return SummarizationResponse(**result)
    except (ValueError, TypeError) as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as ex:
        logger.exception(f"Prediction exception: {ex}")
        raise HTTPException(
            status_code=500,
            detail=f"Inference failure: {str(ex)}. Check logs for detailed trace."
        )


@app.post("/train")
async def train_route():
    """Triggers the complete 5-stage training pipeline."""
    try:
        logger.info("Triggering full training pipeline via API endpoint...")
        DataIngestionTrainingPipeline().main()
        DataValidationTrainingPipeline().main()
        DataTransformationTrainingPipeline().main()
        ModelTrainerTrainingPipeline().main()
        scores = ModelEvaluationTrainingPipeline().main()
        return {
            "status": "success",
            "message": "Training and evaluation completed successfully.",
            "metrics": scores
        }
    except Exception as ex:
        logger.exception(f"Training pipeline execution failed: {ex}")
        raise HTTPException(status_code=500, detail=f"Training pipeline error: {str(ex)}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)

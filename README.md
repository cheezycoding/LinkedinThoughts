# LinkedIn Thoughts 💭

AI-powered satirical LinkedIn post generator. Create r/LinkedInLunatics-worthy content with a custom-trained MLA (Multi-Head Latent Attention) transformer.

## 🚀 Live Demo

- **Frontend:** https://linkedin-thoughts.vercel.app
- **Backend API:** https://linkedin-lunatics-708213822442.asia-southeast1.run.app

## 🏗️ Architecture

```
├── frontend/          # Next.js app (Vercel)
│   └── src/app/       # React components
└── backend/           # FastAPI + PyTorch (Cloud Run)
    ├── api.py         # REST API
    ├── model_mla.py   # LuhGPT model definition
    ├── mla_v2.py      # Multi-Head Latent Attention
    └── Dockerfile     # Container config
```

## 🧠 Model Details

- **Architecture:** DeepSeek-style MLA Transformer
- **Parameters:** ~114M
- **Training Data:** Satirical LinkedIn posts
- **Features:**
  - Multi-Head Latent Attention (KV compression)
  - Weight tying (embedding ↔ lm_head)
  - Pre-norm architecture
  - RoPE positional encoding

## 🛠️ Local Development

### Backend
```bash
cd backend
pip install -r requirements.txt
# Download model weights to backend/linkedin_mla_best.pt
python api.py
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## 📝 API Usage

```bash
curl -X POST https://linkedin-lunatics-708213822442.asia-southeast1.run.app/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "I just fired", "temperature": 0.3, "max_tokens": 100}'
```

## ⚠️ Disclaimer

This is satire. Please don't actually post AI-generated content on LinkedIn. Greg Hustleworth III is not a real person.

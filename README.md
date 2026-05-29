# Review-Consultant-AI

> GPT-5.4-mini 기반 AI 이슈 진단 & 실행 처방전 대시보드

## 구성

```
Review-Consultant-AI/
├── backend/
│   ├── api_server.py       ← Flask 백엔드 (OpenAI API 호출)
│   ├── requirements.txt
│   ├── .env.example       ← 환경변수 템플릿 (복사 후 .env로 사용)
│   └── shoes_data/
│       ├── llm_scores_by_product.json
│       └── llm_sentences.json
│
├── frontend/
│   ├── index.html           ← React 프론트엔드 (단일 파일)
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── main.js
│       ├── App.js
│       └── App.css
│
├── .gitignore
└── README.md
```

## 시작하기

### 1. 환경변수 설정

```bash
# .env.example을 복사하여 .env 파일 생성
copy .env.example .env
```

`.env` 파일을 열어 본인의 API 키를 입력하세요.

```
OPENAI_API_KEY=sk-여기에_본인의_API_키를_입력하세요
OPENAI_MODEL=gpt-5.4-mini
```

> ⚠️ `.env` 파일은 `.gitignore`에 포함되어 있어 GitHub에 업로드되지 않습니다.

### 2. 패키지 설치

```bash
pip install flask flask-cors openai python-dotenv
```

### 3. 백엔드 서버 실행

```bash
python api_server.py
```

### 4. 프론트엔드 실행

`index.html` 파일을 브라우저로 열면 됩니다.

```bash
# 또는 간단한 HTTP 서버로 실행
python -m http.server 3000
# 브라우저에서 http://localhost:3000 접속
```

## 기능

- **전체 개요 분석**: 모든 속성을 종합하여 가장 시급한 문제 + 우선순위별 개선 방향
- **속성별 상세 진단**: 선택한 속성의 부정 리뷰 기반 🔍 진단 → 💊 처방전 출력
- 토큰 사용량 및 예상 비용 표시

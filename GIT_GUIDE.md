# 🌿 GitHub 팀 협업 가이드

> 뉴런스 팀 | 리뷰 기반 상품 분석 인사이트 대시보드

---

## 📦 레포지토리 구성

우리 팀은 기능별로 **4개의 레포지토리**를 운영합니다.

| 레포지토리 | 기능 설명 |
|---|---|
| `Review-Aspect-Scoreboard` | 다차원 속성별 만족도 스코어보드 |
| `Review-Sum-Map` | 시맨틱 리뷰 요약 및 키워드 맵 |
| `Review-Risk-Radar` | 부정 리스크 감지 및 조기 경보 |
| `Review-Consultant-AI` | AI 이슈 진단 및 실행 처방전 |

---

## 📁 브랜치 전략 (Branch Strategy)

우리 팀은 **GitHub Flow** 방식을 사용합니다.  
`main` 브랜치는 항상 배포 가능한 상태를 유지하고, 작업은 별도 브랜치에서 진행합니다.  
**각 레포지토리마다 동일한 브랜치 전략을 적용합니다.**

### 브랜치 구조 (레포지토리별 공통)

```
main
├── dev                          ← 통합 개발 브랜치 (PR 대상)
│   ├── feat/기능명/이름         ← 기능 개발 (개인 작업)
│   └── docs/문서명              ← 문서 작업
```

브랜치명은 `feat/기능명/본인이름` 형식으로 만듭니다.  
누가 어떤 작업을 하고 있는지 브랜치명만 봐도 알 수 있어요.

```bash
# 예시
feat/keyword-map/chaewon
feat/radar-chart/junhyung
feat/alert-logic/jiwung
```

| 브랜치 | 용도 | 비고 |
|--------|------|------|
| `main` | 최종 배포용 | 직접 push ❌, PR만 허용 |
| `dev` | 기능 통합 브랜치 | 모든 PR은 여기로 |
| `feat/기능명/이름` | 새 기능 개발 | 개인 작업 브랜치 |
| `docs/문서명` | 문서 작업 | |
| `refactor/내용` | 코드 리팩토링 | |

---

## 🌱 브랜치 생성 & 작업 흐름

### 1단계 — 최신 `dev` 브랜치를 받아온다

```bash
git checkout dev
git pull origin dev
```

### 2단계 — 내 작업 브랜치를 만든다

```bash
git checkout -b feat/attribute-extraction/chaewon
# 형식: feat/기능명/본인이름
```

### 3단계 — 작업 후 커밋

```bash
git add .
git commit -m "feat: 신발 속성(착용감·사이즈·디자인) 추출 로직 구현"
```

### 4단계 — 원격 저장소에 push

```bash
git push origin feat/attribute-extraction/chaewon
```

### 5단계 — GitHub에서 Pull Request (PR) 생성

- **base:** `dev` ← **compare:** `feat/attribute-extraction/chaewon`
- PR 제목, 설명 작성 후 팀원에게 리뷰 요청
- 리뷰 후 `dev`에 Merge

---

## ✏️ 커밋 메시지 규칙

### 형식

```
타입: 변경 내용 요약 (50자 이내)

(선택) 상세 설명
```

### 커밋 타입

| 타입 | 설명 | 예시 |
|------|------|------|
| `feat` | 새로운 기능 추가 | `feat: 속성별 만족도 스코어 집계 로직 구현` |
| `fix` | 버그 수정 | `fix: 부정 키워드 빈도 계산 오류 수정` |
| `docs` | 문서 수정 | `docs: README 파이프라인 구조 업데이트` |
| `refactor` | 리팩토링 (기능 변경 없음) | `refactor: 리뷰 속성 추출 함수 모듈화` |
| `style` | 코드 포맷, 세미콜론 등 | `style: 불필요한 공백 제거` |
| `test` | 테스트 코드 추가/수정 | `test: 스코어보드 집계 단위 테스트 추가` |
| `chore` | 빌드, 패키지 관련 | `chore: requirements.txt 업데이트` |
| `data` | 데이터셋, 전처리 관련 | `data: Amazon 신발 리뷰 샘플 데이터 추가` |

### ✅ 좋은 커밋 예시

```
feat: 다차원 속성별 만족도 레이더 차트 구현

- 착용감·디자인·사이즈·내구성·가격 5개 속성 시각화
- Streamlit에서 plotly 레이더 차트 렌더링
- 브랜드별 필터링 기능 연동
```

```
fix: 리뷰 텍스트 없는 데이터 처리 시 NoneType 오류 수정
```

### ❌ 피해야 할 커밋 예시

```
수정함          ← 뭘 수정했는지 모름
asdf            ← 의미 없는 메시지
작업중          ← 상태가 아닌 변경 내용을 써야 함
일단 올림       ← 금지
```

---

## 🔄 PR (Pull Request) 규칙

### PR 제목
커밋 메시지와 동일한 형식 사용
```
feat: LLM 기반 리뷰 속성 감성 분류 모듈 구현
```

### PR 설명 템플릿
```markdown
## 변경 사항
- 무엇을 구현/수정했는지 간략히

## 관련 이슈
- closes #이슈번호 (해당되면)

## 테스트 방법
- 어떻게 확인하면 되는지

## 스크린샷 (선택)
```

### 리뷰 규칙
- PR은 **최소 1명 이상** 리뷰 후 Merge
- Merge는 작업자 본인이 직접 (리뷰 승인 후)
- `main` 브랜치로의 Merge는 팀장이 최종 확인

---

## ⚠️ 주의사항

```
❌ git push origin main    ← main에 직접 push 금지
❌ 하나의 커밋에 여러 기능  ← 기능 단위로 쪼개기
❌ 리뷰 없이 본인 PR Merge ← 반드시 리뷰 먼저
✅ 작업 시작 전 항상 git pull origin dev 먼저!
```

---

## 🆘 자주 쓰는 Git 명령어 모음

```bash
# 현재 브랜치 및 상태 확인
git status
git branch

# 원격 최신 변경사항 가져오기
git pull origin dev

# 스테이징 & 커밋
git add 파일명        # 특정 파일만
git add .             # 전체
git commit -m "메시지"

# 브랜치 이동
git checkout 브랜치명
git checkout -b 새브랜치명   # 생성 + 이동

# 로그 확인
git log --oneline --graph

# 충돌(conflict) 발생 시
# → 파일 열어서 직접 수정 → git add → git commit
```

---

## 📌 한눈에 보는 작업 흐름

```
[dev pull] → [브랜치 생성] → [작업] → [커밋] → [push] → [PR 생성] → [리뷰] → [dev merge]
```

---

> 궁금한 점은 팀 단톡방에 올려주세요 😊  
> 마지막 업데이트: 2026.05

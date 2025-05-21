# Task Description

- 올거나이즈 RAG-Evaluation-Dataset-Ko 데이터셋을 활용해서 RAG 시스템 구축하기
  - https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO

- 사용자가 질문할 때, 5개의 도메인 중 하나를 선택해서 질문한다고 가정
  - finance, public, medical, law, commerce

- 사용자의 질문에 대한 답을 관련 문서에 기반해서 생성하기
## WorkFlow

- 도메인별 PDF 문서 로딩 및 전처리

- 문서 분할 (Chunking)
- KoE5 임베딩 및 Chroma 벡터스토어 생성
- Retriever로 유사 문서 검색 (Top-10)
- Re-ranker로 정확도 높은 문서 재정렬 (Top-6)
- Prompt 구성
- GPT-4o-mini로 정답 생성
- 결과 저장 및 평가

# 데이터 분석 및 필요한 PDF 다운로드  
- 사용된 pdf만 다운로드 했습니다.
  - 총 63개 중 53개 사용
# RAG 시스템 구축

- 프레임워크: LangChain

## PDF로부터 Domain 별 vector DB & Retriever 만들기
- PDF 로더 : PyPDFLoader -> PyMuPDFLoader
  - 한국어에 강하고 처리 속도 빠름
  - 특수문자 인코딩에 강함
  - 파싱 및 page 별 분할: loader.load_and_split()
- 청킹: MarkdownTextSplitter, #RecursiveCharacterTextSplitter(그냥 길이 기준 분할)
  - 마크다운 문법 인식 및 문서 구조를 인식하여 자연스럽게 분할
- vector store: Chroma
- embedding model: KoE5
  - https://huggingface.co/nlpai-lab/KoE5

### Retriever
- search_type: 벡터 유사도 기반
- top_k: 10개
  - 몇개 찾을건지 k와 chunk 사이즈에 따른 실험을 통해 정해야됨
- 참고: search_type 설정에 bm25가 없어서, 하이브리드 서치 하려면 직접 bm25 구현해서 가중치 주거나 혹은, 다른 솔루션(Elastic search 등) 사용

### Reranker
- reranker : Dongjin-kr/ko-reranker
  - https://huggingface.co/Dongjin-kr/ko-reranker
- reranked_k : 5개

## 답변 생성하기
- LLM : Chatgpt-4o-mini
- 테스트 데이터셋: 도메인 당 10개
- 생성 prompt : rlm/ rag-prompt

# 평가

## 1.ragas score:
- https://docs.ragas.io/en/v0.1.21/concepts/metrics/index.html
- ![image](https://github.com/user-attachments/assets/b2f49f60-e24a-4bea-94ce-635b1b2badb0)
  - default llm: gpt-3.5-turbo

- answer_relevancy, # 생성 평가: 질문과 답변의 관계정도
- faithfulness, # 생성 평가: 답변이 얼마나 Context에 근거한 정확한 답변인지
- context_recall, # 검색 평가: 정답을 n개의 문장으로 쪼개고, 각 문장이 retrieved context에 있는지 확인
- context_precision # 검색 평가: 검색된 k개의 context 중에 정답(ground truth 만드는 데 도움이 되는지 확인) 비율

## 2.allganize 자동평가:
- https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO 의 Auto Evaluate Colab 참조
- 총 4개의 LLM Eval을 사용하여 평가한 후, voting 하여 "O" 혹은 "X"를 결정했습니다.
- llm_evaluate() 함수
  - 밑의 4가지 평가 결과를 종합
  - O (정답) / X (오답)으로 통합 판단
  1. TonicAI : answer_similarity (threshold=4)
    - OpenAI 모델을 사용하여 생성된 답변과 기준 정답의 유사도를 0~5 점수로 출력
    - 평가 모델은 openai:/gpt-4o-mini 사용
  2. MLflow : answer_similarity/v1/score (threshold=4)
    - 0~5 스칼라 출력인듯
    - 평가 모델은 openai:/gpt-4o-mini 사용
  3. MLflow : answer_correctness/v1/score (threshold=4)
    - 0~5 스칼라 출력 인듯
    - 평가 모델은 openai:/gpt-4o-mini 사용
  4. Allganize Eval : answer_correctness/claude3-opus
    - 생성된 답변의 정확성을 기준 답변과 비교하여 검증
      - 맞으면 1, 틀리면 0 출력
    - 평가 모델은 "claude-3-opus-20240229" 사용

# 개선할 수 있는 부분
## pre-filtering(검색 풀 미리 줄이기)
- 실험 결과 효과 있음
  - BERT 같은 Classifier
    - 데이터 증강 필요  
  - 혹은 LLM 활용가능
    - 문서에 대한 메타 데이터 및 설명 or 요약

## 파싱 및 청킹
- PDF 파서
  - ERROR:pypdf._cmap:Advanced encoding /KSCpc-EUC-H not implemented yet 해결 위해 다른 파서 사용
    - PyMuPDFLoader로 대체 했음

- 텍스트 분할기 실험
  - Content-Aware Splitting이 검색 성능 더 좋아서 대체함.
- **Contextual chunking**
 - **문서 hierarchy 기반으로 contextual를 추가한 chunk 구성**
- chunk 사이즈 및 top k 실험
  - 1. LLM 선정
  - 2. chunk 사이즈와 LLM의 Context Length 에 따라 k를 정하기
  - 3. retriever의 top-k recall 점수 측정
- 이미지랑 표 처리하는 기법들 


## retriever

- 임베딩 모델 실험
- reranker 모델 실험
- 임베딩, reranker 학습

#### 검색방법
- bm25 추가 및 하이브리드 서치 방법
  - chroma 에서 지원 안 하니까, FAISS +BM25Retriever

- 검색 방법 개선:
  - HyDE
    - https://aclanthology.org/2023.acl-long.99/
    - 1. **사용자의 질문 입력**: 사용자가 특정 질문을 입력한다.
      2. **가상 문서 생성**: 질문을 기반으로 LLM을 사용하여 관련성이 높은 가상의 문서를 생성한다.
      3. **임베딩 생성**: 생성된 가상 문서를 임베딩 모델로 벡터화한다.
      4. **유사도 비교 및 검색**: 생성된 벡터와 데이터베이스 문서의 벡터 간 유사도를 계산하여 가장 관련성이 높은 문서를 반환한다

  - HyQE
    - https://arxiv.org/abs/2410.15262

## generation
- 모델 선정
- prompt engineering

## 비용 측면
- prompt caching : 반복적으로 사용되는 프롬프트를 서버에 저장하여, 이후 동일한 프롬프트가 다시 들어왔을 때 이를 빠르게 처리하는 기능
  - https://www.anthropic.com/news/contextual-retrieval





---
applyTo: '**'
---
## 목표

- 모든 HTTP API는 **OpenAPI 3.1** 스펙으로 계약(Contract)을 명시하고, **JSON Schema 2020-12**로 데이터 구조를 정의한다. OpenAPI 3.1은 2020-12를 공식적으로 호환한다. [spec.openapis.org+2openapis.org+2](https://spec.openapis.org/oas/v3.1.0.html?utm_source=chatgpt.com)
- 에러 응답은 **Problem Details(RFC 9457)** 형식( `application/problem+json`)으로 표준화한다(기존 에러 명령문과 합치). [rfc-editor.org](https://www.rfc-editor.org/rfc/rfc9457.html?utm_source=chatgpt.com)
- 문서는 **Docs-as-Code** 방식(버전관리/리뷰/자동검증/빌드)을 따른다. [Write the Docs](https://www.writethedocs.org/guide/docs-as-code.html?utm_source=chatgpt.com)

## 범위

- 서버의 모든 공개/내부 API, 데이터 구조(class/DTO/스키마), 오류 스키마, 변경 이력.

## 폴더 구조(레포 내)

```
/docs
  /api
    openapi.yaml            # 단일 진실 원천(Single Source of Truth)
    /examples               # 요청/응답 예시(JSON)
    /style                  # API 스타일가이드(네이밍/상태코드/버전 규칙)
  /schemas                  # JSON Schema 2020-12 정의(공유 모델)
    problem.json            # RFC 9457 호환 에러 스키마
    conversation.json
    message.json
    graph-node.json
    graph-edge.json
  /guides                   # How-to (동기화 흐름, OAuth 플로우 등)
  /reference                # 엔드포인트/모델 레퍼런스(MD, OAS에서 생성해도 됨)
  /adr                      # 아키텍처 결정 기록(ADR)
  CHANGELOG.md              # Keep a Changelog 포맷

```

- Notion은 요약/튜토리얼을 복제하되, **계약·스키마의 원본은 레포**가 기준이다.
- 문서 유형은 **Diátaxis**(Tutorials/How-to/Reference/Explanation) 구분을 따른다. [diataxis.fr](https://diataxis.fr/?utm_source=chatgpt.com)

## 필수 규칙

### 1) OpenAPI 3.1 (계약 우선)

- 모든 엔드포인트/메서드/상태코드/헤더/쿼리/바디/응답을 **`/docs/api/openapi.yaml`** 에 정의한다.
- 공통 스키마는 **`components/schemas/*`** 에 두고, 가능한 한 **JSON Schema** `$ref` 를 사용한다. (3.1은 2020-12 스펙을 직접 사용) [openapis.org](https://www.openapis.org/blog/2021/02/18/openapi-specification-3-1-released?utm_source=chatgpt.com)
- 에러 응답은 `application/problem+json` 을 **기본 오류 미디어타입**으로 선언한다. [rfc-editor.org](https://www.rfc-editor.org/rfc/rfc9457.html?utm_source=chatgpt.com)

### 2) JSON Schema 2020-12 (데이터 구조)

- `/docs/schemas/*.json` 은 모델 단위로 작성하며, **$id**(고유 URI)와 `$schema` 를 명시한다.
- 스키마는 예시(`examples`)와 함께 제공하고, 릴리스마다 **변경 이력**을 남긴다(아래 변경 관리 참조). [json-schema.org](https://json-schema.org/specification?utm_source=chatgpt.com)

### 3) 표준 에러(Problem Details)

- `/docs/schemas/problem.json` 으로 RFC 9457 호환 스키마 정의(`type,title,status,detail,instance` + 확장 필드).
- OpenAPI 모든 에러 응답은 이 스키마를 `$ref` 로 사용한다. [rfc-editor.org](https://www.rfc-editor.org/rfc/rfc9457.html?utm_source=chatgpt.com)

### 4) 스타일·일관성

- 경로/리소스/쿼리 네이밍, 상태코드 매핑, 페이징/정렬 규약 등 **스타일가이드**를 `/docs/api/style/*` 에 문서화하고, OpenAPI에 반영한다.
- 페이로드 예시는 **실제 가능한 값**으로 최신화한다.

### 5) 자동 검증 & 미리보기

- **Spectral** 룰셋으로 OpenAPI/JSON Schema를 린트하고 CI에서 강제한다(네이밍/필수필드/상태코드 규칙). [stoplight.io+1](https://stoplight.io/open-source/spectral?utm_source=chatgpt.com)
- 로컬/CI에서 **문서 미리보기**(Swagger UI/Redoc 등)를 제공한다. [Swagger+1](https://swagger.io/specification/?utm_source=chatgpt.com)

### 6) 변경 관리

- 파괴적 변경은 **/v{n} 상승**(경로 버전) 후 병행 기간 운영(REST 명령문과 합치).
- 모든 변경은 **CHANGELOG.md** 를 **Keep a Changelog** 형식으로 기록하고, SemVer를 명시한다. [keepachangelog.com](https://keepachangelog.com/en/1.1.0/?utm_source=chatgpt.com)

## 작성 지침(요약 스니펫)

### (A) OpenAPI 3.1 기본 뼈대

```yaml
openapi: 3.1.0
info:
  title: GraphNode API
  version: 1.0.0
jsonSchemaDialect: https://json-schema.org/draft/2020-12/schema
paths:
  /v1/conversations:
    post:
      summary: Create conversation
      requestBody:
        required: true
        content:
          application/json:
            schema: { $ref: "../schemas/conversation.json" }
      responses:
        "201":
          description: Created
          headers:
            Location: { description: Resource URL, schema: { type: string } }
          content:
            application/json:
              schema: { $ref: "../schemas/conversation.json" }
        "400":
          description: Bad Request
          content:
            application/problem+json:
              schema: { $ref: "../schemas/problem.json" }

```

### (B) Problem Details 스키마(요약)

```json
{
  "$id": "https://graphnode.dev/schemas/problem.json",
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": ["type", "title", "status", "detail", "instance"],
  "properties": {
    "type": { "type": "string", "format": "uri" },
    "title": { "type": "string" },
    "status": { "type": "integer" },
    "detail": { "type": "string" },
    "instance": { "type": "string" },
    "correlationId": { "type": "string" },
    "errors": { "type": "array", "items": { "type": "object" } }
  },
  "additionalProperties": true}

```

## 도구/자동화

- **NPM Scripts**
    - `docs:lint` — Spectral로 `openapi.yaml`/`schemas/*.json` 린트. [docs.stoplight.io](https://docs.stoplight.io/docs/spectral/9ffa04e052cc1-spectral-cli?utm_source=chatgpt.com)
    - `docs:build` — Redoc/Swagger UI로 정적 문서 생성. [Swagger+1](https://swagger.io/specification/?utm_source=chatgpt.com)
    - `docs:preview` — 로컬 프리뷰 서버.
- **CI 게이트**
    - 린트 통과 없이는 머지 금지(Spectral).
    - OAS/Schema JSON 스키마 검증.
    - 예시 payload 유효성 테스트(샘플 호출에 스키마 적용).

### 퍼블리싱(GitHub Pages) 및 외부 공유(Plan A: Notion 임베드)

- 퍼블리싱 대상: `/docs` 폴더(포털 `docs/index.html`, OpenAPI HTML, TypeDoc, 가이드/스키마 링크).
- 워크플로우: `.github/workflows/docs-pages.yml` — `main` 푸시 시 `npm ci` → `npm run docs:build` → Pages 배포.
- 산출 URL: GitHub Actions `deploy-pages` 출력의 `page_url` (예: `https://<org>.github.io/<repo>/`).
- Notion(Plan A): 포털/Redoc/TypeDoc URL을 Notion 페이지에 Embed 블록으로 1회 추가. 이후 CI 배포만으로 항상 최신 상태 노출.

### 문서 포털(index.html) 운영 규칙

- `docs/index.html`은 수동 유지되는 문서 허브(포털)이며, **매 Day 작업 종료 시 반드시 갱신**한다.
- 포털은 `docs/` 하위의 **모든 문서 파일(Markdown/HTML 등)** 을 **직접 또는 간접 링크**로 탐색 가능하게 참조해야 한다(신규/변경 문서 누락 금지).
- 주요 고정 섹션 권장: API(OpenAPI), 타입 레퍼런스(TypeDoc), Guides 인덱스, Schemas, CHANGELOG, 추가 스타일가이드/ADR 링크.
- 경로/파일명 변경 시 포털 링크도 즉시 갱신한다.


---
applyTo: '**'
---
# Day별 개발 일지(Daily Dev Log) 작성 및 만든 API와 type, DTO들에 대한 문서화 작성명령문

목표
- 매 Day 작업 완료 시, 누구나 같은 절차로 변경 내용을 이해·재현 가능하도록 **표준 템플릿 문서**를 남긴다. 

API 및 type 문서 작성 항목




Day별 개발 일지 항목
위치/명명
- 위치: `docs/guides/`
- 파일명: `DAY<n>-<주제-케밥>.md` (예: `DAY3-db-connection.md`)
- 템플릿: `docs/guides/templates/DAYn-devlog-template.md`

필수 포함 섹션
- TL;DR(목표/결과/영향)
- 산출물(추가/수정/삭제 파일)
- 메서드/클래스 변경 상세(시그니처/예외/로깅)
- 실행/온보딩(사전 조건, 명령어, 검증 절차)
- 구성/가정/제약(DB·ENV·보안 전제)
- 리스크/부채/트러블슈팅
- 다음 Day 목표/후속 작업
- 참고/링크, 변경 이력

프로세스(매 Day 종료 시)
1) 템플릿 복사 → 파일 생성
   - Windows PowerShell:
     - `Copy-Item docs\guides\templates\DAYn-devlog-template.md docs\guides\DAY<n>-<주제>.md`
2) 내용 채우기(빈 섹션 금지, N/A 허용)
3) 필요한 경우 OpenAPI/Schema/ADR/CHANGELOG **및 `docs/index.html`(문서 포털)** 도 갱신(신규/변경 문서 링크 반영)
4) PR 시 체크리스트
   - [ ] 본 Day 일지 파일 경로를 PR 설명에 링크
   - [ ] 에러 응답은 Problem Details 규격 확인
   - [ ] 온보딩/명령어가 최신(재현 가능)

승인 기준(AC)
- `docs/guides/DAY<n>-*.md` 존재, 필수 섹션 채움
- 재현 절차로 로컬에서 서버기동 및 검증 가능(`/healthz`, 404 Problem Details)
- 변경된 공개 API는 `/docs/api/openapi.yaml` 업데이트/린트 통과
- DB/스키마 변경 시 마이그레이션/인덱스 절차 명시
 - `docs/index.html` 포털이 **docs/ 하위의 모든 문서(MD/HTML)** 를 참조(직/간접 링크)하고, 당일 변경분(추가/수정/삭제)이 반영됨(누락 0건)

주의
- 시크릿/토큰은 문서·로그에 노출 금지
- 레이어 규칙 및 에러/로깅 표준은 기존 명령문과 일치해야 함


## 승인 기준(AC)

- **[정적]** `/docs/api/openapi.yaml` 이 존재하고, 모든 공개 API가 등재되어 있으며 **Spectral 린트 0 경고**. [stoplight.io](https://stoplight.io/open-source/spectral?utm_source=chatgpt.com)
- **[정적]** `/docs/schemas/*.json` 이 JSON Schema 2020-12로 검증 통과(`$schema`/`$id` 포함). [json-schema.org](https://json-schema.org/draft/2020-12?utm_source=chatgpt.com)
- **[런타임]** 에러 응답이 전부 `application/problem+json` 스키마에 부합한다(자동 테스트). [rfc-editor.org](https://www.rfc-editor.org/rfc/rfc9457.html?utm_source=chatgpt.com)
- **[문서]** CHANGELOG가 Keep a Changelog 포맷으로 최신 상태이며 릴리스마다 갱신. [keepachangelog.com](https://keepachangelog.com/en/1.1.0/?utm_source=chatgpt.com)
- **[구성]** 문서 타입(튜토리얼/하우투/레퍼런스/설명)이 **Diátaxis** 구분으로 정리되어 탐색 가능. [diataxis.fr](https://diataxis.fr/?utm_source=chatgpt.com)


---

## API·타입 레퍼런스 생성(보강)

- TypeDoc으로 코드 주석(JSDoc)을 HTML/MD 참조로 산출:
  - 출력 위치: `docs/reference/api`
  - 소스 링크: 각 심볼에 GitHub 소스 링크 표시
  - 공개 범위: `@public` 만 문서화, `@internal`은 제외
- NPM Scripts(권장)
  - `docs:typedoc` — TypeDoc 실행
  - `docs:preview` — Swagger UI/Redoc + TypeDoc 정적 미리보기
- 네비게이션
  - Reference는 “코드 심볼(함수/타입/클래스)” 중심, OpenAPI/Schema는 “계약/데이터 모델” 중심
  - Guide(How-to) 문서와 교차 링크: 컨트롤러/서비스 설명에서 관련 심볼/스키마로 연결

## 문서화 범주·책임(보강, Diátaxis와 합치)

- Tutorials(튜토리얼): 빠른 시작/온보딩/환경설정
- How-to(가이드): OAuth 플로우, 동기화 절차, DB 마이그레이션
- Reference(참조): OpenAPI, JSON Schema, TypeDoc(API/타입)
- Explanation(설명): 아키텍처 결정(ADR), 에러 코드 레지스트리 배경
- 변경 시 원칙
  - 엔드포인트/스키마 변경 → OpenAPI/Schema + 예시 업데이트
  - 코드 심볼 변경 → TypeDoc 재생성(공개 API 변경 시 CHANGELOG 항목 추가)
  - 에러 코드 추가 → 에러 레지스트리 + Problem Details 스키마/예시 반영

## 예시/샘플 데이터(보강)

- 모든 엔드포인트는 최소 1개 이상의 요청/응답 예시(JSON)를 `/docs/api/examples`에 둔다.
- 에러 응답 예시는 `application/problem+json` 형식으로, `correlationId` 포함.
- JSON Schema 예시(`examples`)는 실제 유효한 값으로 Spectral/Ajv 검증 통과.

## 교차 참조 정책(보강)

- OpenAPI → Schema: `$ref` 우선
- OpenAPI/Schema → TypeDoc: description 또는 externalDocs로 참조 링크 추가
- Guides → OpenAPI/TypeDoc: “참고” 섹션에 직접 링크
- Error Registry → RFC 9457 스키마 → 중앙 에러 핸들러 흐름으로 연결 고리 유지

## PR 체크리스트(추가)

- [ ] OpenAPI 변경사항 반영 및 Spectral 린트 통과
- [ ] JSON Schema `$id/$schema`·예시 추가/수정
- [ ] 공개 API/JSDoc 갱신(TypeDoc 재생성)
- [ ] Problem Details 응답 예시 최신화
- [ ] Guides/README 온보딩 명령 최신화

## 승인 기준(AC, 보강)

- [정적] TypeDoc 빌드 경고 0건, `docs/reference/api`가 최신화.
- [정적] 공개 export 심볼 JSDoc 커버리지 100%(@public 기준).
- [계약] OpenAPI/Schema/Spectral 통과 + 예시 유효성(Ajv) 통과.
- [상호참조] Guides ↔ Reference ↔ OpenAPI 간 최소 1개 이상의 교차링크 보유.

```// filepath: c:\Users\KANG\Desktop\BIT_Uni_record\TACO 4\TeamProject\GraphNode\GraphNode_BE\.github\instructions\Documentation.instructions.md
// ...existing code...

---

## API·타입 레퍼런스 생성(보강)

- TypeDoc으로 코드 주석(JSDoc)을 HTML/MD 참조로 산출:
  - 출력 위치: `docs/reference/api`
  - 소스 링크: 각 심볼에 GitHub 소스 링크 표시
  - 공개 범위: `@public` 만 문서화, `@internal`은 제외
- NPM Scripts(권장)
  - `docs:typedoc` — TypeDoc 실행
  - `docs:preview` — Swagger UI/Redoc + TypeDoc 정적 미리보기
- 네비게이션
  - Reference는 “코드 심볼(함수/타입/클래스)” 중심, OpenAPI/Schema는 “계약/데이터 모델” 중심
  - Guide(How-to) 문서와 교차 링크: 컨트롤러/서비스 설명에서 관련 심볼/스키마로 연결

## 문서화 범주·책임(보강, Diátaxis와 합치)

- Tutorials(튜토리얼): 빠른 시작/온보딩/환경설정
- How-to(가이드): OAuth 플로우, 동기화 절차, DB 마이그레이션
- Reference(참조): OpenAPI, JSON Schema, TypeDoc(API/타입)
- Explanation(설명): 아키텍처 결정(ADR), 에러 코드 레지스트리 배경
- 변경 시 원칙
  - 엔드포인트/스키마 변경 → OpenAPI/Schema + 예시 업데이트
  - 코드 심볼 변경 → TypeDoc 재생성(공개 API 변경 시 CHANGELOG 항목 추가)
  - 에러 코드 추가 → 에러 레지스트리 + Problem Details 스키마/예시 반영

## 예시/샘플 데이터(보강)

- 모든 엔드포인트는 최소 1개 이상의 요청/응답 예시(JSON)를 `/docs/api/examples`에 둔다.
- 에러 응답 예시는 `application/problem+json` 형식으로, `correlationId` 포함.
- JSON Schema 예시(`examples`)는 실제 유효한 값으로 Spectral/Ajv 검증 통과.

## 교차 참조 정책(보강)

- OpenAPI → Schema: `$ref` 우선
- OpenAPI/Schema → TypeDoc: description 또는 externalDocs로 참조 링크 추가
- Guides → OpenAPI/TypeDoc: “참고” 섹션에 직접 링크
- Error Registry → RFC 9457 스키마 → 중앙 에러 핸들러 흐름으로 연결 고리 유지

## PR 체크리스트(추가)

- [ ] OpenAPI 변경사항 반영 및 Spectral 린트 통과
- [ ] JSON Schema `$id/$schema`·예시 추가/수정
- [ ] 공개 API/JSDoc 갱신(TypeDoc 재생성)
- [ ] Problem Details 응답 예시 최신화
- [ ] Guides/README 온보딩 명령 최신화

## 승인 기준(AC, 보강)

- [정적] TypeDoc 빌드 경고 0건, `docs/reference/api`가 최신화.
- [정적] 공개 export 심볼 JSDoc 커버리지 100%(@public 기준).
- [계약] OpenAPI/Schema/Spectral 통과 + 예시 유효성(Ajv) 통과.
- [상호참조] Guides ↔ Reference ↔ OpenAPI 간 최소 1개 이상의 교차링크 보유.
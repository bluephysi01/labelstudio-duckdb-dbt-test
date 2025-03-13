# Label Studio DuckDB DBT Test

## 프로젝트 개요
- **목적**: Label Studio에서 YOLO 모델로 사전 어노테이션을 진행하고, 어노테이션 결과를 로컬에 저장/불러오는 과정을 사전 테스트 합니다.
- **핵심 라이브러리**:  
  - [Label Studio SDK](https://github.com/heartexlabs/label-studio-sdk)  
  - [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)  

## 폴더 구조
```bash
labelstudio-duckdb-dbt-test
├── archive
├── data
│   ├── export
│   ├── samples_generalpictures
│   ├── samples_tires
│   ├── tasks_generalpictures
│   └── tasks_tires
├── ls_ui_snaps
├── .env
├── .gitignore
├── create_tasks.ipynb
├── ls_sdk_preannotation.ipynb
├── requirements.txt
├── README.md
```
## 주요 파일 및 폴더 설명
- `archive`: 테스트과정의 파일들이 있는 폴더
- `data`: Label Studio용 태스크 및 내보내기/가져오기 파일들
- `export`: 내보낸 어노테이션 결과 저장 폴더
- `samples_generalpictures`, `samples_tires`: 샘플 이미지 저장 폴더
- `tasks_generalpictures`, `tasks_tires`: Label Studio에서 사용될 태스크 JSON 
- `ls_ui_snaps`: UI 스냅샷(예: Label Studio 인터페이스 캡처)
- `.env`: Label Studio API 키 등 환경 변수를 저장
- `create_tasks.ipynb`: 태스크를 생성하기 위한 Jupyter 노트북
- `ls_sdk_preannotation.ipynb`: Label Studio SDK를 활용해 사전 어노테이션을 수행하는 노트북
- `requirements.txt`: 필요한 Python 패키지 목록

## Label Studio 설정 방법
1. Label Studio 실행
   ```sh
   label-studio start
2. 환경 변수 설정(Windows 기준)
   ```sh
   setx LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED "true" /M
   setx LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT "C:\Users\82102\dev\labelstudio-duckdb-dbt-test" /M

3. 터미널에서 라벨스튜디오를 실행하고 `ls_sdk_preannotation.ipynb` 파일을 열어 셀을 실행합니다.

## 참고사항
- JSON 파일의 경로는 LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT 환경 변수와 일치해야 합니다.
- JSON에서 이미지 경로를 지정할 때 /data/local-files/?d=data/samples/<파일명> 형식으로 설정해야 합니다.
- 폴더 구조를 변경할 경우 LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT 값을 적절히 조정해야 합니다.

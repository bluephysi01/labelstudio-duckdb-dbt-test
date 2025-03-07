# Label Studio DuckDB DBT Test

## 프로젝트 개요
이 프로젝트는 Label Studio와 DuckDB, DBT(Data Build Tool)를 활용하여 데이터 처리 및 라벨링을 수행하는 환경을 구성하는 데 목적이 있습니다.

## 폴더 구조
```
LABELSTUDIO-DUCKDB-DBT-TEST/
├── .venv/              # 가상 환경 폴더
├── data/               # 원본 데이터 저장소
│   ├── json/           # JSON 형식의 데이터 파일 저장 폴더
│   ├── samples/        # 라벨링할 이미지 샘플 폴더
├── output/             # 처리된 데이터 저장소
│   ├── json/           # 라벨링된 JSON 데이터 저장 폴더
│   ├── samples/        # 처리된 이미지 데이터 저장 폴더
├── .gitignore          # Git에서 제외할 파일 목록
├── README.md           # 프로젝트 설명 문서
├── test.ipynb          # 데이터 확인 및 분석을 위한 Jupyter Notebook
```

## 주요 파일 및 폴더 설명
- `.venv/` : 프로젝트에서 사용할 Python 가상 환경을 저장하는 폴더입니다.
- `data/json/` : Label Studio에서 사용할 JSON 파일을 저장하는 폴더입니다.
- `data/samples/` : Label Studio에서 처리할 이미지 샘플이 위치하는 폴더입니다.
- `output/json/` : Label Studio에서 처리 완료된 JSON 파일을 저장하는 폴더입니다.
- `output/samples/` : Label Studio에서 처리된 이미지 데이터를 저장하는 폴더입니다.
- `test.ipynb` : Jupyter Notebook을 사용하여 데이터를 분석하거나 시각화하는 데 사용됩니다.

## Label Studio 설정 방법
1. Label Studio 실행
   ```sh
   label-studio start
2. 환경 변수 설정(Windows 기준)
   ```sh
   setx LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED "true" /M
   setx LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT "C:\Users\82102\dev\labelstudio-duckdb-dbt-test" /M

3. Label Studio UI에서 프로젝트 설정 후 data/json 폴더를 데이터 소스로 추가합니다.

4. JSON 파일을 업로드하여 태스크를 생성하고 라벨링을 진행합니다.

## 데이터 처리 흐름
1. data/samples/에 이미지 파일을 추가합니다.
2. data/json/에 Label Studio에서 사용할 JSON 파일을 저장합니다.
3. Label Studio에서 데이터를 불러와 라벨링을 진행합니다.
4. 라벨링된 데이터는 output/json/과 혹은 output/samples/에 저장됩니다.

## 참고사항
- JSON 파일의 경로는 LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT 환경 변수와 일치해야 합니다.
- JSON에서 이미지 경로를 지정할 때 /data/local-files/?d=data/samples/<파일명> 형식으로 설정해야 합니다.
- 폴더 구조를 변경할 경우 LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT 값을 적절히 조정해야 합니다.

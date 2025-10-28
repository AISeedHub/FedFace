# AISEED 파이썬 프로젝트 템플릿

[한국어](./README.md) | [English](./README_EN.md)

## 소개

AI 회사인 AISEED는 파이썬 프로젝트가 많습니다.  
이전과 달리, 연구를 넘어 실제 제품까지 만들기 때문에 유지보수가 필요합니다.

우리는 앞으로 **코드 작성** 이외에도 **코드 관리**까지 힘써야 한다는 뜻입니다.  
하지만 프로젝트 유지보수는 쉽지 않습니다.

- 코드 컨벤션 통일
- 프로젝트 구조 설계
- 테스트 코드 작성
- 가상 환경 및 의존성 관리
- ...

신규 프로젝트를 시작할 때, 혹은 새로운 인력이 프로젝트에 투입됐을 때, 매번 위와 같은 요소들을 설명하고 설정하는 일은 비효율적입니다.

그래서 전사 차원에서 사용할 일관된 템플릿을 만들었습니다.

## 구성요소

템플릿에 사용된 기본적인 구성 요소는 다음과 같습니다.

- [rye](https://rye.astral.sh/guide/) - 프로젝트 관리
  - [uv](https://github.com/astral-sh/uv) - 의존성
  - [ruff](https://docs.astral.sh/ruff/) - 코드 규칙 및 포맷
- [mypy](https://mypy.readthedocs.io/en/stable/) - 타입 지원
- [pytest](https://docs.pytest.org/) - 테스트 코드
- [pre-commit](https://pre-commit.com/) - git commit 작업 시 사전 작업 수행

## 시작하기

### 개발 환경 설정

- `rye`를 설치해주세요. ([설치 가이드](https://rye.astral.sh/guide/installation/))

### 프로젝트 설정

- Github에서 이 템플릿으로 Repository를 생성한 후 Clone 해주세요.
  ![Github Repository's Use this template](./assets/use-this-template.jpeg)
- 프로젝트 폴더 안에 있는 `pyproject.toml` 파일의 `name`, `version`, `description`, `authors`를 각자의 프로젝트에 맞게 수정해주세요.
- 프로젝트 루트 경로에서 다음 스크립트를 실행해주세요.
  ```bash
  $ rye sync
  $ pre-commit install
  # pre-commit 설치 오류가 발생했다면 파이썬 가상 환경이 활성화되지 않았을 가능성이 큽니다.
  # 터미널을 다시 시작해보세요.
  ```

### 환경변수 설정
`.env.sample`를 참고해서 환경변수 파일인 `.env`를 만들어주세요.

### 메인 파일 실행
프로그램의 진입점이 되는 메인 파일은 `src/main.py`입니다.  
메인 파일을 실행하는 스크립트는 2가지 입니다.

1. `rye run dev`

    개발 환경에서 사용합니다.  
    '개발 모드'를 활성화시키며, 메모리 누수 등 여러 경고들을 출력해주어 유용합니다.

2. `rye run prod`

    운영 환경에서 성능 저하를 일으킬 수 있는 '개발 모드'가 비활성화된 스크립트입니다.  
    실제 프로그램 배포시엔 이 스크립트를 이용해주세요.

두 명령어 모두 `.env`에 명시된 `company_name` 환경변수에 따라 `Hello, {company_name}`을 출력합니다.

## 프로젝트 지침

### 의존성 관리

의존성은 `pip` 대신 `rye`에 내장된 `uv`로 관리합니다.  
`uv`, `ruff`, `rye`는 모두 같은 팀에서 만든 도구이기 때문에 대부분의 명령어는 호환됩니다.

> [!IMPORTANT]  
> 개발에 필요한 패키지와 제품에 필요한 패키지를 구분해주세요.

```bash
# install production dependency
$ rye add numpy

# uninstall production dependency
$ rye remove numpy

# install development dependency
$ rye add --dev pytest

# uninstall development dependency
$ rye remove --dev pytest
```

### 타입 체크

`mypy`로 타입 오류가 발생한 지점을 찾습니다.

```bash
$ rye run type
```

### Lint

`ruff`로 코드 컨벤션에 문제가 있는 지점을 찾습니다.

```bash
$ rye lint
```

### 테스트 실행

`pytest`로 `tests/` 폴더에 있는 테스트를 실행합니다.

```bash
# run test
$ rye run test
```

**테스트 코드 작성**은 몹시 어렵고 방대한 주제이기 때문에 테스트 코드 작성법에 대해선 아직 다루지 않습니다.  
대신, 다른 구성원이 쉽게 코드를 파악할 수 있도록 **코드 사용법**을 위주로 작성해주시기 바랍니다.

### Git

작업 내역을 `commit`할 때 `pre-commit`을 이용해 변경된 코드를 검사합니다.  
커밋하기 전에 `ruff`, `mypy`, `pytest`로 코드 컨벤션, 타이핑, 테스트에 문제가 없는지 확인합니다.

## 기타

### 프로젝트 환경 확인

```bash
$ rye show
```

### 실행할 수 있는 스크립트 목록 확인

```bash
$ rye run
```

### 스크립트 관리

`pyproject.toml`의 `[tool.rye.scripts]` 항목에 원하는 스크립트를 추가하거나 수정하시면 됩니다.

### 파이썬 버전 변경

1. `.python-version`에서 원하는 버전으로 수정

   (타겟 버전은 `pyproject.toml`의 `requires-version`을 수정)

2. sync 스크립트 실행

   ```bash
   $ rye sync
   ```

### PyTorch 설치
일반적인 파이썬 패키지들은 **PyPI**에서 호스팅됩니다. 반면, `pytorch`는 별도의 인덱스를 따로 갖고 있습니다.  
게다가 CPU 전용 빌드, CUDA 버전별 빌드 등 같은 패키지더라도 빌드가 여러 개이기 때문에 `uv`에서 `pytorch`를 설치하려면 어떤 빌드를 패키지로 설치할지 명시해줘야 합니다.

다음은 Linux와 Windows 환경에서는 CUDA 12.6 빌드를, macOS 환경에서는 CPU 빌드를 사용하는 예시입니다.  
만약 다른 버전의 CUDA를 사용하고 싶다면 숫자를 바꿔주시면 됩니다. (예: cu126 -> cu128)

```toml
# pyproject.toml

[tool.uv.sources]
torch = [
  { index = "pytorch-cu126", marker = "sys_platform != 'darwin'" },
  { index = "pytorch-cpu", marker = "sys_platform == 'darwin'" },
]
torchvision = [
  { index = "pytorch-cu126", marker = "sys_platform != 'darwin'" },
  { index = "pytorch-cpu", marker = "sys_platform == 'darwin'" },
]

[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

설정 이후 패키지를 설치해주시면 원하는 빌드로 설치가 완료됩니다.

```bash
rye add torch torchvision
```
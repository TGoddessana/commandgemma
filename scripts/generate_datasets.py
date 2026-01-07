import os
import json
import time
from typing import List

from pydantic import BaseModel
from google import genai
from google.genai import types


class CommandExample(BaseModel):
    korean_question: str
    english_question: str
    command: str


class CommandDataset(BaseModel):
    examples: List[CommandExample]


target_tools = [
    "mv",  # 파일 이동/이름 변경
    "cp",  # 파일 복사 (-r)
    "rm",  # 삭제 (-rf)
    "mkdir",  # 디렉토리 생성 (-p)
    "ln",  # 심볼릭 링크 (ln -s)
    "zip",  # 압축
    "unzip",  # 해제
    "echo",  # 파일 쓰기/환경변수 확인
    "touch",  # 빈 파일 생성
    "ls",  # 디렉토리 목록
    "cat",  # 파일 내용 출력
    "tail",  # 파일 끝부분 출력 (로그 분석 필수)
    "head",  # 파일 시작부분 출력
    "find",  # 복잡한 파일 찾기
    "grep",  # 텍스트 검색 (옵션 다양)
    "sed",  # 텍스트 치환
    "awk",  # 텍스트 가공 및 데이터 추출
    "jq",  # JSON 데이터 파싱 (API 응답 분석용)
    "tar",  # 압축 및 해제
    "xargs",  # 명령어 체이닝
    "ps",  # 프로세스 확인
    "kill",  # 프로세스 종료
    "pkill",  # 이름으로 프로세스 종료
    "lsof",  # 열린 포트/파일 확인 (네트워크 디버깅 필수)
    "netstat",  # 네트워크 연결 상태
    "ss",  # netstat의 최신 버전
    "df",  # 디스크 용량 확인
    "du",  # 폴더별 용량 확인
    "top",  # 시스템 리소스 모니터링
    "htop",  # top의 컬러풀한 버전
    "free",  # 메모리 사용량 확인
    "chmod",  # 파일 권한 변경 (자주 헷갈림)
    "chown",  # 파일 소유자 변경
    "env",  # (추가) 환경변수 목록 확인
    "export",  # (추가) 환경변수 설정 (export A=B)
    "systemctl",  # 서비스 관리 (start, stop, status)
    "journalctl",  # 로그 확인 (옵션 복잡함)
    "crontab",  # 스케줄링
    "curl",  # API 요청 테스트 (헤더, 바디 등)
    "wget",  # 파일 다운로드
    "ssh",  # 원격 접속
    "scp",  # 파일 전송
    "rsync",  # (추가) 서버 간 동기화 (필수)
    "ping",  # (추가) 연결 확인
    "traceroute",  # (추가) 경로 추적
    "tcpdump",  # (추가) 패킷 분석 (고급 디버깅용)
    "whois",  # (추가) 도메인 정보
    "docker",  # 도커 기본
    "docker-compose",  # 도커 컴포즈
    "kubectl",  # 쿠버네티스
    "git"  # 깃 (상황별 명령어)
    "terraform",  # 인프라 생성/변경 (plan, apply, state ...)
    "ansible-playbook",  # 서버 설정 배포 (ansible보다는 playbook을 더 많이 씀)
    "helm",  # 차트 설치/업데이트/롤백
    "openssl",  # 인증서 생성, 키 변환 (옵션 매우 복잡함)
    "aws cli",  # AWS CLI (특히 s3, ec2 관련 기본 명령)
    "dig",  # 도메인 DNS 조회 (nslookup보다 더 자세함)
    "nc",  # Netcat (TCP/UDP 포트 연결 테스트용)
    "nginx",  # -t, -s reload, -V (버전확인)
    "caddy",  # run, reload, stop
    "apache2ctl",  # (혹시 아파치 쓴다면) configtest, graceful
    "java",  # -jar, -D옵션(시스템 프로퍼티)
    "javac",  # (잘 안 쓰지만 기본)
    "gradlew",  # clean build, bootRun, tasks
    "mvn",  # (Maven) clean install, -DskipTests
    "npm",  # install, run, ci (clean install)
    "yarn",  # add, install
    "pip",  # install -r requirements.txt
    "python",  # -m venv (가상환경), -m http.server
    "uv",  # (추가) 최신 Python 도구 (사용자 맞춤)
    "pm2",  # (추가) Node.js 프로세스 매니저 (백엔드 필수)
    "celery",  # worker, beat 실행 옵션
    "redis-cli",  # monitor, flushall, keys (간단한 조회)
    "mysql",  # -u, -p, -h, 데이터베이스 dump/import 명령어
    "pg_dump",  # PostgreSQL 백업 명령어 (자주 까먹음)
    "psql",  # (추가) PostgreSQL 클라이언트
    "sqlite3",  # (추가) SQLite 클라이언트
    "mkswap",  # 스왑 메모리 생성 절차
    "swapon",  # 스왑 활성화
    "ufw",  # 방화벽 (ufw allow 80/tcp)
    "iptables",  # (어렵지만 필수) 리스트 확인, 룰 삭제
    "history",  # !숫자, 검색 등
    "nohup",  # 백그라운드 실행 (nohup java -jar ... &)
]


def generate_prompt(tool_name: str, batch_num: int) -> str:
    """
    150개를 한 번에 만들면 품질이 떨어지므로, 50개씩 나누어 생성하기 위한 프롬프트입니다.
    batch_num을 통해 매번 조금씩 다른 뉘앙스를 요구할 수도 있습니다.
    """

    prompt = f"""
    당신은 시니어 백엔드/DevOps 엔지니어이자 리눅스 마스터입니다.
    270M 소형 모델(SLM) 파인튜닝을 위해 **{tool_name}** 명령어 데이터셋을 생성합니다.

    이번 배치의 목표: **{tool_name}** 도구에 대해 **다양한 파라미터가 적용된 고품질 데이터 50개**를 생성하세요.

    ## ⚠️ 절대적인 필수 규칙 (Parameter Randomization)
    모델이 특정 숫자나 파일명을 '정답'으로 외우지 않도록, 아래 값들을 **매 예시마다 무작위로 변경**하세요.
    - **PID**: 1234, 9876, 543, 1001, 777 등 다양하게 섞으세요.
    - **Port**: 80, 443, 8080, 3000, 5432, 6379, 22 등 다양하게 섞으세요.
    - **IP/URL**: 192.168.1.5, localhost, example.com, 10.0.0.1 등 다양하게 섞으세요.
    - **파일명/경로**: error.log, access.log, data.json, /var/log/syslog, /home/user/app 등 다양하게 섞으세요.
    - **날짜/시간**: "10 minutes ago", "yesterday", "2023-01-01" 등 다양하게 섞으세요.
    - **용량/크기**: 100MB, 1GB, 500KB 등 다양하게 섞으세요.

    **[나쁜 예 - 금지]**
    (모든 예시가 PID 1234를 사용함) -> 절대 금지! 모델이 과적합됩니다.

    **[좋은 예]**
    1. "PID 1234 죽여줘" -> kill -9 1234
    2. "999번 프로세스 종료해" -> kill -9 999
    3. "PID 5000 강제 종료" -> kill -9 5000
    4. "용량이 400MB 넘는 폴더 찾아줘" -> find /home/user -type d -size +400M

    ## 작성 가이드
    1. **한국어 질문 (korean_question)**:
       - **말투를 다양화하세요**: "해줘", "하고 싶어", "어떻게 해?", "명령어 뭐야?", "급해 빨리" 등
       - 구어체를 사용하되, 개발자의 의도를 명확히 담으세요.
       - 예: "8080 포트 쓰고 있는 놈 누구야?", "엔진엑스 설정 문법 맞는지 체크해줘"

    2. **영어 질문 (english_question)**:
       - 한국어 질문의 뉘앙스를 살려 자연스러운 영어로 번역하세요.
       - 예: "Who is using port 8080?", "Check if the Nginx configuration syntax is correct."

    3. **명령어 (command)**:
       - **실무형 옵션 조합**: 단순 `ls` 말고 `ls -alh` 처럼 실무에서 쓰는 조합을 우선하세요.
       - **원라이너**: 복잡한 작업은 `&&` 또는 `|` (파이프)를 사용해 한 줄로 작성하세요.

    ## 생성 요청
    - Output Format: JSON Array of objects
    - Count: **50 distinct examples**
    """
    return prompt


def generate(tool_name: str, total_count: int = 150):
    client = genai.Client()
    model_id = "gemini-3-flash-preview"

    all_examples = []
    batch_size = 50
    batches = total_count // batch_size

    print(f"🚀 [{tool_name}] 데이터 생성 시작 (총 {total_count}개 목표, {batches} 배치)")

    for i in range(batches):
        print(f"  - Batch {i + 1}/{batches} 생성 중...", end=" ", flush=True)

        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=generate_prompt(tool_name, i)),
                    ],
                ),
            ]

            generate_content_config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="MEDIUM"),
                response_mime_type="application/json",
                response_schema=CommandDataset,
                temperature=0.7 + (i * 0.1),  # 배치마다 창의성(온도)을 조금씩 높임
            )

            response = client.models.generate_content(
                model=model_id,
                contents=contents,
                config=generate_content_config,
            )

            if response.text:
                dataset = CommandDataset.model_validate_json(response.text)
                all_examples.extend(dataset.examples)
                print(f"성공 ({len(dataset.examples)}개)")
            else:
                print("실패 (응답 없음)")

            time.sleep(1)

        except Exception as e:
            print(f"에러 발생: {e}")

    # 결과 저장
    output_file = f"dataset/{tool_name}.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for example in all_examples:
            json_line = {
                "korean_question": example.korean_question,
                "english_question": example.english_question,
                "command": example.command
            }
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

    print(f"✅ [{tool_name}] 완료! 총 {len(all_examples)}개 저장됨 -> {output_file}\n")
    return all_examples

if __name__ == "__main__":
    os.environ["GOOGLE_API_KEY"] = "..."

    for tool in target_tools:
        generate(tool)
        print("----------------------------------------\n")
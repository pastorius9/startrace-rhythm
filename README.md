# Star Trace Night Race

PLAVE 'Star Trace' 팬 리듬게임. 야간 네온 F1 서킷 위로 노트가 내려오고, 상단 화면에서 공식 MV([TBmE7m0yEwY](https://www.youtube.com/watch?v=TBmE7m0yEwY))가 소리와 함께 재생된다.

- `index.html`: 게임 전체. 음원 파일 없이 MV 소리를 그대로 쓴다.
- `api/scores.js`: 리더보드 API (Vercel 서버리스 함수 + Upstash Redis).

## 게임
- 4레인 낙하형. PC는 D·F·J·K, 모바일은 레인 터치.
- 채보: 175 BPM 격자에 맞춘 고정 채보. NORMAL 728노트(매 박자 노트), HARD 1089노트(8분음표 위주). 후반 "Go" 뒤 EDM 구간(142.63초~)은 HYPERDRIVE: 노트 속도 2배, 밀도 증가(NORMAL 8분, HARD 16분 픽업), 트랙에 보라색 빛. MV 음원에서 직접 만들어서 노트 시간이 곧 MV 시간이다(`MV_OFFSET = 0`). `build_chart.py`로 생성: `python build_chart.py <MV 음원> chart.json --end 205 --hyper-bar 104` (librosa 필요).
- MV는 사용자가 "레이스 시작"을 눌러 소리와 함께 재생한다. 게임 화면이 재생 중인 MV를 가리지 않는다(유튜브 플레이어 최소 200×200 유지).
- 유튜브 조회수 집계 기준은 공개돼 있지 않다. 정상 재생이므로 집계될 가능성이 높지만 보장되지는 않는다.

## 싱크
- 게임 시계는 유튜브 플레이어가 알려 주는 재생 시각을 1.5초 구간 중앙값으로 걸러서 쓴다.
- 레이스가 끝나면 평균적으로 빠르게/늦게 누른 정도(15ms 이상)를 보여 주고 "SYNC 자동 보정" 버튼으로 바로 맞출 수 있다.

## 리더보드
- 결과 화면에서 닉네임(최대 12자)을 적어 제출하면 F1 타이밍 타워(상위 50위)에 오른다.
- 난이도별로 1인 1기록, 최고점만 유지한다.
- 브라우저마다 무작위 플레이어 ID와 비밀값을 만들고, 서버는 비밀값의 해시만 저장한다. 그 브라우저만 자기 기록을 고칠 수 있다.
- 서버 검사: 점수 0~1,000,000, 닉네임 정리, IP당 10분에 20회 제출 제한.
- 점수는 브라우저에서 계산하므로 마음먹고 조작하는 사람을 완전히 막을 수는 없다. 이상한 기록은 Upstash 콘솔에서 `ZREM lb:normal <id>`로 지운다.

## 배포 (Vercel)
1. Vercel → Add New → Project → GitHub 저장소 `pastorius9/startrace-rhythm` Import → Deploy. 빌드 설정은 기본값 그대로 둔다(프레임워크 없음).
2. 프로젝트 → Storage → Upstash(Redis) 추가 → 이 프로젝트에 연결. `KV_REST_API_URL`/`KV_REST_API_TOKEN` 또는 `UPSTASH_REDIS_REST_URL`/`UPSTASH_REDIS_REST_TOKEN` 환경변수가 자동으로 들어간다.
3. Deployments에서 Redeploy (환경변수 적용).

Redis를 연결하기 전에는 리더보드 자리에 "연결하지 못했어요"가 뜨고, 이 기기의 최고 기록만 저장된다.

# Representative failure cases

Cases are selected deterministically from the BGE-M3 main experiment using the predeclared categories and thresholds in `scripts/select_failure_cases.py`. The first pass takes one unique output from every available category; the second fixed-order pass adds at most one more per category up to 12 total. Within a category, candidates are sorted by severity then run ID. These are diagnostic examples, not a hand-picked success/failure narrative. Text is kept in the original Chinese; no translated text is used for metrics.

## content_failure: `major_r1_run_36568e4f483523bec0eb06e2`

- scenario/model/replicate/alpha: `concision_detail_039` / `gpt6_luna_api` / 3 / 0
- Qwen/GPT intensity: 10.0 / 55.0; content pass: False / False; naturalness: 4.0 / 5.0
- self/external coordinate: 0.0000 / 0.0688; self/external ICE: 0.0000 / 0.0688
- character length and allowed range: 71 / [70, 160]
- original Chinese output: 本地景点开放时间、门票价格及交通路线，请以景区官方公布信息为准。出行前建议确认开放安排和票价，并查询适合的公交、地铁或自驾路线，以便规划行程。

## style_under_or_overshoot: `revision_run_044acc20199dd7a94fe9`

- scenario/model/replicate/alpha: `emotion_apology_023` / `qwen35_35b_a3b` / 1 / 1
- Qwen/GPT intensity: 0.0 / 8.0; content pass: True / True; naturalness: 5.0 / 5.0
- self/external coordinate: 1.0000 / 0.2803; self/external ICE: 0.0000 / 0.7197
- character length and allowed range: 157 / [80, 140]
- original Chinese output: 办公区空调突发故障导致室内温度过高，已严重干扰大家正常办公。对此造成的不便，我们深表遗憾。行政部已紧急联系维修人员处理，但预计修复时间尚不确定。在此期间，请大家尽量调整工作状态，我们会持续跟进进度并第一时间通报结果。对于此次设备故障带来的影响，我们再次致以诚挚歉意，并承诺将尽快彻底解决该问题，恢复舒适的工作环境。

## external_out_of_range: `revision_run_71ea6a700f6931cb0bb6`

- scenario/model/replicate/alpha: `concision_detail_003` / `qwen36_35b_a3b` / 1 / 0.666667
- Qwen/GPT intensity: 60.0 / 93.0; content pass: False / False; naturalness: 4.0 / 4.0
- self/external coordinate: 0.2845 / 1.3527; self/external ICE: 0.3821 / 0.6860
- character length and allowed range: 145 / [70, 160]
- original Chinese output: 正确使用灭火器扑灭初期火灾需遵循规范流程。首先，应迅速提起灭火器并颠倒摇晃，使内部干粉充分松动。接着，拔掉保险销，一手紧握喷管前端，对准火焰根部而非上部。在距离火源约两米处，用力压下压把，左右扫射推进。务必站在上风口或侧风口位置，避免烟雾灼伤。保持冷静，持续喷射直至火焰完全熄灭，防止复燃。

## non_monotonic: `revision_run_0c77a414917187dfbefa`

- scenario/model/replicate/alpha: `concision_detail_021` / `qwen35_35b_a3b` / 2 / 0.166667
- Qwen/GPT intensity: 65.0 / 88.0; content pass: True / True; naturalness: 5.0 / 5.0
- self/external coordinate: 0.7476 / 0.7940; self/external ICE: 0.5810 / 0.6274
- character length and allowed range: 124 / [70, 160]
- original Chinese output: 节约用水是守护生态环境的关键举措。水资源是自然生态系统的血液，维持着生物多样性和气候稳定。过度取水会导致河流干涸、湿地萎缩，破坏栖息地。通过合理用水，我们能恢复水循环平衡，保护濒危物种，促进生态系统的自我修复与可持续发展，确保地球生命之源长流不息。

## length_violation: `revision_run_3829b8160bda8ead7d7e`

- scenario/model/replicate/alpha: `politeness_refusal_027` / `qwen36_27b` / 1 / 0
- Qwen/GPT intensity: 0.0 / 0.0; content pass: False / True; naturalness: 1.0 / 1.0
- self/external coordinate: 0.0000 / 0.4650; self/external ICE: 0.0000 / 0.4650
- character length and allowed range: 999 / [70, 130]
- original Chinese output: Here's a thinking process:

1.  **Analyze User Input:**
   - **Core Semantic:** Invite a friend to go hiking outdoors on the weekend, but the friend refuses due to a scheduling conflict.
   - **Style A:** Direct, not tactful, slightly stiff, but not insulting.
   - **Style B:** Tactful, polite, considerate of face, provides buffer reasons.
   - **Target Style Ratio:** alpha = 0 (100% Style A + 0% Style B). This means the output must be strictly Style A: direct, blunt, slightly stiff, no tact, no polite buffering.
   - **Hard Requirements:**
     - Preserve core semantic.
     - No conflicting new facts.
     - Length: 70 to 130 Chinese characters.
     - Output ONLY the final Chinese text. First word must be the text itself.
     - No explanations, no titles, no markdown lists, no English, no thinking process.
     - Strictly follow the prompt's negative constraints.

2.  **Deconstruct Core Semantic & Style A:**
   - Core: I invite friend to weekend hiking -> Friend says no because of

## judge_disagreement: `revision_run_0a618f15212931096a52`

- scenario/model/replicate/alpha: `objectivity_cat_038` / `qwen36_27b` / 2 / 1
- Qwen/GPT intensity: 95.0 / 0.0; content pass: True / True; naturalness: 5.0 / 1.0
- self/external coordinate: 1.0000 / 0.4055; self/external ICE: 0.0000 / 0.5945
- character length and allowed range: 932 / [80, 150]
- original Chinese output: Here's a thinking process:

1.  **Analyze User Input:**
    - **Core Semantic:** 斑驳的公告栏前，工作人员小心翼翼抚平纸张的褶皱，居民的目光在文字间流转，汇聚成无声的共鸣。 (In front of the mottled bulletin board, the staff member carefully smooths out the wrinkles on the paper, while residents' gazes flow among the words, converging into a silent resonance.)
    - **Style A:** Objective observation record, like medical/biological records, restrained, emotionless.
    - **Style B:** Lyrical modern prose poetry, rich imagery, strong emotion, ornate language.
    - **Target Style Ratio:** alpha = 1 (0% Style A + 100% Style B). So, purely Style B.
    - **Hard Requirements:**
        1. Must preserve core semantic, no conflicting new facts.
        2. Length: 80 to 150 Chinese characters.
        3. Only output the main text, no explanations, no titles, no markdown lists.
        4. No English analysis, no thinking process. First sentence must be the main text.
    -

## anchor_disagreement: `revision_run_f902d9063cdbc1064f07`

- scenario/model/replicate/alpha: `concision_detail_003` / `qwen35_4b` / 3 / 0
- Qwen/GPT intensity: 60.0 / 88.0; content pass: True / True; naturalness: 5.0 / 5.0
- self/external coordinate: 0.0000 / 1.3192; self/external ICE: 0.0000 / 1.3192
- character length and allowed range: 132 / [70, 160]
- original Chinese output: 正确使用灭火器扑灭初期火灾的关键在于遵循“提、拔、握、压”四字口诀。首先迅速提起灭火器，拔掉保险销，一手握住喷管前端对准火焰根部，另一手压下把手即可喷射。喷射时需注意保持安全距离并左右扫射，待明火熄灭后应继续观察防止复燃，同时确保自身处于上风口位置以避免烟雾侵害。

## content_failure: `major_r1_run_9397556368f75651d2777503`

- scenario/model/replicate/alpha: `concision_detail_039` / `gpt6_luna_api` / 1 / 0.166667
- Qwen/GPT intensity: 60.0 / 70.0; content pass: False / False; naturalness: 5.0 / 5.0
- self/external coordinate: 0.1435 / 0.1310; self/external ICE: 0.0231 / 0.0357
- character length and allowed range: 87 / [70, 160]
- original Chinese output: 该景点开放时间、门票价格及交通路线请以现场公告或官方信息为准。出行前建议确认当天是否开放、票价是否调整，并查询公交、地铁或自驾路线，合理安排到达时间，避免因临时变更影响行程。

## style_under_or_overshoot: `revision_run_0f05f67a98ee0832e03d`

- scenario/model/replicate/alpha: `emotion_apology_033` / `qwen35_9b` / 2 / 1
- Qwen/GPT intensity: 0.0 / 18.0; content pass: True / False; naturalness: 4.0 / 3.0
- self/external coordinate: 1.0000 / 0.5884; self/external ICE: 0.0000 / 0.4116
- character length and allowed range: 128 / [80, 140]
- original Chinese output: 各位请注意，原定会议时间被迫临时调整，对此我深感抱歉。此次变动完全是由于我方内部突发紧急事务所致，绝非为了拖延或轻视各位的时间。我对此事负有全部责任，并愿意接受任何合理的批评与监督。请立刻确认新时间，若因此造成任何不便或损失，我们将全力承担后果，绝不推诿。

## external_out_of_range: `revision_run_d898319a907aa907bffa`

- scenario/model/replicate/alpha: `concision_detail_003` / `qwen36_35b_a3b` / 1 / 0
- Qwen/GPT intensity: 60.0 / 90.0; content pass: True / True; naturalness: 5.0 / 4.0
- self/external coordinate: 0.0000 / 1.2944; self/external ICE: 0.0000 / 1.2944
- character length and allowed range: 150 / [70, 160]
- original Chinese output: 正确使用灭火器扑灭初期火灾，首先需迅速提起灭火器，并快速奔赴至距离火焰约两至三米的上风口或侧上风口安全位置。接着，拔掉保险销，一手紧握喷管前端以对准火焰根部，另一手用力压下压把。在操作过程中，应保持身体稳定，左右扫射，由近及远地推进，直至火焰完全熄灭，切勿直接对准火苗上部喷射，以免无法有效降温窒息。

## non_monotonic: `revision_run_43500472698ce2a1cbdd`

- scenario/model/replicate/alpha: `humor_neutral_031` / `qwen35_4b` / 3 / 0.833333
- Qwen/GPT intensity: 65.0 / 65.0; content pass: True / True; naturalness: 5.0 / 4.0
- self/external coordinate: 0.4349 / 0.3250; self/external ICE: 0.3985 / 0.5083
- character length and allowed range: 122 / [80, 140]
- original Chinese output: 社区公告栏最近贴了个垃圾分类新规定，不少居民看了直挠头，心里嘀咕着咋又搞这么复杂。明明以前扔个瓶子都能分清楚，现在突然多了这么多分类，大家心里都有点打鼓，生怕不小心就违规了，毕竟这新规来得太突然，让人有点摸不着头脑，想吐槽却又不想真的乱丢东西。

## length_violation: `revision_run_4923536b43a514166376`

- scenario/model/replicate/alpha: `formality_request_027` / `qwen36_27b` / 2 / 0.5
- Qwen/GPT intensity: 0.0 / 25.0; content pass: False / True; naturalness: 1.0 / 1.0
- self/external coordinate: 0.5348 / 0.4637; self/external ICE: 0.0348 / 0.0363
- character length and allowed range: 997 / [70, 130]
- original Chinese output: Here's a thinking process:

1.  **Analyze User Input:**
   - **Role:** Chinese short text generator.
   - **Constraint 1 (Crucial):** Do NOT show reasoning, analysis, or intermediate steps. ONLY output the final Chinese text. First sentence must be the text itself.
   - **Constraint 2 (Crucial):** Strictly NO Thinking Process, Analysis, steps, titles, or Markdown lists.
   - **Core Semantics:** 社区需通知居民本周末将暂停供水进行管道维护。(The community needs to notify residents that water supply will be suspended this weekend for pipe maintenance.)
   - **Style A:** Casual, conversational, like friends chatting.
   - **Style B:** Formal, official document style, standardized wording, polite but distant.
   - **Target Style Ratio:** alpha = 3/6 (50% Style A + 50% Style B). This means the text should blend casual/conversational elements with formal/official elements equally, or alternate/combine them to achieve a 50/50 feel.
   - **Hard Requirements:**
     1. Preserve core semantics. No conflicting facts.

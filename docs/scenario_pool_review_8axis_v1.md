# StyleBlend 8-Axis Scenario Pool Review v1

Created: 2026-06-23 UTC

## Output Files

- `data/scenarios/styleblend_8axis_scenarios_v1.jsonl`
- `results/scenario_pool_8axis_v1/tables/scenario_domain_distribution.csv`
- `results/scenario_pool_8axis_v1/tables/scenario_validation_summary.csv`

## Summary

| axis | candidates | filtered | final | final domains | top reject reasons |
|---|---:|---:|---:|---:|---|
| concision_detail | 80 | 50 | 40 | 11 | none |
| emotion_apology | 80 | 50 | 40 | 11 | near_duplicate_content:3 |
| empathy_clinical | 80 | 50 | 40 | 12 | none |
| expertise_explain | 80 | 50 | 40 | 10 | near_duplicate_content:8; content_too_short:1 |
| formality_request | 80 | 50 | 40 | 11 | none |
| humor_neutral | 80 | 50 | 40 | 10 | none |
| objectivity_cat | 90 | 50 | 40 | 11 | none |
| politeness_refusal | 80 | 50 | 40 | 10 | near_duplicate_content:1; risk_pattern:1 |

## Domain Distribution

### concision_detail

community: 3, consumer: 4, daily_life: 3, education: 3, environment: 3, family: 4, public_service: 4, science: 4, technology: 4, travel: 4, workplace: 4

### emotion_apology

community: 4, consumer: 4, daily_life: 3, education: 4, environment: 1, family: 4, housing: 4, public_service: 4, technology: 4, travel: 4, workplace: 4

### empathy_clinical

community: 4, consumer: 4, daily_life: 4, education: 4, environment: 1, family: 3, health_non_diagnostic: 3, housing: 1, public_service: 4, technology: 4, travel: 4, workplace: 4

### expertise_explain

daily_life: 5, education: 2, environment: 5, family: 3, health_non_diagnostic: 4, public_service: 4, science: 5, technology: 5, travel: 5, workplace: 2

### formality_request

community: 4, consumer: 4, daily_life: 3, education: 4, family: 4, health_non_diagnostic: 1, housing: 4, public_service: 4, technology: 4, travel: 4, workplace: 4

### humor_neutral

community: 4, consumer: 4, daily_life: 4, education: 4, family: 4, housing: 4, public_service: 4, technology: 4, travel: 4, workplace: 4

### objectivity_cat

community: 4, consumer: 4, daily_life: 4, education: 4, environment: 4, family: 4, public_service: 4, science: 4, technology: 3, travel: 4, workplace: 1

### politeness_refusal

community: 5, consumer: 4, daily_life: 4, education: 4, family: 4, health_non_diagnostic: 3, public_service: 4, technology: 4, travel: 4, workplace: 4


## Random-Like Samples

### concision_detail

- `concision_detail_001` [consumer]: 说明如何选购适合的冬季保暖衣物。 / checks: 选购冬季衣物、保暖功能
- `concision_detail_002` [family]: 介绍如何制定一份合理的周末家庭出游计划。 / checks: 制定周末出游计划、家庭出游
- `concision_detail_003` [public_service]: 简述如何正确使用灭火器扑灭初期火灾。 / checks: 正确使用灭火器、扑灭初期火灾
- `concision_detail_004` [science]: 解释光合作用中植物如何将光能转化为化学能的过程。 / checks: 光合作用、光能转化、化学能
- `concision_detail_005` [technology]: 说明如何设置路由器密码以保障家庭网络安全。 / checks: 设置路由器密码、保障网络安全
- `concision_detail_006` [travel]: 说明如何规划一条适合徒步的山区旅游路线。 / checks: 规划徒步路线、山区旅游
- `concision_detail_007` [workplace]: 请对方明天下午三点前确认会议时间是否合适。 / checks: 明天下午三点前、确认会议时间
- `concision_detail_008` [daily_life]: 简述如何安全保存和使用家用清洁剂。 / checks: 安全保存、家用清洁剂
- `concision_detail_009` [education]: 简述申请图书馆借阅证的基本流程和所需材料。 / checks: 申请借阅证、基本流程、所需材料
- `concision_detail_010` [environment]: 建议将旧电脑回收处理，避免电子垃圾污染环境。 / checks: 旧电脑回收、避免电子垃圾

### emotion_apology

- `emotion_apology_001` [community]: 邻居装修噪音超出规定时间，希望协调方介入制止并保障休息权利。 / checks: 装修噪音、超出规定时间、介入制止
- `emotion_apology_002` [workplace]: 办公会议因设备故障中断，请求后勤部门紧急修复并重新安排会议。 / checks: 设备故障、紧急修复、重新安排会议
- `emotion_apology_003` [consumer]: 服务延迟导致体验受损，希望对方说明原因并给出合理补偿方案。 / checks: 服务延迟、说明原因、合理补偿方案
- `emotion_apology_004` [education]: 课程安排临时调整未提前通知，请求老师重新确认后续课程时间表。 / checks: 课程调整、未提前通知、确认后续时间表
- `emotion_apology_005` [family]: 家庭聚会因突发情况临时取消，需向亲友致歉并重新商定聚会时间。 / checks: 聚会取消、向亲友致歉、重新商定时间
- `emotion_apology_006` [housing]: 小区公共设施损坏长期未修，呼吁物业尽快处理并公示维修进度。 / checks: 设施损坏、尽快处理、公示维修进度
- `emotion_apology_007` [technology]: 软件系统更新后出现功能异常，要求技术团队排查问题并恢复服务。 / checks: 系统更新、功能异常、排查问题
- `emotion_apology_008` [travel]: 航班延误导致行程受阻，希望航空公司协助安排住宿并说明延误原因。 / checks: 航班延误、协助安排住宿、说明延误原因
- `emotion_apology_009` [public_service]: 公共交通班次取消影响出行计划，要求相关部门尽快提供替代路线指引。 / checks: 班次取消、替代路线指引、尽快提供
- `emotion_apology_010` [daily_life]: 线上活动因网络问题中断，希望主办方提供回放链接并致歉。 / checks: 网络问题、提供回放链接、致歉

### empathy_clinical

- `empathy_clinical_001` [workplace]: 最近工作压力大导致连续失眠，不知道该如何调整作息。 / checks: 工作压力大、连续失眠、调整作息
- `empathy_clinical_002` [community]: 社区新政策实施后，邻里对公共空间使用产生分歧，沟通困难。 / checks: 新政策实施、公共空间使用分歧、沟通困难
- `empathy_clinical_003` [consumer]: 网购的商品与描述不符，商家推诿责任，感到失望和困惑。 / checks: 商品与描述不符、商家推诿责任、感到失望
- `empathy_clinical_004` [education]: 孩子这次考试没考好，情绪低落，家长担心影响后续学习。 / checks: 考试没考好、情绪低落、担心后续学习
- `empathy_clinical_005` [public_service]: 单位突然调整考勤制度，员工对新规定感到不适应和困惑。 / checks: 调整考勤制度、感到不适应、困惑
- `empathy_clinical_006` [technology]: 新购买的智能设备无法连接网络，影响日常使用体验。 / checks: 智能设备、无法连接网络、影响使用体验
- `empathy_clinical_007` [travel]: 出门旅游时行李丢失，行程受阻，感到十分焦虑无助。 / checks: 行李丢失、行程受阻、感到焦虑
- `empathy_clinical_008` [daily_life]: 日常通勤路上遇到突发堵车，担心迟到，心情变得烦躁。 / checks: 突发堵车、担心迟到、心情烦躁
- `empathy_clinical_009` [family]: 孩子连续几天没睡好，看着疲惫的样子让人担心，想了解如何调整作息。 / checks: 孩子没睡好、疲惫、调整作息
- `empathy_clinical_010` [health_non_diagnostic]: 家人最近身体不舒服，但各项检查指标正常，心里始终放不下。 / checks: 身体不舒服、检查指标正常、心里放不下

### expertise_explain

- `expertise_explain_001` [daily_life]: 为什么我们需要在过马路时遵守红绿灯的指示？ / checks: 过马路、红绿灯、遵守指示、安全原因
- `expertise_explain_002` [environment]: 为什么冬天我们会感觉冷而夏天感觉热？ / checks: 冬天冷、夏天热、温度变化、原因解释
- `expertise_explain_003` [technology]: 为什么电脑打开后需要等待一段时间才能使用？ / checks: 电脑开机、等待时间、原因解释、正常使用
- `expertise_explain_004` [travel]: 为什么我们要乘坐公共交通工具而不是每个人都开车？ / checks: 公共交通工具、开车、原因解释、出行选择
- `expertise_explain_005` [science]: 为什么天空在白天是蓝色的而夜晚是黑色的？ / checks: 天空颜色、白天蓝色、夜晚黑色、原因解释
- `expertise_explain_006` [health_non_diagnostic]: 为什么我们需要刷牙来保持牙齿健康？ / checks: 刷牙、牙齿健康、必要性、原因解释
- `expertise_explain_007` [public_service]: 为什么我们要把垃圾分类投放到不同的垃圾桶？ / checks: 垃圾分类、不同垃圾桶、投放原因、处理方式
- `expertise_explain_008` [family]: 为什么小朋友长身体时要多吃饭，少挑食？ / checks: 长身体多吃饭、少挑食
- `expertise_explain_009` [education]: 为什么我们要去学校学习知识而不是在家自学？ / checks: 去学校、学习知识、在家自学、原因解释
- `expertise_explain_010` [workplace]: 为什么工厂排放废气会污染空气，影响呼吸？ / checks: 工厂排放废气、影响呼吸

### formality_request

- `formality_request_001` [community]: 告知邻居明天上午九点将开始楼道维修施工，请提前搬运物品避免堵塞。 / checks: 明天上午九点、楼道维修施工、提前搬运物品
- `formality_request_002` [consumer]: 提醒客户订单已发货，预计三天后送达，请保持电话畅通。 / checks: 订单已发货、预计三天后送达、保持电话畅通
- `formality_request_003` [education]: 通知家长本周五学校将组织春季郊游活动，需提前签署同意书。 / checks: 本周五、春季郊游活动、签署同意书
- `formality_request_004` [family]: 通知家人周末聚会地点改在公园，请大家准时到达准备野餐。 / checks: 周末聚会、改在公园、准时到达
- `formality_request_005` [housing]: 告知租客下个月房租缴纳日期提前两天，请按时转账以免产生滞纳金。 / checks: 下个月房租、提前两天、按时转账
- `formality_request_006` [public_service]: 向政府部门提交关于增加社区停车位需求的正式申请报告。 / checks: 增加停车位、需求申请、正式报告
- `formality_request_007` [technology]: 提醒用户系统将于今晚进行维护升级，期间无法登录使用。 / checks: 今晚、系统维护升级、无法登录
- `formality_request_008` [travel]: 告知游客景区将于下周起调整开放时间，建议提前查询最新时刻表。 / checks: 下周起调整、开放时间、查询最新时刻表
- `formality_request_009` [workplace]: 向同事说明下周项目例会改期至周五，并确认大家是否能参加。 / checks: 下周项目例会、改期至周五、确认是否能参加
- `formality_request_010` [daily_life]: 朋友间建议周末去新开的咖啡馆坐坐聊聊近况。 / checks: 建议周末去咖啡馆、聊聊近况、朋友间闲聊

### humor_neutral

- `humor_neutral_001` [community]: 邻居装修时噪音太大，导致楼上楼下居民无法正常休息。 / checks: 邻居装修、噪音大、影响休息
- `humor_neutral_002` [consumer]: 家里刚买的洗衣机洗了一次就出现漏水，导致地板湿了。 / checks: 洗衣机漏水、地板湿了、新购买产品
- `humor_neutral_003` [daily_life]: 图书馆里有人大声说话，干扰了其他正在看书的人。 / checks: 图书馆、大声说话、干扰他人
- `humor_neutral_004` [education]: 孩子在完成数学作业时，总是把加号看成减号，导致结果错误。 / checks: 数学作业、符号看错、结果错误
- `humor_neutral_005` [family]: 家庭聚餐时，大家想点外卖，但配送员找不到具体楼栋号。 / checks: 家庭聚餐、点外卖、找不到楼栋号
- `humor_neutral_006` [housing]: 租房时房东承诺的家电损坏，维修响应速度很慢。 / checks: 租房、家电损坏、维修慢
- `humor_neutral_007` [public_service]: 社区服务中心的办事流程太繁琐，排队时间过长导致体验不佳。 / checks: 社区服务中心、流程繁琐、排队时间长
- `humor_neutral_008` [technology]: 手机软件更新后，原本熟悉的界面布局变得复杂难用。 / checks: 软件更新、界面变复杂、难用
- `humor_neutral_009` [travel]: 周末去公园野餐时，突然下起大雨，大家来不及收拾食物。 / checks: 公园野餐、突然下雨、食物未收拾完
- `humor_neutral_010` [workplace]: 公司新来的同事经常把咖啡杯放在公共办公桌上，引起大家注意。 / checks: 新同事、咖啡杯放公共桌、引起注意

### objectivity_cat

- `objectivity_cat_001` [community]: 社区广场上老人打着太极，动作舒缓，周围绿树成荫。 / checks: 社区广场、打太极、动作舒缓、绿树
- `objectivity_cat_002` [consumer]: 超市货架上的苹果排列整齐，表面泛着红润的光泽。 / checks: 超市货架、苹果、排列整齐、红润光泽
- `objectivity_cat_003` [daily_life]: 老式座钟的指针缓慢移动，发出有节奏的滴答声，灰尘在光束中飞舞。 / checks: 座钟、指针移动、滴答声、灰尘
- `objectivity_cat_004` [education]: 孩子用蜡笔在纸上涂抹色彩，画出歪歪扭扭的圆形和线条。 / checks: 孩子、蜡笔、涂画、圆形线条
- `objectivity_cat_005` [environment]: 清晨的露珠挂在草叶尖，随着微风轻轻颤动，阳光洒下时瞬间蒸发。 / checks: 清晨、露珠、草叶、蒸发
- `objectivity_cat_006` [family]: 家庭餐桌上摆着刚煮好的米饭和几道家常菜，热气腾腾。 / checks: 家庭餐桌、米饭、家常菜、热气腾腾
- `objectivity_cat_007` [public_service]: 地铁站台的屏蔽门缓缓关闭，广播响起提示列车即将进站。 / checks: 地铁站台、屏蔽门关闭、广播提示、列车进站
- `objectivity_cat_008` [science]: 实验室培养皿中的细菌菌落呈现不规则扩散，边缘清晰可见。 / checks: 培养皿、细菌菌落、扩散、边缘清晰
- `objectivity_cat_009` [travel]: 旅游大巴在蜿蜒的山路上行驶，窗外是连绵起伏的青山。 / checks: 旅游大巴、山路、青山、连绵起伏
- `objectivity_cat_010` [technology]: 办公室的电脑屏幕亮起，显示着正在运行的代码窗口。 / checks: 办公室、电脑屏幕、代码窗口、运行中

### politeness_refusal

- `politeness_refusal_001` [community]: 社区组织呼吁全员参与周末义务打扫，请说明因故无法参加。 / checks: 拒绝参与打扫、周末时间、说明原因
- `politeness_refusal_002` [family]: 家人建议周末全家去外地旅游，请表示因预算不足而拒绝。 / checks: 拒绝旅游、预算不足、周末家庭
- `politeness_refusal_003` [consumer]: 顾客要求对已拆封商品进行无理由全额退款，请根据规则拒绝。 / checks: 拒绝退款、已拆封商品、无理由
- `politeness_refusal_004` [education]: 家长希望孩子参加周末额外的补习班，请委婉表示无法参加。 / checks: 拒绝补习班、周末时间、家长与孩子
- `politeness_refusal_005` [public_service]: 公共场馆临时关闭部分区域进行维护，请通知访客无法进入。 / checks: 无法进入、区域维护、通知访客
- `politeness_refusal_006` [technology]: 朋友提议用共享软件处理隐私数据，请明确拒绝该建议。 / checks: 拒绝软件使用、隐私数据、明确表态
- `politeness_refusal_007` [travel]: 旅行社推荐高价豪华游套餐，请直接表示不需要并拒绝购买。 / checks: 拒绝购买、豪华套餐、直接表达
- `politeness_refusal_008` [workplace]: 同事请求借用你的私人电脑处理非工作文件，请予以拒绝。 / checks: 拒绝借用电脑、私人设备、非工作用途
- `politeness_refusal_009` [daily_life]: 朋友邀请周末去爬山，但我不想去，请给出拒绝理由。 / checks: 拒绝爬山邀请、周末时间、给出理由
- `politeness_refusal_010` [health_non_diagnostic]: 健康中心推荐非必要的营养补充剂，请礼貌说明暂不需要。 / checks: 拒绝推荐、营养补充剂、暂不需要


## Potential A/B Naturalness Issues

- `concision_detail_001` may be too sparse for both endpoints: 说明如何选购适合的冬季保暖衣物。
- `concision_detail_023` may be too sparse for both endpoints: 提供选购有机蔬菜的简单辨别技巧。
- `concision_detail_024` may be too sparse for both endpoints: 分享一份周末家庭聚餐的菜单规划。
- `concision_detail_025` may be too sparse for both endpoints: 说明如何正确分类投放生活垃圾。
- `concision_detail_027` may be too sparse for both endpoints: 描述智能手机电池维护的常见误区。
- `concision_detail_028` may be too sparse for both endpoints: 推荐一款适合长途旅行的便携背包。
- `concision_detail_029` may be too sparse for both endpoints: 起草一份项目延期申请的内部报告。
- `concision_detail_034` may be too sparse for both endpoints: 请向客户解释退款流程及所需材料。
- `expertise_explain_021` may be too sparse for both endpoints: 讲述为什么洗澡时身体会感觉变轻。
- `expertise_explain_024` may be too sparse for both endpoints: 描述为什么旅行时耳朵会有胀痛感。
- `expertise_explain_028` may be too sparse for both endpoints: 为什么冬天穿厚衣服会感觉暖和？
- `expertise_explain_034` may be too sparse for both endpoints: 讲述为什么孩子长牙时容易流口水。
- `expertise_explain_035` may be too sparse for both endpoints: 为什么红绿灯交替闪烁来指挥交通？
- `expertise_explain_036` may be too sparse for both endpoints: 为什么煮饺子时水开了要加冷水？

## Potential Fact or Safety Risks

- No rule-based fact/safety risks were found in the final pool. This does not replace manual review.

from __future__ import annotations

import argparse
from pathlib import Path

from utils import ROOT, read_yaml, write_jsonl


SEED_CONTENTS = {
    "emotion_apology": [
        ("我接受快递延误的道歉，希望包裹能尽快送达。", ["接受道歉", "希望尽快送达"]),
        ("客服承认账单多收费用，我希望今天给出退款时间。", ["承认多收费用", "要求退款时间"]),
        ("餐厅把预约弄错了，我希望他们重新安排今晚的位置。", ["预约错误", "重新安排位置"]),
    ],
    "expertise_explain": [
        ("解释为什么天空在晴天看起来是蓝色的。", ["天空蓝色", "解释原因"]),
        ("解释磁铁为什么能吸住铁钉。", ["磁铁", "吸住铁钉"]),
        ("解释水烧开时为什么会冒泡。", ["水烧开", "冒泡"]),
    ],
    "objectivity_cat": [
        ("一只橘猫趴在窗边晒太阳，尾巴偶尔摆动。", ["橘猫", "窗边晒太阳", "尾巴摆动"]),
        ("雨后院子里的叶片上有水珠，光线变得柔和。", ["雨后叶片", "水珠", "柔和光线"]),
        ("桌上的白色杯子旁边放着一本合上的书。", ["白色杯子", "合上的书"]),
    ],
    "formality_request": [
        ("请对方明天下午三点前确认会议时间是否合适。", ["明天下午三点前", "确认会议时间"]),
        ("通知团队本周五提交项目进度更新。", ["本周五", "提交项目进度更新"]),
        ("请求同事帮忙检查附件中的数据表。", ["请求同事", "检查附件数据表"]),
    ],
    "politeness_refusal": [
        ("我不能参加周末聚会，因为已经有家庭安排。", ["不能参加聚会", "已有家庭安排"]),
        ("我不同意把上线时间提前到本周，因为测试还没完成。", ["不同意提前上线", "测试未完成"]),
        ("我无法借出电脑，因为明天自己还要使用。", ["无法借出电脑", "自己要使用"]),
    ],
    "concision_detail": [
        ("说明如何在出门前检查家里的门窗和电源。", ["检查门窗", "检查电源"]),
        ("介绍新员工第一天到办公室后需要完成的事项。", ["新员工第一天", "办公室事项"]),
        ("说明线上会议开始前需要做哪些准备。", ["线上会议", "会前准备"]),
    ],
    "humor_neutral": [
        ("手机电量只剩百分之五，但我还需要导航回家。", ["手机低电量", "需要导航回家"]),
        ("打印机在会议前突然卡纸，需要尽快处理。", ["打印机卡纸", "会议前", "尽快处理"]),
        ("咖啡洒在笔记本旁边，幸好没有碰到键盘。", ["咖啡洒出", "没有碰到键盘"]),
    ],
    "empathy_clinical": [
        ("朋友因为面试失败感到沮丧，需要一句回应。", ["面试失败", "感到沮丧", "回应朋友"]),
        ("同事最近压力很大，担心自己无法按时完成任务。", ["压力很大", "担心无法按时完成"]),
        ("家人检查后只是轻微不适，但仍然有些焦虑。", ["轻微不适", "仍然焦虑"]),
    ],
}


def build_rows(config: dict, axes: list[str] | None, per_axis: int) -> list[dict]:
    axis_defs = config["style_axes"]
    if axes:
        axis_defs = [axis for axis in axis_defs if axis["axis_id"] in set(axes)]

    rows: list[dict] = []
    for axis in axis_defs:
        axis_id = axis["axis_id"]
        seeds = SEED_CONTENTS.get(axis_id, [])
        if not seeds:
            continue
        for idx in range(per_axis):
            content, checks = seeds[idx % len(seeds)]
            variant = idx // len(seeds)
            suffix = f" 这是第{variant + 1}个语境变体。" if variant else ""
            rows.append(
                {
                    "scenario_id": f"{axis_id}_{idx + 1:03d}",
                    "axis_id": axis_id,
                    "content": content + suffix,
                    "style_a": axis["style_a"],
                    "style_b": axis["style_b"],
                    "attribute_name": axis["attribute_name"],
                    "content_checks": checks,
                    "length_range_zh_chars": axis["default_length_zh_chars"],
                    "safety_note": "避免脏话、威胁、医疗诊断或新增事实。",
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=ROOT / "configs/style_axes.yaml")
    parser.add_argument("--out", default=ROOT / "data/scenarios/scenarios_zh.jsonl")
    parser.add_argument("--per-axis", type=int, default=40)
    parser.add_argument("--axes", nargs="*", default=None)
    args = parser.parse_args()

    config = read_yaml(args.config)
    rows = build_rows(config, args.axes, args.per_axis)
    write_jsonl(args.out, rows)
    print(f"Wrote {len(rows)} scenarios to {Path(args.out)}")


if __name__ == "__main__":
    main()


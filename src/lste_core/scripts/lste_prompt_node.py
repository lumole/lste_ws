#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lste_prompt_node
- 订阅 /lste/task
- 调用 MiniCPM (通过 /home/zrz/Desktop/LSTE/Data_exchange/8B-05B.py) 生成 prompt_A / prompt_B
- 发布 /lste/prompts (latched)，然后主动退出以释放显存/内存
"""

import json
import subprocess

import rospy
from lste_msgs.msg import LstePrompts, LsteTask

from utils import prompts as prompt_utils


def task_to_task_parsed(msg: LsteTask) -> dict:
    """把 LsteTask 转成 8B-05B.py 期待的 task_parsed 结构。"""
    if msg.raw_json:
        try:
            data = json.loads(msg.raw_json)
            return data.get("task_parsed", data)
        except Exception as e:
            rospy.logwarn("Failed to parse raw_json in LsteTask: %s", e)

    # fallback: 用字段拼一个最小结构
    return {
        "target": {
            "name": msg.target_name,
            "attributes": list(msg.target_attributes),
        },
        "env": {
            "env_type_prior": list(msg.env_type_prior),
            "related_structures": list(msg.env_related_structures),
        },
        "obj_related": {
            "key_objects": list(msg.obj_key_objects),
            "negative_clues": list(msg.obj_negative_clues),
        },
        "target_ctx": {
            "left": msg.ctx_left,
            "right": msg.ctx_right,
        },
    }


class PromptNode:
    def __init__(self):
        rospy.init_node("lste_prompt_node")
        self.pub = rospy.Publisher("/lste/prompts", LstePrompts, queue_size=1, latch=True)

        # 允许通过参数覆盖 vLLM 地址和模型
        self.vllm_base_url = rospy.get_param("~vllm_base_url", str(prompt_utils.DEFAULT_VLLM_BASE_URL))
        self.vllm_model_name = rospy.get_param("~vllm_model_name", str(prompt_utils.DEFAULT_VLLM_MODEL_PATH))

        # 可选：生成后执行一个停止 vLLM 的命令（例如 pkill 或 tmux 控制命令）
        self.vllm_stop_command = rospy.get_param("~vllm_stop_command", "")
        self.republish_interval = float(rospy.get_param("~republish_interval", 5.0))
        self.latest_prompts = None

        rospy.loginfo("lste_prompt_node started. Waiting for /lste/task ...")
        self.sub = rospy.Subscriber("/lste/task", LsteTask, self.on_task, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(self.republish_interval), self.on_timer)

    def submit_vllm_stop(self):
        if not self.vllm_stop_command:
            return
        rospy.loginfo("Stopping external vLLM via command: %s", self.vllm_stop_command)
        try:
            subprocess.Popen(["bash", "-lc", self.vllm_stop_command])
        except subprocess.CalledProcessError as exc:
            rospy.logwarn("Stop command failed with return code %s", exc.returncode)
        except Exception as exc:
            rospy.logwarn("Failed to launch stop command: %s", exc)

    def on_task(self, msg: LsteTask):
        rospy.loginfo("Received task_id=%s, generating prompts via MiniCPM...", msg.task_id)
        task_parsed = task_to_task_parsed(msg)

        try:
            llm_prompt = prompt_utils.build_llm_prompt_from_task(task_parsed)
            llm_output = prompt_utils.call_local_llm(llm_prompt, self.vllm_base_url, self.vllm_model_name)
            llm_json = prompt_utils.parse_llm_json_output(llm_output)
        except Exception as e:
            rospy.logerr("Failed to call MiniCPM or parse its output: %s", e)
            rospy.signal_shutdown("prompt_failed")
            return

        # 1) prompt_A
        prompt_a_raw = llm_json.get("prompt_A", "")
        prompt_a = prompt_utils.fix_prompt_a(prompt_a_raw, task_parsed)

        # 2) prompt_B
        prompt_b_raw = llm_json.get("prompt_B", [])
        target_name = task_parsed.get("target", {}).get("name", "")
        prompt_b_list = prompt_utils.clean_prompt_b_list(prompt_b_raw, target_name)
        if not prompt_b_list:
            prompt_b_list = prompt_utils.build_default_prompt_b(task_parsed, target_name)

        # 强制加入上下文邻居
        target_ctx = task_parsed.get("target_ctx") or {}
        ctx_terms_raw = []
        for key in ("left", "right"):
            val = target_ctx.get(key)
            if val:
                ctx_terms_raw.append(str(val))
        if ctx_terms_raw:
            ctx_terms = prompt_utils.clean_prompt_b_list(ctx_terms_raw, target_name, drop_colors=False)
            existing = {p.lower() for p in prompt_b_list}
            for term in ctx_terms:
                if term.lower() not in existing:
                    prompt_b_list.append(term)
                    existing.add(term.lower())

        # 强制加入负向物体
        negative_clues = (task_parsed.get("obj_related") or {}).get("negative_clues", []) or []
        if negative_clues:
            neg_terms = prompt_utils.clean_prompt_b_list(negative_clues, target_name, drop_colors=False)
            existing = {p.lower() for p in prompt_b_list}
            for term in neg_terms:
                if term.lower() not in existing:
                    prompt_b_list.append(term)
                    existing.add(term.lower())

        # 发布
        out = LstePrompts()
        out.task_id = msg.task_id
        out.prompt_a = prompt_a
        out.prompt_b_terms = prompt_b_list
        try:
            out.raw_llm_output = json.dumps(llm_json, ensure_ascii=False)
        except Exception:
            out.raw_llm_output = str(llm_json)

        self.pub.publish(out)
        self.latest_prompts = out
        rospy.loginfo("Published /lste/prompts for task_id=%s. Will continue re-publishing until task completes.", msg.task_id)
        self.submit_vllm_stop()
        # 不关闭节点，继续定期发布

    def on_timer(self, _event):
        if self.latest_prompts is not None:
            self.pub.publish(self.latest_prompts)

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    PromptNode().spin()

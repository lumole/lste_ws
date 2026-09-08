# LSTE 文档索引

`07232026` 是本次归档版本日期，格式为 `MMDDYYYY`。文件名日期表示该文档在
本次目录整理中的记录版本；正文中的更新日期仍保留各文档自身的技术更新时间。

## operations

| 文档 | 内容 |
| --- | --- |
| [setup_07232026.md](operations/setup_07232026.md) | 工作区、依赖与环境搭建 |
| [startup_guide_07232026.md](operations/startup_guide_07232026.md) | 日常启动、controller 切换和 tmux 操作 |

## navigation

| 文档 | 内容 |
| --- | --- |
| [sappo_navigation_07232026.md](navigation/sappo_navigation_07232026.md) | LSTE global goal 到 SA-PPO 的现有集成 |
| [gp_subgoal_business_flow_07232026.md](navigation/gp_subgoal_business_flow_07232026.md) | GP subgoal 的业务流程 |
| [s_env_math_07232026.md](navigation/s_env_math_07232026.md) | S-env 的运行时数学说明 |
| [fixed_goal_navigation_debugging_07232026.md](navigation/fixed_goal_navigation_debugging_07232026.md) | 固定目标绕墙测试、故障定位和 TEB 基线 |
| [fixed_global_goal_integration_08052026.md](navigation/fixed_global_goal_integration_08052026.md) | 固定坐标接入正式 `/lste/final_goal`、配置与验证证据 |
| [online_slam_teb_navigation_08062026.md](navigation/online_slam_teb_navigation_08062026.md) | 未知环境在线 SLAM、frontier、Navfn、TEB 接入和端到端验证 |

## perception

| 文档 | 内容 |
| --- | --- |
| [visual_pipeline_07232026.md](perception/visual_pipeline_07232026.md) | 视觉链路、检测性能和框跟随 |
| [wedetect_tensorrt_integration_07232026.md](perception/wedetect_tensorrt_integration_07232026.md) | WeDetect TensorRT 接入与缓存 |

## research

| 文档 | 内容 |
| --- | --- |
| [open_vocabulary_detector_research_07232026.md](research/open_vocabulary_detector_research_07232026.md) | 开放词汇检测器的调研和选型 |
| [semantic_navigation_architecture_survey_08232026.md](research/semantic_navigation_architecture_survey_08232026.md) | 保留 LSTE global goal 的语义导航架构调研与实施建议 |
| [place_portal_workitem_study_09052026.md](research/place_portal_workitem_study_09052026.md) | Place-Portal-WorkItem 方法边界、基线、指标、复现实验和论文证据规则 |
| [task_conditioned_place_portal_belief_09062026.md](research/task_conditioned_place_portal_belief_09062026.md) | 任务条件语义信念、方向 WorkItem、主动门洞探测、持久 Portal 和目标射线验证架构 |
| [graph_first_architecture_innovation_09062026.md](research/graph_first_architecture_innovation_09062026.md) | 基于近期研究的图优先动作架构、实现边界与验证计划 |
| [portal_probe_value_pareto_09062026.md](research/portal_probe_value_pareto_09062026.md) | Portal probe 的硬约束、动作类别和 Pareto 选择层 |
| [place_phase_branch_transaction_09062026.md](research/place_phase_branch_transaction_09062026.md) | Place phase、Portal transaction、失败缓存失效和近期拓扑探索架构落地 |
| [branch_first_evidence_gated_exploration_09062026.md](research/branch_first_evidence_gated_exploration_09062026.md) | Branch-first 证据门控策略、事务身份校验、可证伪假设与消融指标 |
| [portal_decision_architecture_09062026.md](research/portal_decision_architecture_09062026.md) | 任务条件 Portal 决策层、无权重 Pareto 选择和外部架构依据 |
| [frontier_action_evidence_policy_09062026.md](research/frontier_action_evidence_policy_09062026.md) | Place 内 ObservationWorkItem 的事件类别、证据向量、Pareto 选择和实验边界 |
| [event_driven_evidence_graph_09062026.md](research/event_driven_evidence_graph_09062026.md) | Fast-Slow 事件图、可回放证据投影和目标/Portal 身份边界 |
| [graph_route_planner_09062026.md](research/graph_route_planner_09062026.md) | 持久 Place/Portal/WorkItem 图级路线规划器、第一条边物化和回归验证 |
| [controller_owned_route_lease_09062026.md](research/controller_owned_route_lease_09062026.md) | 持久执行中的控制器失败终端所有权与路线 lease 架构 |
| [recent_active_topology_architecture_09062026.md](research/recent_active_topology_architecture_09062026.md) | 2025-2026 未知环境语义拓扑探索调研、代码可用性和 ECAG 架构建议 |
| [durable_portal_action_compiler_09062026.md](research/durable_portal_action_compiler_09062026.md) | 持久 Portal 证据到 crossing route 的无 frontier 物化、preobserved crossing 和验证边界 |
| [target_viewpoint_option_ledger_09072026.md](research/target_viewpoint_option_ledger_09072026.md) | 目标拥有的视点选项账本：将局部 Navfn/TEB 失败与语义目标身份解耦 |
| [optimistic_slow_planning_09072026.md](research/optimistic_slow_planning_09072026.md) | 慢速图规划与快速执行的提案提交协议，避免长规划锁住终端事件 |

## development

| 文档 | 内容 |
| --- | --- |
| [global_frontier_code_layout_09052026.md](development/global_frontier_code_layout_09052026.md) | Global frontier 主节点与策略模块的职责划分 |
| [goal_manager_code_layout_09052026.md](development/goal_manager_code_layout_09052026.md) | Goal Manager 的检测、投影、路由和跟随模块划分 |
| [telemetry_code_layout_09052026.md](development/telemetry_code_layout_09052026.md) | 导航遥测与 TEB bridge 的模块边界和验证方式 |
| [targeted_navigation_experiments_09072026.md](development/targeted_navigation_experiments_09072026.md) | 从秒级最小故障回放逐步扩展到 Level 4 的实验阶梯 |
| [navigation_failure_evidence_09082026.md](development/navigation_failure_evidence_09082026.md) | 失败 episode、route stall 归因、目标 lease 释放和标准摘要 |

## reporting

| 文档 | 内容 |
| --- | --- |
| [teacher_progress_report_08052026.md](reporting/teacher_progress_report_08052026.md) | 面向老师的阶段性进展汇报：接手边界、完成工作、真实实验结果与后续计划 |
| [project_delivery_report_08052026.md](reporting/project_delivery_report_08052026.md) | 阶段性交差报告：架构、控制器对比、真实截图与视频证据 |
| [fixed_goal_video_evidence_07252026.md](reporting/fixed_goal_video_evidence_07252026.md) | 固定目标控制器录像与日志索引 |

## testing

| 文档 | 内容 |
| --- | --- |
| [office_building_benchmark_plan_08272026.md](testing/office_building_benchmark_plan_08272026.md) | 复杂办公楼 Gazebo 场景、任务路线、难度分层与评测计划 |

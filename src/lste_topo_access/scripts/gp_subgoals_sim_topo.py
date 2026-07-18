#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
局部 GP frontier 与 Access-Topo 探索节点。

不要从上到下逐行阅读本文件。它同时保留了早期“直接发布子目标”的
接口和当前 LSTE/Goal Manager 接口，正确的阅读入口是 VSGPNavGlb.step()。
业务流程图见 docs/gp_subgoal_business_flow.md。

运行时数据流：

    /sph_pcl + /rbt_pose
            |  (雷达球面点云、机器人二维位姿)
            v
    step(): 下采样 -> Sparse GP -> 预测占据/方差 -> 提取 frontier
            |                         |
            |                         +-> gp_nav_var / gp_nav_oc（RViz 调试）
            v
    /lste/gp_frontiers              （全部候选方向和面积）
    /lste/gp_frontier_dir           （旧的单方向接口）
            v
    lste_goal_manager.py            （结合任务状态后发布 /lste/final_goal）

Access-Topo 是与 GP 并行的长期记忆：它把机器人每走一段距离的位置
记为 anchor；在多个稳定 frontier 同时出现时记录路口分支；没有可走
frontier 时发出回退目标。运行快照会定期保存为 access_topo_<run_id>.json。
"""

#from typing import Tuple, Optional
#import tempfile
import faulthandler; faulthandler.enable()
import os
from pathlib import Path
import warnings
import json
import yaml
import sys

#import io
import os
#import math
import numpy as np
import math
from time import time
from scipy import stats
from scipy.spatial import distance
#from datetime import datetime

#import pcl
import rospy
import ros_numpy
from tf.transformations import quaternion_from_euler, euler_from_quaternion

### import ros msgs
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from sensor_msgs import point_cloud2
from std_msgs.msg import Header, Bool, UInt8, String
# defined msg
#from gp_subgoal.msg import PosePcl2
from geometry_msgs.msg import PoseStamped, PointStamped, Vector3Stamped
from geometry_msgs.msg import Pose2D, Point, Pose
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker
from lste_msgs.msg import LsteFrontiers, LsteState

### import opencv
import cv2

#from cv_bridge import CvBridge, CvBridgeError  #after cv2


### to disable GPU for GP training and prediction: 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
### to select GPU
# tf.device("gpu:0")

#######disabled warning when import tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def _resolve_topo_path(rel_path: str) -> str:
    """Resolve a path relative to lste_topo_access package or LSTE_WS."""
    if os.path.isabs(rel_path) and rel_path != "":
        return rel_path
    ws = os.environ.get("LSTE_WS")
    if ws:
        return os.path.join(ws, "src", "lste_topo_access", rel_path) if rel_path else os.path.join(ws, "src", "lste_topo_access", "topo_tree", "tree")
    try:
        import rospkg
        pkg_path = rospkg.RosPack().get_path("lste_topo_access")
        return os.path.join(pkg_path, rel_path) if rel_path else os.path.join(pkg_path, "topo_tree", "tree")
    except Exception:
        # Fallback: resolve relative to this file
        pkg = Path(__file__).resolve().parents[1]  # lste_topo_access/
        return str(pkg / rel_path) if rel_path else str(pkg / "topo_tree" / "tree")

### import tensrflow and gpflow and related lib
import tensorflow as tf
import gpflow
#from gpflow.config import default_float
#from gpflow.ci_utils import ci_niter
#from gpflow.utilities import to_default_float
from gpflow import set_trainable

#from gpflow.utilities import print_summary

# -------- helper --------
def wrap_angle(x: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def angle_diff(a: float, b: float) -> float:
    """Minimal absolute angular difference."""
    return abs(wrap_angle(a - b))


def safe_load_yaml(path):
    """Load YAML file, return dict or {} on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

#### configurations
warnings.filterwarnings("ignore")
gpflow.config.set_default_float(np.float32)
np.random.seed(0)
tf.random.set_seed(0)

""" @brief: class to construct a 2D Sparse GP using gpflow """


class SGP2D:
    """Sparse Gaussian Process 的薄封装。

    输入为球面方向 ``(theta, alpha)``，输出为该方向的占据观测值。
    VSGPNavGlb 每轮都会创建该对象、绑定当前点云训练数据，并在固定的
    球面网格上调用 ``model.predict_f`` 得到占据均值与预测方差。
    """

    def __init__(self):
        self.model = None
        self.data = None
        self.kernel1 = None
        self.kernel2 = None
        self.kernel = None
        self.indpts = None
        self.meanf = gpflow.mean_functions.Constant(0)

    """ @brief: to initialize the parameters to the RQ kernel """

    def set_kernel_param(self, ls1, ls2, var, alpha, noise, noise_var):
        self.kernel1 = gpflow.kernels.RationalQuadratic(
            lengthscales=[ls1, ls2])
        self.kernel1.variance.assign(var)
        self.kernel1.alpha.assign(alpha)
        self.kernel2 = gpflow.kernels.White(noise)
        self.kernel2.variance.assign(noise_var)
        self.kernel = self.kernel1 + self.kernel2

    """ @brief: to intialize training data to empty data set """

    def set_empty_data(self):
        in_init, out_init = np.zeros((0, 2)), np.zeros((0, 1))  # input dim:2
        mdl_in = tf.Variable(in_init, shape=(None, 2), dtype=tf.float32)
        mdl_out = tf.Variable(out_init, shape=(None, 1), dtype=tf.float32)
        self.data = (mdl_in, mdl_out)

    """ @brief: to intialize training data as tf tensors """

    def set_training_data(self, d_in, d_out):
        mdl_in = tf.Variable(d_in, dtype=tf.float32)
        mdl_out = tf.Variable(d_out, dtype=tf.float32)
        self.data = (mdl_in, mdl_out)

    """ @brief: to intialize indusing points as empty set"""

    def set_empty_indpts(self):
        indpts_init = np.zeros((0, 2))
        self.indpts = tf.Variable(indpts_init,
                                  shape=(None, 2),
                                  dtype=tf.float32)

    """ @brief: to intialize indusing points from the training data"""

    def set_indpts_from_training_data(self, indpts_size, in_data):
        data_size = np.shape(in_data)[0]
        pts_idx = range(0, data_size, int(data_size / indpts_size))
        self.indpts = in_data[[idx for idx in pts_idx], :]

    """ @brief: to intialize kernel mean"""

    def set_init_mean(self, init_mean):
        self.meanf = gpflow.mean_functions.Constant(init_mean)

    """ @brief: to intiate a SGP model using gpflow lib"""

    def set_sgp_model(self):
        self.model = gpflow.models.SGPR(self.data,
                                        self.kernel,
                                        self.indpts,
                                        mean_function=self.meanf)

    """ @brief: to select which params set to trainable and which are fixed"""

    def select_trainable_param(self):
        set_trainable(self.kernel1.variance, False)
        set_trainable(self.kernel1.lengthscales, False)
        set_trainable(self.kernel2.variance, False)
        set_trainable(self.model.likelihood.variance, False)

    """ @brief: minimize the loss during training"""

    def minimize_loss(self):
        self.model.training_loss_closure(
        )

    """ @brief: to select optimizer: here we are using adam"""

    def adam_optimize_param(self):
        # tm= time()
        optimizer = tf.optimizers.Adam()
        optimizer.minimize(self.model.training_loss,
                           self.model.trainable_variables)
        # print("SGP2D:: adam_optimize_param time: ", time() - tm)


""" @brief: class to train a 2D Sparse GP using the ocupancy surface, 
    to predict the variance surface, and to learn subgoals around the robot """


class VSGPNavGlb:
    """GP frontier、局部子目标和拓扑回退的总控制器。

    本类同时维护四类状态：

    * 感知缓存：``pose`` 与 ``pcl_arr``，分别由 ROS 回调更新；
    * GP 中间结果：训练点、预测网格、frontier 和候选子目标；
    * Access-Topo 长期状态：anchor、路口分支、回退栈和强制前进方向；
    * LSTE 协作状态：当前 state、Goal Manager 的 final_goal 和 active_mode。

    所有重计算发生在 ``step``；回调只更新最新消息，避免在回调线程里
    训练 GP。
    """

    def __init__(self):
        ### Node initialization
        rospy.init_node("gp_subgoal")
        print("##############################################")
        print("              Initialize gp_subgoal           ")
        print("##############################################")

        # 先定义关键属性，避免回调抢跑时属性不存在
        self.pose = None
        self.pcl_arr = None
        self.header = Header()
        self.header.seq = 0
        self.header.stamp = None
        self.header.frame_id = "odom"

        ### subscriber to pose and occupancy surface
        #        self.pose_pcl_sub = rospy.Subscriber("pose_pcl",
        #                                             PosePcl2,
        #                                             self.pose_pcl_cb,
        #                                             queue_size=1)

        # 入口感知数据。oc_srfc_proj 将雷达原始点云转换为 /sph_pcl，并从
        # 里程计提取 /rbt_pose；本节点只缓存两者，实际计算在 step() 中执行。
        self.rbt_pose_sub = rospy.Subscriber("rbt_pose",
                                             Pose2D,
                                             self.pose_cb,
                                             queue_size=1)

        self.sph_pcl_sub = rospy.Subscriber("sph_pcl",
                                            PointCloud2,
                                            self.sph_pcl_cb,
                                            queue_size=1)
        # 以下大部分是 GP 中间结果和 RViz 调试数据。当前 LSTE 主链最重要
        # 的对外接口位于后面的 /lste/gp_frontiers 与 /lste/gp_frontier_dir。
        ## variance surface
        self.gp_var_pub = rospy.Publisher("gp_nav_var",
                                          PointCloud2,
                                          queue_size=1)
        ## occupancy surface
        self.gp_oc_pub = rospy.Publisher("gp_nav_oc",
                                         PointCloud2,
                                         queue_size=1)
        ## navigation (points) subgoals in the robot frame (Ouster frame)
        self.gp_nav_pts_pub = rospy.Publisher("gp_nav_pts",
                                              PointCloud2,
                                              queue_size=1)

        ## navigation (points) subgoals in the odom frame (odom frame)
        self.gp_actul_xy_subgls_pub = rospy.Publisher("gp_nav_actul_xy_gls",
                                                      PointCloud2,
                                                      queue_size=1)

        ## recommended subgoal in odom frame to DRL
        self.gp_rcmndd_subgl_pub = rospy.Publisher("gp_subgoal",
                                                   PoseStamped,
                                                   queue_size=1)

        ## recommended subgoal in os_sensor frame to DRL
        self.gp_subgl_pub = rospy.Publisher("gp_subgoal_os",
                                            PoseStamped,
                                            queue_size=1)

        ## recommended subgoal in rviz
        self.gp_subgl_rviz = rospy.Publisher("gp_subgl_rviz",
                                             PointCloud2,
                                             queue_size=1)
        # LSTE 接口：为 PASS 态提供 frontier（PoseStamped，odom）
        self.lste_gp_frontier_pub = rospy.Publisher("/lste/gp_frontier",
                                                    PoseStamped,
                                                    queue_size=1)

        # ---- LSTE / Access-Topo 状态 ----
        # 先创建状态，再注册后续订阅者，避免 ROS 回调早到而访问未初始化属性。
        self.visited_positions = []
        self.int_visited_positions = []
        self.closed = []
        self.change_flag = True
        self.final_goal_received = False
        # LSTE state 缓存（仅观察/日志，不影响现有逻辑）
        self.lste_state = 0  # 0:PASS 1:SUSPICIOUS 2:LOCKED 3:EXHAUSTED(未用)
        self.lste_subtype = ""
        # 运行级元信息：确保一次运行只写一个 JSON
        self.run_id = int(time())
        self.run_start = rospy.Time.now().to_sec()
        self.run_end = None
        self.state_events = []
        self.profile_events = []
        self.goal_events = []
        self.visited_events = []
        self.active_mode = "unknown"
        self.vanish_distance = None  # 由消失点估计的前向距离（当前停用）
        # PASS 全局目标 4-方向锚定骨架
        self.origin_set = False
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.origin_yaw = 0.0
        self.headings = []  # H0..H3
        self.dir_idx = 0
        self.last_switch_xy = None
        # 分叉 commit
        self.commit_active = False
        self.commit_heading = 0.0
        self.commit_start_xy = (0.0, 0.0)
        # 初始化检测历史记录：用于 interest subgoal 过滤
        self.detected_interest_points = []
        self.match_threshold = 0.5  # 例如，0.5米
        # --- Access-Topo 参数与数据 ---
        # anchor 是沿机器人实际轨迹定期落下的“路钉”；branch 是在 anchor
        # 附近观察到但尚未选择的方向；backtrack_stack 用于无路可走时回退。
        self.load_access_topo_config()
        # 分支上限/路口冷却
        self.max_pending_per_node = getattr(self, "max_pending_per_node", 2)
        self.junction_cooldown_anchors = getattr(self, "junction_cooldown_anchors", 4)
        self.last_junction_anchor = None  # 最近一次生成“路口候选”的 anchor id
        self.anchor_nodes = []  # [{'id', 'x','y','stamp','prev','branches':[]}]
        self.anchor_last_xy = None
        self.anchor_last_id = None
        self.backtrack_stack = []  # list of (node_id, branch_id, weight)
        self.current_backtrack = None  # (node_id, branch_id)
        self.backtrack_start_pose = None  # {"x","y"}
        self.backtrack_start_anchor = None
        self.backtrack_path = []  # anchor id list for the current backtrack (newly generated path)
        self.commit_goal = None  # {"x","y"} commit阶段的目标点
        # 历史回溯段：[{start_pose,start_anchor,path,end_anchor,status,finished,target}]
        self.backtrack_history = []
        self.no_frontier_since = None
        self.no_frontier_start_pose = None
        self.no_frontier_start_anchor = None
        self.backtrack_recording = False  # whether we are recording new anchors during backtrack
        # 无兴趣点时的“回到起点”目标（anchor id），None 表示未触发
        self.return_home_target = None
        self.access_mode = 0  # 0 forward, 1 backtrack
        self.forced_heading_world = None
        self.forced_start_xy = None
        self.forced_branch = None
        self.mode2_start_time = None  # commit阶段开始时间，用于免回退窗口
        self.cluster_track = []  # tracking clusters for stability
        self.pending_junctions = []  # [{"node_id":int,"candidates":[{"heading":float,"weight":float,"last_seen":float}]}]
        # 路口 finalize 后的冷却（防止连续误触发）
        self.junction_post_finalize_cooldown = getattr(self, "junction_post_finalize_cooldown", 4)
        self.last_finalized_anchor = None
        # “双峰持续存在”判定：用于过滤伪路口
        self.multi_peak_since = None
        # 连续无 frontier 的计时，用于延迟回溯触发
        self.no_frontier_since = None
        self.force_window_rad = np.deg2rad(self.force_window_deg)
        self.cluster_eps_rad = np.deg2rad(self.cluster_eps_deg)
        self.junction_peak_min_sep_rad = np.deg2rad(getattr(self, "junction_peak_min_sep_deg", 30.0))
        self.junction_exit_confirm_sec = getattr(self, "junction_exit_confirm_sec", 0.5)
        # 路口决策：走出一定距离后用实际运动方向确定“被选分支”
        self.junction_decision_dist = getattr(self, "junction_decision_dist", self.anchor_step_dist)
        self.junction_same_dir_gate_rad = np.deg2rad(getattr(self, "junction_same_dir_gate_deg", 30.0))
        self.junction_finalize_dist = getattr(self, "junction_finalize_dist", self.junction_decision_dist * 1.5)
        self.junction_finalize_timeout = getattr(self, "junction_finalize_timeout", 6.0)
        # 路口分支约束：chosen 仅 1 个，pending 最多 2 个
        self.junction_max_pending = 2
        # 保存 topo tree（文件名已在 load_access_topo_config 冻结）
        if not getattr(self, "topo_save_file", None):
            self.topo_save_file = os.path.join(self.topo_save_dir, f"access_topo_{self.run_id}.json")
        self.last_topo_save_time = 0.0

        #zcy
        self.zcyrbt_pose_sub = rospy.Subscriber("rbt_pose",
                                                Pose2D,
                                                self.zcypose_cb,
                                                queue_size=1)
        # 消失点距离（走廊消失点 → 沿机器人前向的距离）
        self.vanish_sub = rospy.Subscriber("/vanish_point",
                                           Pose,
                                           self.vanish_cb,
                                           queue_size=1)

        ## interest subgoal in odom frame to topo
        self.gp_interest_subgl_pub = rospy.Publisher("interest_subgoal",
                                                     PoseStamped,
                                                     queue_size=1)

        self.closed_marker_pub = rospy.Publisher("closed_markers", Marker, queue_size=1)
        #标志位
        self.change_flag_pub = rospy.Publisher("change_flag", Bool, queue_size=1)
        self.change_flag_sub = rospy.Subscriber("change_flag", Bool, self.flag_cb, queue_size=1)
        # Access-Topo 回退接口
        self.backtrack_goal_pub = rospy.Publisher("/lste/access_topo/backtrack_goal", PoseStamped, queue_size=1)
        self.access_mode_pub = rospy.Publisher("/lste/access_topo/mode", UInt8, queue_size=1)
        # 动态 final goal：上层发布 PoseStamped 到 /lste/final_goal
        self.final_goal_sub = rospy.Subscriber("/lste/final_goal",
                                               PoseStamped,
                                               self.final_goal_cb,
                                                queue_size=1)
        # LSTE 状态（仅缓存/日志，后续策略切换再使用）
        self.lste_state_sub = rospy.Subscriber("/lste/state",
                                               LsteState,
                                               self.lste_state_cb,
                                               queue_size=1)
        # goal_manager 可选发布的 active_mode（explore_pass / explore_sus_c / catch_target / catch_ctx）
        self.active_mode_sub = rospy.Subscriber("/lste/access_topo/active_mode",
                                                String,
                                                self.active_mode_cb,
                                                queue_size=1)
        # 对外发布 frontier 最大方向（给 Goal Manager 使用），vector.x=theta_rel(rad)，vector.y=area
        self.frontier_dir_pub = rospy.Publisher("/lste/gp_frontier_dir",
                                                Vector3Stamped,
                                                queue_size=1)
        # 对外发布全量 frontier 列表（theta_rel + area）
        self.frontiers_pub = rospy.Publisher("/lste/gp_frontiers",
                                             LsteFrontiers,
                                             queue_size=1)

        ## variables to store data for GP training and prediction
        # 使用私有参数（~names）并提供默认值，避免未设置参数时直接抛异常
        self.oc_srfc_rds = rospy.get_param('~oc_srfc_rds', 5.0)
        self.pcl_skp = rospy.get_param('~pcl_skp', 3)
        self.pose = None
        self.pcl_arr = None
        self.org_unq_thetas = None
        self.pcl_unq_thetas = None
        self.pcl_thetas = None
        self.pcl_alphas = None
        self.pcl_rds = None
        self.pcl_oc = None
        self.pcl_sz = None

        ## grid to reconstruct the variance surface 
        self.gp_grd = None
        self.gp_grd_w = None
        self.gp_grd_h = None
        self.gp_grd_ths = None
        self.gp_grd_als = None
        self.gp_grd_oc = None
        self.gp_grd_rds = None
        self.gp_grd_var = None
        self.sample_gp_nav_grid()

        ## variables to store GPFrontiers and their cost 
        self.gp_nav_pt = None
        self.gp_nav_pts = None
        self.gap_utlty_fun = None
        self.gp_nav_frntr_cntrs = None
        self.gp_nav_frntr_areas = None

        ## GP param
        self.gp_nav_indpts_sz = rospy.get_param('~gp_nav_indpts_sz', 400)
        self.gp_nav_var_thrshld = rospy.get_param('~gp_nav_var_thrshld', 0.03)

        ## cost function param
        self.gap_k_dir = rospy.get_param('~gap_k_dir', 4.0)
        self.gap_k_dst = rospy.get_param('~gap_k_dst', 5.0)
        self.gp_nav_goal_dst = rospy.get_param('~gp_nav_goal_dst', 5.0)

        ## visualization param
        self.gp_nav_var_img_viz = rospy.get_param('~gp_nav_var_img_viz', False)
        self.gp_nav_var_viz = rospy.get_param('~gp_nav_var_viz', 5.0)
        # frontier 日志（可选）
        self.frontier_log_enabled = rospy.get_param('~frontier_log', False)
        self.frontier_log_dir = rospy.get_param(
            '~frontier_log_dir',
            _resolve_topo_path('topo_tree/frontier_log')
        )
        self.frontier_log_file = None
        self.frontier_log_fh = None
        self._stdout_orig = None
        self._stderr_orig = None
        if self.frontier_log_enabled:
            try:
                self.frontier_log_dir = os.path.expanduser(self.frontier_log_dir)
                os.makedirs(self.frontier_log_dir, exist_ok=True)
                self.frontier_log_file = os.path.join(
                    self.frontier_log_dir,
                    f"frontier_{self.run_id}.log"
                )
                rospy.loginfo("gp_subgoal: frontier_log enabled, writing to %s", self.frontier_log_file)
                self.frontier_log_fh = open(self.frontier_log_file, "a", encoding="utf-8", buffering=1)
                self.frontier_log_fh.write(f"# frontier_log run_id={self.run_id}\n")

                class _TeeStream:
                    """Mirror stdout/stderr to log with timestamp, keep console output."""

                    def __init__(self, main_stream, log_stream):
                        self._main = main_stream
                        self._log = log_stream

                    def write(self, data):
                        try:
                            self._main.write(data)
                        except Exception:
                            pass
                        if not self._log:
                            return
                        try:
                            for chunk in data.splitlines(True):
                                ts = 0.0
                                try:
                                    ts = rospy.Time.now().to_sec()
                                except Exception:
                                    pass
                                if ts <= 0.0:
                                    ts = time()
                                if chunk.endswith("\n"):
                                    content = chunk[:-1]
                                    newline = "\n"
                                else:
                                    content = chunk
                                    newline = ""
                                if content.strip():
                                    self._log.write(f"[{ts:.3f}] {content}{newline}")
                                elif newline:
                                    self._log.write(f"[{ts:.3f}]{newline}")
                            self._log.flush()
                        except Exception:
                            pass

                    def flush(self):
                        try:
                            self._main.flush()
                        except Exception:
                            pass
                        try:
                            if self._log:
                                self._log.flush()
                        except Exception:
                            pass

                self._stdout_orig = sys.stdout
                self._stderr_orig = sys.stderr
                sys.stdout = _TeeStream(sys.stdout, self.frontier_log_fh)
                sys.stderr = _TeeStream(sys.stderr, self.frontier_log_fh)
            except Exception as exc:
                rospy.logwarn("gp_subgoal: frontier_log disabled (init failed: %s)", exc)
                self.frontier_log_enabled = False
                try:
                    if self.frontier_log_fh:
                        self.frontier_log_fh.close()
                except Exception:
                    pass
                self.frontier_log_fh = None
                self.frontier_log_file = None

        ## 全局目标动态更新参数（PASS 默认）：每隔 gl_update_period 秒，沿最“空旷”frontier 方向外推 gl_update_forward_dist 米
        # 默认 5s 更新一次（可通过 ~gl_update_period 覆盖）
        self.gl_update_period = rospy.get_param('~gl_update_period', 5.0)
        self.gl_update_forward_dist = rospy.get_param('~gl_update_forward_dist', 10.0)
        # 让第一次调用时立即可以更新（而不是再等待 gl_update_period）
        self.last_gl_update_time = rospy.Time.now().to_sec() - self.gl_update_period

        ## final goal（默认先用参数，后面根据起始位姿自动调整）
        self.gl_x = rospy.get_param('~gl_x', -4.0)
        self.gl_y = rospy.get_param('~gl_y', -16.0)
        self.gl_yaw = rospy.get_param('~gl_yaw', 0.0)
        self.gl_wrt_odom = np.array([self.gl_x, self.gl_y, 1], dtype="float32")
        self.gp_nav_frame_id = "os_sensor"
        self.gp_nav_var_pblsh = True

        self.gp_nav_xypts = None
        self.goal_published = False

        ## limit the environemnent to maximum dimension
        self.map_2d_h = 500
        self.map_2d_w = 500

        ## for generated pointcloud 
        self.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1)
        ]

        #zcy
        self.visited_positions = []
        self.int_visited_positions = []
        self.closed = []
        self.change_flag = True
        self.final_goal_received = False

        # 初始化检测历史记录
        # 使用列表保存字典，每个字典包含点的坐标和检测次数
        self.detected_interest_points = []

        # 定义匹配的距离阈值（根据实际情况调整）
        self.match_threshold = 0.5  # 例如，0.5米

        # 是否已用起始位姿设置过默认全局 goal（避免重复）
        self._start_goal_seeded = False

    def lste_state_cb(self, state_msg: LsteState):
        """缓存 /lste/state，当前仅用于日志观察。"""
        try:
            new_state = int(state_msg.state)
        except Exception:
            return
        new_sub = state_msg.subtype or ""
        if new_state != self.lste_state or new_sub != self.lste_subtype:
            rospy.loginfo("gp_subgoal: /lste/state changed state=%s subtype=%s",
                          str(new_state), new_sub)
            self.state_events.append({
                "t": rospy.Time.now().to_sec(),
                "state": new_state,
                "subtype": new_sub,
                "pose": {"x": getattr(self.pose, "x", None), "y": getattr(self.pose, "y", None)},
                "anchor_id": self.anchor_last_id,
                "profile": self.current_profile,
                "active_mode": self.active_mode,
            })
        self.lste_state = new_state
        self.lste_subtype = new_sub
        # 根据 state/subtype 切换 explore 配置：Sus-C 用 fine，其余用 coarse
        desired_profile = "sus_c" if (new_state == 1 and new_sub == "Sus-C") else "pass"
        if desired_profile != getattr(self, "current_profile", None):
            self.apply_access_topo_profile(desired_profile, reason="/lste/state")

    def active_mode_cb(self, msg: String):
        mode = msg.data or "unknown"
        if mode != self.active_mode:
            rospy.loginfo("gp_subgoal: active_mode changed to %s", mode)
        self.active_mode = mode

    def final_goal_cb(self, msg: PoseStamped):
        """Update final goal from /lste/final_goal (odom frame PoseStamped)."""
        self.gl_x = msg.pose.position.x
        self.gl_y = msg.pose.position.y
        q = msg.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.gl_yaw = yaw
        self.gl_wrt_odom = np.array([self.gl_x, self.gl_y, 1], dtype="float32")
        self.final_goal_received = True
        self.goal_events.append({
            "t": rospy.Time.now().to_sec(),
            "goal": {"x": self.gl_x, "y": self.gl_y, "yaw": self.gl_yaw},
            "mode": self.active_mode,
            "profile": self.current_profile,
            "state": self.lste_state,
            "subtype": self.lste_subtype,
            "anchor_id": self.anchor_last_id,
            "pose": {"x": getattr(self.pose, "x", None), "y": getattr(self.pose, "y", None)},
        })

    def vanish_cb(self, msg: Pose):
        """消失点推断的前方距离，单位 m。"""
        try:
            dist = float(msg.position.x)
            if dist > 0:
                self.vanish_distance = dist
        except Exception:
            pass

    def load_access_topo_config(self):
        """加载 Access-Topo 配置并冻结本次运行的快照文件路径。

        ``pass`` 与 ``sus_c`` 可以使用不同的拓扑参数。profile 切换会改变
        分支判定/回退等阈值，但不会更换本次运行的 JSON 文件，确保一份
        access_topo_<run_id>.json 记录完整的一次实验。
        """
        # 两套配置路径：未提供 fine 时回落到 coarse
        default_cfg = _resolve_topo_path('topo_tree/cfgs/access_topo.yaml')
        cfg_path_pass = rospy.get_param('~access_topo_config_pass',
                                        rospy.get_param('~access_topo_config', default_cfg))
        cfg_path_sus_c = rospy.get_param('~access_topo_config_sus_c', cfg_path_pass)
        self.cfg_pass = safe_load_yaml(cfg_path_pass) or {}
        self.cfg_sus_c = safe_load_yaml(cfg_path_sus_c) or {}
        # 冻结本次运行的保存路径/文件（不随 profile/state 变化）
        base_root = self.cfg_pass.get('topo_tree_root') or \
                    rospy.get_param('~topo_tree_root', _resolve_topo_path(''))
        default_save_dir = os.path.join(base_root, rospy.get_param('~run_name', "run"))
        self.topo_save_dir = rospy.get_param('~topo_save_dir',
                                             self.cfg_pass.get('topo_save_dir', default_save_dir))
        self.topo_save_file = os.path.join(self.topo_save_dir, f"access_topo_{self.run_id}.json")
        self.current_profile = "pass"
        self.apply_access_topo_profile(self.current_profile, reason="init")

    def apply_access_topo_profile(self, profile: str, reason: str = ""):
        """按 profile 应用配置并更新派生参数。profile: 'pass' or 'sus_c'."""
        cfg_file = self.cfg_pass if profile != "sus_c" else self.cfg_sus_c
        if not isinstance(cfg_file, dict):
            cfg_file = {}
        prev = getattr(self, "current_profile", None)
        self.current_profile = profile
        if prev is not None and prev != profile:
            rospy.loginfo("Access-Topo profile switch: %s -> %s (reason=%s)", prev, profile, reason or "")
            self.profile_events.append({
                "t": rospy.Time.now().to_sec(),
                "from": prev,
                "to": profile,
                "reason": reason,
                "pose": {"x": getattr(self.pose, "x", None), "y": getattr(self.pose, "y", None)},
                "anchor_id": self.anchor_last_id,
            })

        # 基础路径与测试名
        self.test_name = cfg_file.get('test_name', rospy.get_param('~test_name', 'default'))
        self.topo_tree_root = cfg_file.get('topo_tree_root') or \
                               rospy.get_param('~topo_tree_root', _resolve_topo_path(''))
        # 参数读取（YAML 优先，ROS param 次之）
        self.anchor_step_dist = cfg_file.get('anchor_step_dist', rospy.get_param('~anchor_step_dist', 1.0))
        self.backtrack_arrive_dist = cfg_file.get('backtrack_arrive_dist',
                                                 rospy.get_param('~backtrack_arrive_dist', 1.0))
        self.visited_thresh = cfg_file.get('visited_thresh', rospy.get_param('~visited_thresh', 3.0))
        self.force_window_deg = cfg_file.get('force_window_deg', rospy.get_param('~force_window_deg', 25.0))
        # 前方可行距离过滤开关（默认关闭；避免在走廊/路口远处因短暂预测误差把 frontier 过滤掉）
        self.enable_min_frontier_clearance = bool(
            cfg_file.get('enable_min_frontier_clearance',
                         rospy.get_param('~enable_min_frontier_clearance', False))
        )
        # 前方可行距离过滤阈值（小于该距离的 frontier 视为不可走）
        self.min_frontier_clearance = cfg_file.get('min_frontier_clearance',
                                                   rospy.get_param('~min_frontier_clearance', 0.0))
        self.commit_dist = cfg_file.get('commit_dist', rospy.get_param('~commit_dist', 2.0))
        self.cluster_eps_deg = cfg_file.get('cluster_eps_deg', rospy.get_param('~cluster_eps_deg', 15.0))
        self.cluster_stable_duration = cfg_file.get('cluster_stable_duration',
                                                    rospy.get_param('~cluster_stable_duration', 3.0))
        self.cluster_lost_timeout = cfg_file.get('cluster_lost_timeout',
                                                 rospy.get_param('~cluster_lost_timeout', 1.0))
        # 路口决策：走出一定距离后用实际运动方向确定“被选分支”
        self.junction_decision_dist = cfg_file.get(
            'junction_decision_dist', rospy.get_param('~junction_decision_dist', self.anchor_step_dist))
        self.junction_same_dir_gate_deg = cfg_file.get(
            'junction_same_dir_gate_deg', rospy.get_param('~junction_same_dir_gate_deg', 30.0))
        # 路口窗口：离开路口多远/等待多久后固化分支（距离优先）
        self.junction_finalize_dist = cfg_file.get(
            'junction_finalize_dist', rospy.get_param('~junction_finalize_dist', self.anchor_step_dist * 1.5))
        self.junction_finalize_timeout = cfg_file.get(
            'junction_finalize_timeout', rospy.get_param('~junction_finalize_timeout', 6.0))
        # 路口会话触发：只有“双峰持续存在”才认为进入路口会话（用于过滤伪路口）
        self.junction_enter_confirm_sec = cfg_file.get(
            'junction_enter_confirm_sec', rospy.get_param('~junction_enter_confirm_sec', 0.8))
        # 被认为是“峰”的最小持续时间（秒）：单峰出现太短不计入多峰判断
        self.junction_peak_min_age = cfg_file.get(
            'junction_peak_min_age', rospy.get_param('~junction_peak_min_age', 0.6))
        # 双峰最小角度间隔（度）：避免同一方向被误分成两簇
        self.junction_peak_min_sep_deg = cfg_file.get(
            'junction_peak_min_sep_deg',
            rospy.get_param('~junction_peak_min_sep_deg', max(30.0, float(self.cluster_eps_deg) * 1.2)))
        # 第二峰权重阈值：绝对值 + 相对比例（用于过滤很小的伪峰）
        self.junction_peak_min_weight = cfg_file.get(
            'junction_peak_min_weight', rospy.get_param('~junction_peak_min_weight', 0.0))
        self.junction_second_weight_ratio = cfg_file.get(
            'junction_second_weight_ratio', rospy.get_param('~junction_second_weight_ratio', 0.15))
        # 相邻 anchor 的兴趣方向合并距离（米），用于全局去重
        self.branch_merge_dist = cfg_file.get('branch_merge_dist',
                                             rospy.get_param('~branch_merge_dist', 1.0))
        # 连续无可用 frontier 的触发时间（秒）；同时要求 access_mode==1 才真正回溯
        self.no_frontier_trigger_sec = cfg_file.get(
            'no_frontier_trigger_sec', rospy.get_param('~no_frontier_trigger_sec', 3.0))
        # commit 固定目标距离（沿强制方向一次性前推，不再动态 carrot）
        self.commit_goal_dist = cfg_file.get('commit_goal_dist',
                                             cfg_file.get('commit_carrot_lookahead',
                                                          rospy.get_param('~commit_carrot_lookahead', 5.0)))
        # 路口/分支限制
        self.max_pending_per_node = cfg_file.get('max_pending_per_node',
                                                 rospy.get_param('~max_pending_per_node', 2))
        self.junction_cooldown_anchors = cfg_file.get('junction_cooldown_anchors',
                                                      rospy.get_param('~junction_cooldown_anchors', 4))
        # 路口角度调试日志：开启后记录路口会话内每帧 frontier 角度
        self.junction_angle_log = cfg_file.get('junction_angle_log',
                                               rospy.get_param('~junction_angle_log', False))
        self.junction_angle_dir = cfg_file.get('junction_angle_dir') or os.path.join(self.topo_tree_root, "angle")
        self.junction_angle_counter = 0
        # topo tree 保存周期（文件名/目录已在 load_access_topo_config 冻结）
        self.topo_save_period = cfg_file.get('topo_save_period', rospy.get_param('~topo_save_period', 2.0))

        # 衍生参数（角度转弧度等）
        self.force_window_rad = np.deg2rad(self.force_window_deg)
        self.cluster_eps_rad = np.deg2rad(self.cluster_eps_deg)
        self.junction_peak_min_sep_rad = np.deg2rad(getattr(self, "junction_peak_min_sep_deg", 30.0))
        self.junction_decision_dist = getattr(self, "junction_decision_dist", self.anchor_step_dist)
        self.junction_same_dir_gate_rad = np.deg2rad(getattr(self, "junction_same_dir_gate_deg", 30.0))
        self.junction_finalize_dist = getattr(self, "junction_finalize_dist", self.junction_decision_dist * 1.5)
        self.junction_finalize_timeout = getattr(self, "junction_finalize_timeout", 6.0)

    # -------- Access-Topo: anchors / branches / backtrack --------
    def set_access_mode(self, mode: int):
        """发布拓扑层的动作优先级：0 前进、1 回退、2 强制分支前进。"""
        self.access_mode = mode
        self.access_mode_pub.publish(UInt8(mode))

    def ensure_anchor(self, force: bool = False) -> int:
        """在当前位置落一个路钉节点（必要时）。返回当前 anchor_id。"""
        if self.pose is None:
            return self.anchor_last_id
        cur_xy = (self.pose.x, self.pose.y)
        if self.anchor_last_xy is None:
            force = True
        else:
            dx = cur_xy[0] - self.anchor_last_xy[0]
            dy = cur_xy[1] - self.anchor_last_xy[1]
            if np.hypot(dx, dy) > self.anchor_step_dist:
                force = True
        created = False
        if force:
            node_id = len(self.anchor_nodes)
            node = {
                "id": node_id,
                "x": cur_xy[0],
                "y": cur_xy[1],
                "stamp": rospy.Time.now().to_sec(),
                "prev": self.anchor_last_id,
                "branches": [],
                "lste_state": self.lste_state,
                "lste_subtype": self.lste_subtype,
                "profile": self.current_profile,
                "active_mode": self.active_mode,
            }
            self.anchor_nodes.append(node)
            self.anchor_last_xy = cur_xy
            self.anchor_last_id = node_id
            created = True
            # 在回溯记录阶段，将新生成的 anchor 追加到 backtrack_path
            if self.backtrack_recording:
                # 只接受连续递增的 anchor id，防止跳号
                if (not self.backtrack_path) or node_id == self.backtrack_path[-1] + 1:
                    self.backtrack_path.append(node_id)
        return self.anchor_last_id

    def _get_node(self, node_id: int):
        if node_id is None:
            return None
        if 0 <= node_id < len(self.anchor_nodes):
            return self.anchor_nodes[node_id]
        return None

    def _add_branch_if_new(self, node_id: int, heading_world: float, weight: float):
        """向 anchor 登记一个尚未走过的分支，并做局部/相邻节点去重。

        ``heading_world`` 是 odom 世界系方向，不是相对相机或雷达的角度。
        只有状态为 ``PENDING`` 的分支会进入回退栈，之后可被选为强制前进
        的方向。
        """
        node = self._get_node(node_id)
        if node is None:
            return None
        heading_world = wrap_angle(heading_world)
        now_ts = rospy.Time.now().to_sec()
        # 本节点内部去重/合并
        for br in node["branches"]:
            if angle_diff(br.get("heading_world", 0.0), heading_world) <= self.cluster_eps_rad:
                br["weight"] = max(br.get("weight", 0.0), weight)
                return br["id"]
        # 相邻节点全局去重（位置 + 方向）
        merge_dist = getattr(self, "branch_merge_dist", None)
        if merge_dist is not None and node.get("x") is not None and node.get("y") is not None:
            for other in self.anchor_nodes:
                if other is None or other.get("x") is None or other.get("y") is None:
                    continue
                if other.get("id") == node_id:
                    continue
                dx = node["x"] - other["x"]
                dy = node["y"] - other["y"]
                if np.hypot(dx, dy) <= merge_dist:
                    for br in other.get("branches", []):
                        if angle_diff(br.get("heading_world", 0.0), heading_world) <= self.cluster_eps_rad:
                            br["weight"] = max(br.get("weight", 0.0), weight)
                            return None  # 已有相近兴趣点，直接跳过新增
        # 限制同一节点的 PENDING 数量，权重低的会被更高权重的替换
        pending_branches = [
            (idx, br) for idx, br in enumerate(node["branches"])
            if br.get("status") == "PENDING"
        ]
        if len(pending_branches) >= self.max_pending_per_node:
            min_idx, min_br = min(pending_branches, key=lambda t: t[1].get("weight", 0.0))
            if weight <= min_br.get("weight", 0.0):
                return None  # 新分支不如现有最低权重，忽略
            # 用更高权重的 pending 替换最低权重的
            node["branches"][min_idx] = {
                "id": min_idx,
                "heading_world": heading_world,
                "weight": weight,
                "status": "PENDING",
                "created": now_ts,
            }
            # 更新栈：移除旧的，添加新的
            self._remove_from_backtrack_stack(node_id, min_idx)
            self.backtrack_stack.append((node_id, min_idx, weight))
            # 按 (node_id, weight) 升序，pop() 会取到“最近的路口、权重更高”的 pending
            self.backtrack_stack.sort(key=lambda x: (x[0], x[2]))
            return min_idx
        # 新增 branch
        branch_id = len(node["branches"])
        node["branches"].append({
            "id": branch_id,
            "heading_world": heading_world,
            "weight": weight,
            "status": "PENDING",
            "created": now_ts,
        })
        # 以 (node_id, weight) 升序排序，pop() 取到“最近的 pending，若同节点则权重大”
        self.backtrack_stack.append((node_id, branch_id, weight))
        self.backtrack_stack.sort(key=lambda x: (x[0], x[2]))
        return branch_id

    # 路口角度调试日志：在会话期间按帧记录 frontier 角度
    def _maybe_init_junction_log(self, pj: dict):
        if not getattr(self, "junction_angle_log", False):
            return
        if pj.get("log_inited"):
            return
        try:
            os.makedirs(self.junction_angle_dir, exist_ok=True)
        except Exception:
            pass
        self.junction_angle_counter += 1
        fname = f"junction_{int(rospy.Time.now().to_sec())}_{pj.get('node_id','x')}_{self.junction_angle_counter}.json"
        pj["log_path"] = os.path.join(self.junction_angle_dir, fname)
        pj["log_frames"] = []
        pj["log_inited"] = True

    def _append_junction_log(self, pj: dict, stamp: float, thetas_rel, pose):
        if (not getattr(self, "junction_angle_log", False)) or (not pj.get("log_inited")):
            return
        try:
            frame = {"stamp": float(stamp), "thetas_rel": [float(t) for t in thetas_rel]}
            if pose is not None and thetas_rel is not None and len(thetas_rel) > 0:
                frame["headings_world"] = [float(wrap_angle(pose.theta + t)) for t in thetas_rel]
            pj.setdefault("log_frames", []).append(frame)
        except Exception:
            pass

    def _finalize_junction_log(self, pj: dict, status: str = "finalized"):
        if (not getattr(self, "junction_angle_log", False)) or (not pj.get("log_inited")):
            return
        data = {
            "node_id": pj.get("node_id"),
            "created": pj.get("created"),
            "anchor_xy": pj.get("anchor_xy"),
            "status": status,
            "chosen_id": pj.get("chosen_id"),
            "candidates": pj.get("candidates"),
            "frames": pj.get("log_frames", []),
        }
        path = pj.get("log_path", None)
        try:
            os.makedirs(self.junction_angle_dir, exist_ok=True)
            if path:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "save junction angle log failed: %s", exc)

    def _publish_backtrack_goal(self, node_id: int):
        node = self._get_node(node_id)
        if node is None:
            return
        msg = PoseStamped()
        msg.header.frame_id = "odom"
        msg.header.stamp = rospy.Time.now()
        msg.pose.position.x = node["x"]
        msg.pose.position.y = node["y"]
        msg.pose.orientation.w = 1.0
        self.backtrack_goal_pub.publish(msg)

    def _publish_commit_carrot(self):
        """
        固定 commit 目标：进入 mode2 后，一次性在分叉点沿强制方向前推 commit_goal_dist。
        期间不再动态更新，保持该目标覆盖 final_goal。
        """
        if self.forced_heading_world is None or self.forced_start_xy is None:
            return
        dist = getattr(self, "commit_goal_dist", 5.0)
        # 将 pending 方向投影到最接近的骨架方向
        heading_target = self.forced_heading_world
        try:
            if self.headings:
                diffs = [abs(wrap_angle(self.forced_heading_world - h)) for h in self.headings]
                k_near = int(np.argmin(diffs))
                heading_target = self.headings[k_near]
        except Exception:
            pass
        x = self.forced_start_xy[0] + dist * np.cos(heading_target)
        y = self.forced_start_xy[1] + dist * np.sin(heading_target)
        self.commit_goal = {"x": x, "y": y}
        msg = PoseStamped()
        msg.header.frame_id = "odom"
        msg.header.stamp = rospy.Time.now()
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.orientation.w = 1.0
        self.backtrack_goal_pub.publish(msg)

    def _nearest_anchor_dist(self, x: float, y: float) -> float:
        """返回与已有 anchor 的最小距离；无 anchor 则返回 None。"""
        if not self.anchor_nodes:
            return None
        d_min = None
        for nd in self.anchor_nodes:
            dx = x - nd.get("x", 0.0)
            dy = y - nd.get("y", 0.0)
            d = np.hypot(dx, dy)
            if d_min is None or d < d_min:
                d_min = d
        return d_min

    def _finalize_backtrack_session(self, end_anchor: int = None, status: str = "done"):
        """固化当前回溯段到历史，并清空当前记录状态。"""
        # 确保结束 anchor 也被加入路径（保持连续）
        if end_anchor is not None and self.backtrack_recording:
            if (not self.backtrack_path) or end_anchor == self.backtrack_path[-1] + 1:
                self.backtrack_path.append(end_anchor)
        if self.backtrack_path or self.backtrack_start_pose or self.backtrack_start_anchor is not None:
            session = {
                "start_pose": dict(self.backtrack_start_pose) if self.backtrack_start_pose else None,
                "start_anchor": self.backtrack_start_anchor,
                "path": list(self.backtrack_path),
                "end_anchor": end_anchor,
                "status": status,
                "finished": rospy.Time.now().to_sec(),
            }
            if self.current_backtrack is not None:
                session["target"] = {
                    "node": self.current_backtrack[0],
                    "branch": self.current_backtrack[1]
                }
            self.backtrack_history.append(session)
        # 清空当前段状态，准备下一次回溯
        self.backtrack_recording = False
        self.backtrack_path = []
        self.backtrack_start_pose = None
        self.backtrack_start_anchor = None
        self.no_frontier_start_pose = None
        self.no_frontier_start_anchor = None

    def enter_backtrack(self):
        """候选为空时触发：切到 BACKTRACK，并将最近兴趣分支作为回退目标。"""
        if self.access_mode == 1 and self.current_backtrack is not None:
            return
        if len(self.backtrack_stack) == 0:
            # 没有兴趣点：回到起点 (id=0) 作为安全落点
            self.return_home_target = 0
            if self.anchor_nodes:
                self._publish_backtrack_goal(self.return_home_target)
            self.set_access_mode(1)
            self.change_flag = False
            self.change_flag_pub.publish(self.change_flag)
            return
        # 有兴趣点，正常回溯
        self.return_home_target = None
        # 优先使用“最近的” pending（node_id 最大）；同节点内取权重更高的
        node_id, branch_id, weight = self.backtrack_stack.pop()
        node = self._get_node(node_id)
        if node is None:
            return
        # 如果尚未有起点 anchor，使用当前 anchor 作为起点
        if self.backtrack_start_anchor is None:
            start_anchor = self.ensure_anchor(force=False)
            self.backtrack_start_anchor = start_anchor
            if self.backtrack_recording:
                self.backtrack_path = []
                if start_anchor is not None:
                    self.backtrack_path.append(start_anchor)
        brs = node["branches"]
        if 0 <= branch_id < len(brs):
            # 不再修改 PENDING -> TRACKBACK，只做标记，避免把唯一 pending 变“丢失”
            brs[branch_id]["trackback_target"] = True
        self.current_backtrack = (node_id, branch_id)
        self.set_access_mode(1)
        self.change_flag = False
        self.change_flag_pub.publish(self.change_flag)
        self._publish_backtrack_goal(node_id)

    def _remove_from_backtrack_stack(self, node_id: int, branch_id: int):
        """Drop a (node, branch) from回退候选栈，避免重复消费."""
        self.backtrack_stack = [
            t for t in self.backtrack_stack
            if not (len(t) >= 2 and t[0] == node_id and t[1] == branch_id)
        ]

    def _pick_forced_branch(self, node: dict, fallback_branch_id: int = None) -> int:
        """
        选择强制方向：
        - 优先用同一节点上 status==PENDING 的分支（权重最高的一个）
        - 否则退回当前回退的分支 fallback_branch_id
        """
        if not node:
            return fallback_branch_id
        best = None
        for idx, br in enumerate(node.get("branches", [])):
            if br.get("status") != "PENDING":
                continue
            wt = br.get("weight", 0.0)
            if best is None or wt > best[0]:
                best = (wt, idx)
        if best is not None:
            return best[1]
        return fallback_branch_id

    def _start_forced_heading(self, node_id: int, branch_id: int):
        node = self._get_node(node_id)
        if node is None:
            return
        brs = node["branches"]
        if not (0 <= branch_id < len(brs)):
            return
        heading_world = brs[branch_id].get("heading_world")
        # 不再把 PENDING 改成 TRACKBACK，避免同一节点出现多个 TRACKBACK；
        # 如果原本就是 TRACKBACK（回退目标），保持原状态即可。
        self.forced_heading_world = heading_world
        # 分叉点作为强制段的起点参考
        self.forced_start_xy = (node.get("x", self.pose.x), node.get("y", self.pose.y))
        self.forced_branch = (node_id, branch_id)
        self.mode2_start_time = rospy.Time.now().to_sec()
        self.current_backtrack = None
        # commit 阶段：保持 Access 覆盖，使用 commit 目标
        self.set_access_mode(2)
        self.change_flag = True
        self.change_flag_pub.publish(self.change_flag)
        # 初始发布一个固定 commit 目标
        self._publish_commit_carrot()

    def check_backtrack_progress(self):
        """BACKTRACK 模式下，判断是否已到达回退点并切回 FORWARD。

        这不是路径规划器：它只从已记录的 anchor/branch 中选一个回退目标，
        再由 Goal Manager 将该目标作为当前 /lste/final_goal 的高优先级来源。
        """
        if self.access_mode != 1 or self.pose is None:
            return
        # 没有兴趣点时，回到起点 (anchor 0)
        if self.return_home_target is not None:
            target = self._get_node(self.return_home_target)
            if target is not None:
                dist_home = np.hypot(self.pose.x - target["x"], self.pose.y - target["y"])
                if dist_home <= self.backtrack_arrive_dist:
                    end_anchor = self.ensure_anchor(force=True)
                    self._finalize_backtrack_session(end_anchor=end_anchor, status="reach_home")
                    self.return_home_target = None
                    self.current_backtrack = None
                    self.set_access_mode(0)
                    self.change_flag = True
                    self.change_flag_pub.publish(self.change_flag)
                    return
            # 仍在回到起点的路上，无需处理兴趣点回溯
            if self.current_backtrack is None:
                return
        node_id, branch_id = self.current_backtrack
        node = self._get_node(node_id)
        if node is None:
            return
        dist = np.hypot(self.pose.x - node["x"], self.pose.y - node["y"])
        if dist <= self.backtrack_arrive_dist:
            # 回溯段结束，停止记录新路径并固化本段历史
            if self.backtrack_recording:
                end_anchor = self.ensure_anchor(force=True)
                self._finalize_backtrack_session(end_anchor=end_anchor, status="reach_interest")
            # 到达回退点：强制只走该分支方向
            forced_branch_id = self._pick_forced_branch(node, fallback_branch_id=branch_id)
            if forced_branch_id is not None:
                self._remove_from_backtrack_stack(node_id, forced_branch_id)
                self._start_forced_heading(node_id, forced_branch_id)

    def update_forced_heading_progress(self):
        """当强制沿兴趣分支前进到足够距离后，解除强制窗口并标记 DONE。"""
        if self.forced_heading_world is None or self.forced_start_xy is None or self.pose is None:
            return
        dx = self.pose.x - self.forced_start_xy[0]
        dy = self.pose.y - self.forced_start_xy[1]
        progress = dx * np.cos(self.forced_heading_world) + dy * np.sin(self.forced_heading_world)
        # 固定 commit 目标可按需重复发布（目标不随进度变化）
        self._publish_commit_carrot()
        if progress >= self.commit_dist:
            # 标记当前分支 DONE
            if self.forced_branch is not None:
                node = self._get_node(self.forced_branch[0])
                if node:
                    idx = self.forced_branch[1]
                    if 0 <= idx < len(node.get("branches", [])):
                        node["branches"][idx]["status"] = "DONE"
                        node["branches"][idx].pop("trackback_target", None)
            self.forced_heading_world = None
            self.forced_start_xy = None
            self.forced_branch = None
            self.mode2_start_time = None
            self.commit_goal = None
            # 解除 Access 覆盖，恢复正常模式
            self.set_access_mode(0)
            self.change_flag = True
            self.change_flag_pub.publish(self.change_flag)

    def _angle_mean(self, angles, weights):
        c = np.sum(np.cos(angles) * weights)
        s = np.sum(np.sin(angles) * weights)
        return wrap_angle(np.arctan2(s, c))

    def cluster_frontiers(self):
        """把本轮 frontier 按相近方向聚成峰，供路口判定使用。

        单个 frontier 可能只是噪声。多个分离的稳定峰持续出现，才可能表示
        左右岔路等真实分支；最终的时间稳定性判断在
        ``create_interest_from_frontiers``。返回值为 ``center`` 与 ``weight``
        组成的角度簇列表。
        """
        if self.gp_nav_frntr_cntrs is None:
            return []
        try:
            thetas = np.array(self.gp_nav_frntr_cntrs).reshape(-1, 2)[:, 0]
        except Exception:
            return []
        if thetas.size == 0:
            return []
        areas = getattr(self, "gp_nav_frntr_areas", None)
        if areas is None:
            weights = np.ones_like(thetas)
        else:
            try:
                weights = np.array(areas).reshape(-1)
            except Exception:
                weights = np.ones_like(thetas)
            if weights.size == 0:
                weights = np.ones_like(thetas)
            elif weights.size != thetas.size:
                # fallback to uniform if dimension mismatch
                weights = np.ones_like(thetas)
        # 按角度排序
        idx = np.argsort(thetas)
        thetas = thetas[idx]
        weights = weights[idx]
        clusters = []
        current_angles = [thetas[0]]
        current_weights = [weights[0]]
        for ang, w in zip(thetas[1:], weights[1:]):
            center_now = self._angle_mean(np.array(current_angles), np.array(current_weights))
            if angle_diff(ang, center_now) <= self.cluster_eps_rad:
                current_angles.append(ang)
                current_weights.append(w)
            else:
                clusters.append({"angles": current_angles, "weights": current_weights})
                current_angles = [ang]
                current_weights = [w]
        clusters.append({"angles": current_angles, "weights": current_weights})
        # wrap 合并首尾
        if len(clusters) > 1:
            first_c = self._angle_mean(np.array(clusters[0]["angles"]), np.array(clusters[0]["weights"]))
            last_c = self._angle_mean(np.array(clusters[-1]["angles"]), np.array(clusters[-1]["weights"]))
            if angle_diff(first_c, last_c) <= self.cluster_eps_rad:
                merged_angles = clusters[0]["angles"] + clusters[-1]["angles"]
                merged_weights = clusters[0]["weights"] + clusters[-1]["weights"]
                clusters = [{"angles": merged_angles, "weights": merged_weights}] + clusters[1:-1]
        result = []
        for cl in clusters:
            angs = np.array(cl["angles"])
            wts = np.array(cl["weights"])
            result.append({"center": self._angle_mean(angs, wts), "weight": float(wts.sum())})
        return result

    def update_cluster_track(self, clusters):
        """维护 cluster 跟踪，返回当前稳定的 cluster 列表。"""
        now = rospy.Time.now().to_sec()
        # 匹配已有 track
        for tr in self.cluster_track:
            tr["matched"] = False
        for cl in clusters:
            best = None
            best_diff = None
            for tr in self.cluster_track:
                diff = angle_diff(cl["center"], tr["center"])
                if diff <= self.cluster_eps_rad and (best_diff is None or diff < best_diff):
                    best = tr
                    best_diff = diff
            if best is not None:
                best["center"] = cl["center"]
                best["last_seen"] = now
                best["weight"] = cl["weight"]
                best["matched"] = True
            else:
                self.cluster_track.append({
                    "center": cl["center"],
                    "first_seen": now,
                    "last_seen": now,
                    "weight": cl["weight"],
                    "matched": True,
                })
        # 清理长时间未出现的
        self.cluster_track = [
            tr for tr in self.cluster_track
            if (now - tr.get("last_seen", now)) <= self.cluster_lost_timeout
        ]
        stable = []
        for tr in self.cluster_track:
            if (now - tr.get("first_seen", now)) >= self.cluster_stable_duration:
                stable.append({"center": tr["center"], "weight": tr.get("weight", 1.0)})
        return stable

    def create_interest_from_frontiers(self, chosen_theta_rel: float):
        """根据“多峰 -> 单峰”过程更新路口分支状态。

        进入路口时先记录所有稳定峰；机器人继续前进一段距离或超时后，再用
        实际运动方向确认哪一支已走过（CHOSEN），其余方向保留为 PENDING，
        以后可作为回退后的待探索分支。
        """
        now = rospy.Time.now().to_sec()
        clusters = self.cluster_frontiers()
        # 维护 cluster 跟踪，用于判定“多峰是否持续存在”
        self.update_cluster_track(clusters)
        # 只统计当前帧 matched 且已达到最小存在时长的峰，避免用到历史残留
        stable_present = []
        for tr in self.cluster_track:
            if not tr.get("matched", False):
                continue
            if (now - tr.get("first_seen", now)) < getattr(self, "junction_peak_min_age", 0.0):
                continue
            stable_present.append({"center": tr.get("center"), "weight": tr.get("weight", 1.0)})
        min_w = float(getattr(self, "junction_peak_min_weight", 0.0))
        if min_w > 0.0:
            stable_present = [cl for cl in stable_present if float(cl.get("weight", 0.0)) >= min_w]
        stable_present.sort(key=lambda x: float(x.get("weight", 0.0)), reverse=True)
        stable_count = len(stable_present)

        # 多峰持续判定：双峰满足分离/权重比，且持续足够长才进入会话
        multi_peak_now = False
        if stable_count >= 2:
            c0, c1 = stable_present[0], stable_present[1]
            sep = angle_diff(float(c0.get("center", 0.0)), float(c1.get("center", 0.0)))
            if sep >= float(getattr(self, "junction_peak_min_sep_rad", self.cluster_eps_rad)):
                w0 = float(c0.get("weight", 0.0))
                w1 = float(c1.get("weight", 0.0))
                ratio = float(getattr(self, "junction_second_weight_ratio", 0.0))
                if ratio <= 0.0 or (w0 > 1e-6 and (w1 / w0) >= ratio):
                    multi_peak_now = True
        if multi_peak_now:
            if self.multi_peak_since is None:
                self.multi_peak_since = now
        else:
            self.multi_peak_since = None
        enter_junction_session = (
            multi_peak_now
            and (self.multi_peak_since is not None)
            and ((now - self.multi_peak_since) >= float(getattr(self, "junction_enter_confirm_sec", 0.0)))
        )

        try:
            thetas_rel_raw = np.array(self.gp_nav_frntr_cntrs).reshape(-1, 2)[:, 0]
        except Exception:
            thetas_rel_raw = np.array([])

        stable_world = [
            {"heading": wrap_angle(self.pose.theta + float(cl.get("center", 0.0))), "weight": float(cl.get("weight", 0.0))}
            for cl in stable_present
        ]

        # 进入路口会话：满足双峰条件且通过冷却窗口
        if enter_junction_session:
            node_id = self.ensure_anchor(force=False)
            if node_id is not None:
                already = any(item.get("node_id") == node_id for item in self.pending_junctions)
                in_cooldown = False
                if self.last_finalized_anchor is not None and (node_id - self.last_finalized_anchor) <= self.junction_post_finalize_cooldown:
                    in_cooldown = True
                if self.last_junction_anchor is not None and (node_id - self.last_junction_anchor) < self.junction_cooldown_anchors:
                    in_cooldown = True
                if (not already) and (not in_cooldown):
                    anchor = self._get_node(node_id)
                    anchor_xy = (anchor["x"], anchor["y"]) if anchor else (
                        getattr(self.pose, "x", 0.0),
                        getattr(self.pose, "y", 0.0)
                    )
                    pj = {
                        "node_id": node_id,
                        "start_anchor": node_id,
                        "anchor_xy": anchor_xy,
                        "created": now,
                        "observations": [],
                        "actual_headings": [],
                        "peak_count_max": stable_count,
                        "was_multi": True,
                    }
                    self.pending_junctions.append(pj)
                    self.last_junction_anchor = node_id
                    self._maybe_init_junction_log(pj)

        # 会话中：持续记录峰与位移方向，双峰降为单峰时收尾
        new_pending = []
        for pj in self.pending_junctions:
            node_id = pj.get("node_id")
            anchor_xy = pj.get("anchor_xy", (0.0, 0.0))
            dist_from_anchor = None
            dx = dy = 0.0
            if self.pose is not None:
                dx = self.pose.x - anchor_xy[0]
                dy = self.pose.y - anchor_xy[1]
                dist_from_anchor = np.hypot(dx, dy)
                if dist_from_anchor > 0.05:  # 抖动过滤
                    try:
                        heading_actual = math.atan2(dy, dx)
                        pj.setdefault("actual_headings", []).append(heading_actual)
                    except Exception:
                        pass

            if stable_world:
                pj.setdefault("observations", []).extend(
                    [(float(sw["heading"]), float(sw["weight"])) for sw in stable_world]
                )
            pj["peak_count_max"] = max(pj.get("peak_count_max", 0), stable_count)
            # 单峰稳定计时：稳定 1 个峰持续到阈值才允许退出
            if stable_count == 1:
                if pj.get("single_peak_since") is None:
                    pj["single_peak_since"] = now
            else:
                pj["single_peak_since"] = None

            if thetas_rel_raw is not None and thetas_rel_raw.size > 0:
                self._append_junction_log(pj, now, thetas_rel_raw, self.pose)

            ended = (
                pj.get("was_multi", False)
                and stable_count == 1
                and pj.get("single_peak_since") is not None
                and (now - pj["single_peak_since"]) >= self.junction_exit_confirm_sec
            )
            if ended:
                self._finalize_junction_session(pj)
                self._finalize_junction_log(pj, status="finalized")
                continue  # 会话已结束，不再保留

            new_pending.append(pj)

        self.pending_junctions = new_pending

    def _prune_neighbor_branches(self, target_anchor: int, window: int = 3):
        """在 anchor 窗口内只保留 id 较大的分叉，清除窗口内更小 id 的分支。"""
        if target_anchor is None:
            return
        low = target_anchor - window
        for node in self.anchor_nodes:
            nid = node.get("id")
            if nid is None or nid == target_anchor:
                continue
            if nid < low or nid > target_anchor:
                continue
            if node.get("branches"):
                node["branches"] = []

    def _cluster_angles_with_weights(self, angles, weights):
        """把角度列表按 cluster_eps_rad 聚成簇，返回 [{'center','weight'}]."""
        if not angles:
            return []
        pairs = sorted(zip(angles, weights), key=lambda t: t[0])
        clusters = []

        def flush(cur_angles, cur_weights):
            if not cur_angles:
                return None
            center = self._angle_mean(np.array(cur_angles), np.array(cur_weights))
            weight = float(np.sum(cur_weights))
            return {"center": center, "weight": weight}

        cur_angles = []
        cur_weights = []
        for ang, w in pairs:
            if not cur_angles:
                cur_angles = [ang]
                cur_weights = [w]
                continue
            if angle_diff(ang, cur_angles[-1]) <= self.cluster_eps_rad:
                cur_angles.append(ang)
                cur_weights.append(w)
            else:
                cl = flush(cur_angles, cur_weights)
                if cl:
                    clusters.append(cl)
                cur_angles = [ang]
                cur_weights = [w]
        cl = flush(cur_angles, cur_weights)
        if cl:
            clusters.append(cl)

        # 合并首尾（环状角度），避免 0/2pi 断裂
        if len(clusters) > 1:
            first = clusters[0]
            last = clusters[-1]
            if angle_diff(first["center"], last["center"]) <= self.cluster_eps_rad:
                merged_center = self._angle_mean(
                    np.array([first["center"], last["center"]]),
                    np.array([first["weight"], last["weight"]])
                )
                merged_weight = first["weight"] + last["weight"]
                clusters = clusters[1:-1]
                clusters.append({"center": merged_center, "weight": merged_weight})

        clusters.sort(key=lambda x: x.get("weight", 0.0), reverse=True)
        return clusters

    def _finalize_junction_session(self, session: dict):
        """会话结束：按退出方向定 chosen，其余为 pending（限制数量）。"""
        start_anchor = session.get("start_anchor", session.get("node_id"))
        observations = session.get("observations", [])
        if not observations:
            rospy.logwarn_throttle(5.0, "junction finalize: no observations for node %s", start_anchor)
            return
        angles, weights = zip(*observations)
        clusters = self._cluster_angles_with_weights(list(angles), list(weights))
        if not clusters:
            rospy.logwarn_throttle(5.0, "junction finalize: no clusters for node %s", start_anchor)
            return

        # 用会话后半段位移方向作为退出方向
        heading_exit = None
        actual_headings = session.get("actual_headings", [])
        if actual_headings:
            tail = actual_headings[-4:] if len(actual_headings) >= 4 else actual_headings
            try:
                heading_exit = self._angle_mean(np.array(tail), np.ones(len(tail)))
            except Exception:
                heading_exit = None
        if heading_exit is None:
            heading_exit = clusters[0]["center"]

        chosen_idx = min(range(len(clusters)), key=lambda i: angle_diff(clusters[i]["center"], heading_exit))
        chosen = clusters.pop(chosen_idx)

        pending_limit = 2 if session.get("peak_count_max", 2) >= 3 else 1
        pending_clusters = sorted(clusters, key=lambda c: c.get("weight", 0.0), reverse=True)[:pending_limit]
        if not pending_clusters and clusters:
            pending_clusters = [clusters[0]]

        if not pending_clusters:
            rospy.logwarn_throttle(5.0, "junction finalize: no pending branches for node %s", start_anchor)

        # 目标 anchor：起始 anchor 与当前 anchor 的中位 id，找不到则退回 start
        end_anchor = self.anchor_last_id if self.anchor_last_id is not None else start_anchor
        target_anchor = start_anchor
        if end_anchor is not None and start_anchor is not None:
            try:
                if end_anchor < start_anchor:
                    end_anchor = start_anchor
                target_anchor = int((start_anchor + end_anchor) // 2)
            except Exception:
                target_anchor = start_anchor
        if self._get_node(target_anchor) is None:
            if end_anchor is not None and self._get_node(end_anchor) is not None:
                target_anchor = end_anchor
            else:
                target_anchor = start_anchor

        # 清除邻近 anchor（id 差 <=3）里更小 id 的分支，保留当前 anchor
        self._prune_neighbor_branches(target_anchor, window=3)

        for pc in pending_clusters:
            self._add_branch_if_new(target_anchor, pc.get("center"), pc.get("weight", 1.0))

        node = self._get_node(target_anchor)
        if node is not None:
            node["chosen_heading"] = chosen.get("center")
        # 记录用于冷却
        self.last_finalized_anchor = target_anchor

    def dump_access_topo(self, force: bool = False):
        """周期性保存 topo tree 为 JSON，便于离线可视化和实验复盘。

        写入同一个 run_id 文件而不是每周期新建文件；下次重新启动节点才会
        生成新的 run_id 和新文件。该 JSON 是运行产物，不是启动配置。
        """
        if not self.topo_save_dir:
            return
        now = rospy.Time.now().to_sec()
        if (not force) and (now - self.last_topo_save_time < self.topo_save_period):
            return
        self.last_topo_save_time = now
        try:
            os.makedirs(self.topo_save_dir, exist_ok=True)
            data = {
                "stamp": now,
                "run_id": self.run_id,
                "run_start": self.run_start,
                "run_end": self.run_end,
                "test_name": self.test_name,
                "topo_tree_root": self.topo_tree_root,
                "pose": {"x": getattr(self.pose, "x", None), "y": getattr(self.pose, "y", None)},
                "access_mode": self.access_mode,
                "forced_heading_world": self.forced_heading_world,
                "forced_branch": self.forced_branch,
                "current_backtrack": self.current_backtrack,
                "backtrack_start_pose": self.backtrack_start_pose,
                "backtrack_path": self.backtrack_path,
                "backtrack_history": self.backtrack_history,
                "return_home_target": self.return_home_target,
                "current_profile": self.current_profile,
                "nodes": self.anchor_nodes,
                "pending_junctions": self.pending_junctions,
                "backtrack_stack": self.backtrack_stack,
                "state_events": self.state_events,
                "profile_events": self.profile_events,
                "goal_events": self.goal_events,
                "visited_events": self.visited_events,
            }
            with open(self.topo_save_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "dump_access_topo failed: %s", exc)

    def frontier_log(self, msg: str):
        """Append frontier-related log to file when enabled."""
        if not getattr(self, "frontier_log_enabled", False):
            return
        if not getattr(self, "frontier_log_file", None):
            return
        try:
            t = rospy.Time.now().to_sec() if rospy.rostime.is_initialized() else 0.0
            if getattr(self, "frontier_log_fh", None) is not None:
                self.frontier_log_fh.write(f"[{t:.3f}] {msg}\n")
            else:
                with open(self.frontier_log_file, "a", encoding="utf-8") as f:
                    f.write(f"[{t:.3f}] {msg}\n")
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "gp_subgoal: frontier_log write failed: %s", exc)
            self.frontier_log_enabled = False

    def on_shutdown(self):
        """节点退出时强制落盘，确保单个 JSON 包含最终状态。"""
        self.run_end = rospy.Time.now().to_sec()
        self.dump_access_topo(force=True)
        # 关闭 frontier log，并恢复 stdout/stderr
        try:
            if self._stdout_orig is not None:
                sys.stdout = self._stdout_orig
            if self._stderr_orig is not None:
                sys.stderr = self._stderr_orig
        except Exception:
            pass
        try:
            if self.frontier_log_fh:
                self.frontier_log_fh.flush()
                self.frontier_log_fh.close()
        except Exception:
            pass

    #zcy
    # 切换模式，true为高斯开启发布，flase为topo，高斯subgoal停止发布
    def flag_cb(self, bool_msg):
        self.change_flag = bool_msg.data


    def zcypose_cb(self, rbt_pose_msg):
        self.pose = rbt_pose_msg
        # 初始化全局方向骨架：记录起始位姿，定义 4 个固定世界方向
        if not self.origin_set:
            self.origin_x = self.pose.x
            self.origin_y = self.pose.y
            self.origin_yaw = wrap_angle(self.pose.theta)
            self.headings = [
                self.origin_yaw,
                wrap_angle(self.origin_yaw + np.pi * 0.5),
                wrap_angle(self.origin_yaw + np.pi),
                wrap_angle(self.origin_yaw + np.pi * 1.5),
            ]
            self.dir_idx = 0
            self.last_switch_xy = (self.origin_x, self.origin_y)
            self.origin_set = True

        int_position = [int(self.pose.x), int(self.pose.y)]
        real_position = [self.pose.x, self.pose.y]
        if int_position not in self.int_visited_positions:
            self.int_visited_positions.append(int_position)
            self.visited_positions.append(real_position)
            print(real_position)
        self.ensure_anchor()

    def is_visited(self, point, threshold):
        for traj_pt in self.visited_positions:
            distance = np.sqrt((point[0] - traj_pt[0]) ** 2 + (point[1] - traj_pt[1]) ** 2)
            if distance < threshold:
                return True
        return False

    def publish_closed_targets(self):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "closed_points"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        # 设置点的大小（直径）
        marker.scale.x = 0.2  # 根据需要调整
        marker.scale.y = 0.2  # 根据需要调整

        # 设置颜色为红色
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0  # 透明度

        # 添加点到Marker消息中
        for point in self.visited_positions:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = 0.0  # 2D
            marker.points.append(p)

        # 发布Marker消息
        self.closed_marker_pub.publish(marker)

    """ @brief: to print all parameters"""

    def print_ros_param(self):
        print("######## OC SRFC ######")
        print("oc_srfc_rds                : ", self.oc_srfc_rds)
        print("######## Nav ########")
        print("gp_nav_indpts_sz    : ", self.gp_nav_indpts_sz)
        print("gap_k_dir        : ", self.gap_k_dir)
        print("gap_k_dst       : ", self.gap_k_dst)
        print("gp_nav_var_thrshld  : ", self.gp_nav_var_thrshld)
        print("gp_nav_goal_dst     : ", self.gp_nav_goal_dst)
        print("gp_nav_var_img_viz  : ", self.gp_nav_var_img_viz)
        print("gl_x         : ", self.gl_x)
        print("gl_y         : ", self.gl_y)
        print("##########################################")

    """ @brief: to initiate and train 2D SGP using the SGP2D class"""

    def gp_nav_fit(self, ls1, ls2, var, alpha, noise, noise_var):
        """用当前球面观测创建 Sparse GP 模型。

        输入 ``gp_nav_din`` 的每行是 ``[theta, alpha]``，``gp_nav_dout`` 是
        该方向的占据值。随后 step() 会用此模型预测 ``gp_grd`` 全网格，从
        方差中找尚未观测充分的 frontier。
        """
        self.gp_nav = SGP2D()
        self.gp_nav.set_kernel_param(ls1, ls2, var, alpha, noise, noise_var)
        self.gp_nav.set_training_data(self.gp_nav_din, self.gp_nav_dout)
        self.gp_nav.set_indpts_from_training_data(
            self.gp_nav_indpts_sz, self.gp_nav_din)  #(indpts_size, data_size)
        self.gp_nav.set_sgp_model()
        self.gp_nav.select_trainable_param()
        self.gp_nav.minimize_loss()
        # self.gp_nav.adam_optimize_param()

    """ @brief: callback funtion to process the occupancy surface"""

    def pose_pcl_cb(self, pose_pcl_msg):
        #################### GP Nav ####################
        print("\n\n\n##### OBSV: seq=", pose_pcl_msg.header.seq, ", RosTime=", pose_pcl_msg.header.stamp.to_sec(),
              " #####")
        msg_rcvd_time = time()
        self.header.stamp = pose_pcl_msg.header.stamp
        self.pose = pose_pcl_msg.pose

        ## transformation matrices between robot and world
        self.tf_rbt_2_odom()
        self.tf_odom_2_rbt()
        self.rbt2gl_error()


        ## retrieve th, al, rds points from pointcloud
        pcl_arr = ros_numpy.point_cloud2.pointcloud2_to_array(
            pose_pcl_msg.pcl2, squeeze=True)
        pcl_arr = np.round(np.array(pcl_arr.tolist(), dtype='float'), 4)

        print("org_pcl_size: ", np.shape(pcl_arr))
        ## Downsample and assign thetas, alphas, occs, rds variables
        self.downsample_pcl(pcl_arr)

        ## define input and output data for training  
        self.gp_nav_din = np.column_stack((self.pcl_thetas, self.pcl_alphas))
        self.gp_nav_dout = np.array(self.pcl_oc, dtype='float').reshape(-1, 1)
        self.gp_nav_fit(0.09, 0.11, 0.7, 10, 10,
                        0.005)  #(ls1, ls2, var, alpha, noise, noise_var)
        nav_grd_oc, nav_grd_var = self.gp_nav.model.predict_f(self.gp_grd)
        self.gp_grd_var = nav_grd_var.numpy()
        self.gp_grd_oc = nav_grd_oc.numpy()
        self.gp_grd_rds = self.oc_srfc_rds - self.gp_grd_oc

        ## process the variance surface to define GPFrontiers and assign subgoals
        self.gp_nav_mask_thrshld()
        self.gp_grd_var_img()
        self.gp_nav_pkup_nav_pt()
        ####calculate navigation point in world frame
        ####self.gp_nav_xypts_actul_pcl()

        #### occupancy and variance surfaces visualization
        self.gp_grd_oc_pcl()
        self.gp_grd_var_pcl()

        # goal in polar coordinate wrt to velodyne
        self.gl_wrt_rbt = self.tf_2d_inv @ self.gl_wrt_odom

        # predict occupancy in direction of final goal using GP occupancy model
        self.gl_th = np.arctan2(self.gl_wrt_rbt[1], self.gl_wrt_rbt[0])
        self.gl_al = np.pi / 2
        gl_oc, gl_var = self.gp_nav.model.predict_f(
            np.array([self.gl_th, self.gl_al], dtype="float32").reshape(1, 2))
        gl_rds = self.oc_srfc_rds - gl_oc.numpy()
        # print("gl_var, var_thrshld: ", gl_var.numpy(), self.gp_nav_var_thrshld)
        # print("gl_dst_err, gl_rds: ", self.gl_dst_err, gl_rds)

        ### mode: no obstacle betwen goal and robot, robots go directly to final goal
        if (self.gl_dst_err < gl_rds):
            if not self.goal_published:
                self.glbl_gl_pbl()  ## not sure if this correct situation
                self.goal_published = False
                print("Navigation Mode: FinalGoal ")

            else:
                quit()

        ### mode: there is obstacle betwen goal and robot, robots follow recommended subgoal        
        else:
            print("Navigation Mode: SubGoal ")
            self.gp_nav_glbl_gl_pbl()

        print("Total Processing Time (with visualization): ", time() - msg_rcvd_time)

    """ @brief: error bet. robot and goal"""

    def rbt2gl_error(self):
        self.gl_dst_err = np.sqrt((self.gl_x - self.pose.x) ** 2 +
                                  (self.gl_y - self.pose.y) ** 2)
        self.gl_dir = np.arctan2(self.gl_y - self.pose.y,
                                 self.gl_x - self.pose.x)
        self.gl_dir_err = self.gl_dir - self.pose.theta
        # print("gl_dst_err, gl_dir_err: ", self.gl_dst_err, self.gl_dir_err )

    """ @brief: varinace threshoild based on the variance distribution (mean and variance)"""

    def gp_nav_mask_thrshld(self):
        gp_nav_var_stats = stats.describe(self.gp_grd_var)
        gp_nav_var_mean = gp_nav_var_stats.mean[0]
        gp_nav_var_var = gp_nav_var_stats.variance[0]
        self.gp_nav_var_thrshld = 0.6 * (gp_nav_var_mean - 3 * gp_nav_var_var)
        ##print("variance distribution mean and var: ", gp_nav_var_mean, gp_nav_var_var)
        ##print("variance threshold: ", self.gp_nav_var_thrshld)

    """ @brief: convert variance to cv image and detect high variance regions usign the variance threshold"""

    def gp_grd_var_img(self):
        """把 GP 方差网格转为二维图像，并用轮廓中心提取 frontier。

        此处的 frontier 不是全局地图的路口名称，而是“当前视野中方差高、
        可能通向未知区域”的局部球面方向。输出写入：

        * ``gp_nav_frntr_cntrs``: 每个候选的 ``[theta_rel, alpha]``；
        * ``gp_nav_frntr_areas``: 对应未知区域面积，用作 Goal Manager 权重。
        """
        img = np.zeros((self.gp_grd_h, 3 * self.gp_grd_w), np.uint8)
        ## normlize variance
        var_xtnd = np.array([]).reshape(-1, 1)
        var_xtnd = np.append(var_xtnd, self.gp_grd_var)
        var_xtnd = np.append(var_xtnd, self.gp_grd_var)
        var_xtnd = np.append(var_xtnd, self.gp_grd_var)
        # print("var_xtnd: ", np.shape(var_xtnd))
        var_norm = np.linalg.norm(var_xtnd)
        normalized_var = var_xtnd / var_norm
        bw_var = (255 / normalized_var[normalized_var.argmax(axis=0)]
                  ) * normalized_var

        ## img[0:self.gp_grd_h, 0:self.gp_grd_w] = 400*self.gp_oc_srfc_grd_var.reshape(self.gp_oc_srfc_grd_w, self.gp_oc_srfc_grd_h).T
        img[0:self.gp_grd_h,
        0:3 * self.gp_grd_w] = bw_var.reshape(3 * self.gp_grd_w, self.gp_grd_h).T

        ## grey to binay
        self.gp_nav_img_thrshld = 0.3 * int(
            np.mean(bw_var) + self.gp_nav_var_thrshld * np.var(bw_var))
        _, bw_img = cv2.threshold(img, self.gp_nav_img_thrshld, 255,
                                  cv2.THRESH_BINARY)
        ### resize image
        scale_factor = 10
        width = int(bw_img.shape[1] * scale_factor)
        height = int(bw_img.shape[0] * scale_factor)
        dsize = (width, height)
        scaled_img = cv2.resize(bw_img, dsize)
        contours, hierarchy = cv2.findContours(scaled_img.copy(),
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        l_frntr_cntrs = []
        l_frntr_ars = []
        # print(">> l_frntr_areas: ", [cv2.contourArea(i) for i in contours])
        for i in contours:
            M = cv2.moments(i)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.drawContours(scaled_img, [i], -1, (0, 255, 0), 2)
                cv2.circle(scaled_img, (cx, cy), 7, (0, 0, 255), -1)

            cx = int(cx / scale_factor)
            cy = int(cy / scale_factor)
            # print(f"x: {cx} y: {cy} area:{cv2.contourArea(i)}")
            if cx >= 513 and cx < 1026 and cv2.contourArea(
                    i) > 2000 * scale_factor and cy < 13 and cy > 3:
                l_frntr_cntrs = l_frntr_cntrs + [
                    self.gp_grd_ths[cx - 513], self.gp_grd_als[cy]
                ]
                l_frntr_ars.append(cv2.contourArea(i))
                ara = l_frntr_ars[0]

        # print(">> l_frntr_cntrs: ", l_frntr_cntrs)
        # print(">> l_frntr_areas: ", l_frntr_ars)

        if (len(l_frntr_ars) == 1
            and l_frntr_ars[0] > 800000000) or len(l_frntr_ars) == 0:
            print("could not find frontier above threshold")
            # 不再兜底四个方向，直接认为无 frontier（交给上层 no_frontier 逻辑处理）
            l_frntr_cntrs = []
            l_frntr_ars = []
        self.gp_nav_frntr_cntrs = np.array(l_frntr_cntrs).reshape(-1, 2)
        self.gp_nav_frntr_areas = np.array(l_frntr_ars)  #.reshape(-1)

        ## draw contours
        if self.gp_nav_var_img_viz:
            cv2.drawContours(scaled_img, contours, -1, 255, 3, cv2.LINE_AA,
                             hierarchy, abs(-1))
            cv2.imshow('contours', scaled_img)
            cv2.waitKey()

    """ @brief: robot to wolrd 2D TF matrix"""

    def tf_rbt_2_odom(self):
        if self.pose is None:
            self.tf_2d = None
            return
        self.tf_2d = np.array(
            [[np.cos(self.pose.theta), -np.sin(self.pose.theta), self.pose.x],
             [np.sin(self.pose.theta),
              np.cos(self.pose.theta), self.pose.y], [0, 0, 1]])

    """ @brief:  odom to robot 2D TF matrix"""

    def tf_odom_2_rbt(self):
        if self.pose is None:
            self.tf_2d_inv = None
            return
        cos_th = np.cos(self.pose.theta)
        sin_th = np.sin(self.pose.theta)
        self.tf_2d_inv = np.array(
            [[cos_th, sin_th, -self.pose.x * cos_th - self.pose.y * sin_th],
             [-sin_th, cos_th, self.pose.x * sin_th - self.pose.y * cos_th],
             [0, 0, 1]])

    """ @brief:  robot to odom 2D TF matrix"""

    def gls_in_odom_frame(self, gls):
        xy_gls = np.empty([0, 3])
        for gl in gls:
            xy_gls = np.vstack((xy_gls, self.tf_2d @ gl))
        # limit to map  border,  here consider  w=h (this part is not important)
        # xy_gls = np.clip(xy_gls, -self.map_2d_h / 2 + 1, self.map_2d_h / 2 - 1) 
        return xy_gls

    """ @brief:  calculate xy location in odom frame for subgoals (navigation points)"""

    def xy_actul_nav_gls(self):
        actul_gls_rds = self.gp_nav_goal_dst * np.ones(
            self.gp_nav_gls_sz, dtype='float32').reshape(-1, 1)
        x, y, z = self.convert_spherical_2_cartesian(
            self.gp_nav_pts.T[0].reshape(-1, 1),
            self.gp_nav_pts.T[1].reshape(-1, 1), actul_gls_rds)
        z = np.ones(self.gp_nav_gls_sz).reshape(-1, 1)
        actul_gls = np.hstack((x, y, z))
        self.gp_nav_actul_xy_gls = self.gls_in_odom_frame(actul_gls)

    """ @brief:  calculate the recommended subgoal (navigation point) based on cost function"""

    def gp_nav_pkup_nav_pt(self):
        """将 frontier 变为机器人周围的候选点，并选出旧接口的推荐点。

        当前 LSTE 主链会把“全部 frontier”交给 Goal Manager 统一选择；本函数
        仍会根据距内部全局目标的距离、偏离正前方的角度和拓扑强制窗口选出
        ``chsn_gl_idx``，以兼容旧的 gp_subgoal/gp_subgoal_os 发布接口。
        """
        self.gp_nav_pts = self.gp_nav_frntr_cntrs
        # print(f"gp_nav_frntr_cntrs={self.gp_nav_frntr_cntrs}")
        self.gp_nav_gls_sz = np.shape(self.gp_nav_pts)[0]
        # print("# goals: ", self.gp_nav_gls_sz)
        self.xy_actul_nav_gls()

        #zcy
        new_gp_nav_actul_xy_gls = []
        new_gp_nav_frntr_cntrs = []
        new_gp_nav_frntr_areas = []
        closed = []  # 用于存储阻塞的目标点

        for idx, subgoal in enumerate(self.gp_nav_actul_xy_gls):
            x, y, _ = subgoal
            theta_rel = None
            try:
                theta_rel = float(self.gp_nav_frntr_cntrs[idx][0])
            except Exception:
                pass
            in_forced_window = False
            if self.forced_heading_world is not None and theta_rel is not None:
                heading_world = wrap_angle(self.pose.theta + theta_rel)
                if angle_diff(heading_world, self.forced_heading_world) <= self.force_window_rad:
                    in_forced_window = True
                else:
                    continue  # 强制窗口下，非兴趣方向直接丢弃
            # 可行距离过滤：用 GP 预测该方向的可行半径（oc_srfc_rds - oc），过近则视为不可走
            clearance_ok = True
            try:
                if getattr(self, "enable_min_frontier_clearance", False) and self.min_frontier_clearance > 0.0 and self.gp_nav is not None:
                    oc_pred, _ = self.gp_nav.model.predict_f(
                        np.array([theta_rel, np.pi / 2], dtype="float32").reshape(1, 2)
                    )
                    rds = float(self.oc_srfc_rds - oc_pred.numpy())
                    if rds < float(self.min_frontier_clearance):
                        clearance_ok = False
            except Exception:
                clearance_ok = True
            if not clearance_ok:
                continue
            if (not in_forced_window) and self.is_visited([x, y], self.visited_thresh):
                closed_position = [int(x), int(y)]
                if closed_position not in self.closed:
                    closed.append(closed_position)
                continue
            new_gp_nav_actul_xy_gls.append(subgoal)
            new_gp_nav_frntr_cntrs.append(self.gp_nav_frntr_cntrs[idx])
            if idx < len(self.gp_nav_frntr_areas):
                new_gp_nav_frntr_areas.append(self.gp_nav_frntr_areas[idx])

        #对齐形状
        self.gp_nav_actul_xy_gls = np.array(new_gp_nav_actul_xy_gls)
        self.gp_nav_frntr_cntrs = np.array(new_gp_nav_frntr_cntrs)
        self.gp_nav_frntr_areas = np.array(new_gp_nav_frntr_areas)
        self.gp_nav_pts = self.gp_nav_frntr_cntrs
        self.gp_nav_gls_sz = len(new_gp_nav_actul_xy_gls)

        # 全部的候选目标点
        print(f"gp_nav_actul_xy_gls={self.gp_nav_actul_xy_gls}")
        self.frontier_log(f"candidates={self.gp_nav_actul_xy_gls.tolist()} areas={self.gp_nav_frntr_areas.tolist() if hasattr(self, 'gp_nav_frntr_areas') else []}")

        # 回到起点途中：方案 A——不再因新 frontier 打断，全部忽略
        if self.return_home_target is not None and self.gp_nav_gls_sz > 0:
            # 直接丢弃 frontier，继续回家
            self.gp_nav_actul_xy_gls = np.array([])
            self.gp_nav_frntr_cntrs = np.array([])
            self.gp_nav_frntr_areas = np.array([])
            self.gp_nav_pts = self.gp_nav_frntr_cntrs
            self.gp_nav_gls_sz = 0
            return

        if len(self.gp_nav_actul_xy_gls) == 0:
            now = rospy.Time.now().to_sec()
            # mode2 免回退窗口：给强制阶段至少 5 秒探索 frontier
            if self.access_mode == 2 and self.forced_heading_world is not None:
                if self.mode2_start_time is not None and (now - self.mode2_start_time) < 5.0:
                    return
            if self.no_frontier_since is None:
                self.no_frontier_since = now
                if self.pose is not None:
                    self.no_frontier_start_pose = {"x": self.pose.x, "y": self.pose.y}
                self.no_frontier_start_anchor = self.ensure_anchor(force=False)
            # 无 frontier 连续 >=T 触发回退
            if self.no_frontier_since is not None:
                if (now - self.no_frontier_since) >= self.no_frontier_trigger_sec:
                    if self.no_frontier_start_pose is not None and self.backtrack_start_pose is None:
                        self.backtrack_start_pose = dict(self.no_frontier_start_pose)
                    if self.backtrack_start_anchor is None:
                        self.backtrack_start_anchor = self.no_frontier_start_anchor
                    if not self.backtrack_recording:
                        self.backtrack_path = []
                        if self.backtrack_start_anchor is not None:
                            self.backtrack_path.append(self.backtrack_start_anchor)
                        self.backtrack_recording = True
                    self.enter_backtrack()
            return
        # 恢复有 frontier，清除计时
        self.no_frontier_since = None
        self.no_frontier_start_pose = None
        self.no_frontier_start_anchor = None
        if not self.backtrack_recording:
            # 事件未触发回溯，清空起点信息
            self.backtrack_start_pose = None
            self.backtrack_start_anchor = None
            self.backtrack_path = []

        # Access commit 阶段：只保留“每个 PENDING 分支方向最近的一个 frontier”
        if self.access_mode == 2 and self.pose is not None:
            pend_headings = []
            if self.forced_branch is not None:
                node = self._get_node(self.forced_branch[0])
                if node:
                    for br in node.get("branches", []):
                        if br.get("status") == "PENDING" and br.get("heading_world") is not None:
                            pend_headings.append(float(br["heading_world"]))
            if pend_headings and self.gp_nav_frntr_cntrs is not None and self.gp_nav_frntr_cntrs.size > 0:
                try:
                    thetas_rel = np.array(self.gp_nav_frntr_cntrs).reshape(-1, 2)[:, 0]
                except Exception:
                    thetas_rel = np.array([])
                if thetas_rel.size > 0:
                    headings_world = [wrap_angle(float(self.pose.theta) + float(th)) for th in thetas_rel]
                    keep_idx = set()
                    for ph in pend_headings:
                        best = None
                        for idx, hw in enumerate(headings_world):
                            diff = angle_diff(hw, ph)
                            if best is None or diff < best[0]:
                                best = (diff, idx)
                        if best is not None:
                            keep_idx.add(best[1])
                    if keep_idx:
                        keep_idx = sorted(list(keep_idx))
                        self.gp_nav_actul_xy_gls = self.gp_nav_actul_xy_gls[keep_idx]
                        self.gp_nav_frntr_cntrs = self.gp_nav_frntr_cntrs[keep_idx]
                        try:
                            if len(self.gp_nav_frntr_areas) >= len(self.gp_nav_frntr_cntrs):
                                self.gp_nav_frntr_areas = self.gp_nav_frntr_areas[keep_idx]
                        except Exception:
                            self.gp_nav_frntr_areas = np.array([])
                        self.gp_nav_pts = self.gp_nav_frntr_cntrs
                        self.gp_nav_gls_sz = len(keep_idx)

        gap_to_gl_dst = np.sqrt((self.gl_y -
                                 self.gp_nav_actul_xy_gls.T[1]) ** 2 +
                                (self.gl_x - self.gp_nav_actul_xy_gls.T[0]) ** 2)

        self.gap_utlty_fun = self.gap_k_dst * (
                self.gp_nav_goal_dst + gap_to_gl_dst) + self.gap_k_dir * abs(
            self.gp_nav_frntr_cntrs.T[0]) ** 2
        print("gap_dir: ", abs(self.gp_nav_frntr_cntrs.T[0]))
        print("gap_to_gl_dst : ", gap_to_gl_dst)
        print("self.gap_utlty_fun: ", self.gap_utlty_fun)

        self.chsn_gl_idx = self.gap_utlty_fun.argmin(axis=0)
        self.gp_nav_pt = self.gp_nav_pts[self.chsn_gl_idx]
        print("recommended subgoal id and direction: ", self.chsn_gl_idx, self.gp_nav_pt[0])
        try:
            chosen_theta_rel = float(self.gp_nav_pt[0])
        except Exception:
            chosen_theta_rel = None
        if chosen_theta_rel is not None:
            self.create_interest_from_frontiers(chosen_theta_rel)
        self.closed.extend(closed)

    def publish_frontiers(self):
        """发布全量 frontier（theta_rel + area）供 GoalManager 聚合使用。

        不在这里决定车辆最终走哪条路：Goal Manager 会再根据 LSTE 状态、
        运动先验、激光安全距离和回退覆盖规则选出唯一 final_goal。
        """
        if self.gp_nav_frntr_cntrs is None:
            return
        try:
            thetas = np.array(self.gp_nav_frntr_cntrs).reshape(-1, 2)[:, 0]
        except Exception:
            return
        if thetas.size == 0:
            return
        areas_raw = getattr(self, "gp_nav_frntr_areas", None)
        if areas_raw is None:
            areas = np.ones_like(thetas, dtype=float)
        else:
            areas_raw = np.array(areas_raw).reshape(-1)
            if areas_raw.size != thetas.size:
                areas = np.ones_like(thetas, dtype=float)
                m = min(len(areas_raw), len(areas))
                areas[:m] = areas_raw[:m]
            else:
                areas = areas_raw

        msg = LsteFrontiers()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_footprint"
        msg.theta_rel = [float(t) for t in thetas]
        msg.area = [float(a) for a in areas]
        self.frontiers_pub.publish(msg)

    def update_global_goal_periodic(self):
        """
        基于 4-方向锚定 + 分叉 commit 的内部全局目标更新。
        这里维护的是 GP 的评分参考目标，不等于 Goal Manager 对外发布的
        /lste/final_goal。

        - commit_active=True: 沿 commit_heading 前推固定距离 D。
        - 否则用 frontier 按四个骨架方向聚合投票，满足门槛后切换。
        """
        if self.pose is None or not self.origin_set:
            return
        now = rospy.Time.now().to_sec()

        # 非 commit 状态下：按照 gl_update_period 节流更新频率
        # （commit 阶段仍然每帧更新，方向固定，只是沿既定方向前推）
        if (not self.commit_active) and (now - self.last_gl_update_time < self.gl_update_period):
            return

        # 1) 分叉检测 + commit 触发
        if (not self.commit_active) and self.gp_nav_frntr_cntrs is not None and len(self.gp_nav_frntr_cntrs) > 0:
            thetas = np.array(self.gp_nav_frntr_cntrs).reshape(-1, 2)[:, 0]
            areas = np.array(getattr(self, "gp_nav_frntr_areas", []))
            if len(areas) != len(thetas):
                areas = np.ones_like(thetas)
            thr_angle = np.deg2rad(25.0)
            sep_angle = np.deg2rad(60.0)
            left_mask = thetas < -thr_angle
            right_mask = thetas > thr_angle
            if np.any(left_mask) and np.any(right_mask):
                theta_L = thetas[left_mask]
                theta_R = thetas[right_mask]
                area_L = areas[left_mask]
                area_R = areas[right_mask]
                idx_L = int(np.argmax(area_L))
                idx_R = int(np.argmax(area_R))
                best_theta_L = theta_L[idx_L]
                best_theta_R = theta_R[idx_R]
                best_area_L = area_L[idx_L]
                best_area_R = area_R[idx_R]
                if abs(best_theta_R - best_theta_L) > sep_angle and abs(best_theta_L) > thr_angle and abs(best_theta_R) > thr_angle:
                    if max(best_area_L, best_area_R) >= 1.25 * min(best_area_L, best_area_R):
                        theta_best = best_theta_L if best_area_L >= best_area_R else best_theta_R
                        # 将 commit 方向也投射到 4 个固定世界方向，保证全局目标始终落在骨架射线上
                        heading_world = wrap_angle(self.pose.theta + theta_best)
                        diffs_commit = [abs(wrap_angle(heading_world - h)) for h in self.headings]
                        k_commit = int(np.argmin(diffs_commit))
                        self.commit_active = True
                        self.commit_heading = self.headings[k_commit]
                        self.commit_start_xy = (self.pose.x, self.pose.y)
                        rospy.loginfo_throttle(5.0, "Commit to heading %.2f rad (dir %d)", self.commit_heading, k_commit)

        # 2) commit 模式：沿 commit_heading 前推
        D = 6.0
        if self.commit_active:
            dx = self.pose.x - self.commit_start_xy[0]
            dy = self.pose.y - self.commit_start_xy[1]
            progress = dx * np.cos(self.commit_heading) + dy * np.sin(self.commit_heading)
            if progress >= 3.0:
                self.commit_active = False
            else:
                self.gl_x = self.pose.x + D * np.cos(self.commit_heading)
                self.gl_y = self.pose.y + D * np.sin(self.commit_heading)
                self.gl_yaw = self.commit_heading
                self.gl_wrt_odom = np.array([self.gl_x, self.gl_y, 1], dtype="float32")
                self.last_switch_xy = (self.pose.x, self.pose.y)
                self.last_gl_update_time = now
                return

        # 3) 非 commit：frontier → 骨架方向选择
        if self.gp_nav_frntr_cntrs is None or len(self.gp_nav_frntr_cntrs) == 0:
            return
        areas = np.array(getattr(self, "gp_nav_frntr_areas", []))
        if areas.size == 0:
            areas = np.ones(len(self.gp_nav_frntr_cntrs))

        # 对外发布“最大 frontier 方向”供 Goal Manager 使用（相对机器人角度）
        try:
            best_idx = int(np.argmax(areas))
            if best_idx >= 0 and best_idx < len(self.gp_nav_frntr_cntrs):
                theta_best = float(self.gp_nav_frntr_cntrs[best_idx][0])
                area_best = float(areas[best_idx]) if best_idx < len(areas) else 0.0
                msg = Vector3Stamped()
                msg.header.stamp = rospy.Time.now()
                msg.header.frame_id = "base_footprint"
                msg.vector.x = theta_best  # rad，相对机器人
                msg.vector.y = area_best
                msg.vector.z = 0.0
                self.frontier_dir_pub.publish(msg)
        except Exception:
            pass

        # 将所有 frontier 按最近的骨架方向聚合（只接受在扇区范围内的）
        sector_half = np.deg2rad(30.0)  # 扇区半宽，控制哪些 frontiers 参与投票
        sum_area = np.zeros(4, dtype=float)
        max_area = np.zeros(4, dtype=float)
        heading_weighted = np.zeros(4, dtype=float)

        for i, f in enumerate(self.gp_nav_frntr_cntrs):
            theta_rel = float(f[0])  # 相对机器人角
            heading_world = wrap_angle(self.pose.theta + theta_rel)
            diffs = [abs(wrap_angle(heading_world - h)) for h in self.headings]
            k = int(np.argmin(diffs))
            if diffs[k] > sector_half:
                continue  # 超出扇区范围的 frontiers 不参与该方向投票
            a = float(areas[i]) if i < len(areas) else 1.0
            sum_area[k] += a
            heading_weighted[k] += a * heading_world
            if a > max_area[k]:
                max_area[k] = a

        # 选票最多的骨架方向
        k_star = int(np.argmax(sum_area))
        if sum_area[k_star] <= 0:
            k_star = self.dir_idx  # 没有有效投票则保持当前方向
        # 该方向的加权平均朝向，用于一致性判断
        if sum_area[k_star] > 0:
            heading_f = wrap_angle(heading_weighted[k_star] / sum_area[k_star])
        else:
            heading_f = self.headings[k_star]

        # 门槛：角度一致性 + 行进距离
        diff_heading = abs(wrap_angle(heading_f - self.headings[k_star]))
        allow_switch = diff_heading < np.deg2rad(25.0)
        if self.last_switch_xy is None:
            traveled = 0.0
        else:
            dx = self.pose.x - self.last_switch_xy[0]
            dy = self.pose.y - self.last_switch_xy[1]
            traveled = np.hypot(dx, dy)
        if allow_switch and traveled >= 2.0 and k_star != self.dir_idx:
            self.dir_idx = k_star
            self.last_switch_xy = (self.pose.x, self.pose.y)

        heading = self.headings[self.dir_idx]
        self.gl_x = self.pose.x + D * np.cos(heading)
        self.gl_y = self.pose.y + D * np.sin(heading)
        self.gl_yaw = heading
        self.gl_wrt_odom = np.array([self.gl_x, self.gl_y, 1], dtype="float32")
        self.last_gl_update_time = now

    """ @brief:  publish recommended goal to DRL"""

    def gp_nav_glbl_gl_pbl(self):
        if self.gp_nav_actul_xy_gls is None or np.size(self.gp_nav_actul_xy_gls) == 0:
            return
        n = int(np.shape(self.gp_nav_actul_xy_gls)[0])
        if n <= 0:
            return
        if getattr(self, "chsn_gl_idx", None) is None:
            return
        if int(self.chsn_gl_idx) < 0 or int(self.chsn_gl_idx) >= n:
            return
        nav_xy_gl = self.gp_nav_actul_xy_gls[self.chsn_gl_idx].reshape(3, -1)
        yaw = np.arctan2(nav_xy_gl[1], nav_xy_gl[0])
        qtrn = quaternion_from_euler(0, 0, yaw)
        gl_msg = PoseStamped()
        gl_msg.pose.position.x = nav_xy_gl[0]
        gl_msg.pose.position.y = nav_xy_gl[1]
        gl_msg.pose.position.z = 0
        gl_msg.pose.orientation.x = qtrn[0]
        gl_msg.pose.orientation.y = qtrn[1]
        gl_msg.pose.orientation.z = qtrn[2]
        gl_msg.pose.orientation.w = qtrn[3]

        self.header.frame_id = "odom"
        gl_msg.header = self.header
        self.gp_rcmndd_subgl_pub.publish(gl_msg)
        print("recomended subgoal in odom frame: ", nav_xy_gl[0], nav_xy_gl[1])

        rbt_gl_msg = PoseStamped()
        rbt_gl_msg.pose.position.x = np.cos(self.pose.theta) * (gl_msg.pose.position.x - self.pose.x) + \
                                     np.sin(self.pose.theta) * (gl_msg.pose.position.y - self.pose.y)
        rbt_gl_msg.pose.position.y = - np.sin(self.pose.theta) * (gl_msg.pose.position.x - self.pose.x) + \
                                     np.cos(self.pose.theta) * (gl_msg.pose.position.y - self.pose.y)
        self.header.frame_id = "os_sensor"
        rbt_gl_msg.header = self.header
        self.gp_subgl_pub.publish(rbt_gl_msg)
        print("recomended subgoal in os_sensor frame: ", rbt_gl_msg.pose.position.x, rbt_gl_msg.pose.position.y)
        gp_nav_pcl = np.column_stack((rbt_gl_msg.pose.position.x, rbt_gl_msg.pose.position.y, 0, 1))
        self.header.frame_id = self.gp_nav_frame_id
        pc2 = point_cloud2.create_cloud(self.header, self.fields, gp_nav_pcl)
        self.gp_subgl_rviz.publish(pc2)
        # 同步发布到 LSTE 接口，供 PASS 态作为 frontier 使用
        self.lste_gp_frontier_pub.publish(gl_msg)

    #zcy
    def gp_interest_subgoal_publish(self):
        interest_count = self.gp_nav_gls_sz - 1
        if interest_count <= 0:
            rospy.loginfo("No interest subgoals to publish.")
            return

        # 获取剩余的候选点（排除已选择的子目标点）
        interest_points = self.gp_nav_actul_xy_gls.copy()
        interest_points = np.delete(interest_points, self.chsn_gl_idx, axis=0)

        # 临时列表用于存储当前帧检测到的点
        current_frame_points = []

        # 遍历剩余的候选点，逐个匹配并更新检测历史
        for idx, point in enumerate(interest_points):
            x, y, _ = point
            current_frame_points.append((x, y))
            matched = False
            for tracked_point in self.detected_interest_points:
                dist = distance.euclidean((x, y), (tracked_point['x'], tracked_point['y']))
                if dist < self.match_threshold:
                    tracked_point['count'] += 1
                    tracked_point['x'] = x  # 更新点的位置（可选）
                    tracked_point['y'] = y
                    matched = True
                    break
            if not matched:
                # 新检测到的点，初始化检测次数为1
                self.detected_interest_points.append({'x': x, 'y': y, 'count': 1})

        # 清理未在当前帧中检测到的点
        # 创建一个新列表，只保留当前帧检测到的点
        new_detected_interest_points = []
        for tracked_point in self.detected_interest_points:
            # 检查该点是否在当前帧的检测点中
            in_current_frame = False
            for (x, y) in current_frame_points:
                dist = distance.euclidean((tracked_point['x'], tracked_point['y']), (x, y))
                if dist < self.match_threshold:
                    in_current_frame = True
                    break
            if in_current_frame:
                new_detected_interest_points.append(tracked_point)
            else:

                pass
        self.detected_interest_points = new_detected_interest_points

        # 遍历检测历史，发布满足条件的点，连续出现3次以上才认为有效
        for tracked_point in self.detected_interest_points:
            if tracked_point['count'] >= 3:
                interest_gl_msg = PoseStamped()
                interest_gl_msg.pose.position.x = tracked_point['x']
                interest_gl_msg.pose.position.y = tracked_point['y']

                interest_gl_msg.header.frame_id = "odom"
                interest_gl_msg.header.stamp = rospy.Time.now()

                self.gp_interest_subgl_pub.publish(interest_gl_msg)
                print(f"Published filtered interest subgoal: x={tracked_point['x']}, y={tracked_point['y']}")


    """ @brief:  publish final goal"""

    def glbl_gl_pbl(self):
        # yaw = np.arctan2(self.gl_x, self.gl_y)
        # qtrn = quaternion_from_euler(0, 0, yaw)
        rbt_gl_msg = PoseStamped()
        rbt_gl_msg.pose.position.x = np.cos(self.pose.theta) * (self.gl_x - self.pose.x) + \
                                     np.sin(self.pose.theta) * (self.gl_y - self.pose.y)
        rbt_gl_msg.pose.position.y = - np.sin(self.pose.theta) * (self.gl_x - self.pose.x) + \
                                     np.cos(self.pose.theta) * (self.gl_y - self.pose.y)
        self.header.frame_id = "os_sensor"
        rbt_gl_msg.header = self.header
        self.gp_subgl_pub.publish(rbt_gl_msg)
        print("recomended subgoal is global goal, in os_sensor frame: ", rbt_gl_msg.pose.position.x,
              rbt_gl_msg.pose.position.y)
        # print("glbl_nav goal: ", nav_xy_gl[0], nav_xy_gl[1])

    """ @brief:  stop command when robot reach final goal"""

    def stop_cmd(self):
        gl_msg = PoseStamped()
        gl_msg.pose.position.x = self.pose.x
        gl_msg.pose.position.y = self.pose.y
        gl_msg.pose.position.z = 0
        gl_msg.pose.orientation.x = 0
        gl_msg.pose.orientation.y = 0
        gl_msg.pose.orientation.z = 0
        gl_msg.pose.orientation.w = 1
        self.header.frame_id = "odom"
        gl_msg.header = self.header
        self.gp_rcmndd_subgl_pub.publish(gl_msg)

    """ @brief:  publish variance surface for visualization"""

    def gp_grd_var_pcl(self):
        rds = self.gp_nav_var_viz * np.ones(np.shape(self.gp_grd_var)[0],
                                            dtype='float32').reshape(-1, 1)
        x, y, z = self.convert_spherical_2_cartesian(
            self.gp_grd.T[:][0].reshape(-1, 1),
            self.gp_grd.T[:][1].reshape(-1, 1), rds)
        intensity = np.array(self.gp_grd_var,
                             dtype='float32').reshape(-1, 1)
        gp_var_pcl = np.column_stack((x, y, z, intensity))
        self.header.frame_id = self.gp_nav_frame_id
        pc2 = point_cloud2.create_cloud(self.header, self.fields, gp_var_pcl)
        self.gp_var_pub.publish(pc2)

    """ @brief:  publish occupancy surface for visualization"""

    def gp_grd_oc_pcl(self):
        rds = self.gp_nav_var_viz * np.ones(np.shape(self.gp_grd_rds)[0],
                                            dtype='float32').reshape(-1, 1)
        x, y, z = self.convert_spherical_2_cartesian(
            self.gp_grd.T[:][0].reshape(-1, 1),
            self.gp_grd.T[:][1].reshape(-1, 1), rds)
        intensity = np.array(self.gp_grd_rds,
                             dtype='float32').reshape(-1, 1)
        gp_oc_pcl = np.column_stack((x, y, z, intensity))
        self.header.frame_id = self.gp_nav_frame_id
        pc2 = point_cloud2.create_cloud(self.header, self.fields, gp_oc_pcl)
        self.gp_oc_pub.publish(pc2)

    """ @brief:  publish navigation points for visualization"""

    def gp_nav_pts_pcl(self):
        rds = self.gp_nav_goal_dst * np.ones(self.gp_nav_gls_sz,
                                             dtype='float32').reshape(-1, 1)
        if self.gp_nav_pts is None or self.gp_nav_pts.size == 0:
            return
        x, y, z = self.convert_spherical_2_cartesian(
            self.gp_nav_pts.T[0].reshape(-1, 1),
            self.gp_nav_pts.T[1].reshape(-1, 1), rds)
        intensity = np.array(self.gap_utlty_fun,
                             dtype='float32').reshape(-1, 1)
        nav_pts_pcl = np.column_stack((x, y, z, intensity))
        self.header.frame_id = self.gp_nav_frame_id
        pc2 = point_cloud2.create_cloud(self.header, self.fields, nav_pts_pcl)
        self.gp_nav_pts_pub.publish(pc2)

    """ @brief:  publish navigation points in odom frame for visualization"""

    def gp_nav_xypts_actul_pcl(self):
        # print(">> gp_nav_xypts_actul_pcl:: ")
        if self.gp_nav_actul_xy_gls is None or len(self.gp_nav_actul_xy_gls) == 0:
            return
        intensity = np.array(self.gap_utlty_fun,
                             dtype='float32').reshape(-1, 1)
        if intensity.shape[0] != self.gp_nav_actul_xy_gls.shape[0]:
            intensity = np.zeros((self.gp_nav_actul_xy_gls.shape[0], 1), dtype='float32')
        nav_pts_pcl = np.column_stack((self.gp_nav_actul_xy_gls, intensity))
        self.header.frame_id = "odom"
        pc2 = point_cloud2.create_cloud(self.header, self.fields, nav_pts_pcl)
        self.gp_actul_xy_subgls_pub.publish(pc2)

    #zcy
    def clear_points(self):
        # 创建一个空的 PointCloud2 消息
        self.header.frame_id = "odom"
        empty_pcl = point_cloud2.create_cloud(self.header, self.fields, [])

        # 发布空的 PointCloud2 消息以清空 gp_nav_pts_pub
        self.gp_nav_pts_pub.publish(empty_pcl)

        # 发布空的 PointCloud2 消息以清空 gp_actul_xy_subgls_pub
        self.gp_actul_xy_subgls_pub.publish(empty_pcl)

    """ @brief:  downsample pointcloud"""

    def downsample_pcl(self, pcl_arr):
        """按方位角下采样球面点云，并生成 GP 训练样本。

        输入点格式来自 oc_srfc_proj：x=theta、y=alpha、z=distance、
        intensity=占据相关量。``pcl_skp`` 越大，训练点越少、计算越快，
        但方向分辨率越低。
        """
        pcl_arr = pcl_arr[np.argsort(pcl_arr[:, 0])]  ## sort based on thetas
        thetas = pcl_arr.transpose()[:][0].reshape(-1, 1)
        self.org_unq_thetas = np.array(sorted(set(
            thetas.flatten())))  #.reshape(-1,1)
        #### percentage to keep  or fraction to delete
        keep_th_ids = [
            t for t in range(0, np.shape(self.org_unq_thetas)[0], self.pcl_skp)]
        ids = []
        for t in keep_th_ids:
            ids = ids + list(np.where(thetas == self.org_unq_thetas[t])[0])
        pcl_arr = pcl_arr[ids]
        pcl_arr = pcl_arr.transpose()
        self.pcl_thetas = np.round(pcl_arr[:][0].reshape(-1, 1), 4)
        self.pcl_alphas = np.round(pcl_arr[:][1].reshape(-1, 1), 4)
        self.pcl_rds = np.round(pcl_arr[:][2].reshape(-1, 1), 4)
        self.pcl_oc = np.round(pcl_arr[:][3].reshape(-1, 1), 4)
        self.pcl_sz = np.shape(self.pcl_thetas)[0]
        self.pcl_unq_thetas = np.array(sorted(set(
            self.pcl_thetas.flatten())))  #.reshape(-1,1)
        self.unq_smpld_th_size = np.shape(self.pcl_unq_thetas)[0]

    """ @brief:  grid to reconstruct variance surface"""

    def sample_gp_nav_grid(self):
        th_rsltion = 0.01227  #0.00174  # 0.02 # #from -pi to pi rad -> 0 35999
        al_rsltion = 0.0349  #0.0349  #vpl16 resoltuion is 2 deg (from -15 to 15 deg)
        self.gp_grd_ths = np.arange(-np.pi + 0.0, np.pi,
                                    th_rsltion, dtype='float32')
        self.gp_grd_als = np.arange(np.pi / 2 - 0.261799, np.pi / 2 + 0.261799,
                                    al_rsltion, dtype='float32')
        self.gp_grd = np.array(np.meshgrid(self.gp_grd_ths,
                                           self.gp_grd_als)).T.reshape(-1, 2)
        self.gp_grd_w = np.shape(self.gp_grd_ths)[0]
        self.gp_grd_h = np.shape(self.gp_grd_als)[0]
        # print("\ngrid: ", np.shape(self.gp_grd))
        # print("grid w, h: ", self.gp_grd_w, self.gp_grd_h)

    """ @brief:  convert spherical to cartesian coordinates"""

    def convert_spherical_2_cartesian(self, theta, alpha, dist):
        x = np.array(dist * np.sin(alpha) * np.cos(theta),
                     dtype='float32').reshape(-1, 1)
        y = np.array(dist * np.sin(alpha) * np.sin(theta),
                     dtype='float32').reshape(-1, 1)
        z = np.array(dist * np.cos(alpha), dtype='float32').reshape(-1, 1)
        return x, y, z

    """ @brief:  convert  cartesian  to spherical coordinates"""

    def convert_cartesian_2_spherical(self, x, y, z):
        dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arctan2(y, x)
        alpha = np.arccos(z / dist)
        return theta, alpha, dist

    """ @brief:  callback of sph pointcloud """

    def sph_pcl_cb(self, sph_pcl_msg):
        """缓存最新球面点云；不在回调中执行昂贵的 GP 训练。"""
        # 防御：若 header 尚未初始化（极早期回调），先建一个
        if not hasattr(self, "header") or self.header is None:
            self.header = Header()
            self.header.frame_id = "odom"
        self.header.stamp = sph_pcl_msg.header.stamp
        pcl_arr = ros_numpy.point_cloud2.pointcloud2_to_array(
            sph_pcl_msg, squeeze=True)
        self.pcl_arr = np.round(np.array(pcl_arr.tolist(), dtype='float'), 4)

        """ @brief:  callback of robot pose """

    def pose_cb(self, rbt_pose_msg):
        """缓存最新二维位姿，供下一轮 step() 使用。"""
        self.pose = rbt_pose_msg

        """ @brief:  main process function """

    def step(self):
        """节点主循环：从最新传感器缓存计算 frontier、拓扑和调试输出。

        每轮可按以下顺序理解：

        1. 等待 pose 与 sph_pcl；
        2. 推进旧的回退/强制分支状态；
        3. 点云下采样并训练 GP；
        4. 在球面网格预测占据与不确定度；
        5. 提取并发布全部 frontier；
        6. 更新拓扑快照和 RViz 点云；
        7. 保留旧子目标接口的直达/绕障分支。

        回调只覆盖最新输入，因此本循环可能对同一帧运行多次；循环频率由
        GP 计算耗时决定，而不是由 /sph_pcl 的发布频率直接限定。
        """
        while not rospy.is_shutdown():
            # [1] 没有位姿无法把相对方向放进 odom；没有球面点云无法训练 GP。
            if self.pose is None or self.pcl_arr is None:
                rospy.logwarn_throttle(5.0, "gp_subgoal: waiting for rbt_pose and sph_pcl ...")
                rospy.sleep(0.05)
                continue
            # [2] 启动初期 Goal Manager 尚未发布 final_goal 时，先给 GP 一个
            # “正前方 10 m”的内部评分参考。它不会直接驱动车辆。
            if (not self._start_goal_seeded) and (not self.final_goal_received) and self.pose is not None:
                try:
                    # Pose2D: x, y, theta（theta 为朝向）
                    forward_dist = 10.0
                    self.gl_x = self.pose.x + forward_dist * np.cos(self.pose.theta)
                    self.gl_y = self.pose.y + forward_dist * np.sin(self.pose.theta)
                    self.gl_yaw = self.pose.theta
                    self.gl_wrt_odom = np.array([self.gl_x, self.gl_y, 1], dtype="float32")
                    self._start_goal_seeded = True
                except Exception:
                    pass
            # [3] Access-Topo 先兑现上轮决定：检查回退是否到点、强制方向
            # 是否已走够距离。可能更新 /lste/access_topo/mode 或回退目标。
            self.check_backtrack_progress()
            self.update_forced_heading_progress()
            # [4] 构造 robot <-> odom 的二维变换，并计算到内部参考目标的距离。
            self.tf_rbt_2_odom()
            self.tf_odom_2_rbt()
            self.rbt2gl_error()


            # [5] 将最新 /sph_pcl 整理为稀疏训练样本：(theta, alpha) -> occupancy。
            self.downsample_pcl(self.pcl_arr)

            # [6] 每轮以当前局部观测创建 GP，并在固定球面网格预测：
            # gp_grd_oc 为占据均值，gp_grd_var 为模型不确定度。
            self.gp_nav_din = np.column_stack((self.pcl_thetas, self.pcl_alphas))
            self.gp_nav_dout = np.array(self.pcl_oc, dtype='float').reshape(-1, 1)

            self.gp_nav_fit(0.09, 0.11, 0.7, 10, 10,
                            0.005)  #(ls1, ls2, var, alpha, noise, noise_var)
            nav_grd_oc, nav_grd_var = self.gp_nav.model.predict_f(self.gp_grd)
            self.gp_grd_var = nav_grd_var.numpy()
            self.gp_grd_oc = nav_grd_oc.numpy()
            self.gp_grd_rds = self.oc_srfc_rds - self.gp_grd_oc

            # [7] 方差图 -> 多个 frontier；随后发布给 Goal Manager，并同步维护
            # 路口/分支/回退快照。Goal Manager 而非这里发布最终 final_goal。
            self.gp_nav_mask_thrshld()
            self.gp_grd_var_img()
            self.gp_nav_pkup_nav_pt()
            self.publish_frontiers()
            self.update_global_goal_periodic()  # 每隔 gl_update_period 秒基于当前 frontier 更新全局目标
            self.dump_access_topo()
            ####calculate navigation point in world frame zcy topo模式下不发布subgoal及可视化
            # self.gp_nav_xypts_actul_pcl()
            # self.gp_nav_pts_pcl()

            # [8] 将 GP 预测转换回 PointCloud2，仅用于 RViz/调试观察。
            self.gp_grd_oc_pcl()
            self.gp_grd_var_pcl()

            # [9] 以下是保留的旧子目标接口：判断内部参考目标方向的预测可行
            # 距离，并据此发布 gp_subgoal/gp_subgoal_os 等话题。当前 LSTE 主链
            # 主要使用第 [7] 步的 /lste/gp_frontiers + Goal Manager。
            # 将内部目标从 odom 转回机器人坐标，得到它对应的球面方向。
            self.gl_wrt_rbt = self.tf_2d_inv @ self.gl_wrt_odom

            # predict occupancy in direction of final goal using GP occupancy model
            self.gl_th = np.arctan2(self.gl_wrt_rbt[1], self.gl_wrt_rbt[0])
            self.gl_al = np.pi / 2
            gl_oc, gl_var = self.gp_nav.model.predict_f(
                np.array([self.gl_th, self.gl_al], dtype="float32").reshape(1, 2))
            gl_rds = self.oc_srfc_rds - gl_oc.numpy()
            # print("gl_var, var_thrshld: ", gl_var.numpy(), self.gp_nav_var_thrshld)
            # print("gl_dst_err, gl_rds: ", self.gl_dst_err, gl_rds)

            # 若预测的自由距离足以覆盖目标距离，旧接口认为可直接朝目标走。
            if (self.gl_dst_err < gl_rds):
                if not self.goal_published:
                    self.glbl_gl_pbl()  ## not sure if this correct situation
                    self.goal_published = False
                    ##print("Navigation Mode: FinalGoal ")
                else:
                    quit()

            # 否则旧接口发布推荐子目标和其余兴趣点；若拓扑标志禁止旧接口，
            # 则清空其可视化点并进入下一轮。
            else:
                #zcy
                self.publish_closed_targets()
                #zcy如果标志位为1，则启用topo，停止发送gpsubgoal
                if not self.change_flag:
                    print("false,change topo")
                    self.clear_points()
                    continue
                ##print("Navigation Mode: SubGoal ")
                self.gp_nav_xypts_actul_pcl()
                self.gp_nav_pts_pcl()
                self.gp_nav_glbl_gl_pbl()
                self.gp_interest_subgoal_publish()



if __name__ == "__main__":
    VSGPNavGlb()
    try:
        VSGPNavGlb().step()
    except rospy.ROSInterruptException:
        pass

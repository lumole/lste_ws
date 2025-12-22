#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#from typing import Tuple, Optional
#import tempfile
#import pathlib
import warnings

#import io
import os
#import math
import numpy as np
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
from std_msgs.msg import Header, Bool
# defined msg
#from gp_subgoal.msg import PosePcl2
from geometry_msgs.msg import PoseStamped, PointStamped, Vector3Stamped
from geometry_msgs.msg import Pose2D, Point, Pose
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker
from lste_msgs.msg import LsteFrontiers

### import opencv
import cv2

#from cv_bridge import CvBridge, CvBridgeError  #after cv2


### to disable GPU for GP training and prediction: 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
### to select GPU
# tf.device("gpu:0")

#######disabled warning when import tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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


#### configurations
warnings.filterwarnings("ignore")
gpflow.config.set_default_float(np.float32)
np.random.seed(0)
tf.random.set_seed(0)

""" @brief: class to construct a 2D Sparse GP using gpflow """


class SGP2D:

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

    def __init__(self):
        ### Node initialization
        rospy.init_node("gp_subgoal")
        print("##############################################")
        print("              Initialize gp_subgoal           ")
        print("##############################################")

        ### subscriber to pose and occupancy surface
        #        self.pose_pcl_sub = rospy.Subscriber("pose_pcl",
        #                                             PosePcl2,
        #                                             self.pose_pcl_cb,
        #                                             queue_size=1)

        ### subscriber to pose and occupancy surface by two topics
        self.rbt_pose_sub = rospy.Subscriber("rbt_pose",
                                             Pose2D,
                                             self.pose_cb,
                                             queue_size=1)

        self.sph_pcl_sub = rospy.Subscriber("sph_pcl",
                                            PointCloud2,
                                            self.sph_pcl_cb,
                                            queue_size=1)
        ### publishers
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

        # ---- zcy 相关状态先初始化，再注册订阅者，避免回调在属性创建前触发 ----
        self.visited_positions = []
        self.int_visited_positions = []
        self.closed = []
        self.change_flag = True
        self.final_goal_received = False
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
        # 动态 final goal：上层发布 PoseStamped 到 /lste/final_goal
        self.final_goal_sub = rospy.Subscriber("/lste/final_goal",
                                               PoseStamped,
                                               self.final_goal_cb,
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

        ##variables to deal with odom msg and oc srfr msg
        self.pcl_arr = None

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
        self.header = Header()
        self.header.seq = 0
        self.header.stamp = None
        self.header.frame_id = "odom"
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

        ## waite for robot pose and oc_srfc
        while self.pose is None or self.pcl_arr is None:
            pass

        # 如果还没有收到 /lste/final_goal，则把“默认全局目标”
        # 设置为【起步位置正前方 10 m】（odom 坐标系）
        if not self.final_goal_received and self.pose is not None:
            try:
                # Pose2D: x, y, theta（theta 为朝向）
                forward_dist = 10.0
                self.gl_x = self.pose.x + forward_dist * np.cos(self.pose.theta)
                self.gl_y = self.pose.y + forward_dist * np.sin(self.pose.theta)
                self.gl_yaw = self.pose.theta
                self.gl_wrt_odom = np.array([self.gl_x, self.gl_y, 1], dtype="float32")
            except Exception as e:
                rospy.logwarn("Failed to set default final goal from start pose: %s", e)

        ## print all params
        self.print_ros_param()
        #rospy.spin()

    def final_goal_cb(self, msg: PoseStamped):
        """Update final goal from /lste/final_goal (odom frame PoseStamped)."""
        self.gl_x = msg.pose.position.x
        self.gl_y = msg.pose.position.y
        q = msg.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.gl_yaw = yaw
        self.gl_wrt_odom = np.array([self.gl_x, self.gl_y, 1], dtype="float32")
        self.final_goal_received = True

    def vanish_cb(self, msg: Pose):
        """消失点推断的前方距离，单位 m。"""
        try:
            dist = float(msg.position.x)
            if dist > 0:
                self.vanish_distance = dist
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
            l_frntr_cntrs = [[0, np.pi / 2], [np.pi / 2, np.pi / 2],
                             [np.pi, np.pi / 2], [-np.pi / 2, np.pi / 2]]
            l_frntr_ars = [12500, 12500, 12500, 12500]
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
        self.tf_2d = np.array(
            [[np.cos(self.pose.theta), -np.sin(self.pose.theta), self.pose.x],
             [np.sin(self.pose.theta),
              np.cos(self.pose.theta), self.pose.y], [0, 0, 1]])

    """ @brief:  odom to robot 2D TF matrix"""

    def tf_odom_2_rbt(self):
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
        # print("gp_nav_pkup_nav_pt:: ")
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
            if not self.is_visited([x, y], 3):
                new_gp_nav_actul_xy_gls.append(subgoal)
                new_gp_nav_frntr_cntrs.append(self.gp_nav_frntr_cntrs[idx])
                # 保留对应的 frontier 面积，后续用于“最空旷方向”判断
                if idx < len(self.gp_nav_frntr_areas):
                    new_gp_nav_frntr_areas.append(self.gp_nav_frntr_areas[idx])
            else:
                # closed.append(subgoal)
                # print(f"close_list =  {closed} ")

                closed_position = [int(x), int(y)]
                if closed_position not in self.closed:
                    closed.append(closed_position)

        #对齐形状
        self.gp_nav_actul_xy_gls = np.array(new_gp_nav_actul_xy_gls)
        self.gp_nav_frntr_cntrs = np.array(new_gp_nav_frntr_cntrs)
        self.gp_nav_frntr_areas = np.array(new_gp_nav_frntr_areas)
        self.gp_nav_pts = self.gp_nav_frntr_cntrs
        self.gp_nav_gls_sz = len(new_gp_nav_actul_xy_gls)

        # 全部的候选目标点
        print(f"gp_nav_actul_xy_gls={self.gp_nav_actul_xy_gls}")

        if len(self.gp_nav_actul_xy_gls) == 0:
            if not self.change_flag:
                return
            self.change_flag = False
            self.change_flag_pub.publish(self.change_flag)
            return

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

        self.closed.extend(closed)

    def publish_frontiers(self):
        """发布全量 frontier（theta_rel + area）供 GoalManager 聚合使用。"""
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
        基于 4-方向锚定 + 分叉 commit 的全局目标更新。
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
        intensity = np.array(self.gap_utlty_fun,
                             dtype='float32').reshape(-1, 1)
        # print("intensity: ", intensity)
        # print("self.gp_nav_actul_xy_gls: ", self.gp_nav_actul_xy_gls)
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
        self.header.stamp = sph_pcl_msg.header.stamp
        pcl_arr = ros_numpy.point_cloud2.pointcloud2_to_array(
            sph_pcl_msg, squeeze=True)
        self.pcl_arr = np.round(np.array(pcl_arr.tolist(), dtype='float'), 4)

        """ @brief:  callback of robot pose """

    def pose_cb(self, rbt_pose_msg):
        self.pose = rbt_pose_msg

        """ @brief:  main process function """

    def step(self):
        while not rospy.is_shutdown():
            ## transformation matrices between robot and world
            self.tf_rbt_2_odom()
            self.tf_odom_2_rbt()
            self.rbt2gl_error()


            ## Downsample and assign thetas, alphas, occs, rds variables
            self.downsample_pcl(self.pcl_arr)

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
            self.publish_frontiers()
            self.update_global_goal_periodic()  # 每隔 gl_update_period 秒基于当前 frontier 更新全局目标
            ####calculate navigation point in world frame zcy topo模式下不发布subgoal及可视化
            # self.gp_nav_xypts_actul_pcl()
            # self.gp_nav_pts_pcl()

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
                    ##print("Navigation Mode: FinalGoal ")
                else:
                    quit()

            ### mode: there is obstacle betwen goal and robot, robots follow recommended subgoal
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

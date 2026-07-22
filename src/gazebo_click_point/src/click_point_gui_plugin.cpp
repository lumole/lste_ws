#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <std_msgs/Empty.h>
#include <std_msgs/String.h>

#include <cstdlib>
#include <memory>
#include <string>

#include <QCursor>
#include <QHBoxLayout>
#include <QKeySequence>
#include <QShortcut>
#include <QToolButton>
#include <QToolTip>
#include <QTimer>

#include <gazebo/common/MouseEvent.hh>
#include <gazebo/gui/GuiIface.hh>
#include <gazebo/gui/GuiPlugin.hh>
#include <gazebo/gui/MouseEventHandler.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/rendering/UserCamera.hh>
#include <gazebo/transport/transport.hh>

namespace gazebo
{
class ClickPointGuiPlugin : public GUIPlugin
{
public:
  ClickPointGuiPlugin()
  {
    int argc = 0;
    char **argv = nullptr;
    if (!ros::isInitialized())
      ros::init(argc, argv, "gazebo_click_point", ros::init_options::NoSigintHandler);

    this->node_ = std::make_unique<ros::NodeHandle>();
    this->node_->setCallbackQueue(&this->callbackQueue_);
    this->publisher_ = this->node_->advertise<geometry_msgs::PointStamped>("/clicked_point", 1);
    this->goalPublisher_ = this->node_->advertise<geometry_msgs::PoseStamped>("/move_base/current_goal", 1);
    this->controllerTogglePublisher_ =
        this->node_->advertise<std_msgs::Empty>("/lste/controller_toggle", 1);
    this->controllerModeSubscriber_ = this->node_->subscribe(
        "/lste/controller_mode", 1, &ClickPointGuiPlugin::OnControllerMode, this);
    this->gazeboNode_ = transport::NodePtr(new transport::Node());
    this->gazeboNode_->Init();
    this->saveControlPublisher_ =
        this->gazeboNode_->Advertise<msgs::ServerControl>("/gazebo/server/control");

    this->controllerButton_ = new QToolButton(this);
    this->controllerButton_->setText("RL / KB");
    this->controllerButton_->setToolTip("Switch controller (Ctrl+Shift+K)");
    this->controllerButton_->setFixedSize(72, 28);
    QObject::connect(this->controllerButton_, &QToolButton::clicked,
                     [this]() { this->PublishControllerToggle(); });

    this->saveButton_ = new QToolButton(this);
    this->saveButton_->setText("Save");
    this->saveButton_->setToolTip("Overwrite the current world file");
    this->saveButton_->setFixedSize(52, 28);
    QObject::connect(this->saveButton_, &QToolButton::clicked,
                     [this]() { this->SaveCurrentWorld(); });

    auto *layout = new QHBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(this->controllerButton_);
    layout->addWidget(this->saveButton_);
    this->setLayout(layout);
    this->setFixedSize(130, 28);
    this->move(10, 10);

    this->controllerShortcut_ = new QShortcut(QKeySequence("Ctrl+Shift+K"), this);
    this->controllerShortcut_->setContext(Qt::ApplicationShortcut);
    QObject::connect(this->controllerShortcut_, &QShortcut::activated,
                     [this]() { this->PublishControllerToggle(); });

    this->rosSpinTimer_ = new QTimer(this);
    QObject::connect(this->rosSpinTimer_, &QTimer::timeout, [this]()
    {
      this->callbackQueue_.callAvailable(ros::WallDuration(0.0));
    });
    this->rosSpinTimer_->start(50);

    gui::MouseEventHandler::Instance()->AddPressFilter(
        "gazebo_click_point", [this](const common::MouseEvent &event)
        {
          return this->OnMousePress(event);
        });
    gui::MouseEventHandler::Instance()->AddReleaseFilter(
        "gazebo_click_point", [](const common::MouseEvent &event)
        {
          if (event.Button() == common::MouseEvent::LEFT)
            QToolTip::hideText();
          return false;
        });
    gui::MouseEventHandler::Instance()->AddMoveFilter(
        "gazebo_click_point", [this](const common::MouseEvent &event)
        {
          if (event.Buttons() & common::MouseEvent::LEFT)
            this->SelectPoint(event);
          return false;
        });
  }

  ~ClickPointGuiPlugin() override
  {
    gui::MouseEventHandler::Instance()->RemovePressFilter("gazebo_click_point");
    gui::MouseEventHandler::Instance()->RemoveReleaseFilter("gazebo_click_point");
    gui::MouseEventHandler::Instance()->RemoveMoveFilter("gazebo_click_point");
  }

  void Load(sdf::ElementPtr) override
  {
    this->setVisible(true);
  }

private:
  void SaveCurrentWorld()
  {
    const char *worldPath = std::getenv("LSTE_WORLD");
    const std::string worldName = gui::get_world();
    if (!worldPath || std::string(worldPath).empty() || worldName.empty())
    {
      QToolTip::showText(QCursor::pos(), "Save unavailable: current world is not ready", this);
      QTimer::singleShot(1800, []() { QToolTip::hideText(); });
      return;
    }

    msgs::ServerControl message;
    message.set_save_world_name(worldName);
    message.set_save_filename(worldPath);
    this->saveControlPublisher_->Publish(message);
    QToolTip::showText(QCursor::pos(), "Saved layout", this);
    QTimer::singleShot(1200, []() { QToolTip::hideText(); });
  }

  void PublishControllerToggle()
  {
    this->pendingControllerMode_ =
        this->currentControllerMode_ == "sappo" ? "teleop" : "sappo";
    std_msgs::Empty message;
    this->controllerTogglePublisher_.publish(message);
    QToolTip::showText(QCursor::pos(), "Switching controller...");
  }

  void OnControllerMode(const std_msgs::String::ConstPtr &message)
  {
    this->currentControllerMode_ = message->data;
    if (this->pendingControllerMode_.empty() ||
        this->currentControllerMode_ != this->pendingControllerMode_)
      return;

    const auto text = this->currentControllerMode_ == "teleop"
                          ? QString("Switched to keyboard control")
                          : QString("Switched to SA-PPO");
    this->pendingControllerMode_.clear();
    QToolTip::showText(QCursor::pos(), text, this);
    QTimer::singleShot(1600, []() { QToolTip::hideText(); });
  }

  bool OnMousePress(const common::MouseEvent &event)
  {
    if (event.Button() == common::MouseEvent::LEFT && event.Shift() &&
        this->hasSelectedPoint_)
    {
      this->PublishGoal();
      return true;
    }

    if (event.Button() == common::MouseEvent::LEFT)
      return this->SelectPoint(event);
    return false;
  }

  void PublishGoal()
  {
    geometry_msgs::PoseStamped goal;
    goal.header.stamp = ros::Time::now();
    goal.header.frame_id = "world";
    goal.pose.position.x = this->selectedPoint_.X();
    goal.pose.position.y = this->selectedPoint_.Y();
    goal.pose.position.z = this->selectedPoint_.Z();
    goal.pose.orientation.w = 1.0;
    this->goalPublisher_.publish(goal);

    const auto text = QString("Goal set: x=%1  y=%2")
                          .arg(this->selectedPoint_.X(), 0, 'f', 2)
                          .arg(this->selectedPoint_.Y(), 0, 'f', 2);
    QToolTip::showText(QCursor::pos(), text, this);
    QTimer::singleShot(1200, []() { QToolTip::hideText(); });
  }

  bool SelectPoint(const common::MouseEvent &event)
  {
    if (event.Dragging() && !(event.Buttons() & common::MouseEvent::LEFT))
      return false;

    auto camera = gui::get_active_camera();
    if (!camera)
      return false;

    const auto position = event.Pos();
    ignition::math::Vector3d point;
    const ignition::math::Planed ground(ignition::math::Vector3d::UnitZ, 0.0);
    if (!camera->WorldPointOnPlane(position.X(), position.Y(), ground, point))
      return false;

    geometry_msgs::PointStamped message;
    message.header.stamp = ros::Time::now();
    message.header.frame_id = "world";
    message.point.x = point.X();
    message.point.y = point.Y();
    message.point.z = point.Z();
    this->publisher_.publish(message);
    this->selectedPoint_ = point;
    this->hasSelectedPoint_ = true;

    const auto text = QString("Selected: x=%1  y=%2  z=%3\nShift + left-click to set goal")
                          .arg(point.X(), 0, 'f', 2)
                          .arg(point.Y(), 0, 'f', 2)
                          .arg(point.Z(), 0, 'f', 2);
    QToolTip::showText(QCursor::pos(), text, this);
    return false;
  }

  std::unique_ptr<ros::NodeHandle> node_;
  ros::CallbackQueue callbackQueue_;
  ros::Publisher publisher_;
  ros::Publisher goalPublisher_;
  ros::Publisher controllerTogglePublisher_;
  ros::Subscriber controllerModeSubscriber_;
  transport::NodePtr gazeboNode_;
  transport::PublisherPtr saveControlPublisher_;
  QToolButton *controllerButton_ = nullptr;
  QToolButton *saveButton_ = nullptr;
  QShortcut *controllerShortcut_ = nullptr;
  QTimer *rosSpinTimer_ = nullptr;
  std::string currentControllerMode_;
  std::string pendingControllerMode_;
  ignition::math::Vector3d selectedPoint_;
  bool hasSelectedPoint_ = false;
};

GZ_REGISTER_GUI_PLUGIN(ClickPointGuiPlugin)
}


roscore
rosrun lste_core lste_task_node.py _json_path:=/home/zrz/lste_ws/model/Data_exchange/vlm_prompt/lab/yellow_cup.json _task_id:=yellow_cup
rostopic echo /lste/task  #能看到发布的task

conda activate minicpm
cd ~/lste_ws/model/MiniCPM/test/
bash start.sh 

conda activate minicpm
rosparam set /lste_prompt_node/vllm_stop_command "pkill -f 'vllm serve'"
rosrun lste_core lste_prompt_node.py \
  _vllm_base_url:=http://localhost:8000/v1 \
  _vllm_model_name:=/home/zrz/lste_ws/model/MiniCPM/OpenBMB/MiniCPM4-0___5B
rostopic echo /lste/prompts  #能看到发布的prompts



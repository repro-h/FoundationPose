# 测试

cd ~/nas/mengxt/Projects/FoundationPose
conda activate foundationpose

CUDA_VISIBLE_DEVICES=1 xvfb-run -a python run_demo.py \
  --mesh_file demo_data/mustard0/mesh/textured_simple.obj \
  --test_scene_dir demo_data/mustard0 \
  --est_refine_iter 5 \
  --track_refine_iter 2 \
  --debug 2 \
  --debug_dir debug/mustard0
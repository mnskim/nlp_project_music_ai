export PYTHONPATH=`pwd`
export CUDA_VISIBLE_DEVICES=3
python eval_xai.py --task xa checkpoints/checkpoint_best_xai_apex_M2PFnP_base_1e-5.pt.pt  xai_data_bin_apex_reg_cls/0

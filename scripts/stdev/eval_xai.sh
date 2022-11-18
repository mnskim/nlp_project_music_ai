export PYTHONPATH=`pwd`
export CUDA_VISIBLE_DEVICES=2
#python eval_xai.py --task xai_M2PFnP --head_name xai_M2PFnP_res --checkpoint_file checkpoints/checkpoint_best_xai_apex_M2PFnP_base_1e-5.pt.pt  --data_dir xai_data_bin_apex_reg_cls/0

#python eval_xai.py --task xai_M2PF --head_name xai_head --checkpoint_file checkpoints/checkpoint_best_xai_apex_M2PF_0.00001_base_adv.pt.pt --data_dir xai_data_bin_apex_reg_cls/0

#python eval_xai.py --task xai_M2PF --head_name xai_head --checkpoint_file checkpoints/checkpoint_best_xai_apex_M2PF_0.0001_base.pt.pt --data_dir xai_data_bin_apex_reg_cls/0

#python eval_xai.py --task xai_M2PF --head_name xai_head --checkpoint_file checkpoints/checkpoint_best_xai_apex_0.00001_base_M2PF_rm_bad_adv.pt.pt --data_dir xai_data_bin_apex_rm_bad/0

python eval_xai.py --task xai_M2PF --head_name xai_head --checkpoint_file /data1/jongho/muzic/musicbert/checkpoints_stdev/checkpoint_best_xai_stdev_M2PF_0.000001_base.pt.pt --data_dir /data1/jongho/muzic/musicbert/processed/xai_data_bin_stdev/0


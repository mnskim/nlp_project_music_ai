export PYTHONPATH=`pwd`
export CUDA_VISIBLE_DEVICES=2
#python eval_xai.py --task xai_M2PFnP --head_name xai_M2PFnP_res --checkpoint_file checkpoints/checkpoint_best_xai_apex_M2PFnP_base_1e-5.pt.pt  --data_dir xai_data_bin_apex_reg_cls/0

#python eval_xai.py --task xai_M2PF --head_name xai_head --checkpoint_file checkpoints/checkpoint_best_xai_apex_M2PF_0.00001_base_adv.pt.pt --data_dir xai_data_bin_apex_reg_cls/0

#python eval_xai.py --task xai_M2PF --head_name xai_head --checkpoint_file checkpoints/checkpoint_best_xai_apex_M2PF_0.00001_base.pt.pt --data_dir processed/xai_data_bin_apex_reg_cls/0

#python eval_xai.py --task xai_M2PF --head_name xai_head --checkpoint_file checkpoints_rm_highstd/checkpoint_best_xai_apex_M2PF_0.00001_base.pt.pt --data_dir processed/xai_data_bin_apex_reg_cls_rm_highstd/0

# python eval_xai.py --task xai_M2PF --head_name xai_head --checkpoint_file checkpoints/checkpoint_best_xai_apex_M2PF_0.0001_base.pt.pt --data_dir processed/xai_data_bin_apex_reg_cls_rm_highstd/0 #0.36
# python eval_xai.py --task xai_M2PF --head_name xai_head --checkpoint_file checkpoints/checkpoint_best_xai_apex_M2PF_0.00001_base.pt.pt --data_dir processed/xai_data_bin_apex_reg_cls_rm_highstd/0 #0.40
# python eval_xai.py --task xai_M2PF --head_name xai_head --checkpoint_file checkpoints/checkpoint_best_xai_apex_M2PF_0.000001_base.pt.pt --data_dir processed/xai_data_bin_apex_reg_cls_rm_highstd/0 #0.3946

# python eval_xai.py --task xai_M2PF --head_name xai_head --checkpoint_file checkpoints_rm_highstd/checkpoint_best_xai_apex_M2PF_0.0001_base.pt.pt --data_dir processed/xai_data_bin_apex_reg_cls_rm_highstd/0 #0.37
# python eval_xai.py --task xai_M2PF --head_name xai_head --checkpoint_file checkpoints_rm_highstd/checkpoint_best_xai_apex_M2PF_0.00001_base.pt.pt --data_dir processed/xai_data_bin_apex_reg_cls_rm_highstd/0 #0.389
# python eval_xai.py --task xai_M2PF --head_name xai_head --checkpoint_file checkpoints_rm_highstd/checkpoint_best_xai_apex_M2PF_0.000001_base.pt.pt --data_dir processed/xai_data_bin_apex_reg_cls_rm_highstd/0 #0.3928

# python eval_xai.py --task xai_M2PF --head_name xai_head --checkpoint_file checkpoints_rm_highstd/checkpoint_best_xai_apex_M2PF_0.0001_base.pt.pt --data_dir processed/xai_data_bin_apex_reg_cls/0 #0.341
# python eval_xai.py --task xai_M2PF --head_name xai_head --checkpoint_file checkpoints_rm_highstd/checkpoint_best_xai_apex_M2PF_0.00001_base.pt.pt --data_dir processed/xai_data_bin_apex_reg_cls/0 #0.362
# python eval_xai.py --task xai_M2PF --head_name xai_head --checkpoint_file checkpoints_rm_highstd/checkpoint_best_xai_apex_M2PF_0.000001_base.pt.pt --data_dir processed/xai_data_bin_apex_reg_cls/0 #0.369

# # baseline
# python eval_xai.py --task xai_M2PF --head_name xai_head --checkpoint_file checkpoints/checkpoint_best_xai_apex_M2PF_0.0001_base.pt.pt --data_dir processed/xai_data_bin_apex_reg_cls/0    #0.372
# python eval_xai.py --task xai_M2PF --head_name xai_head --checkpoint_file checkpoints/checkpoint_best_xai_apex_M2PF_0.00001_base.pt.pt --data_dir processed/xai_data_bin_apex_reg_cls/0    #0.396
# python eval_xai.py --task xai_M2PF --head_name xai_head --checkpoint_file checkpoints/checkpoint_best_xai_apex_M2PF_0.000001_base.pt.pt --data_dir processed/xai_data_bin_apex_reg_cls/0  #0.3863


#python eval_xai.py --task xai_M2PF --head_name xai_head --checkpoint_file checkpoints/checkpoint_best_xai_apex_0.00001_base_M2PF_rm_bad_adv.pt.pt --data_dir xai_data_bin_apex_rm_bad/0
#python eval_xai.py --task xai_M2PF --head_name xai_head --checkpoint_file checkpoints_base_split/checkpoint_best_xai_apex_M2PF_0.00001_base.pt.pt --data_dir processed/xai_data_bin_apex_reg_cls_base_split/0

# python eval_xai.py --task xai_M2PF --head_name xai_head --checkpoint_file checkpoints_16bars/checkpoint_best_xai_apex_M2PF_0.0001_base_16.pt.pt --data_dir processed/xai_data_bin_apex_reg_cls_16bars/0
# python eval_xai.py --task xai_M2PF --head_name xai_head --checkpoint_file checkpoints_16bars/checkpoint_best_xai_apex_M2PF_0.00001_base_16.pt.pt --data_dir processed/xai_data_bin_apex_reg_cls_16bars/0
# python eval_xai.py --task xai_M2PF --head_name xai_head --checkpoint_file checkpoints_16bars/checkpoint_best_xai_apex_M2PF_0.000001_base_16.pt.pt --data_dir processed/xai_data_bin_apex_reg_cls_16bars/0

# python eval_xai.py --task xai_M2PF --head_name xai_head --checkpoint_file checkpoints_8_16_match/checkpoint_best_xai_apex_M2PF_0.0001_base_8.pt.pt --data_dir processed/subset_8_16/xai_data_bin_apex_8bars/0
# python eval_xai.py --task xai_M2PF --head_name xai_head --checkpoint_file checkpoints_8_16_match/checkpoint_best_xai_apex_M2PF_0.00001_base_8.pt.pt --data_dir processed/subset_8_16/xai_data_bin_apex_8bars/0
# python eval_xai.py --task xai_M2PF --head_name xai_head --checkpoint_file checkpoints_8_16_match/checkpoint_best_xai_apex_M2PF_0.000001_base_8.pt.pt --data_dir processed/subset_8_16/xai_data_bin_apex_8bars/0

python eval_xai.py --task xai_M2PF --head_name xai_head --checkpoint_file checkpoints_8bars/checkpoint_best_xai_apex_M2PF_0.0001_base_8.pt.pt --data_dir processed/xai_data_bin_apex_reg_cls_8bars/0
python eval_xai.py --task xai_M2PF --head_name xai_head --checkpoint_file checkpoints_8bars/checkpoint_best_xai_apex_M2PF_0.00001_base_8.pt.pt --data_dir processed/xai_data_bin_apex_reg_cls_8bars/0
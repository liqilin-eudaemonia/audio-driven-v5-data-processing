exp_folder=$1
pred_folder="${2:-${exp_folder}/pred_gs3.0_scale-cond1.0-ref1.0}"
gt_folder=${3:-${exp_folder}/gt}
# pred_folder="/data/UniAnimate-DiT/uni_outputs/eval_pose"
# pred_folder="/data/Moore-AnimateAnyone/output/moore"
# gt_folder="/data/UniAnimate-DiT/uni_outputs/eval_gtpose"
pred_folder="/data/pipline/human_centric_vw_model/benchmark/out_8_0_32_VA_eval_5s_dialog_pretrain_step_51000"
gt_folder="/data/pipline/human_centric_vw_model/benchmark/testset_gt"
exp_folder="/data/pipline/human_centric_vw_model/benchmark/video_me_dialog"
echo "pred_folder" ${pred_folder}
echo "gt_folder" ${gt_folder}

CUDA_VISIBLE_DEVICES="0" python vid_metrics/metric_center.py --path_gen ${pred_folder}/ --path_gt ${gt_folder} --type fid  --write_metric_to ${exp_folder}/fid.json
CUDA_VISIBLE_DEVICES="0" python vid_metrics/metric_center.py --path_gen ${pred_folder}/ --path_gt ${gt_folder} --type ssim psnr lpips  --write_metric_to ${exp_folder}/l1_ssim.json
# Pytorch-FID
# python -m pytorch_fid ${pred_folder}/ ${gt_folder} --device cuda:0
#  FVD eval
# the root dir should be the dir containing the fvd pretrain model (resnet-50-kinetics and i3d)
python  vid_metrics/metric_center.py --root_dir ./weights --path_gen ${pred_folder}/ --path_gt ${gt_folder}/ --type fid-vid  --write_metric_to ${exp_folder}/fid-vid_fvd.json --number_sample_frames 81

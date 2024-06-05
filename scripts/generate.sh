source scripts/functions.sh

# Speaker & exp.
speaker=m001_trump
exp=decmp
exp_name=animnet-decmp

# Add audio in config/test_media/gen_$speaker.yaml

# Generate 3D results.
GenAnimNet --generating --data_src=celebtalk --wanna_fps=25 \
  --exp=$exp \
  --exp_name=$exp_name \
  --load='epoch_50.pth' \
  --speaker="$speaker" \
  --test_media="gen_$speaker" \
  --dump_offsets \
  --dump_audio \
  model.visualizer.video_grid_size=512 \
;

# Neural Rendering.
# Please change the arguments.
for audio_name in \
  speech@clip0 \
  speech@clip1 \
  speech@clip2 \
  speech@clip3 \
  speech@clip4
do
  python -m scripts.neural_render \
    --nr_ckpt="runs/neural_renderer/$speaker/checkpoints/epoch_60.pth" \
    --out_path="runs/anime/$speaker/$exp_name/generated/[50][test]test-clips/${audio_name}-nr.mp4" \
    --audio_path="runs/anime/$speaker/$exp_name/generated/[50][test]test-clips/${audio_name}/audio.wav" \
    --offsets_npy="runs/anime/$speaker/$exp_name/generated/[50][test]test-clips/${audio_name}/dump-offsets-final.npy" \
    --iden_path="assets/datasets/talk_video/celebtalk/data/$speaker/fitted/identity/identity.obj" \
    --reenact_video="assets/datasets/talk_video/celebtalk/data/$speaker/avoffset_corrected/vld-000-fps25.mp4" \
    --reenact_coeff="assets/datasets/talk_video/celebtalk/data/$speaker/fitted/vld-000" \
    --reenact_static_frame=0 \
  ;
done

echo "\n"
echo "Done! Videos are generated at: runs/anime/$speaker/$exp_name/generated/[50][test]test-clips/"

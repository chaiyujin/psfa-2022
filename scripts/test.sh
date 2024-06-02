source scripts/functions.sh

spk_args=" --data_src=celebtalk --speaker=m001_trump --wanna_fps=25"
trn_args=" --same_idle"
gen_args=" --generating"

GenAnimNet --exp='decmp' --exp_name='animnet-decmp-abl_no_reg' --load='epoch_50.pth' $spk_args $gen_args \
  --test_media=test_clips;

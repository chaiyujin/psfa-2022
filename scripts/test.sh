source scripts/functions.sh

spk_args=" --data_src=celebtalk --speaker=m001_trump --wanna_fps=25"
trn_args=" --same_idle"
gen_args=" --generating"

GenAnimNet --exp='decmp' --exp_name='animnet-decmp-abl_no_reg' --load='epoch_50.pth' $spk_args $gen_args \
  --test_media=regen_m001_trump \
  --dump_offsets \
  --dump_audio \
  data.audio.deepspeech.graph_pb=$HOME/assets/pretrain_models/deepspeech-0.1.0-models/output_graph.pb \
;

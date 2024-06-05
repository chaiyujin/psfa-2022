#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

# * ---------------------------------------------------------------------------------------------------------------- * #
# * Args
# * ---------------------------------------------------------------------------------------------------------------- * #

function GetArgs() {
  local exp=
  local exp_name=
  local extra_name=
  # shared
  local data_src=celebtalk
  local speaker=
  local wanna_fps=30
  local avoffset_ms=0  # control output avoffset, if not 0, should match speaker's data
  local correct_avoffset=true
  # for generating
  local is_gen=false
  local test_media=
  local vis_data_src=
  local vis_speaker=
  local same_idle=false
  local dump_offsets=false
  local dump_metrics=false
  local dump_audio=false
  local video_postfix=''
  # other
  local other=
  for i in "$@"; do
    case $i in
      --exp=*          ) exp=${i#*=}            ;;
      --exp_name=*     ) exp_name=${i#*=}       ;;
      --extra_name=*   ) extra_name=${i#*=}     ;;
      # shared
      --data_src=*     ) data_src=${i#*=}       ;;
      --speaker=*      ) speaker=${i#*=}        ;;
      --wanna_fps=*    ) wanna_fps=${i#*=}      ;;
      --avoffset_ms=*  ) avoffset_ms=${i#*=}    ;;
      --keep_avoffset  ) correct_avoffset=false ;;
      # generating
      --generating     ) is_gen=true            ;;
      --test_media=*   ) test_media=${i#*=}     ;;
      --vis_data_src=* ) vis_data_src=${i#*=}   ;;
      --vis_speaker=*  ) vis_speaker=${i#*=}    ;;
      --same_idle      ) same_idle=true         ;;
      --dump_offsets   ) dump_offsets=true      ;;
      --dump_metrics   ) dump_metrics=true      ;;
      --dump_audio     ) dump_audio=true        ;;
      --video_postfix=*) video_postfix=${i#*=}  ;;
      # other
      *) other="${other} ${i}";;
    esac
  done

  # > required
  [ -n "${exp}"      ] || { echo "[ERROR] --exp is not given!"; exit 1; }
  [ -n "${exp_name}" ] || { echo "[ERROR] --exp_name is not given!"; exit 1; }
  if [[ ! "${exp_name}" =~ .*"${exp}".* ]]; then
    echo "[ERROR] --exp_name '${exp_name}' not match with --exp '${exp}'"; exit 1;
  fi

  # > set default values depends on other
  [ -n "${test_media}"   ] || { test_media="training_${data_src}"; }
  [ -n "${vis_data_src}" ] || { vis_data_src=${data_src};          }
  [ -n "${vis_speaker}"  ] || { vis_speaker=${speaker};            }
  if [ "$same_idle" == "true" ]; then
    video_postfix+='-same_idle'
  fi

  # > some values depends on mode
  local max_duration=20
  if [ "$is_gen" == "true" ]; then
    max_duration=1000000
  fi

  # > speakers from vocaset
  local speakers_voca='${data.VOCASET.REAL3D_SPEAKERS}'
  if [[ "$exp" == "track" ]]; then
    speakers_voca='[]'
  fi

  # > get exp_dir
  local root='${path.runs_dir}/anime'

  # > args string
  local shared=" experiment=animnet_${exp} path.exp_dir=${root}/$speaker/${exp_name}${extra_name}"
  # data
  shared+=" data=all_vocaset dataset=talk_voca datamodule=talk_voca"
  shared+=" data.audio.sliding_window.deepspeech=16"
  shared+=" dataset.data_src_talk=${data_src}"
  shared+=" dataset.speakers_talk=[${speaker}]"
  shared+=" dataset.speakers_voca=${speakers_voca}"
  shared+=" dataset.random_shift_alpha=true"
  shared+=' +dataset.VOCASET=${data.VOCASET}'
  shared+=" datamodule.batch_size=4"
  shared+=" datamodule.train_source_voca=[vocaset_all.csv]"
  # training
  shared+=" trainer.max_epochs=60"
  shared+=" callbacks.model_checkpoint.monitor=epoch-val_metric/mvd-avg:lips"
  # model
  shared+=" model.wanna_fps=${wanna_fps}"
  shared+=" model.data_info.correct_avoffset=${correct_avoffset}"
  shared+=" model.neural_renderer.load_from.path=dont_build"
  shared+=" +model.dont_render=true"
  # loss
  shared+=" model.loss.mesh.part=face_noeyeballs"
  # testing
  shared+=" test_media@model.test_media=${test_media}"
  shared+=" model.test_media.do=[misc]"
  # visualizer
  shared+=" ++model.visualizer.data_src=${vis_data_src}"
  shared+=" ++model.visualizer.speaker=${vis_speaker}"
  shared+=" ++model.visualizer.avoffset_ms=${avoffset_ms}"
  shared+=" ++model.visualizer.video_1st_epoch=true"
  shared+=" ++model.visualizer.video_gap_epochs=10"
  shared+=" ++model.visualizer.draw_gap_steps=1000"
  shared+=" ++model.visualizer.same_idle=${same_idle}"
  shared+=" ++model.visualizer.video_postfix=${video_postfix}"
  shared+=" ++model.visualizer.max_duration=${max_duration}"
  shared+=" ++model.visualizer.generate_test=true"
  shared+=" ++model.visualizer.generate_valid=false"
  shared+=" ++model.visualizer.generate_train=false"
  shared+=" ++model.visualizer.dump_offsets=${dump_offsets}"
  shared+=" ++model.visualizer.dump_metrics=${dump_metrics}"
  shared+=" ++model.visualizer.dump_audio=${dump_audio}"
  shared+=' ++model.visualizer.dkwargs.style_id="${index_of:${dataset.speakers},${model.visualizer.speaker}}"'

  # > Extra args for decmp
  if [ "$exp" == "decmp" ]; then
    shared="${shared} $(GetArgsForDecmp)"
  fi

  # > Concate with other unknown args
  echo "${shared} ${other}"
}

function GetArgsForDecmp() {
  local shared=''
  shared+=" dataset.gt_for_swap=true"
  shared+=' +dataset.default_swap=${model.test_media.default_extra}'
  shared+=" datamodule.sampler=BatchSamplerWithDtwPair"
  shared+=" datamodule.sampler_kwargs.pairs_from_same_speaker=true"
  shared+=" datamodule.sampler_kwargs.pairs_from_diff_speaker=true"
  shared+=' +model.datamodule_sampler_kwargs=${datamodule.sampler_kwargs}'
  shared+=" ++model.visualizer.video_gap_epochs=10"
  shared+=" ++model.visualizer.draw_gap_steps=2000"
  echo ${shared}
}

# * ---------------------------------------------------------------------------------------------------------------- * #
# * Train AnimNet
# * ---------------------------------------------------------------------------------------------------------------- * #

# * Baselines (Ablation of structure)
function TrainAnimNet() {
  # * Baseline1: Only Tracked data
  python3 -m src mode=train $(GetArgs --exp=track --exp_name=animnet-track "$@");
  # * Baseline2: VOCASET + Tracked data
  python3 -m src mode=train $(GetArgs --exp=cmb3d --exp_name=animnet-cmb3d "$@");
}

function TrainAnimNetTrack() {
  python3 -m src mode=train $(GetArgs --exp=track --exp_name=animnet-track "$@");
}

# * Ablation of structure: naive style encoder from renference animation
function TrainAnimNetRefer() {
  python3 -m src mode=train $(GetArgs --exp=refer --exp_name=animnet-refer "$@") \
    model.data_info.pad_tgt=true \
    dataset.gt_for_swap=true \
    dataset.swap_using_self=true \
    +dataset.default_swap='${model.test_media.default_extra}' \
  ;
}

function TrainAnimNetCmb3dDecmp() {
  python3 -m src mode=train $(GetArgs --exp=cmb3d --exp_name=animnet-cmb3d "$@");
  TrainAnimNetDecmpAblation "$@" --ablations="no_reg";
}

function TrainAnimNetReferE2e() {
  TrainAnimNetRefer "$@";
  TrainAnimNetDecmpAblation "$@" --ablations="e2e";
}

function TrainAnimNetDecmpAblDtwSwp() {
  TrainAnimNetDecmpAblation "$@" --ablations="no_reg no_dtw";
  TrainAnimNetDecmpAblation "$@" --ablations="no_reg no_swp";
}

# * ---------------------------------------------------------------------------------------------------------------- * #
# * Train Decmp
# * ---------------------------------------------------------------------------------------------------------------- * #

function TrainAnimNetDecmp() {
  python3 -m src mode=train $(GetArgs --exp=decmp --exp_name=animnet-decmp "$@");
}

# * Loss ablations
function TrainAnimNetDecmpAblation() {
  local ablations=
  local other=
  for i in "$@"; do
    case $i in
      --ablations=*  ) ablations=${i#*=} ;;
      *) other="${other} ${i}"           ;;
    esac
  done

  local extra_args=
  local abl_name=
  for abl in $ablations; do
    [ -n "${abl_name}" ] && { abl_name="${abl_name}_"; }
    abl_name="${abl_name}${abl}"
    case $abl in
      no_dtw) extra_args+=" model.ablation.${abl}=true datamodule.sampler_kwargs.pairs_from_diff_speaker=false" ;;
      no_swp) extra_args+=" model.ablation.${abl}=true model.loss.swp=0" ;;
      no_ph2) extra_args+=" model.ablation.${abl}=true model.loss.cyc=0" ;;
      no_cyc) extra_args+=" model.ablation.${abl}=true model.loss.cyc=0" ;;
      no_reg) extra_args+=" model.ablation.${abl}=true model.loss.reg_z_ctt_close=0 model.loss.reg_z_sty_close=0 model.loss.reg_z_aud_close=0" ;;
      e2e   ) extra_args+=" model.ablation.${abl}=true dataset.gt_for_swap=true +dataset.swap_for_e2e=true datamodule.sampler_kwargs.pairs_from_diff_speaker=false datamodule.sampler_kwargs.pairs_from_same_speaker=false model.loss.swp=0 model.loss.cyc=0 model.loss.reg_z_ctt_close=0 model.loss.reg_z_sty_close=0 model.loss.reg_z_aud_close=0" ;;
    esac
  done

  local args=$(GetArgs --exp=decmp --exp_name=animnet-decmp-abl_${abl_name} ${other})
  python3 -m src mode=train ${args} ${extra_args};
}

# * ---------------------------------------------------------------------------------------------------------------- * #
# * Generating
# * ---------------------------------------------------------------------------------------------------------------- * #

function GenAnimNet() {
  local load=
  local other="--generating"
  for i in "$@"; do
    case $i in
      --load=*) load=${i#*=} ;;
      *) other="${other} ${i}";;
    esac
  done

  [ -n "$load" ] || { echo "--load is not set!"; exit 1; }

  local args=$(GetArgs $other);
  python3 -m src mode=generate hydra.run.dir=.snaps/run \
    ${args} \
    load_from='${path.exp_dir}/checkpoints/'${load} \
    strict=false \
  ;
}

function GenAll() {
  local load_decmp=
  local load_cmb3d=
  local load_refer=
  local load_track=
  local ablation=
  local shared=
  for i in "$@"; do
    case $i in
      --load_decmp=*) load_decmp=${i#*=} ;;
      --load_cmb3d=*) load_cmb3d=${i#*=} ;;
      --load_refer=*) load_refer=${i#*=} ;;
      --load_track=*) load_track=${i#*=} ;;
      --ablation    ) ablation=true       ;;
      *) shared="${shared} ${i}";;
    esac
  done

  [ -n "${load_cmb3d}" ] && GenAnimNet --exp='cmb3d' --exp_name='animnet-cmb3d' --load=${load_cmb3d} ${shared};
  [ -n "${load_refer}" ] && GenAnimNet --exp='refer' --exp_name='animnet-refer' --load=${load_refer} ${shared};
  [ -n "${load_track}" ] && GenAnimNet --exp='track' --exp_name='animnet-track' --load=${load_track} ${shared};

  if [ -n "${load_decmp}" ]; then
    GenAnimNet --exp='decmp' --exp_name='animnet-decmp-abl_no_reg' --load=${load_decmp} ${shared};
    if [ -n "${ablation}" ]; then
      GenAnimNet --exp='decmp' --exp_name='animnet-decmp-abl_no_reg_no_dtw' --load=${load_decmp} ${shared};
      GenAnimNet --exp='decmp' --exp_name='animnet-decmp-abl_no_reg_no_swp' --load=${load_decmp} ${shared};
      GenAnimNet --exp='decmp' --exp_name='animnet-decmp-abl_no_reg_no_cyc' --load=${load_decmp} ${shared};
      GenAnimNet --exp='decmp' --exp_name='animnet-decmp-abl_e2e'           --load=${load_decmp} ${shared};
      # GenAnimNet --exp='decmp' --exp_name='animnet-decmp-abl_no_reg' --load=${load_decmp} ${shared};
      # GenAnimNet --exp='decmp' --exp_name='animnet-decmp-abl_no_ph2' --load=${load_decmp} ${shared};
      # GenAnimNet --exp='decmp' --exp_name='animnet-decmp-abl_no_reg_no_ph2' --load=${load_decmp} ${shared};
    fi
  fi
}

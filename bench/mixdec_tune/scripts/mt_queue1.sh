#!/bin/bash
# MIXDEC_TUNE part 1 queue: one measurement at a time, each only with >= 16 GB available
while [ ! -f ~/mt_meas/dump.done ]; do sleep 20; done
run() {
  while [ "$(free -g | awk "/^Mem:/{print \$7}")" -lt 16 ]; do sleep 60; done
  ~/mt_eval.sh "$@"
}
B="ratio=0.3,copy=0"
run M0 "$B"
run CLS "$B,cls=1"
run DOC "$B,doc=1"
run NG "$B,ngo=1:2:3:4:5:6:7:8:12:16,ngcap=8,nglr=0.000625"
run KN "$B,bkn=10:4:2:1"
run MX "$B,floor=1.67652e-05,mlr=0.00181686,decay=0.0001"
run T1 "$B,t1=1"
run MIX2a "$B,mix2=1"
run MIX2b "$B,mix2=2"
run PRE "$B,ngpre=8388608,ngtb=20"
run GEN "$B" --mixdec-generic-bytes 2097152
run ART4 "$B,artk=4,art=32768"
run LSTM0 "$B,lstm=0"
while [ ! -f ~/mt_data/lstm32.done ]; do sleep 30; done
run LSTMx "$B" --mixdec-lstm $HOME/mt_data/clean64_32m.lstm
echo done > ~/mt_meas/queue1.done

#!/usr/bin/env bash
SESSION=1
PARA1=$1
EPOCH=${PARA1:=6}
PARA2=$2
CHECKPOINT=${PARA2:=18197}
python test_net.py --dataset pascal_voc --net res101 \
                   --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
                   --cuda
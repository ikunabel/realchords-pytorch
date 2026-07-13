rsync -avz --progress \
  --exclude-from "$PWD/data/exclude.txt" \
  $PWD/data/ \
  yh522379@copy23-1.hpc.itc.rwth-aachen.de:/home/yh522379/realchords-pytorch/data/

rsync -avz --progress \
  $PWD/journal/ \
  yh522379@copy23-1.hpc.itc.rwth-aachen.de:/home/yh522379/realchords-pytorch/journal/

rsync -avz --progress \
  $PWD/logs/eval/ \
  yh522379@copy23-1.hpc.itc.rwth-aachen.de:/home/yh522379/realchords-pytorch/logs/eval/

rsync -avz --progress \
  $PWD/logs/eval/ \
  yh522379@copy23-1.hpc.itc.rwth-aachen.de:/home/yh522379/realchords-pytorch/logs/eval/

rsync -avz --progress \
  $PWD/logs/paired_eval/ \
  yh522379@copy23-1.hpc.itc.rwth-aachen.de:/home/yh522379/realchords-pytorch/logs/paired_eval/

rsync -avz --progress \
  $PWD/logs/generated/ \
  yh522379@copy23-1.hpc.itc.rwth-aachen.de:/home/yh522379/realchords-pytorch/logs/generated/



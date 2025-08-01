# bash ./train_LEDaug_BART_10train_tune_tensorboard_5.sh ../config/config-bert.txt > train_LEDaug_BART_10train_tune_tensorboard_5_results.txt &
#!/usr/bin/env bash
# train_data= "C:\Users\CSE RGUKT\Downloads\TTS\TTS\TTS_zeroshot\data\raw_train_all_subset10_onecol.csv"
# dev_data= "C:\Users\CSE RGUKT\Downloads\TTS\TTS\TTS_zeroshot\data\raw_val_all_onecol.csv"
# test_data= "C:\Users\CSE RGUKT\Downloads\TTS\TTS\TTS_zeroshot\data\raw_test_all_onecol.csv"

# kg_data= "C:\Users\CSE RGUKT\Downloads\TTS\TTS\TTS_zeroshot\data\raw_train_all_subset_kg_epoch_onecol.csv"

train_data=../data/raw_train_all_subset10_onecol.csv
dev_data=../data/raw_val_all_onecol.csv
test_data=../data/raw_test_all_onecol.csv

kg_data=../data/raw_train_all_subset_kg_epoch_onecol.csv

for seed in {0..3}
do
    echo "start random seed ${seed}......"
    rm ../data/raw_train_all_subset_kg_epoch_onecol.csv
    cp ../data/raw_train_all_subset10_kg_epoch_led_onecol.csv ../data/raw_train_all_subset_kg_epoch_onecol.csv
    for epoch in {0..1}
    do
        echo "start training Gen ${epoch}......"
        CUDA_LAUNCH_BLOCKING=1 python train_model.py -c $1 -train ${train_data} -dev ${dev_data} -test ${test_data} -kg ${kg_data} \
                              -g ${epoch} -s ${seed} -d 0.1 -d2 0.7 -clipgrad True -step 3  --earlystopping_step 5 -p 10
    done
done
###################################################################################################################
###################################################################################################################
###################################################################################################################

# sriram
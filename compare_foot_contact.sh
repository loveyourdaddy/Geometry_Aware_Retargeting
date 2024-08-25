# model
models=("240808_Gt1Fk100Anchor10_Root10Foot10_char01" "240424_FK100GtDist10_scaled")
epochs=("2000" "8000")

# character 
test_chars=("small" "fat")

# motion
motion0_s1=("greeting002_S1" "one_leg_back_stretch_S1")
motion1_s1=("greeting002_S2" "one_leg_back_stretch_S2")

for i in "${!models[@]}"; do
    model="${models[$i]}"
    epoch="${epochs[$i]}"
    echo "Processing model: $model with epoch: $epoch"

    for test_char in "${test_chars[@]}"; do
        for j in "${!motion0_s1[@]}"; do
            motion0="${motion0_s1[$j]}"
            motion1="${motion1_s1[$j]}"
            python test.py --test_proj $model --test_epoch $epoch --test_char $test_char --motion0 $motion0 --motion1 $motion1
        done
    done
done

# python test.py --test_proj model0 --test_epoch $epoch --test_char $test_model --motion0 $motion0 --motion1 $motion1
# python test.py --test_proj model0 --test_epoch $epoch --test_char $test_model --motion0 $motion0 --motion1 $motion1

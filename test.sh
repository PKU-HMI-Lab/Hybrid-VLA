export COPPELIASIM_ROOT=${HOME}/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
Xvfb :0 -screen 0 1024x768x24 &  
export DISPLAY=:0

export PYTHONPATH=<your_path>/Hybrid-VLA:$PYTHONPATH
python scripts/sim.py \
    --model-path '<your_model>' \
    --task-name 'close_box' \
    --exp-name '<exp_name>' \
    --replay-or-predict 'predict' \
    --cuda 0 \
    --use-diff 1 \
    --use-ar 1 \
    --threshold 5.8 \
    --max-steps 10 \
    --num-episodes 20 \
    --ddim-steps 4   # 4, 6, 8

# task names:
# "close_box" "close_laptop_lid" "toilet_seat_down" "sweep_to_dustpan" "close_fridge" "phone_on_base" "take_umbrella_out_of_umbrella_stand" "take_frame_off_hanger" "place_wine_at_rack_location" "water_plants"


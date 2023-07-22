python run_model.py --BSZ 3 --DATA_DIR ./data/predev/ train --EPCHS 1 --TRN_LAB_SET ./data/predev/train.pkl --VAL_LAB_SET ./data/predev/valid.pkl

python run_model.py --BSZ 3 --DATA_DIR ./data/predev/ test --TST_LAB_SET ./data/predev/test.pkl --SAVED_MOD ./checkpoints/mod0.pt

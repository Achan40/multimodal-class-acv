python run_model.py --BSZ 3 --DATA_DIR ./data/predev/ train --EPCHS 1 --TRN_LAB_SET ./data/predev/train.pkl --VAL_LAB_SET ./data/predev/valid.pkl

python run_model.py --BSZ 3 --DATA_DIR ./data/predev/ test --TST_LAB_SET ./data/predev/test.pkl --SAVED_MOD ./checkpoints/mod0.pt

python run_model.py --BSZ 16 --DATA_DIR ./data/actual/ train --EPCHS 2 --TRN_LAB_SET ./data/actual/train_0.pkl ./data/actual/train_2.pkl ./data/actual/train_3.pkl ./data/actual/train_4.pkl ./data/actual/train_5.pkl --VAL_LAB_SET ./data/actual/valid.pkl

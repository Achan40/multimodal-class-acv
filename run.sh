python -W ignore train_model.py --BSZ 3 --EPCHS 1 --DATA_DIR ./data/predev/ --TRN_LAB_SET ./data/predev/train.pkl --VAL_LAB_SET ./data/predev/valid.pkl


python -W ignore train_model.py --BSZ 3 --EPCHS 1 --DATA_DIR ./data/predev/ --TST_LAB_SET ./data/predev/test.pkl --SAVED_MOD ./checkpoints/mod0.pt

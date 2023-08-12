############# IRENE ###################
# run training loop (predev)
python run_model.py --BSZ 3 --DATA_DIR ./data/predev/ train --EPCHS 5 --TRN_LAB_SET ./data/predev/train.pkl --VAL_LAB_SET ./data/predev/valid.pkl

# run testing (predev)
python run_model.py --BSZ 3 --DATA_DIR ./data/predev/ test --TST_LAB_SET ./data/predev/test.pkl --SAVED_MOD ./checkpoints/set_0_epc_1.pt

# run full training loop
python run_model.py --BSZ 24 --DATA_DIR ./data/actual/ train --EPCHS 10 --TRN_LAB_SET ./data/actual/train_0.pkl ./data/actual/train_1.pkl ./data/actual/train_2.pkl ./data/actual/train_3.pkl ./data/actual/train_4.pkl ./data/actual/train_5.pkl ./data/actual/train_6.pkl ./data/actual/train_7.pkl ./data/actual/train_8.pkl ./data/actual/train_9.pkl ./data/actual/train_10.pkl ./data/actual/train_11.pkl ./data/actual/train_12.pkl ./data/actual/train_13.pkl ./data/actual/train_14.pkl ./data/actual/train_15.pkl ./data/actual/train_16.pkl ./data/actual/train_17.pkl ./data/actual/train_18.pkl ./data/actual/train_19.pkl --VAL_LAB_SET ./data/actual/valid.pkl

# run testing 
python run_model.py --BSZ 24 --DATA_DIR ./data/actual/ test --TST_LAB_SET ./data/actual/test_0.pkl --SAVED_MOD ./checkpoints/set_16_epc_1.pt

############ CNN #################

# CNN training (predev)
python run_model_cnn.py --BSZ 3 --DATA_DIR ./data/predev/ train --EPCHS 5 --TRN_LAB_SET ./data/predev/train.pkl --VAL_LAB_SET ./data/predev/valid.pkl

# CNN run testing (predev)
python run_model_cnn.py --BSZ 3 --DATA_DIR ./data/predev/ test --TST_LAB_SET ./data/predev/test.pkl --SAVED_MOD ./checkpoints/set_0_epc_1.pt

# CNN run full training loop
python run_model_cnn.py --BSZ 24 --DATA_DIR ./data/actual/ train --EPCHS 10 --TRN_LAB_SET ./data/actual/train_0.pkl ./data/actual/train_1.pkl ./data/actual/train_2.pkl ./data/actual/train_3.pkl ./data/actual/train_4.pkl ./data/actual/train_5.pkl ./data/actual/train_6.pkl ./data/actual/train_7.pkl ./data/actual/train_8.pkl ./data/actual/train_9.pkl ./data/actual/train_10.pkl ./data/actual/train_11.pkl ./data/actual/train_12.pkl ./data/actual/train_13.pkl ./data/actual/train_14.pkl ./data/actual/train_15.pkl ./data/actual/train_16.pkl ./data/actual/train_17.pkl ./data/actual/train_18.pkl ./data/actual/train_19.pkl --VAL_LAB_SET ./data/actual/valid.pkl

############### EXTRA ################
# continue training on a saved model with a specified offset
python run_model.py --BSZ 24 --DATA_DIR ./data/actual/ train --EPCHS 10 --SAVED_MOD ./checkpoints/set_0mod.pt --SAVED_MOD_OFFSET 5 --TRN_LAB_SET ./data/actual/train_0.pkl ./data/actual/train_1.pkl ./data/actual/train_2.pkl ./data/actual/train_3.pkl ./data/actual/train_4.pkl ./data/actual/train_5.pkl ./data/actual/train_6.pkl ./data/actual/train_7.pkl ./data/actual/train_8.pkl ./data/actual/train_9.pkl ./data/actual/train_10.pkl ./data/actual/train_11.pkl ./data/actual/train_12.pkl ./data/actual/train_13.pkl ./data/actual/train_14.pkl ./data/actual/train_15.pkl ./data/actual/train_16.pkl ./data/actual/train_17.pkl ./data/actual/train_18.pkl ./data/actual/train_19.pkl --VAL_LAB_SET ./data/actual/valid.pkl

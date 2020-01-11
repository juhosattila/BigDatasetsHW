from DataLoader import PascalDataLoader

train, valid, test = PascalDataLoader(minidataset=True).get_train_valid_test_iterators(
    img_target_size=(299, 299), batch_size=32
)




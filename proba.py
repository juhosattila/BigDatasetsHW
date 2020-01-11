from DataLoader import PascalDataLoader

image_dir, df = PascalDataLoader(minidataset=True).load()
print(df.head(10))


# base_dir = './Pascal_dir'
# tarfile_name = 'Pascal_dir.tar'
#
#
# def __load_tarfile(self):
#     # Check if data is already deleted by checking if inner path is present.
#     if os.path.exists(os.path.join(dataset_url, inner_path)):